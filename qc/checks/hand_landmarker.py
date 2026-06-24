"""Shared MediaPipe **Tasks-API** Hand Landmarker.

Why this file exists
--------------------
The palm pipeline needs ONE hand detector that every palm check can share —
the same single-detect discipline `face_landmarker.py` established for the face
pipeline (build the model once, pass it to each image, never instantiate a
model per call). This module is the hand-side mirror of that file.

It uses the **Tasks API** (`mp.tasks.vision.HandLandmarker`), not the legacy
`mp.solutions.hands` API, to stay consistent with the deliberate Tasks-API
migration already done for the face landmarker. The Tasks API returns, per
detected hand:

    - 21 hand landmarks in IMAGE space (x, y normalized to [0,1]; z = depth
      with the wrist as origin),
    - handedness (a "Left"/"Right" label + score),
    - 21 hand landmarks in WORLD space (meters, origin at the hand's center).

The 21 landmarks follow the standard MediaPipe HandLandmark order (WRIST=0,
THUMB_TIP=4, INDEX_FINGER_TIP=8, MIDDLE_FINGER_TIP=12, RING_FINGER_TIP=16,
PINKY_TIP=20, ...). The named indices are exposed below as `HandLandmark` so the
palm checks (size, finger spread, angle) can read specific keypoints by name
instead of magic numbers.

What detect_hand() returns (the adapter shape)
----------------------------------------------
`detect_hand(...)` runs the model once and returns a small dataclass shaped to
match `FaceResult` as closely as possible, so the palm pipeline reads the same
fields the face pipeline does:

    HandResult(
        ok:             bool,
        message:        str,
        landmarks_px:   list[(x, y, z)] | None,   # PIXEL space (int x/y, rel z)
        landmarks_norm: <raw normalized landmark list> | None,  # .x/.y/.z in [0,1]
        bbox:           (x, y, w, h) | None,       # px, 10% margin (same as face)
        norm_box:       (x, y, w, h) | None,       # normalized
        handedness:     str | None,                # "Left" / "Right"
        handedness_score: float | None,            # confidence of that label
        world_landmarks: <raw world landmark list> | None,  # meters, for angle math
    )

`landmarks_px` matches the face contract: int pixel x/y plus relative z, so a
bbox can be derived the same way and the size check reuses the same tuple shape.
`landmarks_norm` exposes objects with `.x/.y/.z` for any normalized-space math.
`world_landmarks` is hand-specific: real-world 3D coords are the cleanest signal
for the roll/pitch angle check (image-space coords distort with perspective).

Model bundle
------------
The Tasks API loads a `.task` bundle from disk. Path comes from config:

    models.hand_landmarker.model_path   (default: models/hand_landmarker.task)

Download once (the canonical Google-hosted bundle):
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
`verify_env.py` should check the file is present and loadable before any run
(same as it does for the face bundle).
"""

from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal, Optional

import cv2
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)

# Same 10% bbox margin face_landmarker uses, so the palm bbox handed to the size
# check is built in the identical spirit (landmark extent + 10% padding).
_BBOX_MARGIN_RATIO = 0.1

# Default model location, relative to the project root (where config.yml and
# run_face.py live). Overridable via config models.hand_landmarker.model_path.
DEFAULT_MODEL_PATH = os.path.join("models", "hand_landmarker.task")


class HandLandmark(enum.IntEnum):
    """The 21 MediaPipe hand landmark indices, exposed by name so the palm
    checks can read specific keypoints without magic numbers."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass
class HandResult:
    """One image's hand detection, carrying every representation the palm
    checks consume so each check changes minimally. Shaped to mirror
    FaceResult (face_landmarker.py)."""
    ok: bool
    message: str
    landmarks_px: Optional[list] = None        # [(x, y, z)] pixel space
    landmarks_norm: Optional[Any] = None       # raw normalized landmark list (.x/.y/.z)
    bbox: Optional[tuple] = None               # (x, y, w, h) px, 10% margin
    norm_box: Optional[tuple] = None           # (x, y, w, h) normalized
    handedness: Optional[str] = None           # "Left" / "Right"
    handedness_score: Optional[float] = None   # confidence of the handedness label
    world_landmarks: Optional[Any] = None      # raw world landmark list (.x/.y/.z, meters)


def create_hand_landmarker(
    model_path: str = DEFAULT_MODEL_PATH,
    *,
    num_hands: int = 1,
    min_hand_detection_confidence: float = 0.5,
    min_hand_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """Create ONE Tasks-API HandLandmarker.

    Mirrors the shared-detector pattern: the pipeline builds this once and
    passes it to every image. Settings map onto the existing config keys in
    models.hands:
      num_hands                     <- models.hands.max_num_hands (default 1)
      min_hand_detection_confidence <- models.hands.min_detection_confidence

    running_mode=IMAGE: each palm photo is an independent still (there is no
    video tracking to exploit), which is exactly the palm pipeline's input.

    Raises FileNotFoundError with a fix hint if the .task bundle is missing, so
    the failure is a clear one-line message, not a deep MediaPipe stack trace
    (same behaviour as create_face_landmarker).
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Hand Landmarker model bundle not found: {model_path}\n"
            f"Download it once with:\n"
            f"  mkdir -p {os.path.dirname(model_path) or '.'}\n"
            f"  curl -L -o {model_path} \\\n"
            f"    https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            f"hand_landmarker/float16/1/hand_landmarker.task"
        )

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Load the .task bundle as BYTES and pass model_asset_buffer, NOT
    # model_asset_path — the SAME Windows fix face_landmarker.py documents:
    # model_asset_path re-resolves the path relative to MediaPipe's own
    # site-packages dir, so an absolute Windows path fails with errno=22.
    # Reading the bytes ourselves sidesteps MediaPipe's path handling and works
    # identically on Windows, Linux, and macOS.
    with open(model_path, "rb") as f:
        model_bytes = f.read()
        
    """
    The question is that we should evaluate whether the option to extract the per-point confidence score to evalute the face occlusion is the right method to implement by now.
    VisionRunningMode: IMAGE, VIDEO, LIVE_STREAM
    """

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return HandLandmarker.create_from_options(options)


def detect_hand(
    image: Any,
    *,
    detector,
    input_color_space: Literal["BGR", "RGB"] = "BGR",
    require_single_hand: bool = True,
) -> HandResult:
    """Run the shared HandLandmarker once on one image.

    Args:
        image: file path (str) OR a decoded image array.
        detector: a HandLandmarker from create_hand_landmarker (REQUIRED — like
            detect_face, there is no make-your-own fallback, because building a
            Tasks detector means loading the .task bundle from disk every call,
            which is far too slow per image).
        input_color_space: "BGR" (OpenCV default) or "RGB" (already-RGB array).
        require_single_hand: if True, >1 detected hand -> ok=False with a
            "Multiple hands" message (same contract shape as detect_face's
            multiple-faces case; the pipeline maps that to FAIL).

    Returns:
        HandResult — see module docstring.

    Handedness note: MediaPipe assigns Left/Right assuming the image is mirrored
    (selfie/front-camera, flipped horizontally). The palm spec encodes the hand
    in the FILENAME (..._palm_L_... / ..._palm_R_...), so the pipeline should
    trust the filename as ground truth and use this handedness only as a sanity
    signal, NOT swap files based on it.
    """
    # --- resolve input to a BGR array (array-first, like detect_face) ---
    if isinstance(image, str):
        frame = cv2.imread(image)
        if frame is None:
            return HandResult(False, "Failed to load image")
    else:
        frame = image

    if frame is None or getattr(frame, "size", 0) == 0:
        return HandResult(False, "Empty or invalid image")

    height, width = frame.shape[:2]

    # MediaPipe Tasks wants an SRGB mp.Image. Convert to RGB if needed.
    if input_color_space == "BGR":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif input_color_space == "RGB":
        rgb = frame
    else:
        return HandResult(False, f"Unsupported color space: {input_color_space}")

    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb))
        result = detector.detect(mp_image)
    except Exception as e:  # keep the no-crash contract of detect_face
        logger.debug("HAND_LANDMARKER | detect error: %s", e)
        return HandResult(False, f"Error during hand detection: {e}")

    hands = result.hand_landmarks or []
    if not hands:
        return HandResult(False, "No hands detected")

    if require_single_hand and len(hands) > 1:
        return HandResult(False, f"Multiple hands detected: {len(hands)}")

    # --- first hand -> the representations downstream needs ---
    norm_landmarks = hands[0]   # list of NormalizedLandmark (.x/.y/.z in [0,1])

    landmarks_px = [
        (int(lm.x * width), int(lm.y * height), lm.z) for lm in norm_landmarks
    ]

    xs = [p[0] for p in landmarks_px]
    ys = [p[1] for p in landmarks_px]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w, h = x_max - x_min, y_max - y_min

    mx = int(w * _BBOX_MARGIN_RATIO)
    my = int(h * _BBOX_MARGIN_RATIO)
    x_min = max(0, x_min - mx)
    y_min = max(0, y_min - my)
    x_max = min(width, x_max + mx)
    y_max = min(height, y_max + my)
    w, h = x_max - x_min, y_max - y_min

    bbox = (x_min, y_min, w, h)
    norm_box = (x_min / width, y_min / height, w / width, h / height)

    # --- handedness for THIS (first) hand: label + score ---
    handedness = None
    handedness_score = None
    if result.handedness and result.handedness[0]:
        top = result.handedness[0][0]   # highest-scoring category for hand 0
        handedness = top.category_name
        handedness_score = top.score

    # --- world landmarks for THIS (first) hand, if present ---
    world_landmarks = None
    if getattr(result, "hand_world_landmarks", None):
        world_landmarks = result.hand_world_landmarks[0]

    return HandResult(
        ok=True,
        message="Hand detected successfully",
        landmarks_px=landmarks_px,
        landmarks_norm=norm_landmarks,
        bbox=bbox,
        norm_box=norm_box,
        handedness=handedness,
        handedness_score=handedness_score,
        world_landmarks=world_landmarks,
    )