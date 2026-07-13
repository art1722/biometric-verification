"""Shared MediaPipe **Tasks-API** Pose Landmarker.

Why this file exists
--------------------
The gait pipeline needs ONE pose detector that every walk check can share — the
same single-detect discipline `face_landmarker.py` established for the face
pipeline and `hand_landmarker.py` for the palm pipeline (build the model once,
pass it to each frame, never instantiate a model per call). This module is the
gait-side mirror of those files.

It uses the **Tasks API** (`mp.tasks.vision.PoseLandmarker`), not the legacy
`mp.solutions.pose` API, to stay consistent with the deliberate Tasks-API
migration already done for the face and hand landmarkers. The Tasks API returns,
per detected pose:

    - 33 pose landmarks in IMAGE space (x, y normalized to [0,1]; z = depth
      with the hips as origin; PLUS a per-landmark `visibility` in [0,1]),
    - 33 pose landmarks in WORLD space (meters, origin at the hip center).

The 33 landmarks follow the standard MediaPipe PoseLandmark order (NOSE=0,
LEFT_SHOULDER=11, RIGHT_SHOULDER=12, LEFT_HIP=23, RIGHT_HIP=24, LEFT_ANKLE=27,
RIGHT_ANKLE=28, ...). The named indices are exposed below as `PoseLandmark` so
the gait checks (person height, full-body visibility, walk direction) can read
specific keypoints by name instead of magic numbers.

The per-landmark `visibility` score is the key extra Pose gives that Hand does
NOT: it lets `check_full_body_visible` distinguish a landmark that is genuinely
in-frame from one MediaPipe merely GUESSED at off-screen. The face/palm results
have no equivalent, so this field is pose-specific.

What detect_pose() returns (the adapter shape)
----------------------------------------------
`detect_pose(...)` runs the model once and returns a small dataclass shaped to
match `HandResult`/`FaceResult` as closely as possible, so the gait pipeline
reads the same fields the palm pipeline does:

    PoseResult(
        ok:              bool,
        message:         str,
        landmarks_px:    list[(x, y, z)] | None,   # PIXEL space (int x/y, rel z)
        landmarks_norm:  <raw normalized landmark list> | None,  # .x/.y/.z/.visibility
        bbox:            (x, y, w, h) | None,       # px, 10% margin (same as hand)
        norm_box:        (x, y, w, h) | None,       # normalized
        visibility:      list[float] | None,        # per-landmark visibility [0,1]
        world_landmarks: <raw world landmark list> | None,  # meters
    )

`landmarks_px` matches the hand/face contract: int pixel x/y plus relative z, so
the body bbox can be derived the same way and `check_brightness_walk` can reuse
`check_brightness(image, bbox, ...)` exactly like `check_brightness_palm`.
`landmarks_norm` exposes objects with `.x/.y/.z/.visibility` for normalized-space
math (person-height ratio, centroid traversal). `world_landmarks` is available
for any metric 3D reasoning, mirroring the hand world-landmark field.

There is deliberately NO handedness field (that is hand-specific); the pose
equivalent — which way the person faces / walks — is derived downstream by the
gait direction check from the landmark trajectory, not reported here.

Model bundle
------------
The Tasks API loads a `.task` bundle from disk. Path comes from config:

    models.pose_landmarker.model_path   (default: models/pose_landmarker.task)

Download once (the canonical Google-hosted bundle; `_lite` / `_full` / `_heavy`
trade speed for accuracy — pick per the 1,500-volunteer runtime budget):
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
`verify_env.py` should check the file is present and loadable before any run
(same as it does for the face and hand bundles).
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

# Same 10% bbox margin face_landmarker / hand_landmarker use, so the body bbox
# handed to check_brightness_walk is built in the identical spirit (landmark
# extent + 10% padding).
_BBOX_MARGIN_RATIO = 0.1

# Default model location, relative to the project root (where config.yml and
# run_face.py live). Overridable via config models.pose_landmarker.model_path.
DEFAULT_MODEL_PATH = os.path.join("models", "pose_landmarker.task")


class PoseLandmark(enum.IntEnum):
    """The 33 MediaPipe pose landmark indices, exposed by name so the gait
    checks can read specific keypoints without magic numbers."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


@dataclass
class PoseResult:
    """One frame's pose detection, carrying every representation the gait
    checks consume so each check changes minimally. Shaped to mirror
    HandResult (hand_landmarker.py); the pose-specific addition is the
    per-landmark `visibility` list."""
    ok: bool
    message: str
    landmarks_px: Optional[list] = None        # [(x, y, z)] pixel space
    landmarks_norm: Optional[Any] = None       # raw normalized landmark list (.x/.y/.z/.visibility)
    bbox: Optional[tuple] = None               # (x, y, w, h) px, 10% margin
    norm_box: Optional[tuple] = None           # (x, y, w, h) normalized
    visibility: Optional[list] = None          # per-landmark visibility [0,1]
    world_landmarks: Optional[Any] = None      # raw world landmark list (.x/.y/.z, meters)


def create_pose_landmarker(
    model_path: str = DEFAULT_MODEL_PATH,
    *,
    num_poses: int = 1,
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """Create ONE Tasks-API PoseLandmarker.

    Mirrors the shared-detector pattern: the pipeline builds this once and
    passes it to every frame. Settings map onto config keys in
    models.pose_landmarker:
      num_poses                     <- (gait protocol has one walker; default 1)
      min_pose_detection_confidence <- models.pose_landmarker.min_detection_confidence
      min_pose_presence_confidence  <- models.pose_landmarker.min_pose_presence_confidence
      min_tracking_confidence       <- models.pose_landmarker.min_tracking_confidence

    running_mode=IMAGE: the gait pipeline samples frames and detects each one
    independently (the same per-frame still discipline the face pipeline uses on
    sampled frames — no cross-frame tracking is exploited inside the detector;
    the walk-direction check does its own temporal reasoning over the timeline).

    Raises FileNotFoundError with a fix hint if the .task bundle is missing, so
    the failure is a clear one-line message, not a deep MediaPipe stack trace
    (same behaviour as create_hand_landmarker / create_face_landmarker).
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Pose Landmarker model bundle not found: {model_path}\n"
            f"Download it once with:\n"
            f"  mkdir -p {os.path.dirname(model_path) or '.'}\n"
            f"  curl -L -o {model_path} \\\n"
            f"    https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            f"pose_landmarker_full/float16/1/pose_landmarker_full.task"
        )

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Load the .task bundle as BYTES and pass model_asset_buffer, NOT
    # model_asset_path — the SAME Windows fix face_landmarker.py / hand_landmarker.py
    # document: model_asset_path re-resolves the path relative to MediaPipe's own
    # site-packages dir, so an absolute Windows path fails with errno=22. Reading
    # the bytes ourselves sidesteps MediaPipe's path handling and works
    # identically on Windows, Linux, and macOS.
    with open(model_path, "rb") as f:
        model_bytes = f.read()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        # output_segmentation_masks left at its False default — the gait checks
        # need landmarks + visibility, not a silhouette mask.
    )
    return PoseLandmarker.create_from_options(options)


def detect_pose(
    image: Any,
    *,
    detector,
    input_color_space: Literal["BGR", "RGB"] = "BGR",
    require_single_pose: bool = True,
) -> PoseResult:
    """Run the shared PoseLandmarker once on one image.

    Args:
        image: file path (str) OR a decoded image array.
        detector: a PoseLandmarker from create_pose_landmarker (REQUIRED — like
            detect_hand, there is no make-your-own fallback, because building a
            Tasks detector means loading the .task bundle from disk every call,
            which is far too slow per frame).
        input_color_space: "BGR" (OpenCV default) or "RGB" (already-RGB array).
        require_single_pose: if True, >1 detected pose -> ok=False with a
            "Multiple poses" message (same contract shape as detect_hand's
            multiple-hands case; the pipeline maps that to FAIL). The gait
            protocol films one walker, so a second person in frame is a defect.

    Returns:
        PoseResult — see module docstring.
    """
    # --- resolve input to a BGR array (array-first, like detect_hand) ---
    if isinstance(image, str):
        frame = cv2.imread(image)
        if frame is None:
            return PoseResult(False, "Failed to load image")
    else:
        frame = image

    if frame is None or getattr(frame, "size", 0) == 0:
        return PoseResult(False, "Empty or invalid image")

    height, width = frame.shape[:2]

    # MediaPipe Tasks wants an SRGB mp.Image. Convert to RGB if needed.
    if input_color_space == "BGR":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif input_color_space == "RGB":
        rgb = frame
    else:
        return PoseResult(False, f"Unsupported color space: {input_color_space}")

    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb))
        result = detector.detect(mp_image)
    except Exception as e:  # keep the no-crash contract of detect_hand
        logger.debug("POSE_LANDMARKER | detect error: %s", e)
        return PoseResult(False, f"Error during pose detection: {e}")

    poses = result.pose_landmarks or []
    if not poses:
        return PoseResult(False, "No poses detected")

    if require_single_pose and len(poses) > 1:
        return PoseResult(False, f"Multiple poses detected: {len(poses)}")

    # --- first pose -> the representations downstream needs ---
    norm_landmarks = poses[0]   # list of NormalizedLandmark (.x/.y/.z/.visibility)

    landmarks_px = [
        (int(lm.x * width), int(lm.y * height), lm.z) for lm in norm_landmarks
    ]
    visibility = [getattr(lm, "visibility", None) for lm in norm_landmarks]

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

    # --- world landmarks for THIS (first) pose, if present ---
    world_landmarks = None
    if getattr(result, "pose_world_landmarks", None):
        world_landmarks = result.pose_world_landmarks[0]

    return PoseResult(
        ok=True,
        message="Pose detected successfully",
        landmarks_px=landmarks_px,
        landmarks_norm=norm_landmarks,
        bbox=bbox,
        norm_box=norm_box,
        visibility=visibility,
        world_landmarks=world_landmarks,
    )