"""Shared MediaPipe **Tasks-API** Face Landmarker.

Why this file exists
--------------------
The project originally used the LEGACY solutions API
(`mp.solutions.face_mesh.FaceMesh`). That API returns landmarks ONLY. The
eyes-open check was therefore a geometric Eye-Aspect-Ratio (EAR) over six
landmarks per eye — a ratio that is fragile: its scale depends on the face
mesh, drifts with head pose, and needed per-video threshold calibration
(see the old config note: repo 0.37 sliced through video 002's open-eye EAR).

We were advised NOT to keep the landmark-ratio method and to use a model
instead. MediaPipe's own answer is the **Face Landmarker Tasks API** with
`output_face_blendshapes=True`: the same model that emits the mesh ALSO emits
52 blendshape coefficients, two of which — ``eyeBlinkLeft`` and
``eyeBlinkRight`` — are a trained closed-ness score per eye. That is the new
eyes-open signal (see check_eye.py).

Blendshapes are NOT available from the legacy solutions API, so this is a
genuine API migration, not a drop-in. To avoid running two face models per
frame, this ONE detector now feeds every face check:

    get_lm            (landmarks + bbox)      -> face_size, head_fully, ...
    estimate_head_pose(raw normalized lm)     -> turn classification
    check_eye_status  (blendshapes)           -> eyes open / closed

One model, one inference per frame, three consumers — same single-detect
discipline the pipeline always had, just on the newer API.

What detect() returns (the adapter shape)
-----------------------------------------
`detect_face(...)` runs the model once and returns a small dataclass that
carries BOTH representations the downstream checks already expect, so those
checks change as little as possible:

    FaceResult(
        ok:            bool,
        message:       str,
        landmarks_px:  list[(x, y, z)] | None,   # PIXEL space — get_lm contract
        landmarks_norm: <raw normalized landmark list> | None,  # head-pose contract
        bbox:          (x, y, w, h) | None,
        norm_box:      (x, y, w, h) | None,
        blendshapes:   dict[str, float] | None,  # name -> score, e.g. eyeBlinkLeft
    )

`landmarks_px` reproduces exactly what the old get_lm returned (int pixel x/y,
relative z), so check_face_size / check_head_fully / the old EAR all still read
the same tuples. `landmarks_norm` exposes objects with `.x/.y/.z` so the
head-pose math (which reads normalized coords) is untouched.

Model bundle
------------
The Tasks API loads a `.task` model bundle from disk (the legacy API bundled
its model inside the wheel). Path comes from config:

    models.face_landmarker.model_path   (default: models/face_landmarker.task)

Download once (the canonical Google-hosted bundle):
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
`verify_env.py` checks the file is present and loadable before any run.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Literal, Optional

import cv2
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)

# Same 10% bbox margin the old get_lm used, so the bbox handed to face_size is
# identical in spirit to before.
_BBOX_MARGIN_RATIO = 0.1

# Default model location, relative to the project root (where config.yml and
# run_face.py live). Overridable via config models.face_landmarker.model_path.
DEFAULT_MODEL_PATH = os.path.join("models", "face_landmarker.task")


@dataclass
class FaceResult:
    """One frame's face detection, carrying every representation the face
    checks consume so each check changes minimally."""
    ok: bool
    message: str
    landmarks_px: Optional[list] = None      # [(x, y, z)] pixel space
    landmarks_norm: Optional[Any] = None     # raw normalized landmark list (.x/.y/.z)
    bbox: Optional[tuple] = None             # (x, y, w, h) px, 10% margin
    norm_box: Optional[tuple] = None         # (x, y, w, h) normalized
    blendshapes: Optional[dict] = None       # {name: score}


def create_face_landmarker(
    model_path: str = DEFAULT_MODEL_PATH,
    *,
    num_faces: int = 10,
    min_face_detection_confidence: float = 0.6,
    min_face_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """Create ONE Tasks-API FaceLandmarker, blendshapes enabled.

    Mirrors the old shared-detector pattern: the pipeline builds this once and
    passes it to every frame. Settings map onto config keys:
      num_faces                     <- models.face_landmarker.num_faces
      min_face_detection_confidence <- models.face_landmarker.min_face_detection_confidence

    static_image_mode is expressed here as running_mode=IMAGE (each frame is
    treated independently — the previous behaviour with static_image_mode=True).

    Raises FileNotFoundError with a fix hint if the .task bundle is missing, so
    the failure is a clear one-line message, not a deep MediaPipe stack trace.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Face Landmarker model bundle not found: {model_path}\n"
            f"Download it once with:\n"
            f"  mkdir -p {os.path.dirname(model_path) or '.'}\n"
            f"  curl -L -o {model_path} \\\n"
            f"    https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            f"face_landmarker/float16/1/face_landmarker.task"
        )

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Load the .task bundle as BYTES and pass model_asset_buffer, NOT
    # model_asset_path. On Windows, model_asset_path re-resolves the given path
    # relative to MediaPipe's own site-packages dir, so an ABSOLUTE path like
    # 'C:\proj\models\face_landmarker.task' becomes the impossible
    # 'site-packages/C:\proj\...' and fails with errno=22 (EINVAL). Reading the
    # bytes ourselves sidesteps MediaPipe's path handling entirely and works the
    # same on Windows, Linux, and macOS.
    with open(model_path, "rb") as f:
        model_bytes = f.read()

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=num_faces,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_face_blendshapes=True,          # <-- the whole point of the migration
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


def detect_face(
    image: Any,
    *,
    detector,
    input_color_space: Literal["BGR", "RGB"] = "BGR",
    require_single_face: bool = True,
) -> FaceResult:
    """Run the shared FaceLandmarker once on one frame.

    Args:
        image: file path (str) OR a decoded image array (a video frame).
        detector: a FaceLandmarker from create_face_landmarker (REQUIRED — unlike
            the old get_lm there is no make-your-own fallback, because building a
            Tasks detector means loading the .task bundle from disk every call,
            which is far too slow per frame).
        input_color_space: "BGR" (OpenCV default) or "RGB" (already-RGB frame).
        require_single_face: if True, >1 detected face -> ok=False with a
            "Multiple faces" message (same contract as the old get_lm; the
            pipeline maps that to FAIL).

    Returns:
        FaceResult — see module docstring.
    """
    # --- resolve input to a BGR array (array-first, like the old get_lm) ---
    if isinstance(image, str):
        frame = cv2.imread(image)
        if frame is None:
            return FaceResult(False, "Failed to load image")
    else:
        frame = image

    if frame is None or getattr(frame, "size", 0) == 0:
        return FaceResult(False, "Empty or invalid image")

    height, width = frame.shape[:2]

    # MediaPipe Tasks wants an SRGB mp.Image. Convert to RGB if needed.
    if input_color_space == "BGR":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif input_color_space == "RGB":
        rgb = frame
    else:
        return FaceResult(False, f"Unsupported color space: {input_color_space}")

    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb))
        result = detector.detect(mp_image)
    except Exception as e:  # keep the no-crash contract of the old get_lm
        logger.debug("FACE_LANDMARKER | detect error: %s", e)
        return FaceResult(False, f"Error during face detection: {e}")

    faces = result.face_landmarks or []
    if not faces:
        return FaceResult(False, "No faces detected")

    if require_single_face and len(faces) > 1:
        return FaceResult(False, f"Multiple faces detected: {len(faces)}")

    # --- first face -> the two landmark representations downstream needs ---
    norm_landmarks = faces[0]   # list of NormalizedLandmark (.x/.y/.z), head-pose reads this

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

    # --- blendshapes: category_name -> score, for THIS (first) face ---
    blendshapes = None
    if result.face_blendshapes:
        blendshapes = {
            c.category_name: c.score for c in result.face_blendshapes[0]
        }

    return FaceResult(
        ok=True,
        message="Face detected successfully",
        landmarks_px=landmarks_px,
        landmarks_norm=norm_landmarks,
        bbox=bbox,
        norm_box=norm_box,
        blendshapes=blendshapes,
    )