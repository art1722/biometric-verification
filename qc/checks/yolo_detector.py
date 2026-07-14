"""YOLO (COCO) object detector wrapper for the occlusion / foreign-object check.

Shaped to mirror ``pose_landmarker.py``: a ``create_yolo_detector(config)`` that
loads the model ONCE, and a ``detect_objects(detector, frame, ...)`` that runs it
on a single frame and returns a small, framework-agnostic result. The caller
(the walk/face pipeline) builds one detector and reuses it across every frame,
exactly like the pose detector, so the model is never reloaded per frame.

Why a wrapper (not raw ultralytics in the pipeline)
---------------------------------------------------
- Keeps ``ultralytics`` a SOFT dependency: it is imported lazily inside
  ``create_yolo_detector`` so importing this module (or the whole qc package)
  never crashes on a machine that has not installed it. The import error is
  raised only if someone actually asks for the detector, with a clear message.
- Returns plain Python (list of ``Detection``) so the check function has no
  ultralytics types leaking into it and is trivial to unit-test.

COCO classes
------------
The pretrained model already knows all 80 COCO classes (0=person ... 79=
toothbrush). We do NOT train anything and do NOT touch the COCO dataset; we only
read back the class ids the model predicts. ``model.names`` maps id -> name.

Model bundle
------------
Weights path comes from config:

    models.yolo.model_path   (default: models/yolov8n.pt)

The ``.pt`` auto-downloads on first use if ultralytics can reach the network,
but for the 1,500-volunteer offline runs the file should be vendored next to the
other model bundles and ``verify_env.py`` should check it is present, same as the
pose/face/hand bundles.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Default weights location, relative to the project root (where config.yml and
# the runners live). Overridable via config models.yolo.model_path.
DEFAULT_MODEL_PATH = os.path.join("models", "yolov8n.pt")

# The COCO id for the ONLY allowed class. Everything else (1..79) is a foreign
# object for occlusion purposes. Kept as a name-independent constant so the check
# never hard-codes the string "person".
PERSON_CLASS_ID = 0


@dataclass
class Detection:
    """One detected object on one frame, framework-agnostic.

    cls_id:   COCO class id (0=person ... 79=toothbrush).
    cls_name: human-readable class name, from model.names.
    conf:     detection confidence in [0, 1].
    bbox:     (x, y, w, h) in PIXEL coords, so it matches the rest of the
              pipeline's bbox convention (pose bbox, brightness bbox).
    """
    cls_id: int
    cls_name: str
    conf: float
    bbox: tuple


class _YoloDetector:
    """Thin holder for the loaded model + its class-name map + default conf.

    Passed around as an opaque ``detector`` handle, mirroring how the pose
    ``PoseLandmarker`` object is created once and reused.
    """

    def __init__(self, model: Any, names: dict, default_conf: float):
        self.model = model
        self.names = names
        self.default_conf = default_conf


# Project root = two levels up from this file (<root>/qc/checks/yolo_detector.py).
# Used to keep ALL ultralytics state inside the project instead of the user home.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _pin_ultralytics_into_project() -> None:
    """Keep ultralytics' config/cache INSIDE the project, on every machine.

    By default ultralytics writes settings.json + cache to the user home
    (~/.config/Ultralytics on Linux, %APPDATA%\\Ultralytics on Windows) and has
    telemetry ON. For a program shipped to other computers / Docker, that state
    must be deterministic and self-contained, so we redirect it to
    <project>/.ultralytics and disable analytics.

    MUST run BEFORE `import ultralytics`: the library reads YOLO_CONFIG_DIR at
    import time. We use setdefault so an explicit env var (e.g. the Dockerfile's
    /app/.ultralytics) always wins over this fallback. Idempotent.
    """
    cfg_dir = os.path.join(_PROJECT_ROOT, ".ultralytics")
    try:
        os.makedirs(cfg_dir, exist_ok=True)
    except OSError:
        # If the project dir is not writable (rare), leave ultralytics to its
        # own default rather than crash the whole run.
        return
    os.environ.setdefault("YOLO_CONFIG_DIR", cfg_dir)
    os.environ.setdefault("ULTRALYTICS_ANALYTICS", "False")


def create_yolo_detector(config: Optional[dict] = None) -> _YoloDetector:
    """Load the YOLO model once and return a reusable detector handle.

    ultralytics is imported HERE (lazily) so the qc package imports cleanly even
    when ultralytics is not installed; the ImportError only fires if a run
    actually needs object detection.

    Config keys (all optional, sensible defaults):
        models.yolo.model_path : path to the .pt weights (default yolov8n.pt).
        models.yolo.conf       : default confidence gate (default 0.35). The
            check can still override this per call.
    """
    cfg = ((config or {}).get("models", {}) or {}).get("yolo", {}) or {}
    model_path = cfg.get("model_path", DEFAULT_MODEL_PATH)
    default_conf = float(cfg.get("conf", 0.35))

    # Pin ultralytics state into the project BEFORE importing it.
    _pin_ultralytics_into_project()

    try:
        from ultralytics import YOLO
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "ultralytics is required for the YOLO occlusion check but is not "
            "installed. Install it with `pip install ultralytics`, or disable "
            "the occlusion check for this run."
        ) from e

    # YOLO() accepts a local path OR a bare name it will auto-download. We do not
    # assert existence here: a bare name like "yolov8n.pt" is valid and triggers
    # the ultralytics download path. verify_env.py is the right place to enforce
    # a vendored file for offline batch runs.
    model = YOLO(model_path)
    names = dict(model.names)  # {0: 'person', 1: 'bicycle', ...}
    logger.info("YOLO detector loaded from %s (%d classes, conf=%.2f)",
                model_path, len(names), default_conf)
    return _YoloDetector(model, names, default_conf)


def detect_objects(
    detector: _YoloDetector,
    frame_bgr: Any,
    *,
    conf: Optional[float] = None,
) -> List[Detection]:
    """Run the detector on ONE BGR frame; return a list of Detection.

    Args:
        detector: handle from create_yolo_detector.
        frame_bgr: the frame as a BGR numpy array (OpenCV default), matching how
            the rest of the pipeline carries frames.
        conf: confidence gate for THIS call. If None, uses the detector's
            default_conf. Detections below conf are dropped by ultralytics, so
            they never reach the check.

    Returns:
        list[Detection] for every object at or above the confidence gate,
        including any person(s). The caller decides which classes matter; this
        function does not filter by class, so it stays reusable.
    """
    use_conf = detector.default_conf if conf is None else float(conf)

    # verbose=False keeps ultralytics quiet in the per-frame loop.
    results = detector.model(frame_bgr, conf=use_conf, verbose=False)
    if not results:
        return []

    r = results[0]
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    out: List[Detection] = []
    # .xywh is center-x, center-y, w, h; we want top-left x/y to match the
    # (x, y, w, h) top-left convention used elsewhere, so convert.
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        c = float(boxes.conf[i].item())
        cx, cy, w, h = (float(v) for v in boxes.xywh[i].tolist())
        x = cx - w / 2.0
        y = cy - h / 2.0
        out.append(Detection(
            cls_id=cls_id,
            cls_name=detector.names.get(cls_id, str(cls_id)),
            conf=c,
            bbox=(x, y, w, h),
        ))
    return out


def close_yolo_detector(detector: Any) -> None:
    """Symmetry with the pose detector's close path. ultralytics has no explicit
    handle to release, so this is a no-op kept so the pipeline's _maybe_close
    logic can treat both detectors identically."""
    return None