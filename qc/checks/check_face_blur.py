"""Face blur / sharpness check.

Why this is not raw Laplacian anymore
-------------------------------------
The old repo-style metric used ``variance(Laplacian(gray))`` with a fixed
threshold. That is fragile for this project because the score changes with
camera processing, exposure, face crop size, compression, dark lighting, and
low-contrast facial texture. In practice it false-fails many usable frames.

This version is a more stable QC signal:
  1. Reuse the already-detected face bbox when available; do not re-detect.
  2. Crop the inner face region so background/hair edges do not dominate.
  3. Resize to a fixed size so score is less dependent on face size in pixels.
  4. Use the luminance channel and CLAHE contrast normalization.
  5. Score sharpness with Tenengrad/Sobel gradient energy.
  6. Return None when blur should not be judged, e.g. too dark / too low
     contrast. The pipeline maps None -> SKIP, not FAIL.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import cv2
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)


def _to_bgr(image: Any, input_color_space: str):
    """Resolve input to a BGR array."""
    if isinstance(image, str):
        return cv2.imread(image)
    if input_color_space == "RGB":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def _detect_bbox_with_legacy_detector(img_bgr: np.ndarray, detector: Optional[Any]):
    """Fallback only for standalone use. The face pipeline should pass bbox."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    owns = detector is None
    if owns:
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        )
    try:
        results = detector.process(img_rgb)
    finally:
        if owns:
            detector.close()

    if not results.detections:
        return None

    rb = results.detections[0].location_data.relative_bounding_box
    x = max(0, int(rb.xmin * w))
    y = max(0, int(rb.ymin * h))
    bw = max(0, int(rb.width * w))
    bh = max(0, int(rb.height * h))
    return (x, y, bw, bh)


def _inner_crop_from_bbox(
    img_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    crop_margin: float,
):
    """Return a central face crop from (x, y, w, h), clamped to image bounds."""
    h_img, w_img = img_bgr.shape[:2]
    x, y, bw, bh = bbox

    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(w_img, int(round(x + bw)))
    y1 = min(h_img, int(round(y + bh)))

    if x1 <= x0 or y1 <= y0:
        return None

    # Trim inside the face bbox. This avoids hair/background edges dominating
    # the sharpness score while keeping enough eyes/nose/mouth texture.
    m = max(0.0, min(0.35, float(crop_margin)))
    dx = int(round((x1 - x0) * m))
    dy = int(round((y1 - y0) * m))
    x0 += dx
    x1 -= dx
    y0 += dy
    y1 -= dy

    if x1 <= x0 or y1 <= y0:
        return None

    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop


def check_face_blur(
    image: Any,
    threshold: float = 35.0,
    *,
    bbox: Optional[tuple[int, int, int, int]] = None,
    detector: Optional[Any] = None,
    input_color_space: Literal["BGR", "RGB"] = "BGR",
    resize_px: int = 224,
    crop_margin: float = 0.15,
    min_luma: float = 35.0,
    min_contrast: float = 4.0,
):
    """Check whether the face region is sharp enough.

    Args:
        image: path or decoded frame.
        threshold: Tenengrad/Sobel sharpness threshold. Higher = stricter.
        bbox: preferred; (x, y, w, h) from the shared face detector.
        detector: fallback legacy face detector only when bbox is None.
        input_color_space: BGR or RGB.
        resize_px: fixed crop size before scoring.
        crop_margin: fraction trimmed from each side of bbox.
        min_luma: below this, blur is not judged; brightness should fail it.
        min_contrast: below this, blur is not judged because edge evidence is
            insufficient.

    Returns:
        (True, msg)  = sharp enough
        (False, msg) = judged blurry
        (None, msg)  = not judged; caller should map to SKIP
    """
    if threshold <= 0:
        return (False, "invalid blur threshold; must be positive")

    img = _to_bgr(image, input_color_space)
    if img is None or getattr(img, "size", 0) == 0:
        return (None, "blur not judged: invalid image")

    if bbox is None:
        bbox = _detect_bbox_with_legacy_detector(img, detector)
        if bbox is None:
            return (None, "blur not judged: no face bbox")

    face = _inner_crop_from_bbox(img, bbox, crop_margin=crop_margin)
    if face is None:
        return (None, f"blur not judged: invalid face crop bbox={bbox}")

    # Use luminance, not raw BGR/RGB. This is the channel relevant to edge detail.
    ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]

    mean_luma = float(np.mean(y))
    contrast = float(np.std(y))

    # Do not turn exposure/contrast problems into blur failures. The brightness
    # check owns too-dark/too-bright decisions.
    if mean_luma < min_luma:
        return (
            None,
            f"blur not judged: face too dark; "
            f"luma={mean_luma:.1f} < {min_luma:.1f}",
        )

    if contrast < min_contrast:
        return (
            None,
            f"blur not judged: too little local contrast; "
            f"contrast={contrast:.1f} < {min_contrast:.1f}",
        )

    size = max(32, int(resize_px))
    y = cv2.resize(y, (size, size), interpolation=cv2.INTER_AREA)

    # Normalize local contrast so different cameras/exposures are less dominant.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)

    # Tenengrad / Sobel sharpness. sqrt(mean(gx^2 + gy^2)) keeps the score scale
    # easier to read than raw energy.
    gx = cv2.Sobel(y_eq, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(y_eq, cv2.CV_64F, 0, 1, ksize=3)
    sharpness = float(np.sqrt(np.mean(gx * gx + gy * gy)))

    reason = (
        f"sharpness={sharpness:.1f} threshold={threshold:.1f}; "
        f"luma={mean_luma:.1f}; contrast={contrast:.1f}; "
        f"method=tenengrad_clahe; bbox={bbox}"
    )

    if sharpness < threshold:
        return (False, reason)
    return (True, reason)
