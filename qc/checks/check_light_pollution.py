"""Brightness / light-pollution check (ported from func/check_light_pollution.py).

Spec requirement (source of truth): the face must be well-lit, bright enough to
see facial detail clearly (not too dark, not blown out).

Method (from the repo): detect the face, take the central face region (trimmed
by a margin), and measure mean brightness on the HSV V channel. Classify as:
  - too_dark   if face brightness < dark_threshold
  - too_bright if face brightness > bright_threshold
  - backlight  if |face - background| brightness differs by > diff_threshold
  - normal     otherwise (the only PASS state)

Changes from the repo:
- ARRAY-FIRST + DETECTOR REUSE, same pattern as the other checks.
- Returns the measured brightness/status in the message.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import cv2
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)


def _to_bgr(image: Any, input_color_space: str):
    if isinstance(image, str):
        return cv2.imread(image)
    if input_color_space == "RGB":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def check_lightpol(
    image: Any,
    dark_threshold: float = 35.0,
    bright_threshold: float = 200.0,
    diff_threshold: float = 20.0,
    margin: float = 0.1,
    *,
    detector: Optional[Any] = None,
    input_color_space: Literal["BGR", "RGB"] = "BGR",
):
    """Check the face is neither too dark, too bright, nor backlit.

    Returns:
        (success, message). success True only when status == "normal".
    """
    img = _to_bgr(image, input_color_space)
    if img is None or getattr(img, "size", 0) == 0:
        return (False, "invalid_image")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    owns = detector is None
    if owns:
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
    try:
        result = detector.process(rgb)
    finally:
        if owns:
            detector.close()

    if not result.detections:
        return (False, "no_face")

    bbox = result.detections[0].location_data.relative_bounding_box
    x_min = int(bbox.xmin * w)
    y_min = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)

    # central face region (trim a margin off each side)
    x_start = max(0, int(x_min + bw * margin))
    y_start = max(0, int(y_min + bh * margin))
    x_end = min(w, int(x_min + bw * (1 - margin)))
    y_end = min(h, int(y_min + bh * (1 - margin)))
    if x_end <= x_start or y_end <= y_start:
        return (False, "invalid_face_crop")

    face_v = hsv[y_start:y_end, x_start:x_end, 2]
    if face_v.size == 0:
        return (False, "empty_face_region")

    # background = everything outside the face box
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[y_min:y_min + bh, x_min:x_min + bw] = 0
    background_v = cv2.bitwise_and(hsv[:, :, 2], hsv[:, :, 2], mask=mask)
    background_vals = background_v[background_v > 0]

    face_brightness = float(np.mean(face_v))
    bg_brightness = float(np.mean(background_vals)) if background_vals.size else None
    diff = abs(face_brightness - bg_brightness) if bg_brightness is not None else None

    if face_brightness < dark_threshold:
        status = "too_dark"
    elif face_brightness > bright_threshold:
        status = "too_bright"
    elif diff is not None and diff > diff_threshold:
        status = "backlight"
    else:
        status = "normal"

    msg = f"brightness={face_brightness:.0f} status={status}"
    return (status == "normal", msg)
