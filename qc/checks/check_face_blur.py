"""Face blur check (ported from func/check_face_blur.py).

Spec requirement (source of truth): the face must be clear/sharp enough to see
facial detail.

Method (from the repo): detect the face, crop to its bounding region, convert
to grayscale, and compute the variance of the Laplacian. Low variance = few
sharp edges = blurry. A threshold (config: checks.blur.threshold, repo default
90) separates sharp from blurry.

Changes from the repo:
- ARRAY-FIRST: accepts a path or a BGR/RGB frame.
- DETECTOR REUSE: optional `detector` (a MediaPipe FaceDetection) can be passed
  in; otherwise one is created and closed locally.
- Returns the measured variance in the message so the report can show the value.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import cv2
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)


def _to_bgr(image: Any, input_color_space: str):
    """Resolve input to a BGR array (MediaPipe face detection wants RGB; OpenCV
    crop/grayscale below is colorspace-agnostic for the Laplacian)."""
    if isinstance(image, str):
        img = cv2.imread(image)
        return img
    if input_color_space == "RGB":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def check_face_blur(
    image: Any,
    threshold: float = 90.0,
    *,
    detector: Optional[Any] = None,
    input_color_space: Literal["BGR", "RGB"] = "BGR",
):
    """Check whether the face region is blurry.

    Returns:
        (success, message). success is False when blurry, None-style failures
        (no face, unreadable) also return False with a descriptive message.
    """
    if threshold <= 0:
        return (False, "Threshold must be positive")

    img = _to_bgr(image, input_color_space)
    if img is None or getattr(img, "size", 0) == 0:
        return (False, "Cannot read image")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    owns = detector is None
    if owns:
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
    try:
        results = detector.process(img_rgb)
    finally:
        if owns:
            detector.close()

    if not results.detections:
        return (False, "No face detected")

    bbox = results.detections[0].location_data.relative_bounding_box
    xmin = max(0, int(bbox.xmin * w))
    ymin = max(0, int(bbox.ymin * h))
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)
    xmax = min(w, xmin + bw)
    ymax = min(h, ymin + bh)

    if xmax <= xmin or ymax <= ymin:
        return (False, "Invalid face region")

    face = img[ymin:ymax, xmin:xmax]
    if face.size == 0:
        return (False, "Empty face region")

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if variance < threshold:
        return (False, f"blur variance={variance:.1f} < {threshold}")
    return (True, f"blur variance={variance:.1f} >= {threshold}")
