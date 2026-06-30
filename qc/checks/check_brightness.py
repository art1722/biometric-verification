"""Brightness check — generic core + per-modality wrappers.

History
-------
This replaces the face-only `check_light_pollution.check_lightpol`. That function
welded TWO jobs together: (1) FIND the region (face_detection-specific) and
(2) MEASURE brightness in it (totally generic). Palm/walk need the SAME
measurement over a DIFFERENT region, so the two jobs are split here:

    check_brightness()        the pure CORE: given an image + a bbox, measure
                              mean V and classify too_dark/too_bright/normal.
                              Knows nothing about faces, hands, or detectors.

    check_brightness_face()   face wrapper: runs (or reuses) FaceDetection to get
                              the bbox, then calls the core. Drop-in replacement
                              for the old check_lightpol — same signature, same
                              messages, same (success, message) contract.

    check_brightness_palm()   palm wrapper: takes the bbox the shared
                              HandLandmarker already produced (HandResult.bbox),
                              then calls the core. No second detector.

    check_brightness_walk()   STUB — not implemented yet. Raises so a premature
                              wiring is a loud, obvious error, not a silent pass.

Why this seam (bbox, not a `mode` string)
------------------------------------------
A `mode="face"|"palm"|...` argument would force every modality's detector and
result-shape INTO this one function (if mode == ... call face_detection; elif
... read HandResult), so each new modality edits this file. Passing the bbox
instead keeps modality knowledge in the caller — the SAME pattern check_occlusion
uses (it takes `landmarks`, never re-detects) and check_metadata uses (it takes a
metadata object). Adding `walk` becomes a new 3-line wrapper, zero edits to the
core.

The core's contract mirrors the other checks: (success, message), success True
only when status == "normal", and the measured value embedded in the message as
`brightness=NN` so the existing _BRIGHTNESS_RE extractor in face_rgb.py keeps
working unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)


def _to_bgr(image: Any, input_color_space: str):
    """Resolve a path or array to a BGR array (array-first, like the other
    checks). Returns None on an unreadable path."""
    if isinstance(image, str):
        return cv2.imread(image)
    if input_color_space == "RGB":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


# ---------------------------------------------------------------------------
# CORE — modality-agnostic. Given an image and a bbox, judge brightness.
# ---------------------------------------------------------------------------

def check_brightness(
    image: Any,
    bbox: Tuple[int, int, int, int],
    dark_threshold: float = 35.0,
    bright_threshold: float = 200.0,
    margin: float = 0.1,
    *,
    input_color_space: Literal["BGR", "RGB"] = "BGR",
) -> Tuple[bool, str]:
    """Measure mean brightness inside `bbox` and classify it.

    This is the shared measurement every modality reuses. It does NOT detect
    anything — the caller supplies the region (a face bbox, a hand bbox, ...).

    Args:
        image: file path (str) OR a decoded image array.
        bbox: (x, y, w, h) in PIXELS — the region to measure. The same shape
            mp.solutions.face_detection's box is converted to AND the shape
            HandResult.bbox already is, so both wrappers pass it directly.
        dark_threshold: mean V below this -> too_dark.
        bright_threshold: mean V above this -> too_bright.
        margin: trim this fraction off each side of the bbox before measuring,
            so the reading comes from the region's centre and not its edges
            (the old check_lightpol did this on the face box).
        input_color_space: "BGR" (OpenCV default) or "RGB".

    Returns:
        (success, message). success is True only when status == "normal".
        message always contains "brightness=NN" so face_rgb's _BRIGHTNESS_RE
        can pull the value for the timeline.
    """
    img = _to_bgr(image, input_color_space)
    if img is None or getattr(img, "size", 0) == 0:
        return (False, "invalid_image")

    h_img, w_img = img.shape[:2]
    x, y, bw, bh = bbox

    # Central region: trim `margin` off each side of the supplied bbox, then
    # clamp to the image. Identical arithmetic to the old face version, but on
    # an arbitrary bbox instead of the face-detection box.
    x_start = max(0, int(x + bw * margin))
    y_start = max(0, int(y + bh * margin))
    x_end = min(w_img, int(x + bw * (1 - margin)))
    y_end = min(h_img, int(y + bh * (1 - margin)))
    if x_end <= x_start or y_end <= y_start:
        return (False, "invalid_crop")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    region_v = hsv[y_start:y_end, x_start:x_end, 2]
    if region_v.size == 0:
        return (False, "empty_region")

    brightness = float(np.mean(region_v))

    if brightness < dark_threshold:
        status = "too_dark"
    elif brightness > bright_threshold:
        status = "too_bright"
    else:
        status = "normal"

    b = int(round(brightness))
    lo = int(round(dark_threshold))
    hi = int(round(bright_threshold))

    if status == "normal":
        msg = f"brightness={b}; normal; {lo} <= {b} <= {hi}"
    elif status == "too_dark":
        msg = f"brightness={b} < {lo}; too_dark"
    else:  # too_bright
        msg = f"brightness={b} > {hi}; too_bright"

    return (status == "normal", msg)


# ---------------------------------------------------------------------------
# FACE wrapper — drop-in replacement for the old check_lightpol.
# ---------------------------------------------------------------------------

def check_brightness_face(
    image: Any,
    dark_threshold: float = 35.0,
    bright_threshold: float = 200.0,
    margin: float = 0.1,
    *,
    detector: Optional[Any] = None,
    input_color_space: Literal["BGR", "RGB"] = "BGR",
) -> Tuple[bool, str]:
    """Face brightness: detect the face, then measure its central region.

    Signature-compatible with the old check_lightpol so face_rgb.py changes only
    the import + call name. The `owns = detector is None` reuse pattern is kept:
    pass the pipeline's shared FaceDetection in, or let this build a throwaway
    one for standalone use.

    Returns (success, message). "no_face" when nothing is detected.
    """
    img = _to_bgr(image, input_color_space)
    if img is None or getattr(img, "size", 0) == 0:
        return (False, "invalid_image")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

    # Convert face_detection's relative box -> pixel (x, y, w, h) bbox, then hand
    # it to the core. The core re-applies `margin`, so pass the FULL face box.
    rel = result.detections[0].location_data.relative_bounding_box
    bbox = (int(rel.xmin * w), int(rel.ymin * h),
            int(rel.width * w), int(rel.height * h))

    # img is already BGR here; tell the core so it doesn't re-convert.
    return check_brightness(
        img, bbox, dark_threshold, bright_threshold, margin,
        input_color_space="BGR",
    )


# ---------------------------------------------------------------------------
# PALM wrapper — reuses the bbox the shared HandLandmarker already produced.
# ---------------------------------------------------------------------------

def check_brightness_palm(
    image: Any,
    hand_bbox: Optional[Tuple[int, int, int, int]],
    dark_threshold: float = 35.0,
    bright_threshold: float = 200.0,
    margin: float = 0.1,
    *,
    input_color_space: Literal["BGR", "RGB"] = "BGR",
) -> Tuple[bool, str]:
    """Palm brightness: measure inside the hand bbox the detector already found.

    There is NO detector call here — the palm pipeline runs detect_hand ONCE and
    every check (present, size, angle, now brightness) consumes that one result.
    Pass HandResult.bbox straight through.

    NOTE on thresholds: the defaults here are the FACE defaults. Palm skin under
    the capture lighting may sit in a different V range, and the spec's
    "veins visible to the naked eye" is a CONTRAST requirement mean-V alone does
    not capture. Drive palm thresholds from config palm.brightness.* and validate
    on real palm samples before trusting the defaults. (Not asserting they
    differ — just don't assume they transfer.)

    Returns (success, message). "no_hand" when the caller passes bbox=None
    (i.e. detect_hand failed upstream; check_palm_present already reported why).
    """
    if hand_bbox is None:
        return (False, "no_hand")
    return check_brightness(
        image, hand_bbox, dark_threshold, bright_threshold, margin,
        input_color_space=input_color_space,
    )


# ---------------------------------------------------------------------------
# WALK wrapper — STUB. Not implemented; raises so it can't be wired silently.
# ---------------------------------------------------------------------------

def check_brightness_walk(*args, **kwargs) -> Tuple[bool, str]:
    """Gait/walk brightness — NOT IMPLEMENTED YET.

    Placeholder so the per-modality set is complete and the intended seam is
    visible. When the gait pipeline gains a person/body bbox (e.g. from a pose
    detector), this becomes a 3-line wrapper exactly like check_brightness_palm:
    get the body bbox, call check_brightness(image, body_bbox, ...).
    """
    raise NotImplementedError(
        "check_brightness_walk is not implemented yet; the gait pipeline has no "
        "body bbox source. Mirror check_brightness_palm once it does."
    )