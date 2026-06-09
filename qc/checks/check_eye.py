"""Eyes-open check (EAR — Eye Aspect Ratio).

Spec requirement (source of truth): the volunteer's eyes must be open in the
face capture.

Method (from the repo): for each eye, take 6 Face Mesh landmarks and compute
the Eye Aspect Ratio = (vertical_1 + vertical_2) / (2 * horizontal). A low EAR
means the lid gap is small relative to the eye width, i.e. the eye is closed.
A threshold (config: face.checks.eyes_open.ear_threshold, repo default 0.37)
separates open from closed. Both eyes must exceed it to PASS.

Contract (matches check_face_size / check_head_fully)
-----------------------------------------------------
This check does NOT touch the image or run a model. It CONSUMES the `landmarks`
produced once by get_lm, so the face is detected a single time per frame. The
pipeline only calls this after get_lm has already succeeded, so landmarks are
guaranteed present here.

    check_eye_status(landmarks, ear_threshold) -> (success, message)

landmarks are get_lm's PIXEL-space (x, y, z) tuples; EAR is a ratio of
distances so the pixel scaling cancels out.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Eye landmark indices (MediaPipe Face Mesh), order [p1, p2, p3, p4, p5, p6]:
# p1, p4 horizontal corners; p2-p6 and p3-p5 vertical pairs.
LEFT_EYE_INDICES = [33, 160, 159, 133, 158, 157]
RIGHT_EYE_INDICES = [362, 387, 386, 263, 385, 384]


def calculate_ear(landmarks: List[Tuple[int, int, float]],
                  eye_indices: List[int]) -> Optional[float]:
    """Eye Aspect Ratio for one eye. Returns None if it cannot be computed."""
    try:
        p1 = np.array(landmarks[eye_indices[0]][:2])
        p2 = np.array(landmarks[eye_indices[1]][:2])
        p3 = np.array(landmarks[eye_indices[2]][:2])
        p4 = np.array(landmarks[eye_indices[3]][:2])
        p5 = np.array(landmarks[eye_indices[4]][:2])
        p6 = np.array(landmarks[eye_indices[5]][:2])

        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)

        if horizontal == 0:
            return None
        return (vertical_1 + vertical_2) / (2.0 * horizontal)
    except (IndexError, TypeError, ValueError) as e:
        logger.debug("EYE | EAR calc failed: %s", e)
        return None


def check_eye_status(landmarks: List[Tuple[int, int, float]],
                     ear_threshold: float = 0.37) -> Tuple[bool, str]:
    """Check whether both eyes are open.

    Args:
        landmarks: get_lm's pixel-space (x, y, z) tuples (guaranteed non-None
            by the pipeline, which only calls this after get_lm succeeds).
        ear_threshold: EAR above which an eye counts as open (config-driven).

    Returns:
        (success, message). success is False when either eye is closed or the
        EAR could not be computed. Message includes the measured EAR values.
    """
    if landmarks is None:
        return (False, "No landmarks provided")

    left_ear = calculate_ear(landmarks, LEFT_EYE_INDICES)
    right_ear = calculate_ear(landmarks, RIGHT_EYE_INDICES)

    if left_ear is None or right_ear is None:
        return (False, "Could not compute EAR")

    if left_ear > ear_threshold and right_ear > ear_threshold:
        return (True, f"eyes open (L={left_ear:.2f}, R={right_ear:.2f} > {ear_threshold})")
    return (False, f"eye(s) closed (L={left_ear:.2f}, R={right_ear:.2f} <= {ear_threshold})")