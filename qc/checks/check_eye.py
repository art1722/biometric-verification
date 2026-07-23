"""Eyes-open check — MediaPipe Face Landmarker **blendshapes** (new method).

Spec requirement (source of truth): the volunteer's eyes must be open in the
face capture.

Why this replaced the EAR ratio
--------------------------------
The previous method computed an Eye Aspect Ratio (EAR) from six face-mesh
landmarks per eye: (vertical_1 + vertical_2) / (2 * horizontal). We were
advised to stop using that landmark ratio and use a model instead. The
problems with EAR here were concrete:
  - its absolute scale is mesh-dependent, so the open/closed threshold had to
    be hand-calibrated per setup (config note: repo 0.37 cut through video
    002's natural open-eye EAR of 0.35-0.40);
  - it degrades off-frontal, exactly where this protocol turns the head.

New signal: the Face Landmarker Tasks API emits 52 blendshape coefficients,
including ``eyeBlinkLeft`` and ``eyeBlinkRight`` — a TRAINED per-eye closed-ness
score in [0, 1]. High score = eye closed (this is the inverse of EAR). An eye
counts as OPEN when its blink score is BELOW the threshold.

    eye open   <=>  eyeBlink < blink_threshold

The model is run ONCE per frame by qc/checks/face_landmarker.py; this check
just CONSUMES the resulting blendshape dict, so it still touches no image and
runs no model of its own — same contract as before, new input.

    check_eye_status(blendshapes, blink_threshold) -> (success, message)

A geometric EAR fallback (check_eye_status_ear) is kept for the rare case the
model returns landmarks but no blendshapes, and so the old behaviour stays
auditable. The pipeline uses the blendshape path.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Blendshape category names emitted by the Face Landmarker model.
BLINK_LEFT = "eyeBlinkLeft"
BLINK_RIGHT = "eyeBlinkRight"


def check_eye_status(blendshapes: Optional[dict],
                     blink_threshold: float = 0.5,
                     *,
                     return_scores: bool = False):
    """Check whether both eyes are open, using blendshape blink scores.

    Args:
        blendshapes: {category_name: score} from the Face Landmarker (see
            face_landmarker.detect_face). May be None if the model produced no
            blendshapes for the frame.
        blink_threshold: blink score AT OR ABOVE which an eye counts as CLOSED
            (config: face.checks.eyes_open.blink_threshold). Both eyes must be
            below it to PASS.

    Returns:
        (success, message). success is False when either eye is closed or the
        blink scores are unavailable. Message includes the measured scores.
    """
    def _result(ok, msg, left=None, right=None):
        if return_scores:
            return (ok, msg, left, right)
        return (ok, msg)

    if not blendshapes:
        return _result(False, "No blendshapes provided")

    left = blendshapes.get(BLINK_LEFT)
    right = blendshapes.get(BLINK_RIGHT)

    if left is None or right is None:
        return _result(False, "Blink blendshapes missing (eyeBlinkLeft/Right)")

    # Low blink score = open. Both eyes must be open to pass.
    if left < blink_threshold and right < blink_threshold:
        return _result(True,
                       f"eyes open (blinkL={left:.2f}, blinkR={right:.2f} < {blink_threshold})",
                       left, right)
    return _result(False,
                   f"eye(s) closed (blinkL={left:.2f}, blinkR={right:.2f} >= {blink_threshold})",
                   left, right)


# --------------------------------------------------------------------------
# Legacy EAR fallback — kept for auditability and for frames where the model
# returns landmarks but no blendshapes. NOT used by the pipeline's main path.
# --------------------------------------------------------------------------

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


def check_eye_status_ear(landmarks: List[Tuple[int, int, float]],
                         ear_threshold: float = 0.25) -> Tuple[bool, str]:
    """Legacy EAR-based eyes-open check (fallback only)."""
    if landmarks is None:
        return (False, "No landmarks provided")

    left_ear = calculate_ear(landmarks, LEFT_EYE_INDICES)
    right_ear = calculate_ear(landmarks, RIGHT_EYE_INDICES)

    if left_ear is None or right_ear is None:
        return (False, "Could not compute EAR")

    if left_ear > ear_threshold and right_ear > ear_threshold:
        return (True, f"eyes open (EAR L={left_ear:.2f}, R={right_ear:.2f} > {ear_threshold})")
    return (False, f"eye(s) closed (EAR L={left_ear:.2f}, R={right_ear:.2f} <= {ear_threshold})")