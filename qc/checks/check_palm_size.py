"""Palm size check (palm-side mirror of check_face_size.py).

Checks the detected HAND bounding box meets the spec's minimum dimensions.

Spec requirement (source of truth): palm region >= 200 x 200 px, measured
"from mid-finger to wrist" (i.e. the hand region, not the whole image). The
metadata-level check_resolution in palm.py only guarantees the FILE is at least
200x200 -- a necessary precondition. THIS check asserts the actual hand region
(the detector's bbox) is large enough, which is the real spec requirement.

Mirrors check_face_min_size exactly:
  - same (bbox, min_width, min_height) signature,
  - same `>=` boundary semantics ("not less than 200" => 200 passes),
  - same (success, message) return with measured size in the message,
  - does NOT touch the image or run a model. It consumes the `bbox` produced
    once by detect_hand (qc/checks/hand_landmarker.py), so the hand is detected
    a single time per image and every palm check reads that one result.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def check_palm_min_size(bbox, min_width: int = 200, min_height: int = 200):
    """Check whether the hand bbox is at least min_width x min_height.

    Args:
        bbox: (x, y, w, h) from detect_hand (HandResult.bbox), or None if no
            hand was found. "No hand" is NOT this check's job to report -- that
            is detect_hand's ok/message (the same way check_face_min_size does
            not re-detect the face); a None bbox here just yields a clear
            "no bounding box" message so the row is never silently wrong.
        min_width: Minimum acceptable width in px (spec: 200).
        min_height: Minimum acceptable height in px (spec: 200).

    Returns:
        (success, message). Message includes the measured size so the report
        can show "palm=140x160 < 200x200" style reasons.
    """
    if bbox is None:
        return (False, "No bounding box provided")

    x, y, w, h = bbox

    # Spec says "not less than", so >= (boundary value passes).
    if w >= min_width and h >= min_height:
        return (True, f"palm={w}x{h} >= {min_width}x{min_height}")
    return (False, f"palm={w}x{h} < {min_width}x{min_height}")