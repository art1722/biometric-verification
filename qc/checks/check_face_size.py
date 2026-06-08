"""Face size check (ported from func/check_face_size.py).

Checks the face bounding box meets the spec's minimum dimensions.

Spec requirement (source of truth): head region >= 180 x 180 px.
Note: the repo used a single `min_size` compared with strict `>`. The spec
says "not less than 180", i.e. 180 is acceptable, so this uses `>=`. The repo's
default was also 150; the spec value 180 must come from config.

This check does NOT touch the image or run a model. It consumes the `bbox`
produced once by get_lm, so the face is detected a single time per frame.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def check_face_min_size(bbox, min_width: int = 180, min_height: int = 180):
    """Check whether the face bbox is at least min_width x min_height.

    Args:
        bbox: (x, y, w, h) from get_lm, or None if no face was found.
        min_width: Minimum acceptable width in px (spec: 180).
        min_height: Minimum acceptable height in px (spec: 180).

    Returns:
        (success, message). Message includes the measured size so the report
        can show "face=140x160 < 180x180" style reasons.
    """
    if bbox is None:
        return (False, "No bounding box provided")

    x, y, w, h = bbox

    # Spec says "not less than", so >= (boundary value passes).
    if w >= min_width and h >= min_height:
        return (True, f"face={w}x{h} >= {min_width}x{min_height}")
    return (False, f"face={w}x{h} < {min_width}x{min_height}")
