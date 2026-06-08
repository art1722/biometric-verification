"""Head-fully-visible check (ported from func/check_head_fully.py).

Spec requirement (source of truth): the video must show the full head down to
the neck; the top of the head and the chin must not be cut off by the frame
edge.

Method (from the repo): landmark 10 is the top of the forehead, landmark 152 is
the bottom of the chin. If the forehead landmark sits within `margin_px` of the
top edge, the top is likely cut; if the chin landmark sits within `margin_px` of
the bottom edge, the chin is likely cut.

Change from the repo: the original re-ran its OWN face mesh on the image path.
This port consumes the landmarks already produced by get_lm, so the face is
detected only once per frame. It also takes pixel-space landmarks (what get_lm
returns) rather than normalized ones.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# MediaPipe Face Mesh indices (same as the repo).
_TOP_OF_HEAD_IDX = 10
_CHIN_IDX = 152


def check_head_fully(landmarks, image_height: int, margin_px: int = 10):
    """Check the head is fully inside the frame (top + chin not cut).

    Args:
        landmarks: list[(x, y, z)] in PIXEL coords from get_lm. y is already in
            pixels, so we compare directly against image_height.
        image_height: frame height in px.
        margin_px: how close to an edge counts as "cut" (config: head_fully
            margin_px, default 10).

    Returns:
        (success, message).
    """
    if not landmarks:
        return (False, "No landmarks provided")

    try:
        top_y = landmarks[_TOP_OF_HEAD_IDX][1]
        chin_y = landmarks[_CHIN_IDX][1]
    except (IndexError, TypeError):
        return (False, "Landmark list missing required points")

    top_cut = top_y < margin_px
    chin_cut = chin_y > image_height - margin_px

    if top_cut and chin_cut:
        return (False, "Top of head and chin might be cut")
    if top_cut:
        return (False, "Top of head might be cut")
    if chin_cut:
        return (False, "Chin might be cut")
    return (True, "Head is fully visible")
