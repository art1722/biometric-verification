"""Palm-fully-visible check (palm-side mirror of check_head_fully.py).

Spec requirement (source of truth): the palm image must show the whole hand;
the wrist end and the fingers must not be cut off by the frame edge. This is the
hand analogue of face's "full head down to the neck, top and chin not cut".

Method (mirrors check_head_fully): two anchor landmarks bound the hand along its
long axis --

    WRIST (landmark 0)             = the base/heel of the hand
    MIDDLE_FINGER_MCP (landmark 9) = the knuckle at the base of the middle
                                     finger, the most central "top of palm"
                                     point that is stable regardless of finger
                                     curl (unlike a fingertip, which moves).

If either anchor sits within `margin_px` of a frame edge, that end of the hand
is likely cut. WRIST near the bottom edge -> heel cut; MIDDLE_FINGER_MCP near
the top edge -> upper palm/fingers cut. Both anchors are also checked against
the left/right edges, mirroring how check_head_fully guards the cheeks, so a
hand pushed sideways out of frame is caught too.

Change from face: face uses distinct top (10) / chin (152) indices for the two
edges. The hand has no single "bottom vs top" pair that maps cleanly to every
frame edge, so BOTH anchors are tested against ALL four edges and the first
violation found is reported. This is deliberately conservative -- any anchor
touching any edge fails the frame.

Coordinate space: consumes landmarks_px (PIXEL coords) from detect_hand, the
same contract check_head_fully uses for get_lm. y is already in pixels, so it
compares directly against image_height. The hand is detected ONCE per image
(single-detect discipline); this check does not re-run any model.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# MediaPipe HandLandmark indices (see qc/checks/hand_landmarker.HandLandmark).
_WRIST_IDX = 0
_MIDDLE_FINGER_MCP_IDX = 9


def check_palm_fully(landmarks, image_height: int, image_width: int,
                     margin_px: int = 10):
    """Check the hand is fully inside the frame (wrist + upper palm not cut).

    Args:
        landmarks: list[(x, y, z)] in PIXEL coords from detect_hand
            (HandResult.landmarks_px). y is already in pixels, so it is compared
            directly against image_height.
        image_height: image height in px.
        image_width: image width in px.
        margin_px: how close to an edge counts as "cut" (config:
            palm.visibility.margin_px, default 10).

    Returns:
        (success, message). Mirrors check_head_fully's return shape so the
        pipeline adds a PASS/FAIL row identically to the face side.
    """
    if not landmarks:
        return (False, "No landmarks provided")

    try:
        wrist_x, wrist_y = landmarks[_WRIST_IDX][0], landmarks[_WRIST_IDX][1]
        mcp_x = landmarks[_MIDDLE_FINGER_MCP_IDX][0]
        mcp_y = landmarks[_MIDDLE_FINGER_MCP_IDX][1]
    except (IndexError, TypeError):
        return (False, "Landmark list missing required points")

    # Wrist: the heel of the hand. Nearest to the bottom edge in a normal
    # upright palm shot, but guard every edge to be safe.
    wrist_top_cut = wrist_y < margin_px
    wrist_bottom_cut = wrist_y > image_height - margin_px
    wrist_left_cut = wrist_x < margin_px
    wrist_right_cut = wrist_x > image_width - margin_px

    # Middle-finger MCP: the top-of-palm anchor. Nearest the top edge normally.
    mcp_top_cut = mcp_y < margin_px
    mcp_bottom_cut = mcp_y > image_height - margin_px
    mcp_left_cut = mcp_x < margin_px
    mcp_right_cut = mcp_x > image_width - margin_px

    # Report the most informative violation first (both ends, then each end).
    if wrist_bottom_cut and mcp_top_cut:
        return (False, "Wrist and upper palm might be cut")
    if wrist_bottom_cut or wrist_top_cut:
        return (False, "Wrist might be cut")
    if mcp_top_cut or mcp_bottom_cut:
        return (False, "Upper palm might be cut")
    if wrist_left_cut or mcp_left_cut:
        return (False, "Left side of hand might be cut")
    if wrist_right_cut or mcp_right_cut:
        return (False, "Right side of hand might be cut")
    return (True, "Palm is fully visible")