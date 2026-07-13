"""Person-fully-visible check for the walk (gait) pipeline.

Spec requirement (source of truth, §4):
    "วิดีโอต้องเห็นทั้งตัวและเห็นท่าทางเดินทุกเฟรม"
    -> the whole body (and the walking pose) must be visible in every frame.

Method (researcher-chosen landmark set, 2026-07-13)
--------------------------------------------------
Eight MediaPipe pose landmarks must ALL be present AND inside the frame:

    2  LEFT_EYE          5  RIGHT_EYE
    7  LEFT_EAR          8  RIGHT_EAR
    19 LEFT_INDEX        20 RIGHT_INDEX        (finger tips = hands)
    31 LEFT_FOOT_INDEX   32 RIGHT_FOOT_INDEX   (toes = feet)

These span the body top-to-bottom (head, hands, feet), so requiring all eight
inside the frame is a compact proxy for "the whole person is in view". If any of
the eight is missing or falls outside the frame bounds, the body is cut off and
the frame FAILs.

This is the pose analog of check_head_fully (which does the same edge-proximity
idea for the FACE mesh with different indices). It is a NEW small check rather
than a reuse of check_head_fully because the landmark set and semantics differ:
head_fully guards the head crop; this guards the whole-body extent.

"In frame" test
---------------
A landmark is in-frame when 0 <= x <= width and 0 <= y <= height. MediaPipe can
report a landmark slightly outside [0,1] when it extrapolates an off-screen
joint, so a normalized coordinate < 0 or > 1 (i.e. pixel < 0 or > resolution)
means that part of the body is outside the recorded frame. An optional
`margin_px` treats a landmark within that many pixels of an edge as cut, matching
check_head_fully's margin idea; default 0 (strict: only truly outside fails).

Returns (success, message), the same (bool, str) contract every check uses.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# The eight required landmarks (MediaPipe pose indices) and human-readable names
# for the failure message. Order is top-to-bottom for a readable reason string.
_REQUIRED = (
    (2, "left_eye"), (5, "right_eye"),
    (7, "left_ear"), (8, "right_ear"),
    (19, "left_hand"), (20, "right_hand"),
    (31, "left_foot"), (32, "right_foot"),
)


def check_person_fully(
    landmarks_px: Optional[Sequence[Any]],
    image_width: int,
    image_height: int,
    *,
    margin_px: int = 0,
):
    """Check the eight required landmarks are all present and inside the frame.

    Args:
        landmarks_px: list[(x, y, z)] in PIXEL coords (PoseResult.landmarks_px).
            None/empty if no pose was detected on the frame (the caller reports
            "no pose" via check_person_detected; here it just yields a clear
            message so the row is never silently wrong).
        image_width, image_height: frame size in px, for the in-frame bounds.
        margin_px: a landmark within this many px of any edge counts as cut.
            Default 0 = strict (only a landmark truly outside the frame fails),
            matching the "not out of the video resolution" wording.

    Returns:
        (success, message). On failure the message names which landmarks were
        missing or out of frame, so the reviewer knows what was cut.
    """
    if not landmarks_px:
        return (False, "No pose landmarks provided")

    n = len(landmarks_px)
    lo_x = margin_px
    hi_x = image_width - margin_px
    lo_y = margin_px
    hi_y = image_height - margin_px

    missing = []
    out_of_frame = []
    for idx, name in _REQUIRED:
        if idx >= n:
            missing.append(name)
            continue
        x, y = landmarks_px[idx][0], landmarks_px[idx][1]
        if x < lo_x or x > hi_x or y < lo_y or y > hi_y:
            out_of_frame.append(name)

    if missing or out_of_frame:
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if out_of_frame:
            parts.append("out of frame: " + ", ".join(out_of_frame))
        return (False, "person not fully visible (" + "; ".join(parts) + ")")

    return (True, "person fully visible (all 8 key landmarks in frame)")
