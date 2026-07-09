"""Body-height check for the gait pipeline (spec #4.2).

Spec requirement (source of truth), Thai:
    "ตัวคนต้องมีความสูงอย่างน้อยครึ่งหนึ่งของความสูงภาพ"
    -> the person's height must be AT LEAST HALF the image height.

Where is it measured?  (researcher decision, 2026-07-09)
    NOT every frame. Only the FIRST frame of each walk video -- the point where
    the person is FARTHEST from the camera (so they appear smallest). If the
    person clears the half-frame bar at their smallest, they clear it for the
    whole clip. The check runs on BOTH camera videos (_F and _S) independently.

    The gait pipeline is responsible for picking that first frame and running
    the pose detector on it; THIS function is pure geometry on the resulting
    landmarks (mirrors how check_face_size / check_palm_size consume a bbox they
    do not compute -- detect once, every check reads that one result).

How is "person height" defined?
    The vertical (y) extent of the pose in NORMALIZED image coordinates: the
    span from the topmost visible landmark to the bottommost visible landmark.
    Because y is already normalized to [0, 1] by the image height, that span IS
    the height-as-a-fraction-of-the-image -- no pixel conversion needed, and it
    is resolution-independent (a _F 1080p clip and an _S 1080p clip compare on
    the same scale). ratio = y_max - y_min; PASS if ratio >= min_ratio (0.5).

    We only count landmarks whose `visibility` clears a floor, so a landmark
    MediaPipe merely guessed at off-screen (low visibility) cannot inflate or
    deflate the span. If too few landmarks are visible to trust the span, the
    check says so rather than returning a misleading number (the caller decides
    the verdict for that degenerate case).

Returns (success, message) with the measured ratio in the message, so the
report can show "body_height=0.42 < 0.50" style reasons -- same (bool, str)
contract every other check uses.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# A landmark below this visibility is treated as "not reliably seen" and left
# out of the height span. 0.5 is MediaPipe's own default confidence floor and
# matches the threshold check_full_body_visible will use. [ASSUMPTION] tune on
# real _F/_S footage.
_DEFAULT_MIN_VISIBILITY = 0.5

# Need at least this many visible landmarks to trust a top-to-bottom span. Two
# (one high, one low) is the bare geometric minimum; requiring a few more avoids
# calling a height off a single stray point. [ASSUMPTION]
_MIN_VISIBLE_LANDMARKS = 4


def check_body_height(
    landmarks_norm: Optional[Sequence[Any]],
    *,
    min_ratio: float = 0.5,
    min_visibility: float = _DEFAULT_MIN_VISIBILITY,
    min_visible_landmarks: int = _MIN_VISIBLE_LANDMARKS,
):
    """Check the person is at least `min_ratio` of the image height.

    Args:
        landmarks_norm: the normalized pose landmarks for ONE frame -- the
            `landmarks_norm` field of a PoseResult (each item has .x/.y in
            [0,1] and a .visibility in [0,1]). None if no pose was detected on
            that frame ("no pose" is the detector's ok/message to report, not
            this check's job -- a None here just yields a clear message so the
            row is never silently wrong, same as check_face_size's None bbox).
        min_ratio: minimum person-height / image-height. Spec: 0.5. Boundary
            passes (spec says "at least half", so >=).
        min_visibility: landmarks below this visibility are excluded from the
            height span.
        min_visible_landmarks: fewer than this many visible landmarks -> the
            span is untrustworthy; return (False, "...") with a reason rather
            than a misleading ratio.

    Returns:
        (success, message). Message always contains "body_height=NN.NN" so a
        report/timeline extractor can pull the value the same way it pulls
        "brightness=NN".
    """
    if not landmarks_norm:
        return (False, "No pose landmarks provided")

    # Collect y of every SUFFICIENTLY VISIBLE landmark. `visibility` may be
    # absent on some inputs (None) -- treat that as "unknown", not "invisible",
    # so a landmark list without visibility still yields a span (degrades to
    # "use all points"), which is the safe permissive default.
    ys = []
    for lm in landmarks_norm:
        vis = getattr(lm, "visibility", None)
        if vis is None or vis >= min_visibility:
            ys.append(float(lm.y))

    if len(ys) < min_visible_landmarks:
        return (
            False,
            f"body_height=0.00; only {len(ys)} visible landmark(s) "
            f"< {min_visible_landmarks} needed to measure height",
        )

    ratio = max(ys) - min(ys)

    if ratio >= min_ratio:
        return (True, f"body_height={ratio:.2f} >= {min_ratio:.2f}")
    return (False, f"body_height={ratio:.2f} < {min_ratio:.2f}")