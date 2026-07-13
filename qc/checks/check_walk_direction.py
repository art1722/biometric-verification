"""Walk-direction (pose) check for the gait pipeline — SEQUENCE level.

Spec requirement (source of truth, §4):
    "บันทึกภาพวิดีโอโดยเดินเข้าหากล้องและเดินหันหลังออกจากกล้อง (กล้องที่ 1)
     และบันทึกภาพวิดีโอเดินผ่านกล้องไปด้านซ้ายและขวา (กล้องที่ 2)"
    -> camera 1 (F): walk TOWARD the camera, then AWAY (front then back).
       camera 2 (S): walk PAST the camera to the LEFT, then to the RIGHT.

Why this is NOT in pose_landmarker
----------------------------------
The pose detector reports 33 landmarks PER FRAME; it does not report which way
the person is walking. Direction is a property of the TRAJECTORY across frames,
so this is a SEQUENCE-level check (one verdict per video, like face's
check_turn_sequence), not a per-frame one. It reads the per-frame timeline the
pipeline already builds and reasons over the whole series.

Signals (per view)
------------------
F (toward/away): the depth proxy is the person's on-screen SCALE -- the body
    bounding-box height (normalized). Walking toward the camera makes the body
    grow; walking away makes it shrink. A conforming F clip therefore shows a
    sustained GROW phase followed by a sustained SHRINK phase (or the reverse if
    they start close). We accept either order: what matters is that both a
    clear approach and a clear recede are present.

S (left/right): the signal is the person's horizontal CENTROID X (normalized).
    Walking left-to-right moves the centroid one way; right-to-left moves it the
    other. A conforming S clip shows the centroid traverse the frame in BOTH
    directions (a left->right pass and a right->left pass), in either order.

Method (mirrors the structure of check_turn_sequence_seg without its yaw math)
-----------------------------------------------------------------------------
1. Pull the ordered per-frame series from the timeline (drop frames with no
   pose -- they carry no scale/centroid).
2. Smooth it (moving average) to suppress per-frame jitter.
3. Take the sign of the smoothed derivative to label each step INCREASING /
   DECREASING / FLAT (flat = |delta| below a small epsilon).
4. Require a sustained run of each polarity (>= min_run frames), so a genuine
   phase is present in both directions, not just noise.

Thresholds come from config walk.direction.* so the researcher can tune them
without code changes. All are [DESIGN] defaults pending validation on real
_F/_S footage -- flag for อ.เหมียว.

Returns (success, message): one (bool, str), same contract as every check. The
message names what was and was not found so the reviewer sees why it passed or
failed.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


def _moving_average(values, window):
    """Simple centered-ish moving average; window<=1 returns the input."""
    if window <= 1 or len(values) < window:
        return list(values)
    out = []
    half = window // 2
    n = len(values)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = values[lo:hi]
        out.append(sum(seg) / len(seg))
    return out


def _has_sustained_run(signs, target, min_run):
    """True if `signs` contains a run of >= min_run consecutive `target` values."""
    run = 0
    for s in signs:
        if s == target:
            run += 1
            if run >= min_run:
                return True
        else:
            run = 0
    return False


def _polarity_series(values, *, smooth_window, eps):
    """Smoothed step-to-step polarity: +1 rising, -1 falling, 0 flat."""
    sm = _moving_average(values, smooth_window)
    signs = []
    for a, b in zip(sm, sm[1:]):
        d = b - a
        if d > eps:
            signs.append(+1)
        elif d < -eps:
            signs.append(-1)
        else:
            signs.append(0)
    return signs


def check_walk_direction(
    timeline: Sequence[dict],
    view: Optional[str],
    *,
    smooth_window: int = 5,
    min_run: int = 3,
    eps: float = 0.002,
    min_frames: int = 8,
):
    """Verify the walk direction sequence for one video.

    Args:
        timeline: the per-frame series the pipeline built. Each entry should
            carry "body_scale" (normalized bbox height, for F) and "centroid_x"
            (normalized, for S); frames with no pose carry None and are skipped.
        view: "F" or "S" (from the filename). Anything else -> SKIP (cannot
            branch without knowing the camera).
        smooth_window: moving-average window over the frame series.
        min_run: minimum consecutive frames of one polarity to count a phase as
            sustained (rejects single-frame jitter).
        eps: |delta| below this (per smoothed step) is treated as FLAT, not
            motion, so a stationary stretch does not register as a direction.
        min_frames: fewer usable (pose-present) frames than this -> SKIP; the
            series is too short to judge a there-and-back trajectory.

    Returns:
        (success, message).
    """
    if view not in ("F", "S"):
        return (False, f"unknown view '{view}'; cannot judge walk direction")

    if view == "F":
        key, rising_name, falling_name, label = (
            "body_scale", "approach (growing)", "recede (shrinking)",
            "toward+away")
    else:  # S
        key, rising_name, falling_name, label = (
            "centroid_x", "move right", "move left", "left+right")

    series = [t[key] for t in timeline
              if t.get(key) is not None]
    if len(series) < min_frames:
        return (False,
                f"{view}: only {len(series)} usable frame(s) "
                f"< {min_frames}; too short to verify {label}")

    signs = _polarity_series(series, smooth_window=smooth_window, eps=eps)

    has_rising = _has_sustained_run(signs, +1, min_run)
    has_falling = _has_sustained_run(signs, -1, min_run)

    if has_rising and has_falling:
        return (True,
                f"{view}: {label} detected "
                f"(both {rising_name} and {falling_name} present)")

    missing = []
    if not has_rising:
        missing.append(rising_name)
    if not has_falling:
        missing.append(falling_name)
    return (False,
            f"{view}: {label} incomplete; missing sustained "
            + " and ".join(missing))
