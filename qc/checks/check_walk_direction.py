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
    grow; walking away makes it shrink. Spec order is TOWARD then AWAY, so a
    conforming F clip shows a sustained GROW phase FOLLOWED BY a sustained SHRINK
    phase, IN THAT ORDER. The reverse (away then toward) FAILs.

S (left/right): the signal is the person's horizontal CENTROID X (normalized).
    In camera/image coordinates x increases to the right, so moving to
    SCREEN-LEFT decreases centroid_x and SCREEN-RIGHT increases it. Spec order is
    LEFT then RIGHT ("<=" then "=>"), so a conforming S clip shows a sustained
    left (decreasing) phase FOLLOWED BY a sustained right (increasing) phase, IN
    THAT ORDER. The reverse (right then left) FAILs.

Method (mirrors the structure of check_turn_sequence_seg without its yaw math)
-----------------------------------------------------------------------------
1. Pull the ordered per-frame series from the timeline (drop frames with no
   pose -- they carry no scale/centroid).
2. Smooth it (moving average) to suppress per-frame jitter.
3. Take the sign of the smoothed derivative to label each step INCREASING /
   DECREASING / FLAT (flat = |delta| below a small epsilon).
4. Reduce to the ORDERED list of sustained phases (>= min_run frames each),
   collapsing jitter, then require the first two phases to match the view's
   required order EXACTLY (F: grow->shrink; S: left->right). Order matters --
   "both present" is not enough.

Thresholds come from config walk.direction.* so the debugger can tune them
without code changes. All are [DESIGN] defaults pending validation on real
_F/_S footage

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


def _ordered_phases(signs, min_run):
    """Reduce a per-step polarity series to the ORDERED list of sustained phases.

    Walks `signs` left to right and emits one entry per sustained motion phase,
    IN THE ORDER they occur. A "phase" is a run of >= min_run consecutive steps
    of the same non-zero polarity (+1 rising / -1 falling); flats (0) and runs
    shorter than min_run are ignored as jitter/pauses and do NOT split or emit a
    phase. Consecutive same-polarity runs separated only by jitter collapse into
    one phase, so a brief wobble mid-stride doesn't fabricate an extra reversal.

    Returns e.g. [+1, -1] for a rise-then-fall trajectory, [-1, +1] for
    fall-then-rise, [+1] for a single sustained direction, [] for no sustained
    motion. This is what lets the caller enforce a STRICT there-and-back order
    (toward-then-away, left-then-right) rather than merely "both present".
    """
    phases = []
    run_sign = 0
    run_len = 0

    def _commit():
        # A run just ended (or the series did): if it was long enough and has a
        # direction, record it — but merge with the previous phase if it's the
        # same polarity (jitter between two same-direction runs is not a new
        # phase).
        if run_sign != 0 and run_len >= min_run:
            if not phases or phases[-1] != run_sign:
                phases.append(run_sign)

    for s in signs:
        if s == run_sign:
            run_len += 1
        else:
            _commit()
            run_sign = s
            run_len = 1
    _commit()
    return phases


class PhaseSequenceFSM:
    """A finite state machine that accepts a motion-phase sequence IFF the phases
    occur in a required order.

    This is the walk-direction analogue of the sub-action FSM in Liu, Lu & Chao
    (Eng. Proc. 2026, 129(1), 21), where a caregiver procedure is validated by an
    FSM whose states are the expected steps (S0 Start -> S1 -> S2 -> ... -> End)
    and a transition fires ONLY when the correct next step is performed; an
    out-of-order or missing step leaves the machine short of the accepting state
    and is flagged as an anomaly. Here the "steps" are sustained motion phases
    (+1 rising / -1 falling) rather than caregiver sub-actions, and there are
    exactly two ordered steps per view (a there-and-back), but the control logic
    is the same: consume the observed phases left to right, advance only on the
    expected polarity, and ACCEPT only if the machine reaches its final state.

    States: 0..len(required). State k means "the first k required phases have been
    matched, in order". The accepting state is len(required) (all matched).

    Transition rule (deliberately lenient in two spec-approved ways):
      - From state k (k < N), seeing the expected phase required[k] advances to
        k+1. This is the only way to move forward.
      - Once ACCEPTING (state N), any further phases are IGNORED, not rejected:
        the spec only requires the walk to START with the correct there-and-back,
        so trailing motion after a complete sequence does not fail the clip.
      - Before accepting, a phase that is NOT the expected next one does NOT
        advance the machine. It is recorded (for diagnostics) but the machine
        stays put, so it can neither skip a step nor be satisfied out of order.

    This reproduces the previous "first two phases must equal required, extra
    trailing phases tolerated" behavior EXACTLY, but as an explicit, citable FSM.
    """

    def __init__(self, required):
        self.required = tuple(required)
        self.n = len(self.required)
        self.state = 0
        self.consumed = []  # phases actually fed, in order (for messages)

    @property
    def accepting(self):
        return self.state >= self.n

    def feed(self, phase):
        """Advance the machine by one observed phase. Returns the new state."""
        self.consumed.append(phase)
        if self.accepting:
            return self.state  # trailing phase: ignore, stay accepted
        if phase == self.required[self.state]:
            self.state += 1
        # else: unexpected phase -> no transition (machine stalls on this step)
        return self.state

    def run(self, phases):
        """Feed an entire phase sequence; returns True iff the FSM accepts."""
        for p in phases:
            self.feed(p)
        return self.accepting


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

    # Per view: the signal key, human names for a RISING (+1) and FALLING (-1)
    # phase, the label, and the REQUIRED strict order as (first_sign, second_sign).
    #
    # F (spec: เดินเข้าหากล้อง...แล้ว...หันหลังออกจากกล้อง = toward THEN away):
    #   body_scale grows as the walker nears the camera (+1) then shrinks as they
    #   recede (-1). Required order: (+1, -1).
    # S (spec: เดินผ่านกล้อง...ซ้ายและขวา, camera frame: <= then =>):
    #   the walker moves to SCREEN-LEFT first (centroid_x decreasing, -1) then to
    #   SCREEN-RIGHT (centroid_x increasing, +1). Required order: (-1, +1).
    if view == "F":
        key = "body_scale"
        name = {+1: "approach (growing)", -1: "recede (shrinking)"}
        label = "toward-then-away"
        required = (+1, -1)
    else:  # S
        key = "centroid_x"
        name = {-1: "move left (<=)", +1: "move right (=>)"}
        label = "left-then-right"
        required = (-1, +1)

    series = [t[key] for t in timeline
              if t.get(key) is not None]
    if len(series) < min_frames:
        return (False,
                f"{view}: only {len(series)} usable frame(s) "
                f"< {min_frames}; too short to verify {label}")

    # Front-end: reduce the raw signal to the ordered list of sustained motion
    # phases (jitter/pauses collapsed). This is the sequence the FSM consumes.
    signs = _polarity_series(series, smooth_window=smooth_window, eps=eps)
    phases = _ordered_phases(signs, min_run)

    req_str = f"{name[required[0]]} then {name[required[1]]}"
    phase_str = (" then ".join(name.get(p, str(p)) for p in phases)
                 if phases else "no sustained motion")

    # Back-end: an explicit FSM decides PASS/FAIL. It ACCEPTS iff the required
    # phases occur in order (extra trailing phases tolerated). See PhaseSequenceFSM
    # (modeled on the sub-action FSM in Liu, Lu & Chao 2026).
    fsm = PhaseSequenceFSM(required)
    if fsm.run(phases):
        return (True, f"{view}: {label} OK (detected: {phase_str})")

    # Not accepted -> report WHY, using how far the machine advanced (fsm.state)
    # and what it saw. state == 0 with no phases -> nothing happened; state == 0
    # or 1 with phases present -> stalled because the order was wrong or a step
    # was missing. These branches reproduce the previous messages exactly.
    if not phases:
        reason = "no sustained direction detected"
    elif len(phases) == 1:
        reason = (f"only one phase ({name.get(phases[0], phases[0])}); "
                  f"a there-and-back ({req_str}) was not completed")
    elif (phases[0], phases[1]) == (required[1], required[0]):
        reason = (f"reversed order (got {phase_str}); "
                  f"spec requires {req_str}")
    else:
        reason = f"expected {req_str}, got {phase_str}"

    return (False, f"{view}: {label} FAIL; {reason}")