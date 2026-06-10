"""check_turn_sequence — PRESENCE-FIRST version.

Question this version answers:
    "Did each required turn happen (held long enough), and did the turns occur
     in an accepted order?"

It does NOT carve the timeline into exact segments or reject 'extra' motion.
It scans the whole timeline, asks per direction whether enough evidence exists,
and checks the order of the detected turn-events by time.

Per-direction PASS rule (uniform for left/right/up/down):
    A turn in direction D is PRESENT if EITHER
      (a) at least `min_hold_frames` DETECTED frames are classified as D, OR
      (b) a detection GAP exists, bracketed by front frames before and after
          (front -> gap -> front), AND the model fallback positively confirms
          direction D for that gap.
    Phase 1: the fallback returns None, so a turn that exists ONLY as a
    bracketed gap is scored REVIEW, not FAIL.

Order rule:
    Collect detected turn-events (direction + first timestamp), sort by time,
    compare against each accepted_order in config (ignoring 'front').
    Subsequence match against ANY accepted order -> PASS; else order_policy.

Emits one CheckRow per direction plus one overall row.
"""

from __future__ import annotations

import math
from collections import defaultdict

from qc.schemas import CheckRow
from qc.checks._turn_common import (
    TurnThresholds, classify_frame, confirm_gap_turn,
    FRONT, LEFT, RIGHT, DOWN, UP, GAP,
)

CHECK = "check_turn_sequence"
DIRECTIONS = (LEFT, RIGHT, DOWN, UP)


def _hold_frames(config: dict, key: str, sample_fps: float, default_sec: float = 2.0) -> int:
    hs = config.get("face", {}).get("turn_sequence", {}).get("hold_seconds", {})
    secs = float(hs.get(key, default_sec))
    return max(1, math.floor(secs * sample_fps))


def _find_bracketed_gaps(labels):
    """(start, end) spans of GAP runs with a FRONT before AND a FRONT after."""
    spans = []
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] != GAP:
            i += 1
            continue
        j = i
        while j < n and labels[j] == GAP:
            j += 1
        front_before = any(labels[k] == FRONT for k in range(0, i))
        front_after = any(labels[k] == FRONT for k in range(j, n))
        if front_before and front_after:
            spans.append((i, j - 1))
        i = j
    return spans


def _is_subsequence(small, big):
    it = iter(big)
    return all(any(x == y for y in it) for x in small)


def check_turn_sequence(timeline, config, *, sample_fps=1.0, emit_row=None):
    th = TurnThresholds.from_config(config)
    ts_cfg = config.get("face", {}).get("turn_sequence", {})
    order_policy = ts_cfg.get("order_policy", "REVIEW")
    accepted_orders = ts_cfg.get("accepted_orders", {}) or {}

    if emit_row is None:
        def emit_row(check_name, status, reason, frame_index=None):
            return CheckRow("", "face_rgb", "", check_name, status, reason, frame_index)

    rows = []
    labels = [classify_frame(e, th) for e in timeline]

    # opening front-hold anchor
    front_initial_frames = _hold_frames(config, "front_initial", sample_fps, default_sec=5)
    opening = labels[:max(front_initial_frames, 1)]
    if FRONT not in labels:
        rows.append(emit_row(CHECK, "FAIL",
                             "no front-facing frame anywhere; cannot validate turns"))
        return rows
    if FRONT not in opening:
        rows.append(emit_row(CHECK + "_front_hold", "REVIEW",
                             "no clear front hold at the start of the video"))

    bracketed_gaps = _find_bracketed_gaps(labels)

    detected_counts = defaultdict(int)
    first_ts = {}
    for lab, e in zip(labels, timeline):
        if lab in DIRECTIONS:
            detected_counts[lab] += 1
            first_ts.setdefault(lab, e.get("timestamp_sec"))

    direction_status = {}
    for d in DIRECTIONS:
        need = _hold_frames(config, d, sample_fps)
        count = detected_counts[d]
        if count >= need:
            direction_status[d] = "PASS"
            rows.append(emit_row(f"{CHECK}_{d}", "PASS",
                                 f"{d}: {count} detected frames >= {need}"))
            continue
        confirmed = None
        for (gs, ge) in bracketed_gaps:
            confirmed = confirm_gap_turn(timeline, gs, ge, expected_direction=d)
            if confirmed == d:
                break
        if confirmed == d:
            direction_status[d] = "PASS"
            rows.append(emit_row(f"{CHECK}_{d}", "PASS",
                                 f"{d}: confirmed via bracketed detection gap"))
        elif bracketed_gaps and count == 0:
            direction_status[d] = "REVIEW"
            rows.append(emit_row(f"{CHECK}_{d}", "REVIEW",
                                 f"{d}: only a bracketed gap present; cannot confirm "
                                 f"without head-pose model (Phase 2)"))
        elif 0 < count < need:
            direction_status[d] = "REVIEW"
            rows.append(emit_row(f"{CHECK}_{d}", "REVIEW",
                                 f"{d}: {count} detected frames < {need} (hold too short?)"))
        else:
            direction_status[d] = "FAIL"
            rows.append(emit_row(f"{CHECK}_{d}", "FAIL",
                                 f"{d}: not detected (no frames, no bracketed gap)"))

    seen = [d for d in DIRECTIONS if direction_status.get(d) == "PASS"]
    seen_in_time = sorted(seen, key=lambda d: (first_ts.get(d) if first_ts.get(d)
                                               is not None else float("inf")))
    order_ok = False
    for _name, seq in accepted_orders.items():
        seq_turns = [s for s in seq if s in DIRECTIONS]
        if _is_subsequence(seen_in_time, seq_turns):
            order_ok = True
            break

    if not seen:
        rows.append(emit_row(f"{CHECK}_order", "FAIL", "no turns to order"))
    elif order_ok:
        rows.append(emit_row(f"{CHECK}_order", "PASS",
                             f"turn order ok: {' -> '.join(seen_in_time)}"))
    else:
        rows.append(emit_row(f"{CHECK}_order", order_policy,
                             f"turn order {' -> '.join(seen_in_time)} matches no accepted order"))

    statuses = list(direction_status.values()) + (
        ["PASS"] if order_ok else [order_policy if seen else "FAIL"])
    if any(s == "FAIL" for s in statuses):
        overall = "FAIL"
    elif any(s == "REVIEW" for s in statuses):
        overall = "REVIEW"
    else:
        overall = "PASS"
    present = [d for d in DIRECTIONS if direction_status.get(d) == "PASS"]
    rows.append(emit_row(CHECK, overall,
                         f"turns present: {present or 'none'}; "
                         f"order={'ok' if order_ok else 'mismatch'}"))
    return rows