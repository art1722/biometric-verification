"""check_turn_sequence_seg — SEGMENTATION-FIRST version.

Question this version answers:
    "Does the video consist of the EXACT scripted sequence of held positions,
     in order, each held long enough, with returns to front between movements
     and nothing extra?"

Difference from the presence-first version
------------------------------------------
Presence-first asks "did each turn happen, in order?" and ignores the structure
between turns. Segmentation-first reconstructs the ACTUAL sequence of positions
the head moved through, then matches that whole structure against the scripted
accepted_order. It is stricter: it can reject a video that contains all four
turns but in a malformed structure (e.g. front->left->right with no return to
front between, or an extra unscripted movement, or a hold that is too short).

How it works
------------
1. Classify every frame (shared core).
2. Collapse the frame labels into SEGMENTS: maximal runs of the same position,
   ignoring short noise and absorbing GAP runs into an adjacent turn when the
   gap is bracketed by that turn's approach (a profile-peak gap belongs to the
   turn it sits inside).
3. Drop segments shorter than their hold minimum (too brief to be a real hold),
   tracking them as 'short_holds' for reporting.
4. Compare the resulting ordered list of segment labels against each
   accepted_order. Exact structural match -> PASS; mismatch -> order_policy.

This version carries far more state (segment building, gap absorption, noise
filtering) and therefore more places for subtle bugs — which is exactly why it
ships with the same test file and why presence-first is the safer default until
hold-duration / structural strictness is confirmed required.

Emits per-segment info rows + an overall row.
"""

from __future__ import annotations

import math

from qc.schemas import CheckRow
from qc.checks._turn_common import (
    TurnThresholds, classify_frame, confirm_gap_turn,
    FRONT, LEFT, RIGHT, DOWN, UP, GAP, NEUTRALish, is_turn,
)

CHECK = "check_turn_sequence"   # same public name; pipeline picks ONE module
DIRECTIONS = (LEFT, RIGHT, DOWN, UP)


def _hold_frames(config, key, sample_fps, default_sec=2.0):
    hs = config.get("face", {}).get("turn_sequence", {}).get("hold_seconds", {})
    secs = float(hs.get(key, default_sec))
    return max(1, math.floor(secs * sample_fps))


def _build_segments(labels, timeline):
    """Collapse per-frame labels into [(label, start_idx, end_idx, n_frames)].

    - Contiguous same-label frames merge into one segment.
    - A GAP run is absorbed into the turn it sits between when both neighbours
      are the SAME turn direction (front -> left -> [gap] -> left ...), or when
      one neighbour is a turn and the other is front (the gap is the peak of
      that turn): the gap takes the turn's label. An unbracketed/ambiguous gap
      keeps label GAP.
    - NEUTRALish (dead-band) frames are treated as transition noise and merged
      into whichever neighbouring segment they touch; a standalone NEUTRALish
      run stays as its own 'mid' segment.
    """
    n = len(labels)
    # First pass: relabel absorbable gaps.
    lab = list(labels)
    i = 0
    while i < n:
        if lab[i] != GAP:
            i += 1
            continue
        j = i
        while j < n and lab[j] == GAP:
            j += 1
        prev_lab = lab[i - 1] if i > 0 else None
        next_lab = lab[j] if j < n else None
        turn_neighbours = {x for x in (prev_lab, next_lab) if x in DIRECTIONS}
        if len(turn_neighbours) == 1:
            # gap is the peak of a single turn direction -> absorb
            d = turn_neighbours.pop()
            for k in range(i, j):
                lab[k] = d
        # else: ambiguous gap (e.g. front..gap..front with no turn shoulder, or
        # two different turn neighbours) -> leave as GAP for honest reporting.
        i = j

    # Second pass: collapse runs.
    segs = []
    i = 0
    while i < n:
        cur = lab[i]
        j = i
        while j < n and lab[j] == cur:
            j += 1
        segs.append((cur, i, j - 1, j - i))
        i = j
    return segs


def check_turn_sequence_seg(timeline, config, *, sample_fps=1.0, emit_row=None):
    th = TurnThresholds.from_config(config)
    ts_cfg = config.get("face", {}).get("turn_sequence", {})
    order_policy = ts_cfg.get("order_policy", "REVIEW")
    accepted_orders = ts_cfg.get("accepted_orders", {}) or {}

    if emit_row is None:
        def emit_row(check_name, status, reason, frame_index=None):
            return CheckRow("", "face_rgb", "", check_name, status, reason, frame_index)

    rows = []
    labels = [classify_frame(e, th) for e in timeline]

    if FRONT not in labels:
        rows.append(emit_row(CHECK, "FAIL",
                             "no front-facing frame anywhere; cannot validate turns"))
        return rows

    segs = _build_segments(labels, timeline)

    # Drop too-short HOLD segments (turns/fronts shorter than their minimum).
    # Keep them recorded as short_holds for the report.
    kept = []
    short_holds = []
    for (label, s, e, nfr) in segs:
        if label in DIRECTIONS:
            need = _hold_frames(config, label, sample_fps)
        elif label == FRONT:
            # front holds: use front_between_movements as the generic minimum
            need = _hold_frames(config, "front_between_movements", sample_fps)
        else:
            need = 1  # GAP / mid: no hold minimum
        if nfr >= need:
            kept.append((label, s, e, nfr))
        else:
            short_holds.append((label, nfr, need))

    # The structural sequence = kept segment labels, dropping 'mid' noise.
    structure = [lab for (lab, _s, _e, _n) in kept if lab != NEUTRALish]

    # Report any unconfirmable GAP segments (Phase 1 cannot positively confirm).
    gap_segs = [(s, e) for (lab, s, e, _n) in kept if lab == GAP]
    has_unconfirmed_gap = False
    for (s, e) in gap_segs:
        if confirm_gap_turn(timeline, s, e) is None:
            has_unconfirmed_gap = True

    # Compare structure against accepted orders (exact match on the turn spine).
    def spine(seq):
        # the meaningful structure: collapse consecutive duplicate 'front's,
        # keep turns; compare turn order AND that fronts separate them.
        out = []
        for x in seq:
            if not out or out[-1] != x:
                out.append(x)
        return out

    got_spine = spine([x for x in structure if x in (FRONT,) + DIRECTIONS])
    match_name = None
    for name, seq in accepted_orders.items():
        if spine(seq) == got_spine:
            match_name = name
            break

    # Emit a structural summary row.
    rows.append(emit_row(f"{CHECK}_structure", "PASS" if match_name else order_policy,
                         f"observed: {' -> '.join(got_spine) or 'none'}"
                         + (f" (matches '{match_name}')" if match_name else
                            "; matches no accepted order")))

    # Short-hold reporting.
    for (label, nfr, need) in short_holds:
        if label in DIRECTIONS or label == FRONT:
            rows.append(emit_row(f"{CHECK}_{label}_hold", "REVIEW",
                                 f"{label} held {nfr} frame(s) < {need} (too short)"))

    # Presence cross-check: which directions appear in the kept structure.
    present = [d for d in DIRECTIONS if d in structure]
    missing = [d for d in DIRECTIONS if d not in structure]
    for d in missing:
        rows.append(emit_row(f"{CHECK}_{d}", "FAIL", f"{d}: not present in segments"))
    for d in present:
        rows.append(emit_row(f"{CHECK}_{d}", "PASS", f"{d}: present as a segment"))

    # Overall verdict.
    if missing and not has_unconfirmed_gap:
        overall = "FAIL"
    elif not match_name or has_unconfirmed_gap or short_holds:
        overall = "REVIEW" if (match_name or has_unconfirmed_gap) else order_policy
    else:
        overall = "PASS"
    rows.append(emit_row(CHECK, overall,
                         f"structure {'ok' if match_name else 'mismatch'}; "
                         f"present={present or 'none'}; "
                         f"missing={missing or 'none'}; "
                         f"short_holds={len(short_holds)}; "
                         f"unconfirmed_gap={has_unconfirmed_gap}"))
    return rows