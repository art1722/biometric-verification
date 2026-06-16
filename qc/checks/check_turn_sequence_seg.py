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


def _absorb_gaps(labels):
    """First pass of segmentation: relabel absorbable GAP runs.

    A GAP run is absorbed into a turn when its neighbours include exactly ONE
    turn direction (front -> left -> [gap] -> left, or left -> [gap] -> front,
    or a trailing/leading gap with a single turn shoulder): the gap takes that
    turn's label, because a detected over-floor frame adjacent to the gap is
    positive evidence the gap is that turn's profile peak. An unbracketed gap
    (front..gap..front) or an ambiguous one (left..gap..right) keeps label GAP.

    Returns the relabeled list (input is not mutated). This is the SINGLE
    definition of "expected gap" — both segment building and the per-frame
    detection-row split (split_gap_frames) use it, so the two can never
    disagree about which gaps a turn explains.
    """
    n = len(labels)
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
    return lab


def split_gap_frames(timeline, config):
    """Classify every no-pose frame of the timeline as an EXPECTED or
    UNEXPECTED gap, using the same absorption rule as segment building.

    expected   : the gap run was absorbed into a turn (>= 1 detected
                 over-floor frame immediately adjacent) -> the protocol asked
                 the face to be unmeasurable here; not a defect.
    unexpected : no adjacent turn evidence -> the face should have been
                 measurable here; a real defect.

    Returns (expected, unexpected):
        expected   : dict {timeline_index: turn_direction}
        unexpected : list [timeline_index]

    Note: a frame counts as a gap here when classify_frame says GAP, i.e.
    face_detected is False OR the pose could not be computed. The caller
    decides which of those frames have a detection row to re-status.
    """
    th = TurnThresholds.from_config(config)
    labels = [classify_frame(e, th) for e in timeline]
    absorbed = _absorb_gaps(labels)
    expected: dict = {}
    unexpected: list = []
    for i, orig in enumerate(labels):
        if orig != GAP:
            continue
        if absorbed[i] in DIRECTIONS:
            expected[i] = absorbed[i]
        else:
            unexpected.append(i)
    return expected, unexpected


_VALID_STATUSES = {"PASS", "FAIL", "REVIEW", "SKIP"}


def apply_gap_split_to_detection_rows(rows, timeline, gap_row_positions, config):
    """Re-status the check_face_detected rows of no-face frames in place.

    Args:
        rows: the pipeline's full list[CheckRow] (mutated in place).
        timeline: the per-frame timeline the rows were built from.
        gap_row_positions: dict {timeline_index: index into rows} pointing at
            the check_face_detected row emitted for each NO-FACE frame.
            (Multiple-face frames must NOT be included — that is a different
            defect class and stays at its original status.)
        config: loaded config dict; policy read from face.checks.detection:
            expected_gap_status   (default SKIP)
            unexpected_gap_status (default FAIL)
    """
    from dataclasses import replace

    det_cfg = (config.get("face", {}).get("checks", {}).get("detection", {}) or {})
    exp_status = str(det_cfg.get("expected_gap_status", "SKIP")).upper()
    unexp_status = str(det_cfg.get("unexpected_gap_status", "FAIL")).upper()
    if exp_status not in _VALID_STATUSES:
        exp_status = "SKIP"
    if unexp_status not in _VALID_STATUSES:
        unexp_status = "FAIL"

    expected, unexpected = split_gap_frames(timeline, config)

    for t_idx, r_idx in gap_row_positions.items():
        row = rows[r_idx]
        if t_idx in expected:
            rows[r_idx] = replace(
                row, status=exp_status,
                reason=f"{row.reason}; expected gap (peak of {expected[t_idx]} turn)")
        elif t_idx in unexpected:
            rows[r_idx] = replace(
                row, status=unexp_status,
                reason=f"{row.reason}; unexpected gap (no adjacent turn evidence)")
        # else: timeline entry wasn't a gap by classify_frame (shouldn't
        # happen for a no-face frame) — leave the row untouched.


def _build_segments(labels, timeline):
    """Collapse per-frame labels into [(label, start_idx, end_idx, n_frames)].

    - Contiguous same-label frames merge into one segment.
    - A GAP run is absorbed into the turn it sits between (see _absorb_gaps).
    - NEUTRALish (dead-band) frames are treated as transition noise and merged
      into whichever neighbouring segment they touch; a standalone NEUTRALish
      run stays as its own 'mid' segment.
    """
    n = len(labels)
    lab = _absorb_gaps(labels)

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