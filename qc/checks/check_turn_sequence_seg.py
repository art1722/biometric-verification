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


def _bridge_short_interruptions(labels, max_frames):
    """Remove tiny label flickers inside a stable segment.

    Example:
        down, down, front, down, down
    becomes:
        down, down, down, down, down

    This is for threshold jitter, not for real protocol holds.
    """
    lab = list(labels)
    stable = (FRONT,) + DIRECTIONS

    changed = True
    while changed:
        changed = False
        n = len(lab)
        i = 0

        while i < n:
            j = i + 1
            while j < n and lab[j] == lab[i]:
                j += 1

            run_label = lab[i]
            run_len = j - i
            prev_label = lab[i - 1] if i > 0 else None
            next_label = lab[j] if j < n else None

            if (
                run_len <= max_frames
                and prev_label == next_label
                and prev_label in stable
                and run_label != prev_label
            ):
                for k in range(i, j):
                    lab[k] = prev_label
                changed = True

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


def _build_segments(labels, timeline, *, config=None, sample_fps=1.0):
    """Collapse per-frame labels into [(label, start_idx, end_idx, n_frames)].

    - Contiguous same-label frames merge into one segment.
    - A GAP run is absorbed into the turn it sits between (see _absorb_gaps).
    - NEUTRALish (dead-band) frames are treated as transition noise and merged
      into whichever neighbouring segment they touch; a standalone NEUTRALish
      run stays as its own 'mid' segment.
    """
    n = len(labels)
    lab = _absorb_gaps(labels)

    ts_cfg = config.get("face", {}).get("turn_sequence", {}) if config else {}
    tol_cfg = ts_cfg.get("tolerance", {}) or {}

    jitter_sec = float(tol_cfg.get("label_jitter_tolerance_sec", 0.15))
    max_jitter_frames = max(1, math.ceil(jitter_sec * sample_fps))

    lab = _bridge_short_interruptions(lab, max_jitter_frames)

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
    """Return only 5 researcher-facing turn rows:

    - check_turn_left
    - check_turn_right
    - check_turn_down
    - check_turn_up
    - check_turn_sequence

    Direction rows merge detection + hold duration.
    Sequence row checks the scripted order/structure.
    """

    sample_fps = float(sample_fps or 1.0)
    if sample_fps <= 0:
        sample_fps = 1.0

    th = TurnThresholds.from_config(config)
    ts_cfg = config.get("face", {}).get("turn_sequence", {}) or {}

    order_policy = str(ts_cfg.get("order_policy", "FAIL")).upper()
    short_hold_policy = str(ts_cfg.get("short_hold_policy", "FAIL")).upper()
    unconfirmed_gap_policy = str(
        ts_cfg.get("unconfirmed_gap_policy", "FAIL")
    ).upper()

    accepted_orders = ts_cfg.get("accepted_orders", {}) or {}
    hold_seconds_cfg = ts_cfg.get("hold_seconds", {}) or {}

    if emit_row is None:
        def emit_row(check_name, status, reason, frame_index=None):
            return CheckRow(
                "",
                "face_rgb",
                "",
                check_name,
                status,
                reason,
                frame_index,
            )

    def hold_seconds(key, default_sec=2.0):
        return float(hold_seconds_cfg.get(key, default_sec))

    def sec_from_frames(n_frames):
        return n_frames / sample_fps

    def fmt_sec(seconds):
        return f"{seconds:.1f}s"

    def spine(seq):
        """Collapse consecutive duplicate labels but preserve front separators."""
        out = []
        for x in seq:
            if not out or out[-1] != x:
                out.append(x)
        return out
    
    def find_contiguous_subsequence(haystack, needle):
        """Return (start, end_exclusive) if needle appears contiguously in haystack.

        Example:
            haystack = [right, front, left, front, right, front, up, front, down, front]
            needle   = [front, left, front, right, front, up, front, down, front]

        returns:
            (1, 10)

        Extra labels before or after the accepted sequence are allowed.
        Extra labels inside the accepted sequence are not allowed.
        """
        if not needle:
            return None

        n = len(needle)
        if len(haystack) < n:
            return None

        for start in range(0, len(haystack) - n + 1):
            if haystack[start:start + n] == needle:
                return start, start + n

        return None

    rows = []

    labels = [classify_frame(e, th) for e in timeline]

    # Always emit the same 5 rows, even when the video is unusable.
    if FRONT not in labels:
        for d in DIRECTIONS:
            rows.append(emit_row(
                f"check_turn_{d}",
                "FAIL",
                f"{d} turn cannot be validated; no front-facing frame found",
            ))

        rows.append(emit_row(
            CHECK,
            "FAIL",
            "no front-facing frame anywhere; cannot validate turn sequence",
        ))
        return rows

    segs = _build_segments(
        labels,
        timeline,
        config=config,
        sample_fps=sample_fps,
    )

    # Split segments into:
    # - valid hold segments
    # - detected-but-too-short segments
    kept = []
    short_holds = []

    for label, s, e, nfr in segs:
        if label in DIRECTIONS:
            need = _hold_frames(config, label, sample_fps)
        elif label == FRONT:
            # Keep existing behavior: all front holds use the generic
            # front-between-movements minimum.
            need = _hold_frames(config, "front_between_movements", sample_fps)
        else:
            # GAP / mid / unknown labels do not have hold requirements.
            need = 1

        if nfr >= need:
            kept.append((label, s, e, nfr, need))
        else:
            short_holds.append((label, s, e, nfr, need))

    # ------------------------------------------------------------------
    # 1) Direction rows: detection + hold duration merged into one row.
    # ------------------------------------------------------------------
    for d in DIRECTIONS:
        valid_turns = [
            (s, e, nfr, need)
            for label, s, e, nfr, need in kept
            if label == d
        ]

        short_turns = [
            (s, e, nfr, need)
            for label, s, e, nfr, need in short_holds
            if label == d
        ]

        if valid_turns:
            # Report the longest valid hold for readability.
            s, e, nfr, need = max(valid_turns, key=lambda x: x[2])
            got_sec = sec_from_frames(nfr)
            need_sec = hold_seconds(d, 2.0)

            rows.append(emit_row(
                f"check_turn_{d}",
                "PASS",
                (
                    f"{d} detected; "
                    f"hold={fmt_sec(got_sec)} >= {fmt_sec(need_sec)}"
                ),
            ))

        elif short_turns:
            # Direction happened, but the hold was too short.
            s, e, nfr, need = max(short_turns, key=lambda x: x[2])
            got_sec = sec_from_frames(nfr)
            need_sec = hold_seconds(d, 2.0)

            rows.append(emit_row(
                f"check_turn_{d}",
                short_hold_policy,
                (
                    f"{d} detected but hold too short: "
                    f"{fmt_sec(got_sec)} < {fmt_sec(need_sec)}"
                ),
            ))

        else:
            rows.append(emit_row(
                f"check_turn_{d}",
                "FAIL",
                f"{d} turn not detected",
            ))

    # ------------------------------------------------------------------
    # 2) Sequence row: order/structure only.
    #
    # Direction short-hold failures are already reported in direction rows,
    # so direction labels are allowed to count for sequence order even when
    # their hold is too short.
    #
    # Front short-hold problems still affect sequence, because there is no
    # separate researcher-facing "check_front_hold" row.
    # ------------------------------------------------------------------
    order_labels = []
    front_short_holds = []

    for label, s, e, nfr in segs:
        if label in DIRECTIONS:
            # Count direction evidence for order even if the hold is short.
            order_labels.append(label)

        elif label == FRONT:
            need = _hold_frames(config, "front_between_movements", sample_fps)
            if nfr >= need:
                order_labels.append(label)

        elif label == GAP:
            # GAP is handled separately below.
            continue

        elif label == NEUTRALish:
            # Transition noise; not part of the protocol spine.
            continue

    got_spine = spine([
        x for x in order_labels
        if x in (FRONT,) + DIRECTIONS
    ])

    match_name = None
    match_span = None
    matched_spine = None

    for name, seq in accepted_orders.items():
        accepted_spine = spine(seq)
        span = find_contiguous_subsequence(got_spine, accepted_spine)

        if span is not None:
            match_name = name
            match_span = span
            matched_spine = accepted_spine
            break

    # Unabsorbed GAP segments are still a protocol uncertainty.
    gap_segs = [(s, e) for (label, s, e, _nfr) in segs if label == GAP]
    has_unconfirmed_gap = False

    for s, e in gap_segs:
        if confirm_gap_turn(timeline, s, e) is None:
            has_unconfirmed_gap = True
            break

    observed = " -> ".join(got_spine) if got_spine else "none"

    if not match_name:
        seq_status = order_policy
        seq_reason = (
            f"sequence order mismatch; observed: {observed}; "
            "accepted sequence not found inside observed sequence"
        )

    elif has_unconfirmed_gap:
        start, end = match_span
        matched = " -> ".join(matched_spine)

        seq_status = unconfirmed_gap_policy
        seq_reason = (
            f"accepted sequence found: '{match_name}' "
            f"at observed window {start}:{end}; "
            f"matched: {matched}; "
            "but contains an unconfirmed gap"
        )

    else:
        start, end = match_span
        matched = " -> ".join(matched_spine)

        seq_status = "PASS"
        seq_reason = (
            f"accepted sequence found: '{match_name}' "
            f"at observed window {start}:{end}; "
            f"matched: {matched}; "
            f"observed: {observed}"
        )

    rows.append(emit_row(CHECK, seq_status, seq_reason))

    return rows