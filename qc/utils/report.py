"""Shared report aggregation + CSV writers for the face QC pipeline.

Extracted from run_face.py so BOTH the single-video CLI (run_face.py) and the
folder/batch runner (run_folder.py) call ONE implementation — no second copy
that can drift out of sync.

REVIEW note: the project eliminated the REVIEW status. The "review" counters
below are retained only so existing result/overall CSV columns stay stable;
in practice they are always 0. The batch global summary (run_folder.py) does
NOT expose a review column.
"""
from collections import Counter, defaultdict
import csv
import os


# PASS outranks SKIP on purpose: SKIP rows are routine (every turning frame
# emits them), so a check that passed on all judged frames aggregates to PASS.
# A check whose rows are ALL SKIP still aggregates to SKIP. REVIEW kept for
# backward-compat ranking only; no check emits it anymore.
STATUS_PRIORITY = {"SKIP": 0, "PASS": 1, "REVIEW": 2, "FAIL": 3}
LEVEL_ORDER = {"video": 0, "sequence": 1, "frame": 2}

REPORT_CHECK_ORDER_FACE = {
    # 1) Video/file-level checks
    "check_container": 10,
    "check_fps": 20,
    "check_duration": 30,
    "check_resolution": 40,
    "frames_sampled": 50,
    "frame_checks": 60,

    # 2) Face evidence / landmark availability
    "check_face_detected": 100,
    "check_head_fully": 110,

    # 3) Frontal-frame quality checks
    "check_face_size": 200,
    "check_eyes_open": 210,
    "check_brightness": 220,
    "check_face_blur": 230,
    "check_occlusion": 240,

    # 4) Turn-protocol checks
    "check_turn_left": 300,
    "check_turn_right": 310,
    "check_turn_down": 320,
    "check_turn_up": 330,
    "check_turn_front": 340,
    "check_turn_sequence": 350,
}

REPORT_CHECK_ORDER_WALK = {
    # 1) Video/file-level checks (same slots as face)
    "check_container": 10,
    "check_fps": 20,
    "check_duration": 30,
    "check_resolution": 40,
    "frames_sampled": 50,
    "frame_checks": 60,

    # 2) Person evidence / landmark availability
    "check_person_detected": 100,
    "check_single_person": 105,  # multiple-person -> whole-video FAIL (ratio-exempt)
    "check_person_fully": 110,

    # 3) Frame-quality checks (body_height takes the size slot, like check_face_size)
    "check_body_height": 200,
    "check_brightness": 230,
    "check_person_blur": 240,
    "check_occlusion": 250,

    # 4) Sequence-level walk-direction verdict
    "check_walk_direction": 300,
}


def report_sort_key(check_name, level, data_type=None):
    """Stable order for result CSV rows.

    Each pipeline has its OWN order dict so numbers are only meaningful within a
    modality (no cross-pipeline collisions on shared names like check_brightness).
    data_type picks the dict: a "walk*" data_type uses the walk order, everything
    else uses the face order.

    Known checks follow that pipeline's QC flow. Unknown checks fall back to
    level + check name so future checks still show.
    """
    if data_type and str(data_type).startswith("walk"):
        order = REPORT_CHECK_ORDER_WALK
    else:
        order = REPORT_CHECK_ORDER_FACE

    if check_name in order:
        return (0, order[check_name])

    return (1, LEVEL_ORDER.get(level, 99), check_name)


def worst_status(statuses):
    statuses = [s for s in statuses if s]
    if not statuses:
        return "SKIP"
    return max(statuses, key=lambda s: STATUS_PRIORITY.get(s, -1))


def first_reason(group):
    for preferred_status in ("FAIL", "REVIEW", "SKIP", "PASS"):
        for r in group:
            if r.status == preferred_status and r.reason:
                return r.reason
    return ""


def _ratio_final_status(counts, fail_ratio_max, review_ratio_max):
    judged = counts.get("PASS", 0) + counts.get("FAIL", 0) + counts.get("REVIEW", 0)
    if judged == 0:
        return "SKIP", None
    fail_ratio = counts.get("FAIL", 0) / judged
    review_ratio = counts.get("REVIEW", 0) / judged
    if fail_ratio > fail_ratio_max:
        return "FAIL", fail_ratio
    if review_ratio > review_ratio_max:
        return "REVIEW", review_ratio
    return "PASS", fail_ratio


def summarize_rows_by_check(rows, aggregation_cfg=None):
    aggregation_cfg = aggregation_cfg or {}
    fail_ratio_max = float(aggregation_cfg.get("frame_fail_ratio", 0.0))
    review_ratio_max = float(aggregation_cfg.get("frame_review_ratio", 0.0))
    per_check = aggregation_cfg.get("per_check_fail_ratio", {}) or {}

    grouped = defaultdict(list)
    for r in rows:
        grouped[r.check_name].append(r)

    summaries = []

    def sort_key(item):
        check_name, group = item
        level = getattr(group[0], "level", "frame")
        data_type = getattr(group[0], "data_type", None)
        return report_sort_key(check_name, level, data_type)

    for check_name, group in sorted(grouped.items(), key=sort_key):
        counts = Counter(r.status for r in group)
        level = getattr(group[0], "level", "frame")

        if level == "frame":
            this_fail_max = float(per_check.get(check_name, fail_ratio_max))
            final_status, ratio = _ratio_final_status(
                counts, this_fail_max, review_ratio_max)
            ratio_note = (f" [judged fail-ratio={ratio:.0%}"
                          f" (max {this_fail_max:.0%})]"
                          if ratio is not None else "")
        else:
            final_status = worst_status(r.status for r in group)
            ratio_note = ""

        summaries.append({
            "check_level": level,
            "check_name": check_name,
            "final_status": final_status,
            "total": len(group),
            "pass": counts.get("PASS", 0),
            "fail": counts.get("FAIL", 0),
            "review": counts.get("REVIEW", 0),
            "skip": counts.get("SKIP", 0),
            "reason": first_reason(group) + ratio_note,
        })

    return summaries


def summarize_overall(check_summaries):
    statuses = [s["final_status"] for s in check_summaries]
    final_status = worst_status(statuses)

    failed = [s["check_name"] for s in check_summaries if s["final_status"] == "FAIL"]
    reviewed = [s["check_name"] for s in check_summaries if s["final_status"] == "REVIEW"]

    if failed:
        reason = "failed checks: " + ", ".join(failed[:10])
        if len(failed) > 10:
            reason += f", ... ({len(failed)} total)"
    elif reviewed:
        reason = "review checks: " + ", ".join(reviewed[:10])
        if len(reviewed) > 10:
            reason += f", ... ({len(reviewed)} total)"
    else:
        reason = "all checks passed"

    counts = Counter(s["final_status"] for s in check_summaries)

    return {
        "check_level": "overall",
        "check_name": "OVERALL",
        "final_status": final_status,
        "total": len(check_summaries),
        "pass": counts.get("PASS", 0),
        "fail": counts.get("FAIL", 0),
        "review": counts.get("REVIEW", 0),
        "skip": counts.get("SKIP", 0),
        "reason": reason,
    }


def summarize_timeline(timeline):
    # MODALITY-AWARE (bug fix 2026-07-20): the old version hardcoded the FACE
    # keys (face_detected / yaw / pitch) but is also called for WALK timelines,
    # whose entries have neither -- so `not t.get("face_detected")` was True for
    # EVERY walk frame and detection_gaps always equalled frames_sampled
    # (observed: 126/126 on a clip where all 126 frames detected a person).
    # A "gap" is a sampled frame with no usable detection:
    #   face: face_detected is False        (face_rgb timeline entries)
    #   walk: body_scale is None (no pose)  (walk timeline entries; walk.py
    #         appends an entry for EVERY sampled frame, None scale on no-pose)
    # Detected by key inspection, not data_type, so no caller changes needed.
    if any("face_detected" in t for t in timeline):
        gaps = [t for t in timeline if not t.get("face_detected")]
    else:
        gaps = [t for t in timeline if t.get("body_scale") is None]

    measured = [t for t in timeline if t.get("yaw") is not None]

    if measured:
        yaws = [t["yaw"] for t in measured]
        pitches = [t["pitch"] for t in measured]
        return {
            "frames_sampled": len(timeline),
            "detection_gaps": len(gaps),
            "yaw_min": f"{min(yaws):.1f}",
            "yaw_max": f"{max(yaws):.1f}",
            "pitch_min": f"{min(pitches):.1f}",
            "pitch_max": f"{max(pitches):.1f}",
        }

    return {
        "frames_sampled": len(timeline),
        "detection_gaps": len(gaps),
        "yaw_min": "", "yaw_max": "", "pitch_min": "", "pitch_max": "",
    }


RESULT_FIELDNAMES = [
    "volunteer_id", "data_type", "filename", "check_level", "check_name",
    "final_status", "total", "pass", "fail", "review", "skip", "reason",
    "frames_sampled", "detection_gaps",
    "yaw_min", "yaw_max", "pitch_min", "pitch_max",
]


def _row_identity(rows):
    if rows:
        return rows[0].volunteer_id, rows[0].data_type, rows[0].filename
    return "", "", ""


def build_overall_record(rows, timeline, config=None):
    """Compute the OVERALL verdict + timeline summary WITHOUT writing a file.

    Used by run_folder.py so the batch summary derives from the SAME
    aggregation as the per-volunteer CSVs (no re-reading of files).
    """
    aggregation_cfg = (config or {}).get("report", {}).get("aggregation", {})
    check_summaries = summarize_rows_by_check(rows, aggregation_cfg)
    overall = summarize_overall(check_summaries)
    timeline_summary = summarize_timeline(timeline)
    vid, dtype, fname = _row_identity(rows)
    failed_checks = [s["check_name"] for s in check_summaries
                     if s["final_status"] == "FAIL"]
    # Same failed checks, but each paired with WHY it failed (the per-check
    # reason string). This is what the customer-facing summaries report: one
    # entry per failed check, so the reason gets its own column/field instead
    # of being crammed together. Crash rows are folded in too so a structural
    # failure still shows a reason.
    failed_checks_detail = [
        {"check_name": s["check_name"], "reason": s["reason"]}
        for s in check_summaries
        if s["final_status"] == "FAIL"
    ]
    return {
        "volunteer_id": vid,
        "data_type": dtype,
        "filename": fname,
        "final_status": overall["final_status"],
        "reason": overall["reason"],
        "pass": overall["pass"],
        "fail": overall["fail"],
        "skip": overall["skip"],
        "failed_checks": failed_checks,
        "failed_checks_detail": failed_checks_detail,
        **timeline_summary,
    }


def write_result_csv(result_path, overall_path, rows, timeline, config=None,
                     quiet=False):
    aggregation_cfg = (config or {}).get("report", {}).get("aggregation", {})
    check_summaries = summarize_rows_by_check(rows, aggregation_cfg)
    overall = summarize_overall(check_summaries)
    timeline_summary = summarize_timeline(timeline)
    volunteer_id, data_type, filename = _row_identity(rows)

    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    with open(result_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        w.writeheader()
        for s in check_summaries:
            w.writerow({
                "volunteer_id": volunteer_id, "data_type": data_type,
                "filename": filename, **s,
                "frames_sampled": "", "detection_gaps": "",
                "yaw_min": "", "yaw_max": "", "pitch_min": "", "pitch_max": "",
            })
    if not quiet:
        print(f"wrote final result CSV to {result_path}")

    os.makedirs(os.path.dirname(overall_path) or ".", exist_ok=True)
    with open(overall_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        w.writeheader()
        w.writerow({
            "volunteer_id": volunteer_id, "data_type": data_type,
            "filename": filename, **overall, **timeline_summary,
        })
    if not quiet:
        print(f"wrote overall CSV to {overall_path}")


def write_detail_header_csv(path, rows, timeline, quiet=False):
    vid, dtype, fname = _row_identity(rows)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Per-region occlusion skin-ratio columns (written by face_rgb.py onto each
    # timeline entry). Listed once here so the header and the row body stay in
    # sync — the dashboard reads these to draw the per-region occlusion chart.
    occ_cols = ["occ_forehead", "occ_left_eye", "occ_right_eye", "occ_nose",
                "occ_mouth", "occ_left_cheek", "occ_right_cheek"]
    fields = ["volunteer_id", "data_type", "file_name", "frame_index",
              "time", "label_width", "label_height", "yaw", "pitch", "roll",
              "brightness", "blink_left", "blink_right", "sharpness"] + occ_cols

    def fmt(v, nd=1):
        return "" if v is None else f"{v:.{nd}f}"

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for t in timeline:
            w.writerow([
                vid, dtype, fname, t["frame_index"],
                fmt(t.get("timestamp_sec"), 3),
                fmt(t.get("label_width"), 0), fmt(t.get("label_height"), 0),
                fmt(t.get("yaw")), fmt(t.get("pitch")), fmt(t.get("roll")),
                fmt(t.get("brightness")),
                fmt(t.get("blink_left"), 3), fmt(t.get("blink_right"), 3),
                fmt(t.get("sharpness")),
                # Occlusion ratios at 2 decimals (0.00-1.00); "" when absent.
                *[fmt(t.get(c), 2) for c in occ_cols],
            ])
    if not quiet:
        print(f"wrote detail header CSV to {path}")


def write_walk_detail_header_csv(path, rows, timeline, quiet=False):
    """Per-frame walk series CSV: brightness + body-box dims per sampled frame.

    The walk analog of write_detail_header_csv (which is face-specific:
    yaw/pitch/occlusion columns). Walk's per-frame measured quantity in this MVP
    is brightness (plus the body box the brightness region came from), so those
    are the columns. Same one-row-per-timeline-entry shape and utf-8-sig CSV as
    the face writer, so the dashboard reads it identically.
    """
    vid, dtype, fname = _row_identity(rows)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["volunteer_id", "data_type", "file_name", "frame_index", "time",
              "brightness", "body_width", "body_height_px"]

    def fmt(v, nd=1):
        return "" if v is None else f"{v:.{nd}f}"

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for t in timeline:
            w.writerow([
                vid, dtype, fname, t["frame_index"],
                fmt(t.get("timestamp_sec"), 3),
                fmt(t.get("brightness"), 0),
                fmt(t.get("body_width"), 0),
                fmt(t.get("body_height_px"), 0),
            ])
    if not quiet:
        print(f"wrote walk detail header CSV to {path}")


def write_detail_csv(path, rows, timeline=None, quiet=False):
    t_by_frame = {}
    if timeline:
        t_by_frame = {t["frame_index"]: t["timestamp_sec"] for t in timeline}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["volunteer_id", "data_type", "filename",
                    "frame_index", "time",
                    "check_level", "check_name", "status", "reason"])
        for r in rows:
            t = t_by_frame.get(r.frame_index, "")
            t_str = "" if t == "" or t is None else f"{t:.3f}"
            vol, dtype, fname, level, name, status, reason, fidx = r.as_tuple()
            w.writerow([vol, dtype, fname, fidx, t_str,
                        level, name, status, reason])
    if not quiet:
        print(f"wrote {len(rows)} rows to {path}")


# --------------------------------------------------------------------------
# Customer-facing summaries — what we hand back to the data provider.
#
# Two grains, shared by every modality (face / palm / walk):
#
#   <modality>_summary.csv  (e.g. face_summary.csv): the per-modality report.
#       ONE ROW PER FAILED CHECK so each failure's reason gets its own cell
#       (no "check: reason | check: reason" cramming). A video that passes
#       everything contributes exactly ONE row with status PASS and blank
#       check/reason. A crashed video contributes one row carrying the error
#       text as the reason. Internal-only stats (yaw/pitch, frames_sampled,
#       timing) are deliberately NOT here — they live in each participant's
#       own report and are not the customer's concern.
#
#   all_summary.csv / all_summary.json: the cross-modal roll-up of PROBLEM
#       cases only (FAIL) from every modality. The CSV is the same flat
#       per-check grain as above (minus PASS rows) so a reviewer can pivot it
#       in Excel; the JSON nests one object per video with a `failures` array,
#       which is the clean shape for a customer's tooling to consume.
#
# Both writers live here (not in run_folder.py) so the future palm/walk
# runners produce IDENTICALLY-shaped output by calling the SAME classes.
# --------------------------------------------------------------------------

# Statuses that mean "needs attention" -> included in the all_* roll-up.
ATTENTION_STATUSES = frozenset({"FAIL"})

# Per-check grain, shared by <modality>_summary.csv and all_summary.csv.
# One row = one failed check (or one PASS placeholder row).
PERCHECK_FIELDNAMES = [
    "data_type",        # face_rgb | palm | walk_* — which modality
    "volunteer_id",
    "filename",
    "overall_status",   # the video's verdict: PASS | FAIL | SKIP
    "check_name",       # the failed check; blank on a PASS placeholder row
    "reason",           # why it failed; blank on PASS
]


def expand_to_check_rows(rec):
    """Turn ONE per-video overall record into per-check summary rows.

    `rec` is a build_overall_record() dict (it carries failed_checks_detail).
    Returns a list of flat dicts with PERCHECK_FIELDNAMES:

      - FAIL video  -> one row per failed check (check_name + reason filled)
      - crashed video -> FAIL rows via the synthetic processing_error check
      - PASS video  -> exactly one row, check_name + reason blank

    Keeping this as a pure function (record -> rows) means the same expansion
    feeds both the per-modality CSV and the all_summary CSV with no second
    copy of the logic.
    """
    base = {
        "data_type": rec.get("data_type") or "",
        "volunteer_id": rec.get("volunteer_id") or "",
        "filename": rec.get("filename") or "",
        "overall_status": rec.get("final_status") or rec.get("overall_status") or "",
    }
    status = base["overall_status"].upper()

    detail = rec.get("failed_checks_detail") or []
    if status == "FAIL" and detail:
        return [{**base, "check_name": d["check_name"], "reason": d["reason"]}
                for d in detail]

    # PASS (or a FAIL with no detail, which shouldn't happen) -> one clean row.
    return [{**base, "check_name": "", "reason": ""}]


class SummaryWriter:
    """Write ONE modality's per-check summary CSV (face_summary.csv, ...).

    Open once per run, call .add(rec) with each video's build_overall_record()
    dict; it expands the record to per-check rows and writes them. PASS videos
    get their single placeholder row, so this file is the COMPLETE per-modality
    report (not FAIL-only — that's what all_summary is for).
    """

    def __init__(self, path, mode="w"):
        self.path = path
        self.rows_written = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        need_header = (mode == "w" or not os.path.exists(path)
                       or os.path.getsize(path) == 0)
        self._f = open(path, mode, newline="", encoding="utf-8-sig")
        self._w = csv.DictWriter(self._f, fieldnames=PERCHECK_FIELDNAMES,
                                 extrasaction="ignore")
        if need_header:
            self._w.writeheader()
            self._f.flush()

    def add(self, rec):
        """Expand one overall record to per-check rows and write them."""
        for row in expand_to_check_rows(rec):
            self._w.writerow(row)
            self.rows_written += 1
        self._f.flush()  # a crash mid-run still leaves a valid partial CSV

    def close(self):
        try:
            self._f.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class AllSummaryWriter:
    """Cross-modal COMPLETE roll-up in BOTH csv and json.

    Fed the SAME per-video records as SummaryWriter, and now keeps EVERY
    processed media file (PASS and FAIL) so all_summary is the full
    result set, not just a problem worklist. Writes two files that stay in
    lock-step:

      <stem>.csv  — flat per-check rows (Excel-pivotable). A FAIL media contributes
                    one row per failed check; a PASS media contributes exactly one
                    row (blank check_name/reason).
      <stem>.json — list of {data_type, volunteer_id, filename, overall_status,
                    failures: [{check_name, reason}, ...]} ; one object per
                    media file. failures is [] for a PASS (which
                    carries `error` instead).

    To get just the problem set, filter overall_status == FAIL (the API's
    /results?status=FAIL does this).

    JSON is rewritten in full on each add() (the registry is held in memory)
    so the file on disk is always valid; for ~1,500 volunteers this is cheap.

    csv_mode/json default to "w" (fresh). Pass mode="a" so palm/walk append to
    the same all_summary after face has run — the JSON is re-loaded and merged.
    """

    def __init__(self, stem, mode="w"):
        self.csv_path = f"{stem}.csv"
        self.json_path = f"{stem}.json"
        self.kept = 0
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)

        # JSON registry: keyed (data_type, volunteer_id, filename) so a re-run
        # of one modality replaces its own rows instead of duplicating them.
        self._records = {}
        if mode == "a" and os.path.exists(self.json_path):
            try:
                import json
                with open(self.json_path, "r", encoding="utf-8") as jf:
                    for obj in json.load(jf):
                        key = (obj.get("data_type"), obj.get("volunteer_id"),
                               obj.get("filename"))
                        self._records[key] = obj
            except (OSError, ValueError):
                self._records = {}  # unreadable/partial -> start clean

        # CSV: append if asked AND a non-empty file exists, else fresh+header.
        csv_append = (mode == "a" and os.path.exists(self.csv_path)
                      and os.path.getsize(self.csv_path) > 0)
        self._cf = open(self.csv_path, "a" if csv_append else "w",
                        newline="", encoding="utf-8-sig")
        self._cw = csv.DictWriter(self._cf, fieldnames=PERCHECK_FIELDNAMES,
                                  extrasaction="ignore")
        if not csv_append:
            self._cw.writeheader()
            self._cf.flush()

    def add(self, rec):
        """Keep `rec` and write its rows: one row per failed check for a FAIL,
        and ONE PASS row for a clean media file.

        Previously all_summary kept only FAIL (a reviewer worklist). It now
        records EVERY processed media file so all_summary is the COMPLETE result
        set: a PASS video/image contributes exactly one row (overall_status=PASS,
        blank check_name/reason) and one JSON object with an empty failures list.
        Callers filter to the problem set with status=FAIL on the API/CSV side.

        Returns True (every record is now kept)."""
        status = (rec.get("final_status") or rec.get("overall_status") or "").upper()

        check_rows = expand_to_check_rows(rec)
        for row in check_rows:
            self._cw.writerow(row)
        self._cf.flush()

        key = (rec.get("data_type") or "", rec.get("volunteer_id") or "",
               rec.get("filename") or "")
        self._records[key] = {
            "data_type": rec.get("data_type") or "",
            "volunteer_id": rec.get("volunteer_id") or "",
            "filename": rec.get("filename") or "",
            "overall_status": status,
            "failures": [
                {"check_name": r["check_name"], "reason": r["reason"]}
                for r in check_rows if r["check_name"]  # drop blank placeholder
            ] if status == "FAIL" else [],
            # Crash text is carried as a "processing_error" failed check (see
            # run_folder.error_row), so it already appears in `failures` above.
            "error": rec.get("error", ""),
        }
        self._flush_json()
        self.kept += 1
        return True

    def _flush_json(self):
        import json
        # Stable order: by modality, then volunteer id, then filename.
        ordered = sorted(self._records.values(),
                         key=lambda o: (o["data_type"], o["volunteer_id"],
                                        o["filename"]))
        tmp = self.json_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as jf:
            json.dump(ordered, jf, ensure_ascii=False, indent=2)
        os.replace(tmp, self.json_path)  # atomic: never a half-written json

    def close(self):
        try:
            self._cf.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False