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

REPORT_CHECK_ORDER = {
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

    # 4) Turn-protocol checks
    "check_turn_left": 300,
    "check_turn_right": 310,
    "check_turn_down": 320,
    "check_turn_up": 330,
    "check_turn_front": 340,
    "check_turn_sequence": 350,
}


def report_sort_key(check_name, level):
    """Stable researcher-facing order for result CSV rows.

    Known checks follow the real QC flow.
    Unknown checks fall back to level + check name so future checks still show.
    """
    if check_name in REPORT_CHECK_ORDER:
        return (0, REPORT_CHECK_ORDER[check_name])

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
        return report_sort_key(check_name, level)

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
    measured = [t for t in timeline if t.get("yaw") is not None]
    gaps = [t for t in timeline if not t.get("face_detected")]

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
    fields = ["volunteer_id", "data_type", "file_name", "frame_index",
              "time", "label_width", "label_height", "yaw", "pitch", "roll",
              "brightness", "blink_left", "blink_right", "sharpness"]

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
            ])
    if not quiet:
        print(f"wrote detail header CSV to {path}")


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