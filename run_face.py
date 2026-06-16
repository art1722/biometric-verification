"""Run the face_rgb pipeline on ONE video and show the results.

Usage (from the project root — the folder containing qc/):

    python run_face.py 001_face_rgb.mp4
    python run_face.py path/to/video.mp4 --id 042 --sample-fps 1
    python run_face.py 001_face_rgb.mp4 --csv reports/face_001.csv

Requires: pip install mediapipe opencv-python pyyaml numpy
"""
import argparse
import csv
import os
import re
import sys
import time

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")

from qc.pipelines.face_rgb import run_face_rgb
from collections import Counter, defaultdict


STATUS_PRIORITY = {
    "SKIP": 0,    # not judged (expected for non-frontal frames / stub checks)
    "PASS": 1,
    "REVIEW": 2,
    "FAIL": 3,
}
# Note: PASS outranks SKIP on purpose. With segment-gated quality checks,
# SKIP rows are ROUTINE (every turning frame emits them), so a check that
# passed on all judged frames must aggregate to PASS, not SKIP. A check whose
# rows are ALL SKIP (never judged anywhere) still aggregates to SKIP.


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to a NNN_face_rgb.mp4")
    ap.add_argument("--id", default=None, help="volunteer id (else parsed from filename)")
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--sample-fps", type=float, default=1.0)
    ap.add_argument("--csv", default=None, help="optional: also write rows to this CSV")
    ap.add_argument("--summary", default=None,
                    help="optional: also write the summary (tally + angle range) to this text file")
    ap.add_argument("--result-csv", default=None,
                    help="optional: write final aggregated result CSV")
    ap.add_argument("--overlay", default=None,
                    help="optional: write a debug overlay video (bbox + landmarks "
                         "+ pose + per-check status drawn on each sampled frame) "
                         "to this .mp4 path")
    return ap.parse_args()

def parse_volunteer_id(path):
    """Pull NNN from a filename like 001_face_rgb.mp4; fallback to '000'."""
    m = re.match(r"(\d+)_face_rgb", os.path.basename(path))
    return m.group(1) if m else "000"

def print_rows(rows):
    print("volunteer_id, data_type, filename, check_level, check_name, status, reason, frame_index")
    for r in rows:
        print(", ".join(str(x) for x in r.as_tuple()))

def print_name_status(rows):
    tally = Counter((r.check_name, r.status) for r in rows)
    lines = ["\n=== tally (check_name -> status x count) ==="]
    for (name, status), n in sorted(tally.items()):
        lines.append(f"  {name:22} {status:7} x{n}")
    text = "\n".join(lines)
    print(text)
    return text

def print_angle(timeline):
    # timeline now has ONE entry per sampled frame, gaps included.
    # Gap frames carry yaw/pitch = None, so filter them out before min/max.
    measured = [t for t in timeline if t.get("yaw") is not None]
    gaps = [t for t in timeline if not t.get("face_detected")]
    lines = [f"\n=== head-pose angles collected: {len(measured)} of "
             f"{len(timeline)} frames ({len(gaps)} detection gaps) ==="]
    if measured:
        yaws = [t["yaw"] for t in measured]
        pitches = [t["pitch"] for t in measured]
        lines.append(f"  yaw   range: {min(yaws):+.1f} .. {max(yaws):+.1f}")
        lines.append(f"  pitch range: {min(pitches):+.1f} .. {max(pitches):+.1f}")
        lines.append("  (a real turn video should show yaw swinging negative->positive as the head"
                     " turns left->right, and pitch swinging as it looks down->up)")
    text = "\n".join(lines)
    print(text)
    return text

def print_csv(rows, args_csv=None):
    if args_csv:
        os.makedirs(os.path.dirname(args_csv) or ".", exist_ok=True)
        with open(args_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["volunteer_id", "data_type", "filename",
                        "check_level", "check_name", "status", "reason",
                        "frame_index"])
            for r in rows:
                # as_tuple() already ends with frame_index — do NOT append it
                # again (the old double-append produced an 8th unnamed column
                # that broke pandas/Excel parsing of the detail CSV).
                w.writerow(r.as_tuple())
        print(f"\nwrote {len(rows)} rows to {args_csv}")

def format_duration(seconds):
    """Human-readable elapsed time: '3.4s' or '1m 05.2s' for longer runs."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:04.1f}s"


def worst_status(statuses):
    """Return the most severe status."""
    statuses = [s for s in statuses if s]
    if not statuses:
        return "SKIP"
    return max(statuses, key=lambda s: STATUS_PRIORITY.get(s, -1))


def first_reason(group):
    """Pick one useful representative reason for a summarized check."""
    for preferred_status in ("FAIL", "REVIEW", "SKIP", "PASS"):
        for r in group:
            if r.status == preferred_status and r.reason:
                return r.reason
    return ""


LEVEL_ORDER = {"video": 0, "sequence": 1, "frame": 2}


def _ratio_final_status(counts, fail_ratio_max, review_ratio_max):
    """Final status for a FRAME-level check under the ratio policy.

    judged = PASS + FAIL + REVIEW (SKIPs are 'not judged' and excluded).
    FAIL    if fail/judged   > fail_ratio_max
    REVIEW  elif review/judged > review_ratio_max
    PASS    elif anything was judged
    SKIP    if nothing was judged at all
    """
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
    """Convert many CheckRow objects into one final row per check_name.

    aggregation_cfg: config["report"]["aggregation"] dict (or None). When
    present, FRAME-level checks use the ratio policy (_ratio_final_status);
    video/sequence checks always keep worst-status aggregation.
    """
    aggregation_cfg = aggregation_cfg or {}
    fail_ratio_max = float(aggregation_cfg.get("frame_fail_ratio", 0.0))
    review_ratio_max = float(aggregation_cfg.get("frame_review_ratio", 0.0))

    grouped = defaultdict(list)

    for r in rows:
        grouped[r.check_name].append(r)

    summaries = []

    def sort_key(item):
        check_name, group = item
        level = getattr(group[0], "level", "frame")
        return (LEVEL_ORDER.get(level, 99), check_name)

    for check_name, group in sorted(grouped.items(), key=sort_key):
        counts = Counter(r.status for r in group)
        level = getattr(group[0], "level", "frame")

        if level == "frame":
            final_status, ratio = _ratio_final_status(
                counts, fail_ratio_max, review_ratio_max)
            ratio_note = (f" [judged fail-ratio={ratio:.0%}"
                          f" (max {fail_ratio_max:.0%})]"
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
    """Create one final OVERALL verdict from all summarized checks."""
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
    """Summarize head-pose/timeline information for the final CSV."""
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
        "yaw_min": "",
        "yaw_max": "",
        "pitch_min": "",
        "pitch_max": "",
    }


def write_result_csv(path, rows, timeline, config=None):
    """Write final aggregated result CSV: one row per check + one OVERALL row."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    aggregation_cfg = (config or {}).get("report", {}).get("aggregation", {})
    check_summaries = summarize_rows_by_check(rows, aggregation_cfg)
    overall = summarize_overall(check_summaries)
    timeline_summary = summarize_timeline(timeline)

    if rows:
        volunteer_id = rows[0].volunteer_id
        data_type = rows[0].data_type
        filename = rows[0].filename
    else:
        volunteer_id = ""
        data_type = ""
        filename = ""

    fieldnames = [
        "volunteer_id",
        "data_type",
        "filename",
        "check_level",
        "check_name",
        "final_status",
        "total",
        "pass",
        "fail",
        "review",
        "skip",
        "reason",
        "frames_sampled",
        "detection_gaps",
        "yaw_min",
        "yaw_max",
        "pitch_min",
        "pitch_max",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for s in check_summaries:
            w.writerow({
                "volunteer_id": volunteer_id,
                "data_type": data_type,
                "filename": filename,
                **s,
                "frames_sampled": "",
                "detection_gaps": "",
                "yaw_min": "",
                "yaw_max": "",
                "pitch_min": "",
                "pitch_max": "",
            })

        w.writerow({
            "volunteer_id": volunteer_id,
            "data_type": data_type,
            "filename": filename,
            **overall,
            **timeline_summary,
        })

    print(f"wrote final result CSV to {path}")
    
    
def write_summary(path, video, vid, sample_fps, tally_text, angle_text, elapsed_text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Running face_rgb pipeline on: {video}\n")
        f.write(f"volunteer id: {vid} | sample_fps: {sample_fps}\n")
        f.write(tally_text + "\n")
        f.write(angle_text + "\n")
        f.write(f"\n=== done in {elapsed_text} ===\n")
    print(f"wrote summary to {path}")


def print_results(rows, timeline, args_csv=None):
    print_rows(rows)
    tally_text = print_name_status(rows)
    angle_text = print_angle(timeline)
    print_csv(rows, args_csv)
    return tally_text, angle_text


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"video not found: {args.video}")
    if not os.path.exists(args.config):
        sys.exit(f"config not found: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    vid = args.id or parse_volunteer_id(args.video)

    print(f"Running face_rgb pipeline on: {args.video}")
    print(f"volunteer id: {vid} | sample_fps: {args.sample_fps}\n")

    start = time.perf_counter()

    overlay = None
    if args.overlay:
        from qc.utils.overlay import OverlayWriter
        # play sampled frames back at the sampling rate (clamped to >=1 fps)
        
        # from qc.utils.overlay_below import OverlayWriter
        overlay = OverlayWriter(
            args.overlay, fps=args.sample_fps,
            volunteer_id=vid, filename=os.path.basename(args.video))

    rows, timeline = run_face_rgb(
        args.video, vid, config, sample_fps=args.sample_fps, overlay=overlay)

    if overlay is not None:
        overlay.close()
        print(f"overlay video: {args.overlay} ({overlay.frames_written} frames)")

    elapsed = time.perf_counter() - start

    tally_text, angle_text = print_results(rows, timeline, args_csv=args.csv)

    if args.result_csv:
        write_result_csv(args.result_csv, rows, timeline, config=config)

    elapsed_text = format_duration(elapsed)
    print(f"\n=== done in {elapsed_text} ===")
    
    if args.summary:
        write_summary(args.summary, args.video, vid, args.sample_fps,
                      tally_text, angle_text, elapsed_text)

if __name__ == "__main__":
    main()