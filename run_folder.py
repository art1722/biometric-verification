"""Batch runner: QC every *_face_rgb.mp4 in a folder.

For the real project scale (~1,500 volunteers) this walks an input folder,
runs the face_rgb pipeline on each RGB video, writes the SAME per-volunteer
report files that run_face.py produces, and appends one row per video to a
single global summary CSV.

Design choices (confirmed with the team):
  - Reuses qc.utils.report for ALL aggregation/writing — no second copy.
  - Overlay is OFF by default (writing overlay video for 1,500 files takes
    hours/days). Enable per-run with --overlay.
  - Default sampling is --sample-fps 1 (fast). Pass --sample-fps 0 / a value
    to change; None (native, every frame) is intentionally NOT the batch
    default because it is far too slow at scale.
  - One bad/corrupt video does NOT stop the run: it is caught, logged, and
    recorded as an ERROR row in the global summary so it stays visible.
  - REVIEW is not used; the global summary has no review column.

Usage:
    python run_folder.py data
    python run_folder.py data --out-root reports --summary-csv reports/face_summary.csv
    python run_folder.py data --sample-fps 2
    python run_folder.py data --overlay        # also write overlay videos (slow)
    python run_folder.py data --limit 10       # process only first 10 (smoke test)
"""
import argparse
import csv
import os
import re
import sys
import time
import traceback

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")

from qc.pipelines.face_rgb import run_face_rgb
from qc.utils.report import (
    build_overall_record,
    write_result_csv,
    write_detail_header_csv,
    write_detail_csv,
)

# Strict match: lowercase '_face_rgb.mp4' only, to match the strict-case
# policy in validate_filenames.py. A wrong-case name like '005_face_RGB.MP4'
# is NOT processed here — it would be flagged unrecognised by the validator,
# so the contractor must fix the casing first. (No re.IGNORECASE on purpose.)
FACE_RGB_RE = re.compile(r"^(\d+)_face_rgb\.mp4$")

# Global summary columns. No review_checks (REVIEW eliminated). ERROR column
# carries the exception text for videos that failed to process.
SUMMARY_FIELDNAMES = [
    "volunteer_id",
    "filename",
    "overall_status",
    "failed_checks",
    "frames_sampled",
    "detection_gaps",
    "yaw_min",
    "yaw_max",
    "pitch_min",
    "pitch_max",
    "processing_time_sec",
    "error",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", help="folder to scan for *_face_rgb.mp4")
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--out-root", default="reports",
                    help="root for per-volunteer report folders (default: reports)")
    ap.add_argument("--summary-csv", default=None,
                    help="global summary CSV (default: <out-root>/face_summary.csv)")
    ap.add_argument("--sample-fps", type=float, default=1.0,
                    help="frames sampled per source second (batch default: 1). "
                         "Native/every-frame is intentionally not offered here.")
    ap.add_argument("--overlay", action="store_true",
                    help="also write overlay videos (SLOW; off by default)")
    ap.add_argument("--fail-fast", action="store_true",
                    help="stop processing a file early on a structural defect "
                         "(bad metadata / zero frames / multiple faces). Off by "
                         "default: every frame is processed so the report has a "
                         "full timeline. Ignored when --overlay is set (the "
                         "overlay always needs a complete 1:1 timeline).")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N videos (smoke test)")
    ap.add_argument("--no-detail", action="store_true",
                    help="skip the big per-frame detail CSV (keep result/overall only)")
    return ap.parse_args()


def find_face_rgb_videos(input_dir):
    """Return (matched, skipped, wrong_case).

    matched    : list of (volunteer_id, path) for strict '<id>_face_rgb.mp4'
    skipped    : other .mp4 files (e.g. walk videos) not handled here
    wrong_case : files that are face_rgb only after ignoring case
                 (e.g. '005_face_RGB.MP4'). Strict policy does NOT process
                 these; they are surfaced so the contractor fixes the casing,
                 matching validate_filenames.py.

    Non-matching names are returned separately so a typo'd file stays visible
    instead of being silently skipped.
    """
    matched = []
    skipped = []
    wrong_case = []
    # Case-insensitive form used ONLY to detect a wrong-case RGB file so it can
    # be surfaced (not processed). Strict FACE_RGB_RE remains the gate.
    rgb_loose = re.compile(r"^(\d+)_face_rgb\.mp4$", flags=re.IGNORECASE)
    for root, _, files in os.walk(input_dir):
        for name in sorted(files):
            path = os.path.join(root, name)
            m = FACE_RGB_RE.match(name)
            if m:
                matched.append((m.group(1), path))
            elif rgb_loose.match(name):
                # right name, wrong case (e.g. '005_face_RGB.MP4'). Strict policy
                # rejects it; surface it so the contractor fixes the casing.
                wrong_case.append(path)
            elif name.lower().endswith(".mp4") and "_face_" in name.lower():
                # other face stream (depth/ir/thermal) -> not this pipeline's job
                continue
            elif name.lower().endswith(".mp4"):
                skipped.append(path)
    matched.sort(key=lambda t: (t[0], t[1]))
    return matched, skipped, wrong_case


def process_one(path, vid, config, args):
    """Run the pipeline on one video and write its per-volunteer files.

    Returns a global-summary row dict. Raises nothing — callers expect a row
    even on error (the row's status is ERROR).
    """
    out_dir = os.path.join(args.out_root, vid)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"face_{vid}"
    detail_path = os.path.join(out_dir, f"{stem}_detail.csv")
    detail_header_path = os.path.join(out_dir, f"{stem}_detail_header.csv")
    result_path = os.path.join(out_dir, f"{stem}_result.csv")
    overall_path = os.path.join(out_dir, f"{stem}_overall.csv")
    overlay_path = os.path.join(out_dir, f"{stem}_overlay.mp4") if args.overlay else None

    start = time.perf_counter()

    overlay = None
    if overlay_path:
        from qc.utils.overlay import OverlayWriter
        overlay = OverlayWriter(
            overlay_path, fps=float(args.sample_fps) if args.sample_fps else 30.0,
            volunteer_id=vid, filename=os.path.basename(path))

    rows, timeline = run_face_rgb(
        path, vid, config,
        sample_fps=args.sample_fps if args.sample_fps else None,
        overlay=overlay, progress=None,
        # Fail-fast is OFF by default (full timeline for every file); opt in
        # with --fail-fast to skip doomed files early at 1,500 scale. When
        # --overlay is on, the overlay video must be a complete 1:1 copy, so
        # fail-fast is force-disabled regardless of the flag.
        fail_fast=(args.fail_fast and overlay is None),
    )

    if overlay is not None:
        overlay.close()

    # Per-volunteer files (quiet=True so we don't spam the batch console).
    if not args.no_detail:
        write_detail_csv(detail_path, rows, timeline, quiet=True)
    write_detail_header_csv(detail_header_path, rows, timeline, quiet=True)
    write_result_csv(result_path, overall_path, rows, timeline,
                     config=config, quiet=True)

    rec = build_overall_record(rows, timeline, config=config)
    elapsed = time.perf_counter() - start

    return {
        "volunteer_id": rec["volunteer_id"] or vid,
        "filename": rec["filename"] or os.path.basename(path),
        "overall_status": rec["final_status"],
        "failed_checks": "; ".join(rec["failed_checks"]),
        "frames_sampled": rec["frames_sampled"],
        "detection_gaps": rec["detection_gaps"],
        "yaw_min": rec["yaw_min"],
        "yaw_max": rec["yaw_max"],
        "pitch_min": rec["pitch_min"],
        "pitch_max": rec["pitch_max"],
        "processing_time_sec": f"{elapsed:.1f}",
        "error": "",
    }


def error_row(path, vid, exc, elapsed):
    return {
        "volunteer_id": vid,
        "filename": os.path.basename(path),
        "overall_status": "ERROR",
        "failed_checks": "",
        "frames_sampled": "",
        "detection_gaps": "",
        "yaw_min": "", "yaw_max": "", "pitch_min": "", "pitch_max": "",
        "processing_time_sec": f"{elapsed:.1f}",
        "error": " ".join(str(exc).split())[:300],
    }


def main():
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"not a folder: {args.input_dir}")
    if not os.path.exists(args.config):
        sys.exit(f"config not found: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    summary_csv = args.summary_csv or os.path.join(args.out_root, "face_summary.csv")
    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)

    matched, skipped, wrong_case = find_face_rgb_videos(args.input_dir)
    if args.limit is not None:
        matched = matched[: args.limit]

    if not matched:
        print(f"No *_face_rgb.mp4 found under {args.input_dir}")
        if wrong_case:
            print(f"({len(wrong_case)} wrong-case file(s) found — fix the casing:)")
            for p in wrong_case:
                print(f"  - {p}")
        if skipped:
            print(f"({len(skipped)} other .mp4 files were ignored)")
        return 1

    total = len(matched)
    print(f"=== run_folder.py ===")
    print(f"input     : {args.input_dir}")
    print(f"videos    : {total} face_rgb files  (sample_fps={args.sample_fps}, "
          f"overlay={'on' if args.overlay else 'off'})")
    print(f"out root  : {args.out_root}")
    print(f"summary   : {summary_csv}\n")

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0}
    run_start = time.perf_counter()

    # Stream rows to the summary CSV as we go, so a crash mid-run still leaves
    # a partial-but-valid summary on disk.
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        w.writeheader()

        for i, (vid, path) in enumerate(matched, start=1):
            t0 = time.perf_counter()
            try:
                row = process_one(path, vid, config, args)
            except Exception as exc:  # noqa: BLE001 — one bad file must not kill the batch
                elapsed = time.perf_counter() - t0
                row = error_row(path, vid, exc, elapsed)
                print(f"[{i}/{total}] {vid:>5}  ERROR  {row['error']}", flush=True)
                traceback.print_exc()
            else:
                status = row["overall_status"]
                print(f"[{i}/{total}] {vid:>5}  {status:<6} "
                      f"({row['processing_time_sec']}s, "
                      f"{row['frames_sampled']} frames, "
                      f"{row['detection_gaps']} gaps)", flush=True)

            counts[row["overall_status"]] = counts.get(row["overall_status"], 0) + 1
            w.writerow(row)
            f.flush()

    elapsed = time.perf_counter() - run_start
    print(f"\n=== done: {total} videos in {elapsed:.1f}s ===")
    print(f"  PASS={counts.get('PASS',0)}  FAIL={counts.get('FAIL',0)}  "
          f"SKIP={counts.get('SKIP',0)}  ERROR={counts.get('ERROR',0)}")
    print(f"  summary: {summary_csv}")
    if wrong_case:
        print(f"\n  WARNING: {len(wrong_case)} wrong-case face_rgb file(s) were "
              f"NOT processed (strict naming). Fix the casing to '_face_rgb.mp4':")
        for p in wrong_case:
            print(f"    - {p}")
    if skipped:
        print(f"\n  note: {len(skipped)} non-RGB .mp4 files were ignored "
              f"(depth/ir/thermal belong to other pipelines).")

    # exit 0 only if nothing failed or errored
    return 0 if (counts.get("FAIL", 0) == 0 and counts.get("ERROR", 0) == 0) else 1


if __name__ == "__main__":
    sys.exit(main())