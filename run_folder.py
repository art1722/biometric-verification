"""Batch runner: QC every *_face_rgb.mp4 in a folder.

For the real project scale (~1,500 volunteers) this walks an input folder,
runs the face_rgb pipeline on each RGB video, writes the SAME per-volunteer
report files that run_face.py produces, and writes two customer-facing
summaries.

Customer-facing output (one row per FAILED check, so each reason is its own
cell — see qc.utils.report):
  - face_summary.csv : this modality's full report. One row per failed check;
    a video that passes everything gets one PASS row; an ERROR video gets one
    row carrying the error text. Columns: volunteer_id, data_type, filename,
    overall_status, check_name, reason. Internal stats (yaw/pitch, frame
    counts, timing) are deliberately NOT here — they stay in each volunteer's
    own report.
  - all_summary.csv + all_summary.json : the cross-modal roll-up of PROBLEM
    cases only (FAIL/ERROR) across face + palm + walk. CSV is the same flat
    per-check grain (Excel-pivotable); JSON nests one object per video with a
    `failures: [{check_name, reason}]` array for a customer's tooling.

Design choices (confirmed with the team):
  - Reuses qc.utils.report for ALL aggregation/writing — no second copy. The
    SummaryWriter / AllSummaryWriter classes live there so the future palm and
    walk runners emit IDENTICALLY-shaped output by calling the same classes.
  - all_summary is APPENDED to by default, so palm/walk runs add to the same
    files after face. Pass --fresh-all to start a new cross-modal roll-up.
  - Overlay is OFF by default (writing overlay video for 1,500 files takes
    hours/days). Enable per-run with --overlay.
  - Default sampling is --sample-fps 1 (fast). Pass --sample-fps 0 / a value
    to change; None (native, every frame) is intentionally NOT the batch
    default because it is far too slow at scale.
  - One bad/corrupt video does NOT stop the run: it is caught, logged, and
    recorded as an ERROR row so it stays visible.
  - REVIEW is not used.

Usage:
    python run_folder.py data
    python run_folder.py data --out-root reports --summary-csv reports/face_summary.csv
    python run_folder.py data --sample-fps 2
    python run_folder.py data --overlay        # also write overlay videos (slow)
    python run_folder.py data --limit 10       # process only first 10 (smoke test)
    python run_folder.py data --fresh-all      # start a new all_summary roll-up
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
from qc.pipelines.palm import run_palm_participant
from qc.utils.report import (
    build_overall_record,
    write_result_csv,
    write_detail_header_csv,
    write_detail_csv,
    summarize_rows_by_check,
    summarize_overall,
    SummaryWriter,
    AllSummaryWriter,
)

# Strict match: lowercase '_face_rgb.mp4' only, to match the strict-case
# policy in validate_filenames.py. A wrong-case name like '005_face_RGB.MP4'
# is NOT processed here — it would be flagged unrecognised by the validator,
# so the contractor must fix the casing first. (No re.IGNORECASE on purpose.)
FACE_RGB_RE = re.compile(r"^(\d+)_face_rgb\.mp4$")

# Palm: <id>_palm_<L|R>_<N|RL|RR|PU|PD>.jpg. Strict case to match
# validate_filenames.py (a wrong-case palm file is surfaced, not processed).
PALM_RE = re.compile(
    r"^(?P<vid>\d+)_palm_(?P<hand>[LR])_(?P<pose>N|RL|RR|PU|PD)\.jpg$"
)
PALM_LOOSE_RE = re.compile(
    r"^(?P<vid>\d+)_palm_(?P<hand>[LR])_(?P<pose>N|RL|RR|PU|PD)\.jpg$",
    flags=re.IGNORECASE,
)

# The customer-facing summaries are per-check (one row per failed check), so
# this runner no longer owns a flat per-video column list — SummaryWriter /
# AllSummaryWriter in qc.utils.report define the shared schema (PERCHECK_*).


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", help="folder to scan for *_face_rgb.mp4")
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--out-root", default="reports",
                    help="root for per-volunteer report folders (default: reports)")
    ap.add_argument("--summary-csv", default=None,
                    help="this modality's per-check summary CSV "
                         "(default: <out-root>/face_summary.csv)")
    ap.add_argument("--all-summary", default=None,
                    help="cross-modal FAIL/ERROR roll-up, written as both .csv "
                         "and .json, shared by all modalities "
                         "(default stem: <out-root>/all_summary). Appended to, "
                         "so palm/walk runs add to the same files; pass "
                         "--fresh-all to truncate first.")
    ap.add_argument("--fresh-all", action="store_true",
                    help="truncate the all_summary files before writing (start "
                         "a new cross-modal roll-up instead of appending).")
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
    ap.add_argument("--palm", action="store_true",
                    help="also run the palm image batch (per-participant grading; "
                         "one summary row per palm image). Writes palm_summary.csv "
                         "and appends FAIL/ERROR rows to the shared all_summary.")
    ap.add_argument("--palm-summary-csv", default=None,
                    help="palm per-check summary CSV "
                         "(default: <out-root>/palm_summary.csv)")
    ap.add_argument("--no-face", action="store_true",
                    help="skip the face batch (e.g. run palm only with --palm --no-face)")
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

    # Normalise the few fields the runner/console rely on, then hand the WHOLE
    # record to the writers — they expand it to per-check rows themselves.
    rec["volunteer_id"] = rec.get("volunteer_id") or vid
    rec["filename"] = rec.get("filename") or os.path.basename(path)
    rec["data_type"] = rec.get("data_type") or "face_rgb"
    rec["error"] = ""
    rec["processing_time_sec"] = f"{elapsed:.1f}"
    return rec


def error_row(path, vid, exc, elapsed):
    """An ERROR record shaped like build_overall_record's output, so the same
    writers expand it (one row, status ERROR, reason = the error text)."""
    return {
        "volunteer_id": vid,
        "data_type": "face_rgb",
        "filename": os.path.basename(path),
        "final_status": "ERROR",
        "failed_checks": [],
        "failed_checks_detail": [],
        "error": " ".join(str(exc).split())[:300],
        "processing_time_sec": f"{elapsed:.1f}",
    }


# ---------------------------------------------------------------------------
# Palm batch
# ---------------------------------------------------------------------------
# Palm differs from face in ONE structural way: the angle check grades each
# rotated pose (RL/RR/PU/PD) relative to that hand's OWN neutral (N), so palm
# must be graded PER PARTICIPANT (all 10 images together), not per file. So we
# GROUP palm JPGs by volunteer id first, then call run_palm_participant once
# per volunteer. Its rows are then split back per image, and each image gets
# its OWN per-image overall record (the grain the team chose) fed to the
# palm_summary / all_summary writers — mirroring write_consolidated_overall in
# run_palm.py so a batch image row is identical to a standalone one.


def find_palm_images(input_dir):
    """Return (by_vid, wrong_case).

    by_vid     : dict {volunteer_id -> sorted list of (filename, path)} for
                 strict '<id>_palm_<L|R>_<N|RL|RR|PU|PD>.jpg'.
    wrong_case : palm files that match only case-insensitively (surfaced, not
                 processed), mirroring the face wrong-case policy.
    """
    by_vid = {}
    wrong_case = []
    for root, _, files in os.walk(input_dir):
        for name in sorted(files):
            path = os.path.join(root, name)
            m = PALM_RE.match(name)
            if m:
                by_vid.setdefault(m.group("vid"), []).append((name, path))
            elif PALM_LOOSE_RE.match(name):
                wrong_case.append(path)
    for vid in by_vid:
        by_vid[vid].sort(key=lambda t: t[0])
    return by_vid, wrong_case


def _image_record(file_rows, config):
    """Build ONE per-image overall record from that image's CheckRows.

    Shaped like build_overall_record's output so SummaryWriter/AllSummaryWriter
    expand it identically to a face record: it carries data_type, volunteer_id,
    filename, final_status, and a failed_checks_detail list (one entry per
    failed check, with check_name + reason) that expand_to_check_rows consumes.
    """
    aggregation_cfg = (config or {}).get("report", {}).get("aggregation", {})
    summaries = summarize_rows_by_check(file_rows, aggregation_cfg)
    overall = summarize_overall(summaries)
    r0 = file_rows[0]
    failed_detail = [
        {"check_name": s["check_name"], "reason": s.get("reason", "")}
        for s in summaries
        if s["final_status"] == "FAIL"
    ]
    return {
        "volunteer_id": r0.volunteer_id,
        "data_type": r0.data_type or "palm",
        "filename": r0.filename,
        "final_status": overall["final_status"],
        "failed_checks": [d["check_name"] for d in failed_detail],
        "failed_checks_detail": failed_detail,
        "error": "",
    }


def process_palm_participant(vid, files, config, args):
    """Run palm QC for ONE participant and write their per-volunteer files.

    Returns (records, counts_delta) where `records` is a list of per-IMAGE
    overall records (the chosen grain) for the summary writers, and
    counts_delta tallies image verdicts. Raises nothing: a participant-level
    failure becomes one ERROR record so the batch keeps going.
    """
    out_dir = os.path.join(args.out_root, vid)
    os.makedirs(out_dir, exist_ok=True)
    image_paths = [p for _name, p in files]
    image_order = [name for name, _p in files]

    rows, _timelines = run_palm_participant(
        vid, image_paths, config, progress=None, detect=True,
    )

    # Group rows back per image (per-file grain the writers expand).
    rows_by_file = {}
    for r in rows:
        rows_by_file.setdefault(r.filename, []).append(r)

    # Per-volunteer detail CSV (every check row, all images) — mirrors run_palm.
    if not args.no_detail:
        write_detail_csv(
            os.path.join(out_dir, f"palm_{vid}_detail.csv"), rows, [], quiet=True
        )

    records = []
    counts_delta = {}
    # Preserve filename order; skip any expected image with no rows (absent file).
    seen = set()
    for name in image_order:
        if name in seen:
            continue
        seen.add(name)
        file_rows = rows_by_file.get(name)
        if not file_rows:
            continue
        rec = _image_record(file_rows, config)
        records.append(rec)
        counts_delta[rec["final_status"]] = counts_delta.get(rec["final_status"], 0) + 1
    return records, counts_delta


def run_palm_batch(input_dir, config, args, all_summary, counts):
    """Discover palm images, grade each participant, feed the shared writers.

    Uses a SEPARATE palm_summary.csv (opened here) but the SAME all_summary
    writer passed in, so palm FAIL/ERROR rows append to the cross-modal roll-up
    after face. Mutates `counts` in place with palm image verdicts.
    """
    by_vid, wrong_case = find_palm_images(input_dir)
    if args.limit is not None:
        keep = dict(sorted(by_vid.items())[: args.limit])
        by_vid = keep

    if not by_vid:
        return 0, wrong_case

    palm_summary_csv = args.palm_summary_csv or os.path.join(
        args.out_root, "palm_summary.csv"
    )
    os.makedirs(os.path.dirname(palm_summary_csv) or ".", exist_ok=True)

    total_p = len(by_vid)
    n_images = 0
    print(f"\n=== palm batch: {total_p} participant(s) ===")
    with SummaryWriter(palm_summary_csv, mode="w") as palm_summary:
        for i, (vid, files) in enumerate(sorted(by_vid.items()), start=1):
            t0 = time.perf_counter()
            try:
                records, counts_delta = process_palm_participant(
                    vid, files, config, args
                )
            except Exception as exc:  # noqa: BLE001 — one bad participant must not kill the batch
                elapsed = time.perf_counter() - t0
                rec = {
                    "volunteer_id": vid,
                    "data_type": "palm",
                    "filename": f"{vid}_palm_*",
                    "final_status": "ERROR",
                    "failed_checks": [],
                    "failed_checks_detail": [],
                    "error": " ".join(str(exc).split())[:300],
                }
                palm_summary.add(rec)
                all_summary.add(rec)
                counts["ERROR"] = counts.get("ERROR", 0) + 1
                print(f"[{i}/{total_p}] {vid:>5}  ERROR  {rec['error']}", flush=True)
                traceback.print_exc()
                continue

            for rec in records:
                palm_summary.add(rec)
                all_summary.add(rec)
                counts[rec["final_status"]] = counts.get(rec["final_status"], 0) + 1
            n_images += len(records)
            elapsed = time.perf_counter() - t0
            statuses = " ".join(
                f"{r['filename'].split('_palm_')[-1].replace('.jpg','')}:{r['final_status']}"
                for r in records
            )
            print(
                f"[{i}/{total_p}] {vid:>5}  {len(records)} img  ({elapsed:.1f}s)  {statuses}",
                flush=True,
            )

    print(f"  palm summary: {palm_summary_csv}  ({n_images} image row(s))")
    return n_images, wrong_case


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

    # all_summary is a STEM: the writer makes <stem>.csv and <stem>.json.
    all_stem = args.all_summary or os.path.join(args.out_root, "all_summary")
    os.makedirs(os.path.dirname(all_stem) or ".", exist_ok=True)

    matched, skipped, wrong_case = find_face_rgb_videos(args.input_dir)
    if args.limit is not None:
        matched = matched[: args.limit]

    run_face = not args.no_face
    run_palm = args.palm

    if run_face and not matched:
        print(f"No *_face_rgb.mp4 found under {args.input_dir}")
        if wrong_case:
            print(f"({len(wrong_case)} wrong-case file(s) found — fix the casing:)")
            for p in wrong_case:
                print(f"  - {p}")
        if skipped:
            print(f"({len(skipped)} other .mp4 files were ignored)")
        # With no face videos, only continue if palm was explicitly requested.
        if not run_palm:
            return 1
        run_face = False

    total = len(matched)
    print(f"=== run_folder.py ===")
    print(f"input     : {args.input_dir}")
    if run_face:
        print(f"videos    : {total} face_rgb files  (sample_fps={args.sample_fps}, "
              f"overlay={'on' if args.overlay else 'off'})")
    if run_palm:
        print(f"palm      : ON  (per-participant grading, one row per image)")
    print(f"out root  : {args.out_root}")
    print(f"summary   : {summary_csv}  (per-check, all videos)")
    print(f"all       : {all_stem}.csv / .json  (FAIL/ERROR only, cross-modal)\n")

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0}
    palm_wrong_case = []
    n_palm_images = 0
    run_start = time.perf_counter()

    # Two writers, opened together so both flush per video and a crash mid-run
    # still leaves valid partial files:
    #   SummaryWriter   -> face_summary.csv  (per-check rows; PASS gets one row)
    #   AllSummaryWriter-> all_summary.csv + .json (FAIL/ERROR only, cross-modal)
    # Both expand the SAME overall record, so face/palm/walk stay identical.
    # The all_summary writer stays open across BOTH the face loop and the palm
    # batch so palm rows append to the same cross-modal roll-up in one process.
    all_mode = "w" if args.fresh_all else "a"
    with AllSummaryWriter(all_stem, mode=all_mode) as all_summary:

        if run_face:
            with SummaryWriter(summary_csv, mode="w") as summary:
                for i, (vid, path) in enumerate(matched, start=1):
                    t0 = time.perf_counter()
                    try:
                        rec = process_one(path, vid, config, args)
                    except Exception as exc:  # noqa: BLE001 — one bad file must not kill the batch
                        elapsed = time.perf_counter() - t0
                        rec = error_row(path, vid, exc, elapsed)
                        print(f"[{i}/{total}] {vid:>5}  ERROR  {rec['error']}", flush=True)
                        traceback.print_exc()
                    else:
                        status = rec["final_status"]
                        print(f"[{i}/{total}] {vid:>5}  {status:<6} "
                              f"({rec['processing_time_sec']}s, "
                              f"{rec.get('frames_sampled', '')} frames, "
                              f"{rec.get('detection_gaps', '')} gaps)", flush=True)

                    counts[rec["final_status"]] = counts.get(rec["final_status"], 0) + 1
                    # Per-modality summary gets EVERY video (PASS included); the
                    # cross-modal all_summary keeps only FAIL/ERROR (self-filtering).
                    summary.add(rec)
                    all_summary.add(rec)

        if run_palm:
            n_palm_images, palm_wrong_case = run_palm_batch(
                args.input_dir, config, args, all_summary, counts
            )

    all_kept = all_summary.kept

    elapsed = time.perf_counter() - run_start
    print(f"\n=== done in {elapsed:.1f}s ===")
    if run_face:
        print(f"  face: {total} video(s)")
    if run_palm:
        print(f"  palm: {n_palm_images} image row(s)")
    print(f"  PASS={counts.get('PASS',0)}  FAIL={counts.get('FAIL',0)}  "
          f"SKIP={counts.get('SKIP',0)}  ERROR={counts.get('ERROR',0)}")
    if run_face:
        print(f"  summary: {summary_csv}")
    print(f"  all    : {all_stem}.csv / .json  ({all_kept} FAIL/ERROR video(s) "
          f"{'written' if args.fresh_all else 'appended'})")
    if wrong_case:
        print(f"\n  WARNING: {len(wrong_case)} wrong-case face_rgb file(s) were "
              f"NOT processed (strict naming). Fix the casing to '_face_rgb.mp4':")
        for p in wrong_case:
            print(f"    - {p}")
    if palm_wrong_case:
        print(f"\n  WARNING: {len(palm_wrong_case)} wrong-case palm file(s) were "
              f"NOT processed (strict naming). Fix the casing:")
        for p in palm_wrong_case:
            print(f"    - {p}")
    if skipped:
        print(f"\n  note: {len(skipped)} non-RGB .mp4 files were ignored "
              f"(depth/ir/thermal belong to other pipelines).")

    # exit 0 only if nothing failed or errored
    return 0 if (counts.get("FAIL", 0) == 0 and counts.get("ERROR", 0) == 0) else 1


if __name__ == "__main__":
    sys.exit(main())