"""Batch runner: QC every *_face_rgb.mp4 in a folder.

For the real project scale (~1,500 volunteers) this walks an input folder,
runs the face_rgb pipeline on each RGB video, writes the SAME per-volunteer
report files that run_face.py produces, and writes two customer-facing
summaries.

Customer-facing output (one row per FAILED check, so each reason is its own
cell — see qc.utils.report):
  - face_summary.csv : this modality's full report. One row per failed check;
    a video that passes everything gets one PASS row; a crashed video gets one
    row carrying the error text. Columns: volunteer_id, data_type, filename,
    overall_status, check_name, reason. Internal stats (yaw/pitch, frame
    counts, timing) are deliberately NOT here — they stay in each volunteer's
    own report.
  - all_summary.csv + all_summary.json : the cross-modal roll-up of PROBLEM
    cases only (FAIL) across face + palm + walk. CSV is the same flat
    per-check grain (Excel-pivotable); JSON nests one object per video with a
    `failures: [{check_name, reason}]` array for a customer's tooling.

Design choices (confirmed with the team):
  - Reuses qc.utils.report for ALL aggregation/writing — no second copy. The
    SummaryWriter / AllSummaryWriter classes live there so the future palm and
    walk runners emit IDENTICALLY-shaped output by calling the same classes.
  - all_summary is APPENDED to by default, so palm/walk runs add to the same
    files after face within a single run. Overwritten each run by default;
    pass --append to add to an existing cross-modal roll-up.
  - Overlay is ON by default so a console user can eyeball results without
    reading the CSVs. Pass --no-overlay for faster runs (no video writing);
    NOTE that with overlays on, fail-fast is disabled (a full 1:1 timeline is
    needed to draw the overlay). At 1,500-file scale, prefer --no-overlay.
  - Default sampling is --sample-fps 1 (fast). Pass --sample-fps 0 / a value
    to change; None (native, every frame) is intentionally NOT the batch
    default because it is far too slow at scale.
  - One bad/corrupt video does NOT stop the run: it is caught, logged, and
    recorded as a FAIL row (synthetic processing_error check) so it stays visible.
  - REVIEW is not used.

Usage:
    python run_folder.py data
    python run_folder.py data --out-root reports --face-summary-csv reports/face_summary.csv
    python run_folder.py data --sample-fps 2
    python run_folder.py data --no-overlay     # faster: skip overlay videos
    python run_folder.py data --limit 10       # process only first 10 (smoke test)
    python run_folder.py data --append         # add to an existing all_summary roll-up
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
from qc.pipelines.walk import run_walk
from qc.checks.pose_landmarker import create_pose_landmarker
from run_walk import find_walk_files, _identity as _walk_identity
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
    ap.add_argument("input_dir",
                    help="folder to scan (recursively) for face_rgb, palm, and "
                         "walk files; all three run by default")
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--out-root", default="reports",
                    help="root for per-volunteer report folders (default: reports)")
    ap.add_argument("--all-summary", default=None,
                    help="cross-modal FAIL roll-up, written as both .csv "
                         "and .json, shared by all modalities "
                         "(default stem: <out-root>/all_summary). Overwritten "
                         "each run by default; pass --append to add to an "
                         "existing roll-up instead.")
    ap.add_argument("--append", action="store_true",
                    help="append to the existing all_summary files instead of "
                         "overwriting them (default is to overwrite / start "
                         "fresh each run).")
    ap.add_argument("--sample-fps", type=float, default=5.0,
                    help="frames sampled per source second (batch default: 1). "
                         "Native/every-frame is intentionally not offered here.")
    ap.add_argument("--no-overlay", action="store_true",
                    help="do NOT write overlay videos. Overlays are ON by default "
                         "so a console user can eyeball results without digging "
                         "through CSVs; pass this for faster runs (no video "
                         "writing). NOTE: with overlays on, fail-fast is disabled "
                         "(the overlay needs a complete 1:1 timeline).")
    ap.add_argument("--fail-fast", action="store_true",
                    help="stop processing a file early on a structural defect "
                         "(bad metadata / zero frames / multiple faces). Off by "
                         "default: every frame is processed so the report has a "
                         "full timeline. Ignored unless --no-overlay is set (the "
                         "overlay always needs a complete 1:1 timeline).")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N participants (by volunteer id) "
                         "across all modalities; e.g. --limit 3 -> ids 001,002,004 "
                         "and all their face/palm/walk files (F/S pairs kept together)")
    ap.add_argument("--no-detail", action="store_true",
                    help="skip the big per-frame detail CSV (keep result/overall only)")

    ap.add_argument("--no-palm", action="store_true",
                    help="skip the palm image batch (palm runs by default)")
    ap.add_argument("--palm-summary-csv", default=None,
                    help="palm per-check summary CSV "
                         "(default: <out-root>/palm_summary.csv)")
    ap.add_argument("--no-walk", action="store_true",
                    help="skip the walk/gait video batch (walk runs by default)")
    ap.add_argument("--walk-summary-csv", default=None,
                    help="walk per-check summary CSV "
                         "(default: <out-root>/walk_summary.csv)")
    ap.add_argument("--no-face", action="store_true",
                    help="skip the face batch (e.g. run palm+walk only with --no-face)")
    ap.add_argument("--face-summary-csv", default=None,
                    help="face per-check summary CSV "
                         "(default: <out-root>/face_summary.csv)")
    return ap.parse_args()


def compute_allowed_ids(matched, input_dir, limit):
    """The first `limit` participant ids across ALL modalities (face+palm+walk).

    --limit is BY PARTICIPANT: `--limit 3` means the first 3 volunteer ids that
    exist anywhere in the dataset (e.g. 001, 002, 004), and every file those 3
    own -- across face, palm, and walk -- is processed, with F/S walk pairs kept
    together. Ids are sorted numerically when they are all digits (so 4 < 004 <
    56789 order is 001,002,004,005,...), else lexicographically.

    Returns None when limit is None (no cap -> every modality runs unfiltered).
    Otherwise returns a set of the allowed id strings; each batch intersects its
    own discovered ids with this set, so all three run on the SAME participants.
    """
    if limit is None:
        return None

    ids = set()
    ids.update(vid for vid, _ in matched)                      # face
    ids.update(find_palm_images(input_dir)[0].keys())          # palm
    for p in find_walk_files(input_dir):                       # walk
        ids.add(_walk_identity(p)[0])

    def _key(v):
        return (0, int(v)) if v.isdigit() else (1, v)

    return set(sorted(ids, key=_key)[:limit])


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
    even on error (the row's status is FAIL, via error_row).
    """
    out_dir = os.path.join(args.out_root, vid)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"face_{vid}"
    detail_path = os.path.join(out_dir, f"{stem}_detail.csv")
    detail_header_path = os.path.join(out_dir, f"{stem}_detail_header.csv")
    result_path = os.path.join(out_dir, f"{stem}_result.csv")
    overall_path = os.path.join(out_dir, f"{stem}_overall.csv")
    overlay_path = os.path.join(out_dir, f"{stem}_overlay.mp4") if not args.no_overlay else None

    start = time.perf_counter()

    overlay = None
    if overlay_path:
        from qc.utils.face_overlay import OverlayWriter
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


def error_row(path, vid, exc, elapsed, data_type="face_rgb"):
    """A crash record shaped like build_overall_record's output, so the same
    writers expand it.

    A file that THREW mid-grading (corrupt video, a library crash, a bug) is
    recorded as FAIL, not a separate ERROR status: the spec's media-level verdict
    is PASS/FAIL only. The exception text is preserved by surfacing it as a
    synthetic failed check named "processing_error", so it flows through the
    normal FAIL -> failed_checks_detail rendering (report rows, all_summary
    failures, the API /results rows) exactly like any other failure reason —
    nothing downstream needs an ERROR branch to show it.

    data_type labels which modality crashed ("face_rgb" default; the walk batch
    passes "walk_F"/"walk_S") so a crash row is attributed to the right pipeline.
    """
    error_text = " ".join(str(exc).split())[:300]
    return {
        "volunteer_id": vid,
        "data_type": data_type,
        "filename": os.path.basename(path),
        "final_status": "FAIL",
        "failed_checks": ["processing_error"],
        "failed_checks_detail": [
            {"check_name": "processing_error", "reason": error_text},
        ],
        "error": error_text,
        "processing_time_sec": f"{elapsed:.1f}",
    }


# ---------------------------------------------------------------------------
# Face batch
# ---------------------------------------------------------------------------
def run_face_batch(matched, config, args, all_summary, counts, total=None):
    """Grade each face_rgb video, feed the shared writers.

    Extracted from main() so face, palm and walk all have the SAME shape:
    open a per-modality SummaryWriter, loop, add every record to BOTH that
    writer and the shared cross-modal all_summary, mutate `counts` in place,
    and return (n_rows, wrong_case).

    Face differs from palm/walk in one way ONLY, and it is deliberate:
    discovery happens in main() (find_face_rgb_videos), because main() needs
    `matched` up front to compute allowed_ids for the OTHER two modalities via
    compute_allowed_ids. So the already-discovered, already-id-filtered list is
    passed IN rather than found here. wrong_case is likewise owned and printed
    by main(); this returns [] to keep the 2-tuple signature uniform with
    run_palm_batch / run_walk_batch.

    A video that THROWS is recorded via error_row() -> final_status "FAIL" with
    a synthetic "processing_error" check, exactly like the palm batch. There is
    no ERROR status anywhere in the pipeline (media verdict is PASS/FAIL/SKIP).
    """
    summary_csv = args.face_summary_csv or os.path.join(
        args.out_root, "face_summary.csv"
    )
    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)

    if total is None:
        total = len(matched)
    n_videos = 0

    with SummaryWriter(summary_csv, mode="w") as summary:
        for i, (vid, path) in enumerate(matched, start=1):
            t0 = time.perf_counter()
            try:
                rec = process_one(path, vid, config, args)
            except Exception as exc:  # noqa: BLE001 — one bad file must not kill the batch
                elapsed = time.perf_counter() - t0
                rec = error_row(path, vid, exc, elapsed)
                print(f"[{i}/{total}] {vid:>5}  FAIL (crash)  "
                      f"{rec['error']}", flush=True)
                traceback.print_exc()
            else:
                status = rec["final_status"]
                print(f"[{i}/{total}] {vid:>5}  {status:<6} "
                      f"({rec['processing_time_sec']}s, "
                      f"{rec.get('frames_sampled', '')} frames, "
                      f"{rec.get('detection_gaps', '')} gaps)", flush=True)

            counts[rec["final_status"]] = counts.get(rec["final_status"], 0) + 1
            # Per-modality summary gets EVERY video (PASS included); the
            # cross-modal all_summary keeps only FAIL (self-filtering).
            summary.add(rec)
            all_summary.add(rec)
            n_videos += 1

    print(f"  face summary: {summary_csv}  ({n_videos} video row(s))")
    return n_videos, []


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
    # data_type carries the specific hand+pose (e.g. palm_L_N, palm_R_PU) so the
    # summary CSV distinguishes the 10 palm images per participant, instead of a
    # bare "palm". Parsed from the image filename (002_palm_L_N.jpg); falls back
    # to whatever the row carried, then to "palm", if the name is unexpected.
    palm_type = None
    m = PALM_RE.match(os.path.basename(r0.filename or ""))
    if m:
        palm_type = f"palm_{m.group('hand')}_{m.group('pose')}"
    return {
        "volunteer_id": r0.volunteer_id,
        "data_type": palm_type or r0.data_type or "palm",
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
    failure becomes one FAIL record so the batch keeps going.
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


def run_palm_batch(input_dir, config, args, all_summary, counts, allowed_ids=None):
    """Discover palm images, grade each participant, feed the shared writers.

    Uses a SEPARATE palm_summary.csv (opened here) but the SAME all_summary
    writer passed in, so palm FAIL rows append to the cross-modal roll-up
    after face. Mutates `counts` in place with palm image verdicts.

    allowed_ids: if given (from --limit), only participants whose id is in this
    set are graded, so palm runs on the SAME volunteers as face/walk.
    """
    by_vid, wrong_case = find_palm_images(input_dir)
    if allowed_ids is not None:
        by_vid = {vid: files for vid, files in by_vid.items() if vid in allowed_ids}

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
                error_text = " ".join(str(exc).split())[:300]
                # A participant that THREW is recorded as FAIL (media verdict is
                # PASS/FAIL only). The exception text is preserved as a synthetic
                # "processing_error" check so it renders through the normal FAIL
                # path everywhere, no ERROR branch needed.
                rec = {
                    "volunteer_id": vid,
                    "data_type": "palm",
                    "filename": f"{vid}_palm_*",
                    "final_status": "FAIL",
                    "failed_checks": ["processing_error"],
                    "failed_checks_detail": [
                        {"check_name": "processing_error", "reason": error_text},
                    ],
                    "error": error_text,
                }
                palm_summary.add(rec)
                all_summary.add(rec)
                counts["FAIL"] = counts.get("FAIL", 0) + 1
                print(f"[{i}/{total_p}] {vid:>5}  FAIL (crash)  {error_text}",
                      flush=True)
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


def run_walk_batch(input_dir, config, args, all_summary, counts, allowed_ids=None):
    """Discover walk videos, grade each (per video, F and S independent), feed
    the shared writers.

    Mirrors run_walk.py's main(): it (1) wires each walk frame-check's fail-ratio
    into report aggregation, (2) builds ONE pose detector and ONE YOLO detector
    for the whole batch (weights load once, not per video), and (3) loops the
    videos calling qc.pipelines.walk.run_walk with those shared detectors.

    Walk grades PER VIDEO (each _F/_S file is its own row), so this loop is
    shaped like the face loop, not palm's per-participant grouping. Uses a
    SEPARATE walk_summary.csv but the SAME all_summary writer passed in, so walk
    FAIL rows append to the cross-modal roll-up after face and palm.
    Mutates `counts` in place.

    allowed_ids: if given (from --limit), only videos whose participant id is in
    this set are graded (BOTH F and S of each kept participant), so walk runs on
    the SAME volunteers as face/palm.
    """
    paths = find_walk_files(input_dir)
    if allowed_ids is not None:
        paths = [p for p in paths if _walk_identity(p)[0] in allowed_ids]

    if not paths:
        return 0, []

    # (1) Wire walk per-check fail-ratios into report aggregation, EXACTLY as
    # run_walk.py main() does, so per-frame walk rows aggregate to a video FAIL
    # at the walk thresholds, independent of the face/global ratio.
    walk_cfg = config.get("walk", {})
    agg = config.setdefault("report", {}).setdefault("aggregation", {})
    per_check = agg.setdefault("per_check_fail_ratio", {})
    per_check["check_brightness"] = (walk_cfg.get("brightness", {})
                                     .get("frame_fail_ratio", 0.2))
    per_check["check_person_blur"] = (walk_cfg.get("blur", {})
                                      .get("frame_fail_ratio", 0.2))
    per_check["check_occlusion"] = (walk_cfg.get("occlusion", {})
                                    .get("frame_fail_ratio", 0.2))
    fc_cfg = walk_cfg.get("frame_checks", {})
    per_check["check_person_detected"] = fc_cfg.get(
        "person_detected_fail_ratio", 0.2)
    per_check["check_person_fully"] = fc_cfg.get(
        "person_fully_fail_ratio", 0.2)

    walk_summary_csv = args.walk_summary_csv or os.path.join(
        args.out_root, "walk_summary.csv"
    )
    os.makedirs(os.path.dirname(walk_summary_csv) or ".", exist_ok=True)

    # (2) Build ONE pose detector + ONE YOLO detector for the whole batch, so the
    # bundles load once. Both are passed into every run_walk call. YOLO disabled
    # in config or ultralytics missing -> stays None; the pipeline then emits
    # SKIP occlusion rows instead of crashing.
    pose_cfg = config.get("models", {}).get("pose_landmarker", {})
    model_path = pose_cfg.get("model_path", "models/pose_landmarker.task")
    detector = create_pose_landmarker(
        model_path,
        # >1 so a second person is returned and detect_pose reports "Multiple
        # poses detected" (-> check_single_person -> video FAIL). Mirrors face's
        # num_faces; see qc/pipelines/walk.py for the full rationale.
        num_poses=pose_cfg.get("num_poses", 10),
        min_pose_detection_confidence=pose_cfg.get("min_pose_detection_confidence", 0.5),
        min_pose_presence_confidence=pose_cfg.get("min_pose_presence_confidence", 0.5),
        min_tracking_confidence=pose_cfg.get("min_tracking_confidence", 0.5),
    )
    yolo_detector = None
    if walk_cfg.get("occlusion", {}).get("enabled", True):
        try:
            from qc.checks.yolo_detector import create_yolo_detector
            yolo_detector = create_yolo_detector(config)
        except ImportError as exc:
            print(f"  [warn] occlusion check disabled: {exc}")

    # sample_fps: 0 / negative -> native (every frame), mirroring run_walk.
    sample_fps = args.sample_fps if args.sample_fps and args.sample_fps > 0 else None
    # Overlay ON by default in batch (run_folder's --no-overlay opts out); when
    # on, the pipeline force-disables fail-fast so the overlay gets a full
    # timeline.
    overlay_on = not args.no_overlay

    total_w = len(paths)
    n_videos = 0
    print(f"\n=== walk batch: {total_w} video(s) ===")
    try:
        with SummaryWriter(walk_summary_csv, mode="w") as walk_summary:
            for i, path in enumerate(paths, start=1):
                vid, view = _walk_identity(path)
                out_dir = os.path.join(args.out_root, vid)
                os.makedirs(out_dir, exist_ok=True)
                stem = f"walk_{vid}_{view}"
                t0 = time.perf_counter()
                try:
                    rows, timeline = run_walk(
                        path, vid, config, view=view, detector=detector,
                        yolo_detector=yolo_detector,
                        overlay=overlay_on, out_root=out_dir,
                        sample_fps=sample_fps,
                        # Fail-fast OFF by default (full timeline); --overlay
                        # force-disables it so the overlay is a complete 1:1 copy.
                        fail_fast=(args.fail_fast and not overlay_on),
                    )
                    # Per-volunteer files, same as face (quiet so the batch
                    # console stays clean). One set per F/S video (stem differs).
                    if not args.no_detail:
                        write_detail_csv(
                            os.path.join(out_dir, f"{stem}_detail.csv"),
                            rows, timeline, quiet=True)
                    write_detail_header_csv(
                        os.path.join(out_dir, f"{stem}_detail_header.csv"),
                        rows, timeline, quiet=True)
                    write_result_csv(
                        os.path.join(out_dir, f"{stem}_result.csv"),
                        os.path.join(out_dir, f"{stem}_overall.csv"),
                        rows, timeline, config=config, quiet=True)

                    rec = build_overall_record(rows, timeline, config=config)
                    elapsed = time.perf_counter() - t0
                    rec["volunteer_id"] = rec.get("volunteer_id") or vid
                    rec["filename"] = rec.get("filename") or os.path.basename(path)
                    rec["data_type"] = rec.get("data_type") or f"walk_{view}"
                    rec["error"] = ""
                    rec["processing_time_sec"] = f"{elapsed:.1f}"
                except Exception as exc:  # noqa: BLE001 — one bad file must not kill the batch
                    elapsed = time.perf_counter() - t0
                    rec = error_row(path, vid, exc, elapsed,
                                    data_type=f"walk_{view}")
                    print(f"[{i}/{total_w}] {vid:>5}  FAIL (crash)  "
                          f"{rec['error']}", flush=True)
                    traceback.print_exc()
                else:
                    print(f"[{i}/{total_w}] {vid:>5}_{view}  "
                          f"{rec['final_status']:<6} ({elapsed:.1f}s)", flush=True)

                walk_summary.add(rec)
                all_summary.add(rec)
                counts[rec["final_status"]] = counts.get(rec["final_status"], 0) + 1
                n_videos += 1
    finally:
        if hasattr(detector, "close"):
            try:
                detector.close()
            except Exception:  # pragma: no cover
                pass

    print(f"  walk summary: {walk_summary_csv}  ({n_videos} video row(s))")
    return n_videos, []


def main():
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"not a folder: {args.input_dir}")
    if not os.path.exists(args.config):
        sys.exit(f"config not found: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    summary_csv = args.face_summary_csv or os.path.join(args.out_root, "face_summary.csv")
    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)

    # all_summary is a STEM: the writer makes <stem>.csv and <stem>.json.
    all_stem = args.all_summary or os.path.join(args.out_root, "all_summary")
    os.makedirs(os.path.dirname(all_stem) or ".", exist_ok=True)

    matched, skipped, wrong_case = find_face_rgb_videos(args.input_dir)
    allowed_ids = compute_allowed_ids(matched, args.input_dir, args.limit)
    if allowed_ids is not None:
        matched = [(vid, path) for vid, path in matched if vid in allowed_ids]

    run_face = not args.no_face
    run_palm = not args.no_palm
    run_walk_flag = not args.no_walk
    # Overlays ON by default (console debugging); --no-overlay opts out. The API
    # layer (wired later) will force this OFF for server jobs regardless of CLI.
    overlay_enabled = not args.no_overlay

    if run_face and not matched:
        print(f"No *_face_rgb.mp4 found under {args.input_dir}")
        if wrong_case:
            print(f"({len(wrong_case)} wrong-case file(s) found — fix the casing:)")
            for p in wrong_case:
                print(f"  - {p}")
        if skipped:
            print(f"({len(skipped)} other .mp4 files were ignored)")
        # With no face videos, only continue if palm or walk still run.
        if not run_palm and not run_walk_flag:
            return 1
        run_face = False

    total = len(matched)
    print(f"=== run_folder.py ===")
    print(f"input     : {args.input_dir}")
    if run_face:
        print(f"videos    : {total} face_rgb files  (sample_fps={args.sample_fps}, "
              f"overlay={'on' if overlay_enabled else 'off'})")
    if run_palm:
        print(f"palm      : ON  (per-participant grading, one row per image)")
    if run_walk_flag:
        print(f"walk      : ON  (per-video grading, one row per F/S video)")
    print(f"out root  : {args.out_root}")
    print(f"summary   : {summary_csv}  (per-check, all videos)")
    print(f"all       : {all_stem}.csv / .json  (FAIL only, cross-modal)\n")

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    palm_wrong_case = []
    n_palm_images = 0
    n_walk_videos = 0
    run_start = time.perf_counter()

    # Two writers, opened together so both flush per video and a crash mid-run
    # still leaves valid partial files:
    #   SummaryWriter   -> face_summary.csv  (per-check rows; PASS gets one row)
    #   AllSummaryWriter-> all_summary.csv + .json (FAIL only, cross-modal)
    # Both expand the SAME overall record, so face/palm/walk stay identical.
    # The all_summary writer stays open across BOTH the face loop and the palm
    # batch so palm rows append to the same cross-modal roll-up in one process.
    all_mode = "a" if args.append else "w"
    with AllSummaryWriter(all_stem, mode=all_mode) as all_summary:

        if run_face:
            n_face_videos, _face_wrong = run_face_batch(
                matched, config, args, all_summary, counts, total=total
            )

        if run_palm:
            n_palm_images, palm_wrong_case = run_palm_batch(
                args.input_dir, config, args, all_summary, counts, allowed_ids
            )

        if run_walk_flag:
            n_walk_videos, _walk_wrong = run_walk_batch(
                args.input_dir, config, args, all_summary, counts, allowed_ids
            )

    all_kept = all_summary.kept

    elapsed = time.perf_counter() - run_start
    print(f"\n=== done in {elapsed:.1f}s ===")
    if run_face:
        print(f"  face: {total} video(s)")
    if run_palm:
        print(f"  palm: {n_palm_images} image row(s)")
    if run_walk_flag:
        print(f"  walk: {n_walk_videos} video row(s)")
    print(f"  PASS={counts.get('PASS',0)}  FAIL={counts.get('FAIL',0)}  "
          f"SKIP={counts.get('SKIP',0)}")
    if run_face:
        print(f"  summary: {summary_csv}")
    print(f"  all    : {all_stem}.csv / .json  ({all_kept} video(s) "
          f"{'appended' if args.append else 'written'})")
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

    # exit 0 only if nothing FAILed (a crash is recorded as FAIL via error_row)
    return 0 if counts.get("FAIL", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())