"""run_walk.py — standalone gait (walk) QC runner (MVP).

Mirrors run_palm.py's shape, but SIMPLER: walk grades PER VIDEO (each
NNN_walk_F.mp4 / NNN_walk_S.mp4 is independent), so there is no
participant-level grouping like palm's N-relative angle. This MVP runs exactly
one graded check per video (check_body_height on the first frame) and writes a
one-row-per-video summary CSV.

Usage (folder searches are RECURSIVE up to MAX_SEARCH_DEPTH levels):
    python run_walk.py data 001        # walk files for id 001 under data/
    python run_walk.py data/001        # all walk files under data/001
    python run_walk.py 001_walk_F.mp4  # explicit file(s)

Output:
    <out-root>/walk_summary.csv  — one row per video: id, data_type (walk_F /
    walk_S), filename, PASS/FAIL/SKIP verdict, and the failed check + reason.

One detector is built ONCE and reused across every video (avoids reloading the
.task bundle per file).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import yaml

from qc.pipelines.walk import run_walk
from qc.checks.pose_landmarker import create_pose_landmarker
from qc.utils.report import (
    build_overall_record,
    write_detail_csv,
    write_result_csv,
    write_walk_detail_header_csv,
    SummaryWriter,
)

# Strict walk filename: NNN_walk_[F|S].mp4 (mirrors config walk_F/walk_S keys).
WALK_RE = re.compile(r"^(?P<vid>\d+)_walk_(?P<view>[FS])\.mp4$")

VIEWS = ("F", "S")
MAX_SEARCH_DEPTH = 5


def find_walk_files(root, *, vid=None, max_depth=MAX_SEARCH_DEPTH):
    """Recursively find walk video files under `root`, up to `max_depth` levels.

    Mirrors run_palm.find_palm_files: same depth cap, same de-dup by basename.

    Args:
        root: folder to search.
        vid: if given, keep only files whose participant id == vid.
        max_depth: directory levels below root to descend (0 = root only).

    Returns:
        sorted list of matching file paths (by basename, case-insensitive).
    """
    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)
    found = {}
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.rstrip(os.sep).count(os.sep) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
        for name in filenames:
            m = WALK_RE.match(name)
            if not m:
                continue
            if vid is not None and m.group("vid") != vid:
                continue
            key = name.lower()
            if key not in found:
                found[key] = os.path.join(dirpath, name)
    return [found[k] for k in sorted(found)]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Gait (walk) QC — MVP: first-frame body-height per video.")
    ap.add_argument(
        "inputs", nargs="+",
        help="EITHER `<folder> <id>` (e.g. `data 001`), OR `<folder>`, OR an "
             "explicit list of walk video paths.")
    ap.add_argument("--config", default="config.yml", help="path to config.yml")
    ap.add_argument("--sample-fps", type=float, default=5.0,
                    help="frames sampled per source second for the debug "
                         "overlay. Default: 5. Pass 0 (or a negative) for "
                         "native fps (every frame), so the overlay is 1:1 with "
                         "the original (same length, same speed). Grading is "
                         "unaffected: the verdict always reads the first frame.")
    ap.add_argument("--id", default=None,
                    help="volunteer id (else from the 2nd positional arg in "
                         "`folder id` form, or parsed from filenames)")
    ap.add_argument("--out-root", default="reports",
                    help="output folder for walk_summary.csv (default: reports)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-row stdout and writer prints")
    ap.add_argument("--overlay", action="store_true",
                    help="(no-op) debug overlays are ON by default; kept for "
                         "back-compat/explicitness. Use --no-overlay to disable.")
    ap.add_argument("--no-overlay", action="store_true",
                    help="do NOT write the per-video debug overlay .mp4 "
                         "(skeleton + bbox + height verdict).")
    ap.add_argument("--fail-fast", action="store_true",
                    help="halt a video early on a STRUCTURAL defect (bad "
                         "metadata, 0 frames, or multiple persons in a frame) "
                         "so the FAIL is ratio-exempt, mirroring run_face. Only "
                         "takes effect with --no-overlay (the overlay needs a "
                         "complete timeline, so an overlay run force-disables "
                         "fail-fast).")
    ap.add_argument("--no-progress", action="store_true",
                    help="hide the live per-check progress stream (each CheckRow "
                         "is printed as it is emitted; pass this to silence it).")
    return ap.parse_args()


def _short_reason(reason, max_len=120):
    """Trim a long reason string for one-line progress output."""
    if not reason:
        return ""
    reason = str(reason).replace("\n", " ")
    return reason if len(reason) <= max_len else reason[: max_len - 1] + "\u2026"


def make_progress_printer():
    """Return a callback(row) that prints one line per CheckRow as the walk
    pipeline emits it, so a console user sees progress in real time (mirrors
    run_face's printer)."""
    def progress(row):
        level = getattr(row, "level", "frame")
        reason = _short_reason(row.reason)
        if level == "frame":
            frame_text = "" if row.frame_index is None else f"frame={row.frame_index}"
            print(f"[frame] {frame_text:>12} | "
                  f"{row.check_name:<22} {row.status:<6} | {reason}", flush=True)
        else:
            print(f"[{level}] {'':>12} | "
                  f"{row.check_name:<22} {row.status:<6} | {reason}", flush=True)
    return progress


def resolve_inputs(inputs, explicit_id):
    """Work out the list of walk video paths from the positional inputs.

    Accepted shapes (folder searches RECURSIVE up to MAX_SEARCH_DEPTH):
      1. `<folder> <id>` -> recursively find <id>_walk_[FS].mp4 under <folder>.
      2. `<folder>`      -> recursively find ALL *_walk_[FS].mp4 under <folder>.
      3. explicit file/folder path(s).

    Returns the sorted, de-duplicated list of walk video paths. Unlike palm this
    runner does NOT require a single participant id — it grades each video
    independently, so a folder with many volunteers is fine.
    """
    # form 1: `<folder> <id>`
    if (len(inputs) == 2 and os.path.isdir(inputs[0])
            and re.fullmatch(r"\d+", inputs[1])):
        folder, vid = inputs[0], (explicit_id or inputs[1])
        return find_walk_files(folder, vid=vid)

    # form 2: `<folder>`
    if len(inputs) == 1 and os.path.isdir(inputs[0]):
        return find_walk_files(inputs[0], vid=explicit_id)

    # form 3: explicit files and/or folders
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            paths.extend(find_walk_files(item, vid=explicit_id))
        elif os.path.isfile(item):
            if WALK_RE.match(os.path.basename(item)):
                paths.append(item)
            else:
                print(f"  (skip, not a walk filename: {os.path.basename(item)})")
        else:
            print(f"  (skip, not found: {item})")
    # de-dup, preserve order
    seen, uniq = set(), []
    for p in paths:
        key = os.path.basename(p).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _identity(path):
    """(volunteer_id, view) from a walk filename, or (None, None)."""
    m = WALK_RE.match(os.path.basename(path))
    return (m.group("vid"), m.group("view")) if m else (None, None)


def main():
    args = parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"config not found: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    videos = resolve_inputs(args.inputs, args.id)
    if not videos:
        sys.exit("no walk videos found (expected NNN_walk_[F|S].mp4)")

    # Wire each walk frame-check's fail-ratio into the report aggregation so the
    # per-frame rows aggregate to a video FAIL at the WALK threshold (>=20%),
    # independent of the face/global ratio. summarize_rows_by_check reads
    # per_check_fail_ratio[check_name]; we set each here rather than hard-coding
    # in report.py. [DESIGN]
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

    os.makedirs(args.out_root, exist_ok=True)
    summary_csv = os.path.join(args.out_root, "walk_summary.csv")

    # Build ONE pose detector and reuse it for every video.
    pose_cfg = config.get("models", {}).get("pose_landmarker", {})
    model_path = pose_cfg.get("model_path", "models/pose_landmarker.task")
    detector = create_pose_landmarker(
        model_path,
        min_pose_detection_confidence=pose_cfg.get("min_pose_detection_confidence", 0.5),
        min_pose_presence_confidence=pose_cfg.get("min_pose_presence_confidence", 0.5),
        min_tracking_confidence=pose_cfg.get("min_tracking_confidence", 0.5),
    )

    # Build ONE YOLO detector for the whole batch too (occlusion check), so the
    # weights load once, not per video. Disabled in config or ultralytics missing
    # -> yolo_detector stays None and the pipeline emits SKIP occlusion rows.
    yolo_detector = None
    if config.get("walk", {}).get("occlusion", {}).get("enabled", True):
        try:
            from qc.checks.yolo_detector import create_yolo_detector
            yolo_detector = create_yolo_detector(config)
        except ImportError as e:
            print(f"[warn] occlusion check disabled: {e}")

    print(f"walk videos ({len(videos)}): "
          + ", ".join(os.path.basename(p) for p in videos))
    print(f"summary: {summary_csv}\n")

    t0 = time.time()
    n = 0
    # Live per-check progress stream (real time), unless silenced. --quiet
    # already suppresses per-row output, so it implies no progress too.
    progress_printer = (None if (args.no_progress or args.quiet)
                        else make_progress_printer())
    try:
        with SummaryWriter(summary_csv, mode="w") as summary:
            for path in videos:
                vid, view = _identity(path)
                stem = os.path.splitext(os.path.basename(path))[0]  # e.g. 002_walk_F

                # Per-participant folder groups BOTH F and S under reports/<id>/,
                # mirroring run_face's reports/<id>/ layout. The overlay .mp4 is
                # written INTO this same folder by the pipeline (out_root points
                # here), so every artifact for one video lives together.
                out_dir = os.path.join(args.out_root, vid)
                os.makedirs(out_dir, exist_ok=True)

                # sample_fps: 0 / negative means "native (every frame)", mirroring
                # run_face's None sentinel; the pipeline writes the overlay at the
                # SAME rate it samples so the output duration matches the source.
                sample_fps = args.sample_fps if args.sample_fps and args.sample_fps > 0 else None
                # fail-fast mirrors run_face: a structural defect (bad metadata,
                # 0 frames, multiple persons) halts the file early so the FAIL is
                # ratio-exempt. But the overlay needs a COMPLETE timeline, so
                # fail-fast is force-disabled whenever an overlay is written
                # (overlay is ON by default) -- pass --no-overlay --fail-fast to
                # actually get the early break. Same tradeoff face documents.
                overlay_on = not args.no_overlay
                rows, timeline = run_walk(
                    path, vid, config, view=view, detector=detector,
                    yolo_detector=yolo_detector,
                    overlay=overlay_on, out_root=out_dir,
                    sample_fps=sample_fps,
                    fail_fast=(args.fail_fast and not overlay_on),
                    progress=progress_printer)

                # Post-hoc row dump ONLY when live progress was off (else it
                # double-prints what the progress stream already showed). With
                # --no-progress but not --quiet, this is how rows still surface.
                if not args.quiet and progress_printer is None:
                    for r in rows:
                        print(", ".join(str(x) for x in r.as_tuple()))

                # Per-participant CSVs, same grain and writers as run_face:
                #   walk_<stem>_detail.csv         one row per (frame, check)
                #   walk_<stem>_detail_header.csv  one row per frame (brightness series)
                #   walk_<stem>_result.csv         per-check verdict (PASS/FAIL/SKIP)
                #   walk_<stem>_overall.csv        one OVERALL row for the video
                detail_csv = os.path.join(out_dir, f"walk_{stem}_detail.csv")
                header_csv = os.path.join(out_dir, f"walk_{stem}_detail_header.csv")
                result_csv = os.path.join(out_dir, f"walk_{stem}_result.csv")
                overall_csv = os.path.join(out_dir, f"walk_{stem}_overall.csv")

                write_detail_csv(detail_csv, rows, timeline, quiet=args.quiet)
                write_walk_detail_header_csv(header_csv, rows, timeline,
                                             quiet=args.quiet)
                write_result_csv(result_csv, overall_csv, rows, timeline,
                                 config=config, quiet=args.quiet)

                # One summary row per video, from the SAME aggregation the
                # face/palm summaries use (build_overall_record -> per-check
                # verdict -> overall PASS/FAIL/SKIP). Pass the timeline so any
                # timeline-derived summary fields are populated.
                overall = build_overall_record(rows, timeline, config=config)
                summary.add(overall)
                n += 1
    finally:
        if hasattr(detector, "close"):
            try:
                detector.close()
            except Exception:
                pass

    dt = time.time() - t0
    print(f"\nwrote {n} video row(s) to {summary_csv}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()