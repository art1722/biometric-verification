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

from qc.pipelines.gait import run_gait
from qc.checks.pose_landmarker import create_pose_landmarker
from qc.utils.report import build_overall_record, write_detail_csv, SummaryWriter

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
    ap.add_argument("--id", default=None,
                    help="volunteer id (else from the 2nd positional arg in "
                         "`folder id` form, or parsed from filenames)")
    ap.add_argument("--out-root", default="reports",
                    help="output folder for walk_summary.csv (default: reports)")
    ap.add_argument("--summary-csv", default=None,
                    help="walk summary CSV path "
                         "(default: <out-root>/walk_summary.csv)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-row stdout and writer prints")
    return ap.parse_args()


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

    os.makedirs(args.out_root, exist_ok=True)
    summary_csv = args.summary_csv or os.path.join(args.out_root, "walk_summary.csv")

    # Build ONE pose detector and reuse it for every video.
    pose_cfg = config.get("models", {}).get("pose_landmarker", {})
    model_path = pose_cfg.get("model_path", "models/pose_landmarker.task")
    detector = create_pose_landmarker(
        model_path,
        min_pose_detection_confidence=pose_cfg.get("min_pose_detection_confidence", 0.5),
        min_pose_presence_confidence=pose_cfg.get("min_pose_presence_confidence", 0.5),
        min_tracking_confidence=pose_cfg.get("min_tracking_confidence", 0.5),
    )

    print(f"walk videos ({len(videos)}): "
          + ", ".join(os.path.basename(p) for p in videos))
    print(f"summary: {summary_csv}\n")

    t0 = time.time()
    n = 0
    try:
        with SummaryWriter(summary_csv, mode="w") as summary:
            for path in videos:
                vid, view = _identity(path)
                rows, _timeline = run_gait(
                    path, vid, config, view=view, detector=detector)

                if not args.quiet:
                    for r in rows:
                        print(", ".join(str(x) for x in r.as_tuple()))

                # One summary row per video, from the SAME aggregation the
                # face/palm summaries use (build_overall_record -> per-check
                # verdict -> overall PASS/FAIL/SKIP).
                overall = build_overall_record(rows, [], config=config)
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