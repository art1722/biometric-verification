"""Run the face_rgb pipeline on ONE video and show the results.

Usage (from the project root — the folder containing qc/):

    python run_face.py 001_face_rgb.mp4
    python run_face.py path/to/video.mp4 --id 042 --sample-fps 1
    python run_face.py 001_face_rgb.mp4 --out-root reports

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
from qc.utils.video import probe_video
from qc.utils.report import (
    STATUS_PRIORITY,
    worst_status,
    first_reason,
    summarize_rows_by_check,
    summarize_overall,
    summarize_timeline,
    write_result_csv,
    write_detail_header_csv,
)
from collections import Counter, defaultdict


FACE_RGB_RE = re.compile(r"^(?P<vid>\d+)_face_rgb\.mp4$", re.IGNORECASE)

# Recursive search depth cap (see run_palm.py): levels below the given folder to
# descend, so an accidental run high in the tree can't walk forever.
MAX_SEARCH_DEPTH = 5


def find_face_rgb_files(root, *, vid=None, max_depth=MAX_SEARCH_DEPTH):
    """Recursively find NNN_face_rgb.mp4 files under `root`, up to max_depth
    levels. If vid is given, keep only that participant's file.

    Returns a sorted list of matching paths (by basename, case-insensitive).
    """
    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)
    found = {}
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.rstrip(os.sep).count(os.sep) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
        for name in filenames:
            m = FACE_RGB_RE.match(name)
            if not m:
                continue
            if vid is not None and m.group("vid") != vid:
                continue
            key = name.lower()
            if key not in found:
                found[key] = os.path.join(dirpath, name)
    return [found[k] for k in sorted(found)]


def resolve_face_video(inputs, explicit_id):
    """Resolve the positional inputs to ONE face_rgb video path + vid.

    Accepted shapes (folder searches are RECURSIVE, up to MAX_SEARCH_DEPTH):
      1. `<folder> <id>` -> recursively find <id>_face_rgb.mp4 under <folder>.
      2. `<folder>`      -> recursively find ALL *_face_rgb.mp4 under <folder>;
                            exactly one participant expected (else pass --id).
      3. `<path.mp4>`    -> use the file directly (original behaviour).

    Returns (video_path, vid). Exits with a helpful message on ambiguity/none.
    """
    # form 1: <folder> <id>
    if (len(inputs) == 2 and os.path.isdir(inputs[0])
            and re.fullmatch(r"\d+", inputs[1])):
        folder, vid = inputs[0], (explicit_id or inputs[1])
        matches = find_face_rgb_files(folder, vid=vid)
        if not matches:
            sys.exit(f"no {vid}_face_rgb.mp4 found under {folder} "
                     f"(searched up to {MAX_SEARCH_DEPTH} levels deep)")
        if len(matches) > 1:
            sys.exit(f"multiple {vid}_face_rgb.mp4 found under {folder}:\n  "
                     + "\n  ".join(matches))
        return matches[0], vid

    # form 2: single folder, no id
    if len(inputs) == 1 and os.path.isdir(inputs[0]):
        folder = inputs[0]
        matches = find_face_rgb_files(folder, vid=explicit_id)
        if not matches:
            sys.exit(f"no *_face_rgb.mp4 found under {folder} "
                     f"(searched up to {MAX_SEARCH_DEPTH} levels deep)")
        ids = {FACE_RGB_RE.match(os.path.basename(p)).group("vid") for p in matches}
        if explicit_id:
            vid = explicit_id
        elif len(ids) == 1:
            vid = next(iter(ids))
        else:
            sys.exit(
                f"multiple participant face videos found under {folder}: "
                f"{sorted(ids)}\nThis runner processes ONE participant. "
                "Pass `<folder> <id>` or --id.")
        vid_matches = [p for p in matches
                       if FACE_RGB_RE.match(os.path.basename(p)).group("vid") == vid]
        return vid_matches[0], vid

    # form 3: explicit path (original behaviour)
    if len(inputs) == 1:
        path = inputs[0]
        vid = explicit_id or parse_volunteer_id(path)
        return path, vid

    sys.exit("expected `<folder> <id>`, `<folder>`, or a path to "
             "NNN_face_rgb.mp4")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Parse arguments for running python file run_face.py"
    )
    ap.add_argument("inputs", nargs="+",
                    help="EITHER `<folder> <id>` (e.g. `data 096`), OR a folder "
                         "to search (e.g. `data/096`), OR a path to a "
                         "NNN_face_rgb.mp4. Folder searches are recursive "
                         f"(up to {MAX_SEARCH_DEPTH} levels).")
    ap.add_argument("--id", default=None, help="volunteer id (else parsed from filename)")
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--sample-fps", type=float, default=5,
                help="frames sampled per source second. Default: None = "
                     "native fps (every frame), so the overlay is 1:1 with "
                     "the original (same length, same speed).")
    ap.add_argument("--overlay", default=None,
                    help="optional: write a debug overlay video (bbox + landmarks "
                         "+ pose + per-check status drawn on each sampled frame) "
                         "to this .mp4 path")
    ap.add_argument(
        "--out-root",
        default=None,
        help="output folder; default: reports/<volunteer_id>",
    )

    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="hide live per-check progress",
    )

    ap.add_argument(
        "--no-overlay",
        action="store_true",
        help="do not write the default overlay video",
    )

    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop processing early on a structural defect (bad metadata / "
             "zero frames / multiple faces). Off by default: every frame is "
             "processed so the report has a full timeline. Ignored whenever an "
             "overlay is written (the overlay needs a complete 1:1 timeline) — "
             "note the overlay is ON by default, so pass --no-overlay to let "
             "--fail-fast take effect.",
    )
    return ap.parse_args()

def parse_volunteer_id(path):
    """Pull volunteer id from filename like 405_face_rgb.mp4.

    Returns None if the filename does not match the expected pattern.
    """
    basename = os.path.basename(path)
    m = re.match(r"^(\d+)_face_rgb\.mp4$", basename, flags=re.IGNORECASE)
    return m.group(1) if m else None

def apply_default_output_paths(args, vid):
    """Fill default report paths based on volunteer id.

    Example:
        vid = "405"

    Defaults:
        reports/405/face_405_detail.csv
        reports/405/face_405_result.csv
        reports/405/face_405_summary.txt
        reports/405/face_405_overlay.mp4
    """
    out_dir = args.out_root or os.path.join("reports", vid)
    stem = f"face_{vid}"

    os.makedirs(out_dir, exist_ok=True)

    # These output paths are no longer user-facing flags (removed for a simpler,
    # consistent CLI: the out-root already determines where everything lands).
    # They are set here as fixed, conventional filenames under out-root and
    # consumed by the writers below exactly as before.
    args.csv = os.path.join(out_dir, f"{stem}_detail.csv")
    args.detail_header_csv = os.path.join(out_dir, f"{stem}_detail_header.csv")
    args.result_csv = os.path.join(out_dir, f"{stem}_result.csv")
    args.overall_csv = os.path.join(out_dir, f"{stem}_overall.csv")
    args.summary = os.path.join(out_dir, f"{stem}_summary.txt")

    if args.overlay is None and not args.no_overlay:
        args.overlay = os.path.join(out_dir, f"{stem}_overlay.mp4")


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

def print_csv(rows, timeline=None, args_csv=None):
    t_by_frame = {}
    if timeline:
        t_by_frame = {t["frame_index"]: t["timestamp_sec"] for t in timeline}
    if args_csv:
        os.makedirs(os.path.dirname(args_csv) or ".", exist_ok=True)
        with open(args_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["volunteer_id", "data_type", "filename",
                        "frame_index", "time",
                        "check_level", "check_name", "status", "reason"])
            for r in rows:
                t = t_by_frame.get(r.frame_index, "")
                t_str = "" if t == "" or t is None else f"{t:.3f}"
                # r.as_tuple() = (vol, dtype, fname, level, name, status, reason, frame_index)
                vol, dtype, fname, level, name, status, reason, fidx = r.as_tuple()
                w.writerow([vol, dtype, fname, fidx, t_str,
                            level, name, status, reason])
        print(f"\nwrote {len(rows)} rows to {args_csv}")


def format_duration(seconds):
    """Human-readable elapsed time: '3.4s' or '1m 05.2s' for longer runs."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:04.1f}s"


def _short_reason(reason, max_len=120):
    reason = " ".join(str(reason).split())
    if len(reason) <= max_len:
        return reason
    return reason[: max_len - 3] + "..."


def make_progress_printer():
    """Print one progress line per CheckRow.

    This intentionally mirrors the detail CSV:
        check_name + status + reason
    """
    def progress(row):
        level = getattr(row, "level", "frame")
        reason = _short_reason(row.reason)

        if level == "frame":
            frame_text = "" if row.frame_index is None else f"frame={row.frame_index}"
            print(
                f"[frame] {frame_text:>12} | "
                f"{row.check_name:<22} {row.status:<6} | {reason}",
                flush=True,
            )
        else:
            print(
                f"[{level}] {'':>12} | "
                f"{row.check_name:<22} {row.status:<6} | {reason}",
                flush=True,
            )

    return progress


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
    print_csv(rows, timeline, args_csv)
    return tally_text, angle_text


def main():
    args = parse_args()

    # Resolve `<folder> <id>` / `<folder>` / `<path.mp4>` -> one video + vid.
    # Recursive folder search (up to MAX_SEARCH_DEPTH) mirrors run_palm.py.
    args.video, resolved_vid = resolve_face_video(args.inputs, args.id)
    if args.id is None:
        args.id = resolved_vid

    if not os.path.exists(args.video):
        sys.exit(f"video not found: {args.video}")
    if not os.path.exists(args.config):
        sys.exit(f"config not found: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    vid = args.id or parse_volunteer_id(args.video)


    if vid is None:
        sys.exit(
            "Could not parse volunteer id from filename.\n"
            f"Got: {os.path.basename(args.video)}\n"
            "Expected format: <volunteer_id>_face_rgb.mp4, e.g. 405_face_rgb.mp4\n"
            "Or pass the id manually, e.g.:\n"
            "  python run_face.py data\\some_video.mp4 --id 405"
    )
    apply_default_output_paths(args, vid)

    # Resolve the source's native fps once: needed for defaulting --sample-fps
    # to "native" (every frame) and for computing the overlay's playback fps.
    meta = probe_video(args.video)
    native_fps = meta.fps if (meta.fps and meta.fps > 0) else None
    sample_fps = args.sample_fps            # None = native / every frame

    # Overlay playback fps MUST equal the rate frames were actually sampled at,
    # or the overlay's duration won't match the source. The overlay receives
    # ONLY the sampled frames (one per kept source frame), so:
    #   - sample_fps given (e.g. 5)  -> ~5*duration frames -> play at 5 fps
    #   - sample_fps None (native)   -> every frame kept    -> play at native fps
    # Writing those sampled frames at native fps (the old behaviour) replays a
    # sparse stream too fast: 48s sampled at 5fps -> 240 frames / 30fps = 8s.
    # This mirrors effective_fps inside the face_rgb pipeline so the two agree.
    if sample_fps is not None and sample_fps > 0:
        overlay_fps = float(sample_fps)
    else:
        overlay_fps = native_fps if native_fps else 30.0

    sample_fps_label = "native (every frame)" if sample_fps is None else sample_fps
    print(f"Running face_rgb pipeline on: {args.video}")
    print(f"volunteer id: {vid} | sample_fps: {sample_fps_label} | native_fps: {native_fps}")
    print(f"output folder: {os.path.dirname(args.csv)}\n")

    start = time.perf_counter()

    overlay = None
    if args.overlay:
        from qc.utils.overlay import OverlayWriter
        # Overlay plays at the rate frames were SAMPLED (overlay_fps, resolved
        # above), so its duration matches the source. Sparse sampling looks
        # choppy but stays time-accurate, which is what the QC reviewer needs.
        overlay = OverlayWriter(
            args.overlay, fps=overlay_fps,
            volunteer_id=vid, filename=os.path.basename(args.video))

    progress = None if args.no_progress else make_progress_printer()

    rows, timeline = run_face_rgb(
        args.video,
        vid,
        config,
        sample_fps=sample_fps,
        overlay=overlay,
        progress=progress,
        # Fail-fast is OFF by default (full timeline); opt in with --fail-fast.
        # When an overlay is written the video must be a complete 1:1 copy, so
        # fail-fast is force-disabled regardless of the flag. NOTE: the overlay
        # is ON by default here, so --fail-fast only takes effect together with
        # --no-overlay.
        fail_fast=(args.fail_fast and overlay is None),
    )

    if overlay is not None:
        overlay.close()
        print(f"overlay video: {args.overlay} ({overlay.frames_written} frames)")

    elapsed = time.perf_counter() - start

    tally_text, angle_text = print_results(rows, timeline, args_csv=args.csv)

    if args.detail_header_csv:
        write_detail_header_csv(args.detail_header_csv, rows, timeline)

    if args.result_csv:
        write_result_csv(args.result_csv, args.overall_csv, rows, timeline, config=config)

    elapsed_text = format_duration(elapsed)
    print(f"\n=== done in {elapsed_text} ===")
    
    if args.summary:
        write_summary(args.summary, args.video, vid, args.sample_fps,
                      tally_text, angle_text, elapsed_text)

if __name__ == "__main__":
    main()