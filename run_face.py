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
from collections import Counter


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to a NNN_face_rgb.mp4")
    ap.add_argument("--id", default=None, help="volunteer id (else parsed from filename)")
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--sample-fps", type=float, default=1.0)
    ap.add_argument("--csv", default=None, help="optional: also write rows to this CSV")
    return ap.parse_args()

def parse_volunteer_id(path):
    """Pull NNN from a filename like 001_face_rgb.mp4; fallback to '000'."""
    m = re.match(r"(\d+)_face_rgb", os.path.basename(path))
    return m.group(1) if m else "000"

def print_rows(rows):
    print("volunteer_id, data_type, filename, check_name, status, reason")
    for r in rows:
        print(", ".join(str(x) for x in r.as_tuple()))

def print_name_status(rows):
    tally = Counter((r.check_name, r.status) for r in rows)
    print("\n=== tally (check_name -> status x count) ===")
    for (name, status), n in sorted(tally.items()):
        print(f"  {name:22} {status:7} x{n}")

def print_angle(angles):
    print(f"\n=== head-pose angles collected: {len(angles)} frames ===")
    if angles:
        yaws = [a["yaw"] for a in angles]
        pitches = [a["pitch"] for a in angles]
        print(f"  yaw   range: {min(yaws):+.1f} .. {max(yaws):+.1f}")
        print(f"  pitch range: {min(pitches):+.1f} .. {max(pitches):+.1f}")
        print("  (a real turn video should show yaw swinging negative->positive as the head"
              " turns left->right, and pitch swinging as it looks down->up)")
        
def print_csv(rows, args_csv=None):
    if args_csv:
        os.makedirs(os.path.dirname(args_csv) or ".", exist_ok=True)
        with open(args_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["volunteer_id", "data_type", "filename",
                        "check_name", "status", "reason", "frame_index"])
            for r in rows:
                w.writerow([*r.as_tuple(), r.frame_index])
        print(f"\nwrote {len(rows)} rows to {args_csv}")    
        
def format_duration(seconds):
    """Human-readable elapsed time: '3.4s' or '1m 05.2s' for longer runs."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:04.1f}s"

def print_results(rows, angles, args_csv=None):
    print_rows(rows)
    print_name_status(rows)
    print_angle(angles)
    print_csv(rows, args_csv)
    
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
    rows, angles = run_face_rgb(args.video, vid, config, sample_fps=args.sample_fps)
    elapsed = time.perf_counter() - start

    print_results(rows, angles, args_csv=args.csv)
    print(f"\n=== done in {format_duration(elapsed)} ===")


if __name__ == "__main__":
    main()