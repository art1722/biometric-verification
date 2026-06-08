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

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")

from qc.pipelines.face_rgb import run_face_rgb


def guess_volunteer_id(path):
    """Pull NNN from a filename like 001_face_rgb.mp4; fallback to '000'."""
    m = re.match(r"(\d+)_face_rgb", os.path.basename(path))
    return m.group(1) if m else "000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to a NNN_face_rgb.mp4")
    ap.add_argument("--id", default=None, help="volunteer id (else guessed from filename)")
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--sample-fps", type=float, default=1.0)
    ap.add_argument("--csv", default=None, help="optional: also write rows to this CSV")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"video not found: {args.video}")
    if not os.path.exists(args.config):
        sys.exit(f"config not found: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    vid = args.id or guess_volunteer_id(args.video)

    print(f"Running face_rgb pipeline on: {args.video}")
    print(f"volunteer id: {vid} | sample_fps: {args.sample_fps}\n")

    rows, angles = run_face_rgb(args.video, vid, config, sample_fps=args.sample_fps)

    # --- print rows in the requested format ---
    print("volunteer_id, data_type, filename, check_name, status, reason")
    for r in rows:
        print(", ".join(str(x) for x in r.as_tuple()))

    # --- quick tally so you can see behaviour at a glance ---
    from collections import Counter
    tally = Counter((r.check_name, r.status) for r in rows)
    print("\n=== tally (check_name -> status x count) ===")
    for (name, status), n in sorted(tally.items()):
        print(f"  {name:22} {status:7} x{n}")

    # --- angle summary (for the future turn-sequence check) ---
    print(f"\n=== head-pose angles collected: {len(angles)} frames ===")
    if angles:
        yaws = [a["yaw"] for a in angles]
        pitches = [a["pitch"] for a in angles]
        print(f"  yaw   range: {min(yaws):+.1f} .. {max(yaws):+.1f}")
        print(f"  pitch range: {min(pitches):+.1f} .. {max(pitches):+.1f}")
        print("  (a real turn video should show yaw swinging negative->positive as the head"
              " turns left->right, and pitch swinging as it looks down->up)")

    # --- optional CSV ---
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["volunteer_id", "data_type", "filename",
                        "check_name", "status", "reason", "frame_index"])
            for r in rows:
                w.writerow([*r.as_tuple(), r.frame_index])
        print(f"\nwrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
