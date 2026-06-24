"""Run the MINIMAL palm pipeline on one image and write the four CSVs.

Mirror of run_face.py, reduced to the metadata-only palm slice. It writes the
SAME four report files run_face.py produces, with a `palm_` stem instead of
`face_`:

    reports/<id>/palm_<id>_detail.csv          (one row per check)
    reports/<id>/palm_<id>_detail_header.csv   (per-frame measurements; here:
                                                header only, a still has no
                                                timeline)
    reports/<id>/palm_<id>_result.csv          (per-check summary)
    reports/<id>/palm_<id>_overall.csv         (one PASS/FAIL row for the file)

Reusing the EXACT report writers (write_detail_csv / write_detail_header_csv /
write_result_csv) is the point: the palm CSVs are structurally identical to the
face CSVs, so anything downstream (the dashboard, a reviewer's eye) reads both
the same way.

Filename: NNN_palm_[L|R]_[N|RL|RR|PU|PD].jpg. The volunteer id, hand, and pose
are parsed from the name; hand/pose are passed through to run_palm (unused by
this slice, wired for later checks).

Usage:
    python run_palm.py 0001_palm_L_N.jpg
    python run_palm.py 0001_palm_L_N.jpg --out-dir reports/0001
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")

from qc.pipelines.palm import run_palm
from qc.utils.report import (
    write_detail_csv,
    write_detail_header_csv,
    write_result_csv,
)

# Strict palm filename: <id>_palm_<L|R>_<N|RL|RR|PU|PD>.jpg, lowercase only,
# matching the patterns in config.filenames.required. Captures id/hand/pose.
PALM_RE = re.compile(r"^(?P<vid>\d+)_palm_(?P<hand>[LR])_(?P<pose>N|RL|RR|PU|PD)\.jpg$")


def parse_args():
    ap = argparse.ArgumentParser(description="Minimal palm QC (container + resolution).")
    ap.add_argument("image", help="path to NNN_palm_[L|R]_[N|RL|RR|PU|PD].jpg")
    ap.add_argument("--config", default="config.yml", help="path to config.yml")
    ap.add_argument("--id", default=None, help="volunteer id (else parsed from filename)")
    ap.add_argument("--out-dir", default=None,
                    help="output folder; default: reports/<volunteer_id>")
    ap.add_argument("--overlay", nargs="?", const="__default__", default=None,
                    help="write an annotated image (landmarks+bbox+stats). "
                         "Optionally give a path; default: <out-dir>/palm_<id>_<hand>_<pose>_overlay.jpg")
    ap.add_argument("--quiet", action="store_true", help="suppress writer prints")
    return ap.parse_args()


def parse_filename(path):
    """Return (vid, hand, pose) from the filename, or (None, None, None)."""
    m = PALM_RE.match(os.path.basename(path))
    if not m:
        return None, None, None
    return m.group("vid"), m.group("hand"), m.group("pose")


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"image not found: {args.image}")
    if not os.path.exists(args.config):
        sys.exit(f"config not found: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    vid_parsed, hand, pose = parse_filename(args.image)
    vid = args.id or vid_parsed
    if vid is None:
        sys.exit(
            "filename does not match NNN_palm_[L|R]_[N|RL|RR|PU|PD].jpg\n"
            f"Got: {os.path.basename(args.image)}\n"
            "Pass --id to override, or fix the filename."
        )

    out_dir = args.out_dir or os.path.join("reports", vid)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"palm_{vid}"
    detail_path = os.path.join(out_dir, f"{stem}_detail.csv")
    detail_header_path = os.path.join(out_dir, f"{stem}_detail_header.csv")
    result_path = os.path.join(out_dir, f"{stem}_result.csv")
    overall_path = os.path.join(out_dir, f"{stem}_overall.csv")

    print(f"Running minimal palm pipeline on: {args.image}")
    print(f"output folder: {out_dir}\n")

    t0 = time.time()
    want_overlay = args.overlay is not None
    rows, timeline, hand_result = run_palm(
        args.image, vid, config, hand=hand, pose=pose, detect=want_overlay)
    elapsed = time.time() - t0

    # Print the rows to stdout (same shape as run_face's print_rows).
    print("volunteer_id, data_type, filename, check_level, check_name, status, reason, frame_index")
    for r in rows:
        print(", ".join(str(x) for x in r.as_tuple()))
    print()

    # ---- the four CSVs, identical writers to face ----
    write_detail_csv(detail_path, rows, timeline, quiet=args.quiet)
    write_detail_header_csv(detail_header_path, rows, timeline, quiet=args.quiet)
    write_result_csv(result_path, overall_path, rows, timeline,
                     config=config, quiet=args.quiet)

    # ---- optional annotated image ----
    if want_overlay:
        from qc.utils.palm_overlay import draw_palm_overlay
        if args.overlay == "__default__":
            tag = "_".join(x for x in (hand, pose) if x)
            ov_name = f"palm_{vid}_{tag}_overlay.jpg" if tag else f"palm_{vid}_overlay.jpg"
            overlay_path = os.path.join(out_dir, ov_name)
        else:
            overlay_path = args.overlay

        # Build a {check_name: status} dict from the rows for the header panel.
        check_status = {r.check_name: r.status for r in rows}
        if hand_result is None:
            # Detection unavailable (e.g. model bundle missing). Still write an
            # annotated image showing just the metadata checks, no landmarks.
            from types import SimpleNamespace
            hand_result = SimpleNamespace(
                ok=False, message="hand model unavailable",
                landmarks_px=None, bbox=None,
                handedness=None, handedness_score=None)
        draw_palm_overlay(args.image, hand_result,
                          checks=check_status, out_path=overlay_path)
        if not args.quiet:
            print(f"wrote overlay image to {overlay_path}")

    print(f"\n=== done in {elapsed:.2f}s ===")


if __name__ == "__main__":
    main()
