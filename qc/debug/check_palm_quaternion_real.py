"""Run the EXPERIMENTAL quaternion palm-orientation path on REAL palm images.

Read-only diagnostic: it does NOT touch the grading pipeline. It detects the
hand in each image with the SAME shared HandLandmarker the pipeline uses, then
prints (and optionally CSV-saves) the quaternion delta-vs-N magnitude alongside
the current plane-fit roll/pitch, so you can judge on real data whether the
quaternion MAGNITUDE is a useful signal.

Usage
-----
    python -m qc.debug.check_palm_quaternion_real \
        --dir data/palm/099 \
        --model models/hand_landmarker.task \
        [--csv out/palm_099_quat.csv]

Expects filenames like  099_palm_L_N.jpg  (…_<L|R>_<N|RL|RR|PU|PD>.jpg),
matching the pipeline's convention. Groups by (volunteer, hand); grades each
rotated pose's quaternion delta against that hand's own N.

What it reports per rotated pose:
    quat_angle_deg : gimbal-lock-free rotation magnitude FROM N TO this pose
    quat_axis      : unit axis of that rotation (labelling NOT yet calibrated)
    plane roll/pitch + delta vs N : the CURRENT grading method, for comparison

Interpretation guidance (from the 2026-07-07 volunteer-099 calibration):
    A valid rotated pose should show quat_angle_deg clearly above the ~5-8 deg
    per-image noise floor. On 099 the magnitudes were RL 16, RR 24, PU 29,
    PD 15 deg -- all well separated from N. The AXIS direction, however, did
    not map cleanly to roll-vs-pitch, so treat magnitude as a "did the hand
    move enough" signal only, NOT as directional grading.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

from qc.checks.hand_landmarker import create_hand_landmarker, detect_hand
from qc.checks.check_palm_angle import (
    calculate_palm_angles,
    calculate_palm_quaternion_delta,
)

_PALM_RE = re.compile(
    r"^(?P<vid>\d+)_palm_(?P<hand>[LR])_(?P<pose>N|RL|RR|PU|PD)\.jpg$",
    re.IGNORECASE,
)
_ROTATED = ("RL", "RR", "PU", "PD")


def _detect_world_landmarks(path, detector):
    """Return (world_landmarks | None, roll_pitch | None, note)."""
    res = detect_hand(path, detector=detector)
    if not res.ok or res.world_landmarks is None:
        return None, None, res.message or "no hand"
    aok, ainfo = calculate_palm_angles(res.world_landmarks)
    rp = {"roll": ainfo["roll"], "pitch": ainfo["pitch"]} if aok else None
    return res.world_landmarks, rp, "ok"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="folder of palm .jpg images")
    ap.add_argument("--model", default=os.path.join("models", "hand_landmarker.task"),
                    help="path to hand_landmarker.task bundle")
    ap.add_argument("--csv", default=None, help="optional CSV output path")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.dir):
        print(f"ERROR: not a directory: {args.dir}", file=sys.stderr)
        return 2

    try:
        detector = create_hand_landmarker(model_path=args.model)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # group files by (vid, hand)
    groups: dict = defaultdict(dict)  # (vid,hand) -> {pose: path}
    for name in sorted(os.listdir(args.dir)):
        m = _PALM_RE.match(name)
        if not m:
            continue
        key = (m.group("vid"), m.group("hand").upper())
        groups[key][m.group("pose").upper()] = os.path.join(args.dir, name)

    if not groups:
        print("No palm images matched the naming pattern in", args.dir)
        return 1

    rows_out = []
    header = f"{'file':28s} {'quat_Δ°':>8s} {'quat_axis(x,y,z)':>20s} " \
             f"{'roll':>7s} {'pitch':>7s} {'d_roll':>7s} {'d_pitch':>8s}  note"
    print(header)
    print("-" * len(header))

    for (vid, hand), poses in sorted(groups.items()):
        n_path = poses.get("N")
        n_wl = n_rp = None
        if n_path:
            n_wl, n_rp, n_note = _detect_world_landmarks(n_path, detector)
            note = "N ok" if n_wl is not None else f"N unusable: {n_note}"
            print(f"{os.path.basename(n_path):28s} {'--':>8s} {'(reference)':>20s} "
                  f"{(n_rp or {}).get('roll', float('nan')):7.1f} "
                  f"{(n_rp or {}).get('pitch', float('nan')):7.1f} "
                  f"{'--':>7s} {'--':>8s}  {note}")

        for pose in _ROTATED:
            p_path = poses.get(pose)
            if not p_path:
                continue
            p_wl, p_rp, p_note = _detect_world_landmarks(p_path, detector)
            fname = os.path.basename(p_path)

            if p_wl is None:
                print(f"{fname:28s} {'--':>8s} {'--':>20s} "
                      f"{'--':>7s} {'--':>7s} {'--':>7s} {'--':>8s}  {p_note}")
                rows_out.append(dict(file=fname, quat_angle_deg="", quat_axis="",
                                     roll="", pitch="", d_roll="", d_pitch="",
                                     note=p_note))
                continue

            if n_wl is None:
                note = "no usable N -> cannot delta"
                qang = ""
                axis_s = ""
                droll = dpitch = ""
            else:
                qok, qinfo = calculate_palm_quaternion_delta(p_wl, n_wl)
                if qok:
                    qang = f"{qinfo['angle_deg']:.1f}"
                    ax = qinfo["axis"]
                    axis_s = f"({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f})"
                else:
                    qang, axis_s = "", ""
                if p_rp and n_rp:
                    droll = f"{p_rp['roll'] - n_rp['roll']:+.1f}"
                    dpitch = f"{p_rp['pitch'] - n_rp['pitch']:+.1f}"
                else:
                    droll = dpitch = ""
                note = "ok" if qok else qinfo.get("error", "quat failed")

            print(f"{fname:28s} {qang:>8s} {axis_s:>20s} "
                  f"{(p_rp or {}).get('roll', float('nan')):7.1f} "
                  f"{(p_rp or {}).get('pitch', float('nan')):7.1f} "
                  f"{droll:>7s} {dpitch:>8s}  {note}")
            rows_out.append(dict(file=fname, quat_angle_deg=qang, quat_axis=axis_s,
                                 roll=f"{(p_rp or {}).get('roll', ''):}",
                                 pitch=f"{(p_rp or {}).get('pitch', ''):}",
                                 d_roll=droll, d_pitch=dpitch, note=note))

    if args.csv and rows_out:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nCSV written: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
