"""Run palm QC for ONE participant across all their palm images.

This is the SINGLE palm runner (it replaces the old per-image run_palm.py and
the previous run_palm_participant.py). It grades the angle check at PARTICIPANT
level -- each rotated pose relative to that hand's own N baseline, which
per-image grading cannot do (see qc.pipelines.palm.run_palm_participant /
check_palm_pose_delta).

Invocation -- folder + participant id
-------------------------------------
    python run_palm.py data 002

`data` is the folder to scan; `002` is the participant id. The runner globs
    002_palm_{L,R}_{N,RL,RR,PU,PD}.jpg
inside that folder and runs QC on whatever subset is present. Missing files are
NORMAL: it warns which of the 10 are absent and continues with the rest.

You can still pass explicit image paths instead of "folder id" if you want:
    python run_palm.py data/002_palm_L_N.jpg data/002_palm_L_RL.jpg --id 002

Output -- THREE consolidated files per participant (not per image)
------------------------------------------------------------------
    reports/<id>/palm_<id>_detail.csv    every check row, all images (per-check
                                         grain; one row per (image, check))
    reports/<id>/palm_<id>_overall.csv   ONE row per image: that image's
                                         PASS/FAIL verdict
    plus one overlay PER IMAGE:
    reports/<id>/palm_<id>_<hand>_<pose>_overlay.jpg

result.csv is intentionally NOT written for palm. For a still image every check
fires exactly once, so result.csv was a near-duplicate of detail.csv (its
per-frame count columns are always 1/0 and its yaw/pitch columns are always
blank). detail.csv + overall.csv carry all the information without the
redundant, video-shaped file. The dashboard tolerates a missing result.csv (it
shows an info note; the raw rows remain in the detail expander).

Overlays -- ON BY DEFAULT
-------------------------
An overlay is written for every image by default. `--overlay` is accepted for
explicitness/back-compat but is now a no-op (overlays already on). Use
`--no-overlay` to turn them off.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")

from qc.pipelines.palm import run_palm as run_palm_image, run_palm_participant
from qc.utils.report import (
    write_detail_csv,
    summarize_rows_by_check,
    summarize_overall,
    RESULT_FIELDNAMES,
)

PALM_RE = re.compile(
    r"^(?P<vid>\d+)_palm_(?P<hand>[LR])_(?P<pose>N|RL|RR|PU|PD)\.jpg$",
    re.IGNORECASE)

# The canonical 10 files we expect per participant (2 hands x 5 poses).
HANDS = ("L", "R")
POSES = ("N", "RL", "RR", "PU", "PD")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Participant-level palm QC (angle graded relative to N).")
    ap.add_argument(
        "inputs", nargs="+",
        help="EITHER `<folder> <participant_id>` (e.g. `data 002`), OR an "
             "explicit list of palm image paths.")
    ap.add_argument("--config", default="config.yml", help="path to config.yml")
    ap.add_argument("--id", default=None,
                    help="volunteer id (else taken from the 2nd positional arg "
                         "in `folder id` form, or parsed from filenames)")
    ap.add_argument("--out-dir", default=None,
                    help="output folder; default: reports/<volunteer_id>")
    ap.add_argument("--overlay", action="store_true",
                    help="(no-op) overlays are ON by default; kept for "
                         "back-compat. Use --no-overlay to disable.")
    ap.add_argument("--no-overlay", action="store_true",
                    help="do NOT write the per-image overlay images.")
    ap.add_argument("--angle-3d", action="store_true",
                    help="write interactive 3D palm-angle HTML debug files to "
                         "the output folder; requires plotly.")
    ap.add_argument("--angle-3d-tabs", action="store_true",
                    help="write ONE combined 3D palm-angle HTML per participant "
                         "with a clickable tab per hand/pose (front-on start "
                         "view, gradient plane); requires plotly.")
    ap.add_argument("--overlay-on-image", action="store_true",
                    help="draw the check panel ON TOP of the image instead of "
                         "on a separate strip below it (default is below, so "
                         "the panel never covers the palm).")
    ap.add_argument("--no-detect", action="store_true",
                    help="metadata-only: skip hand detection (no present/size/"
                         "brightness/spread/angle rows). Detection runs by "
                         "default.")
    ap.add_argument("--quiet", action="store_true", help="suppress writer prints")
    return ap.parse_args()


def resolve_inputs(inputs, explicit_id):
    """Work out (image_paths, vid) from the positional inputs.

    Two accepted shapes:
      1. `<folder> <id>`  -> glob folder for <id>_palm_*.jpg (the headline UX).
      2. explicit image path(s) -> use them directly; id from --id or filenames.

    Returns (image_paths, vid, expected_missing) where expected_missing is the
    list of the 10 canonical filenames NOT found (folder form only; [] otherwise).
    """
    # --- form 1: exactly two args, first is a dir, second looks like an id ---
    if (len(inputs) == 2 and os.path.isdir(inputs[0])
            and re.fullmatch(r"\d+", inputs[1])):
        folder, vid = inputs[0], inputs[1]
        vid = explicit_id or vid
        found = {}
        for name in sorted(os.listdir(folder)):
            m = PALM_RE.match(name)
            if m and m.group("vid") == vid:
                found[name.lower()] = os.path.join(folder, name)
        # Determine which of the canonical 10 are missing (for a friendly warn).
        expected = [f"{vid}_palm_{h}_{p}.jpg" for h in HANDS for p in POSES]
        missing = [e for e in expected if e.lower() not in found]
        paths = [found[e.lower()] for e in expected if e.lower() in found]
        return paths, vid, missing

    # --- form 2: explicit files (and/or folders) ---
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            for name in sorted(os.listdir(item)):
                if PALM_RE.match(name):
                    paths.append(os.path.join(item, name))
        elif os.path.isfile(item):
            if PALM_RE.match(os.path.basename(item)):
                paths.append(item)
            else:
                print(f"  (skip, not a palm filename: {os.path.basename(item)})")
        else:
            print(f"  (skip, not found: {item})")
    # de-dup, preserve order
    seen, uniq = set(), []
    for p in paths:
        key = os.path.basename(p).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)

    ids = {PALM_RE.match(os.path.basename(p)).group("vid") for p in uniq}
    if explicit_id:
        vid = explicit_id
        uniq = [p for p in uniq
                if PALM_RE.match(os.path.basename(p)).group("vid") == vid]
    elif len(ids) == 1:
        vid = next(iter(ids))
    elif not ids:
        vid = None
    else:
        sys.exit(
            f"multiple participant ids found: {sorted(ids)}\n"
            "This runner grades ONE participant. Pass `<folder> <id>` or --id.")
    return uniq, vid, []


def parse_hand_pose(path):
    m = PALM_RE.match(os.path.basename(path))
    return (m.group("hand").upper(), m.group("pose").upper()) if m else (None, None)


def write_consolidated_result_NOT_USED():
    """Placeholder: result.csv is intentionally dropped for palm (see module
    docstring). Kept as a named anchor so a future reviewer grepping for
    'result' finds the explicit decision."""
    raise NotImplementedError


def write_consolidated_detail(out_dir, vid, rows, quiet):
    """One detail.csv for the whole participant: every check row, all images.
    Grain unchanged (one row per (image, check)); timeline is [] (stills)."""
    path = os.path.join(out_dir, f"palm_{vid}_detail.csv")
    write_detail_csv(path, rows, [], quiet=quiet)
    return path


def write_consolidated_overall(out_dir, vid, rows_by_file, image_order, config, quiet):
    """One overall.csv: ONE row PER IMAGE (that image's PASS/FAIL verdict).

    Built per image from summarize_rows_by_check + summarize_overall so the
    verdict logic matches face/walk exactly. yaw/pitch/frames columns are left
    blank -- a still has no timeline.
    """
    aggregation_cfg = (config or {}).get("report", {}).get("aggregation", {})
    path = os.path.join(out_dir, f"palm_{vid}_overall.csv")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        w.writeheader()
        for fname in image_order:
            file_rows = rows_by_file.get(fname, [])
            if not file_rows:
                continue
            summaries = summarize_rows_by_check(file_rows, aggregation_cfg)
            overall = summarize_overall(summaries)
            vid0 = file_rows[0].volunteer_id
            dtype0 = file_rows[0].data_type
            w.writerow({
                "volunteer_id": vid0, "data_type": dtype0, "filename": fname,
                **overall,
                "frames_sampled": "", "detection_gaps": "",
                "yaw_min": "", "yaw_max": "", "pitch_min": "", "pitch_max": "",
            })
    if not quiet:
        print(f"wrote per-image overall CSV to {path}")
    return path


def main():
    args = parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"config not found: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    images, vid, missing = resolve_inputs(args.inputs, args.id)

    if not images:
        sys.exit("no palm images found (expected NNN_palm_[L|R]_[N|RL|RR|PU|PD].jpg)")
    if vid is None:
        sys.exit("could not determine participant id; pass `<folder> <id>` or --id.")

    out_dir = args.out_dir or os.path.join("reports", vid)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Participant: {vid}")
    print(f"images ({len(images)}): " +
          ", ".join(os.path.basename(p) for p in images))
    if missing:
        print(f"WARNING: {len(missing)} of 10 expected files missing "
              f"(continuing with what's present):")
        for m in missing:
            print(f"    - {m}")
    print(f"output folder: {out_dir}\n")

    want_detect = not args.no_detect
    t0 = time.time()

    # --- participant-level run: per-image checks + N-relative angle grading ---
    rows, _timelines = run_palm_participant(
        vid, images, config, detect=want_detect)

    # Group finalised rows back by filename (preserve input image order).
    rows_by_file = {}
    for r in rows:
        rows_by_file.setdefault(r.filename, []).append(r)
    image_order = [os.path.basename(p) for p in images]

    # --- print rows to stdout (same shape as the old runners) ---
    if not args.quiet:
        print("volunteer_id, data_type, filename, check_level, check_name, "
              "status, reason, frame_index")
        for r in rows:
            print(", ".join(str(x) for x in r.as_tuple()))
        print()

    # --- THE TWO consolidated CSVs (result.csv intentionally dropped) ---
    detail_path = write_consolidated_detail(out_dir, vid, rows, args.quiet)
    overall_path = write_consolidated_overall(
        out_dir, vid, rows_by_file, image_order, config, args.quiet)

    # --- visual debug outputs, one per image ---
    # Overlay is ON by default. 3D angle HTML is opt-in via --angle-3d because it
    # adds an optional Plotly dependency and is meant for researcher/debug review,
    # not normal batch QC.
    if (not args.no_overlay) or args.angle_3d or args.angle_3d_tabs:
        from types import SimpleNamespace
        if not args.no_overlay:
            from qc.utils.palm_overlay import draw_palm_overlay
        if args.angle_3d or args.angle_3d_tabs:
            try:
                from qc.debug.visualize_palm_angle import (
                    save_palm_angle_debug_html,
                    save_palm_angle_debug_tabs_html,
                )
                from qc.checks.check_palm_angle import calculate_palm_angles
            except ImportError as e:
                sys.exit(
                    "--angle-3d / --angle-3d-tabs requires plotly. Install it with:\n"
                    "  python -m pip install plotly\n"
                    f"Original import error: {e}"
                )
        # Collected across the image loop for the combined tabbed file.
        tab_entries = []

        for path in images:
            fname = os.path.basename(path)
            hand, pose = parse_hand_pose(path)
            # Re-detect this image just for visual outputs. This keeps the batch
            # grading path clean and gives both overlay + 3D HTML the same
            # HandResult.
            _, _, hand_result, _ = run_palm_image(
                path, vid, config, hand=hand, pose=pose, detect=True)
            # {check_name: (status, reason)} so the overlay's bottom panel can
            # show each check's full reason (face-overlay style), not just the
            # PASS/FAIL word.
            check_status = {r.check_name: (r.status, r.reason)
                            for r in rows_by_file.get(fname, [])}
            if hand_result is None:
                hand_result = SimpleNamespace(
                    ok=False, message="hand model unavailable",
                    landmarks_px=None, bbox=None,
                    norm_box=None, landmarks_norm=None,
                    handedness=None, handedness_score=None,
                    world_landmarks=None)
            tag = "_".join(x for x in (hand, pose) if x)

            if not args.no_overlay:
                ov_path = os.path.join(out_dir, f"palm_{vid}_{tag}_overlay.jpg")
                draw_palm_overlay(path, hand_result,
                                  checks=check_status, out_path=ov_path,
                                  panel_below=not args.overlay_on_image)
                if not args.quiet:
                    print(f"wrote overlay image to {ov_path}")

            has_hand = (getattr(hand_result, "ok", False)
                        and getattr(hand_result, "world_landmarks", None) is not None)

            if args.angle_3d:
                html_path = os.path.join(out_dir, f"palm_{vid}_{tag}_angle3d.html")
                if has_hand:
                    try:
                        save_palm_angle_debug_html(
                            hand_result.world_landmarks,
                            html_path,
                            title=f"{fname} — palm angle 3D debug",
                        )
                        if not args.quiet:
                            print(f"wrote 3D angle debug HTML to {html_path}")
                    except Exception as e:
                        if not args.quiet:
                            print(f"skip 3D angle debug for {fname}: {e}")
                elif not args.quiet:
                    msg = getattr(hand_result, "message", "no hand result")
                    print(f"skip 3D angle debug for {fname}: {msg}")

            if args.angle_3d_tabs and has_hand:
                # One tab per hand/pose in a single combined file (written after
                # the loop). Precompute the angle so a per-hand failure to
                # measure becomes a note tab rather than aborting the file.
                aok, ainfo = calculate_palm_angles(hand_result.world_landmarks)
                tab_entries.append({
                    "label": tag.replace("_", " / ") or fname,
                    "world_landmarks": hand_result.world_landmarks,
                    "angle_info": ainfo if aok else None,
                    "title": f"{fname} — palm angle 3D debug",
                })

        # --- one combined tabbed HTML for this participant ---
        if args.angle_3d_tabs:
            if tab_entries:
                tabs_path = os.path.join(out_dir, f"palm_{vid}_angle3d_tabs.html")
                try:
                    save_palm_angle_debug_tabs_html(
                        tab_entries, tabs_path,
                        page_title=f"{vid} — palm angle 3D debug (all hands)",
                    )
                    if not args.quiet:
                        print(f"wrote combined 3D angle tabs HTML to {tabs_path}")
                except Exception as e:
                    if not args.quiet:
                        print(f"skip combined 3D angle tabs for {vid}: {e}")
            elif not args.quiet:
                print(f"no detectable hands for {vid}; skipped combined tabs HTML")

    elapsed = time.time() - t0

    # --- console summary of the angle verdicts (the participant-level point) ---
    print("\n=== angle results (relative to each hand's N) ===")
    for r in rows:
        if r.check_name == "check_palm_angle":
            print(f"  {r.filename:24s} {r.status:5s} {r.reason}")

    print(f"\nwrote: {detail_path}")
    print(f"wrote: {overall_path}")
    print(f"\n=== done in {elapsed:.2f}s ===")


if __name__ == "__main__":
    main()