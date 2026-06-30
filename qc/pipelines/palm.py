"""Palm pipeline — metadata + Phase-2 hand checks.

This is the palm-side mirror of face_rgb.py. It runs:

    Image-level (once, from metadata):
      - check_container   (.jpg, decodable, 3-channel color)
      - check_resolution  (>= 200x200 FILE size, the palm spec minimum)

    Hand-level (once, from a single detect_hand call) -- Phase 2:
      - check_palm_present  (was a hand found? 0 hands / >1 hand handled here)
      - check_palm_size     (hand BBOX >= 200x200, the real spec requirement)
      - check_palm_angle    (wrist roll/pitch within +/-45 deg) -- GATED OFF by
                             default via palm.angle.check_angle_enabled; emits
                             SKIP until per-pose angles are calibrated.

Note on "present": there is NO check_palm_present.py file, exactly as the face
pipeline has no check_face_present.py. "Hand present?" is the ok/message of the
shared detector (detect_hand in hand_landmarker.py), the hand-side mirror of how
get_lm's success answers "face present?". The pipeline emits a check_palm_present
ROW from that detector result; it is not a separate pure-check module.

Still OUT (later phases): finger spread / all-five-fingers-visible, palm-open,
veins-visible, jewelry. Those need their own detectors/heuristics.

Detection runs ONCE per image (the single-detect discipline face established):
the one HandResult feeds present, size, and angle, plus the overlay.

Why these two checks mirror face cleanly
----------------------------------------
`check_container` and `check_resolution` (qc/checks/check_metadata.py) read a
metadata object, not pixels. `probe_image()` produces the SAME VideoMetadata
shape `probe_video()` does, so the identical check functions are reused -- only
the inputs differ (a still vs a video) and the thresholds (200 vs 180, .jpg vs
.mp4). check_fps / check_duration are NOT run: a still has no fps or duration.

Level note
----------
Face used check_level="video" for these metadata rows. A still has no video
container, so this pipeline tags them "image". CheckRow.level is a free str
(not enforced against the CheckLevel Literal), so "image" is accepted; the
report writers treat level as opaque text. If report.py is later made to branch
on known levels, add "image" there.

Returns
-------
(rows, timeline) -- mirrors run_face_rgb's signature so the runner and report
writers are reused verbatim. `timeline` is ALWAYS [] here: a still has no
per-frame timeline, and the detail_header CSV (which iterates timeline) will
therefore have only its header row. That is correct for a metadata-only slice.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from qc.utils.image import probe_image
from qc.checks import check_metadata as md
from qc.schemas import CheckRow

logger = logging.getLogger(__name__)

DATA_TYPE = "palm"


def run_palm(
    path: str,
    volunteer_id: str,
    config: dict,
    *,
    hand: str | None = None,
    pose: str | None = None,
    progress=None,
    detect: bool = False,
):
    """Run the minimal palm pipeline (container + resolution) on one image.

    Args:
        path: path to NNN_palm_[L|R]_[N|RL|RR|PU|PD].jpg
        volunteer_id: e.g. "0001"
        config: the loaded config.yml dict (thresholds live here).
        hand: "L" / "R" parsed from the filename (carried for later checks;
            unused by this metadata-only slice).
        pose: "N"/"RL"/"RR"/"PU"/"PD" parsed from the filename (ditto).
        progress: optional callback(row) invoked as each CheckRow is emitted.

    Returns:
        (rows, timeline) where rows is list[CheckRow] and timeline is [] (a
        still image has no per-frame timeline).
    """
    filename = os.path.basename(path)
    rows: list[CheckRow] = []

    def add(check_name, status, reason, frame_index=None, level="image"):
        # Same add() helper shape as face_rgb.py: build a CheckRow, append it,
        # fire the progress callback. Default level is "image" (vs face's
        # "video") because a still has no video container.
        row = CheckRow(
            volunteer_id,
            DATA_TYPE,
            filename,
            check_name,
            status,
            reason,
            frame_index,
            level=level,
        )
        rows.append(row)
        if progress is not None:
            progress(row)

    # ---- pull thresholds from config (spec is source of truth) ----
    palm_cfg = config.get("palm", {})
    expected_ext = palm_cfg.get("extension", ".jpg")
    size_cfg = palm_cfg.get("size", {})
    min_w = size_cfg.get("min_width_px", 200)
    min_h = size_cfg.get("min_height_px", 200)

    # Brightness thresholds for the hand region. Defaults mirror face, but palm
    # skin may sit in a different V range and the spec's "veins visible" is a
    # contrast cue mean-V does not capture -- tune palm.brightness.* on real
    # samples. [CONFIRM]
    bright_cfg = palm_cfg.get("brightness", {})
    dark_th = bright_cfg.get("dark_threshold", 35.0)
    bright_th = bright_cfg.get("bright_threshold", 200.0)
    bright_margin = bright_cfg.get("margin", 0.1)
    brightness_enabled = bright_cfg.get("enabled", True)

    # ---- image-level checks (once, from metadata) ----
    meta = probe_image(path)

    # check_container: reuse the face function. require_rgb=True keeps the
    # 3-channel-colour requirement (palm vein imaging is colour); expected_ext
    # comes from config (.jpg for palm, not .mp4).
    status, reason = md.check_container(
        meta, require_rgb=True, expected_ext=expected_ext)
    add("check_container", status, reason, level="image")

    # check_resolution: identical function, palm thresholds (>= 200x200).
    # NOTE: this is the WHOLE-IMAGE resolution, not the hand-bbox size. The
    # spec's "200x200 from mid-finger to wrist" is a HAND-region size that
    # needs the detector (a later check_palm_size). This metadata check only
    # asserts the file itself is at least 200x200, which is a necessary
    # precondition for the hand region to possibly be that large.
    status, reason = md.check_resolution(
        meta, min_width=min_w, min_height=min_h)
    add("check_resolution", status, reason, level="image")

    # ---- hand-level checks (Phase 2): run the detector ONCE, emit rows ----
    # Detection is gated by `detect` (the runner sets it). When on, one
    # detect_hand call produces the HandResult that present/size/angle all read,
    # plus the bbox/landmarks the overlay draws -- the single-detect discipline
    # the face pipeline established (detect once, every check consumes it).
    hand_result = None
    measured_angle = None  # raw {"roll","pitch"} for batch N-delta grading
    if detect:
        angle_cfg = palm_cfg.get("angle", {})
        angle_enabled = angle_cfg.get("check_angle_enabled", False)
        max_roll = angle_cfg.get("max_abs_roll_deg", 45)
        max_pitch = angle_cfg.get("max_abs_pitch_deg", 45)
        # Per-pose directional validation knobs (used when angle is enabled).
        max_abs_angle = angle_cfg.get("max_abs_deg", 45)
        min_rotation = angle_cfg.get("min_rotation_deg", 10)
        neutral_tol = angle_cfg.get("neutral_tol_deg", 10)
        off_axis_tol = angle_cfg.get("off_axis_tol_deg", 20)
        # Sign-calibration overrides per (hand, pose), e.g. {"R": {"RL": 1}}.
        # Empty until calibrated against known-correct reference images. [CONFIRM]
        sign_overrides = angle_cfg.get("hand_sign_overrides", {}) or None

        models_cfg = config.get("models", {})
        hl_cfg = models_cfg.get("hand_landmarker", {})
        hands_cfg = models_cfg.get("hands", {})
        # >1 hand is a capture error for palm (one hand per shot) -> FAIL, vs
        # face's REVIEW for an extra face. Confirm policy with อ.เหมียว.
        multi_policy = hands_cfg.get("multiple_hands_policy", "FAIL")

        try:
            from qc.checks.hand_landmarker import (
                create_hand_landmarker, detect_hand,
            )
            from qc.checks.check_palm_size import check_palm_min_size
            from qc.checks.check_palm_angle import check_palm_angle, check_palm_pose

            detector = create_hand_landmarker(
                model_path=hl_cfg.get("model_path", "models/hand_landmarker.task"),
                num_hands=hands_cfg.get("max_num_hands", 1),
                min_hand_detection_confidence=hands_cfg.get(
                    "min_detection_confidence", 0.5),
            )
            try:
                hand_result = detect_hand(path, detector=detector)
            finally:
                detector.close()

            # --- check_palm_present: derived from the detector, NOT a separate
            # check module (mirrors how face uses get_lm's success; there is no
            # check_face_present.py). 0 hands or >1 hand both land here. ---
            if hand_result.ok:
                add("check_palm_present", "PASS", hand_result.message, level="image")
            else:
                # detect_hand already distinguishes "No hands detected" from
                # "Multiple hands detected: N". A multi-hand frame uses the
                # configured policy (FAIL); any other not-ok (no hand, bad image)
                # is a FAIL too -- a palm shot with no detectable hand cannot
                # satisfy the spec.
                msg = hand_result.message or "No hand detected"
                status = multi_policy if msg.startswith("Multiple hands") else "FAIL"
                add("check_palm_present", status, msg, level="image")

            # --- check_palm_size: hand BBOX >= 200x200 (the real spec size).
            # Only meaningful when a single hand was found; otherwise the bbox is
            # None and present already FAILed, so SKIP to avoid a duplicate
            # failure reason for the same root cause. ---
            if hand_result.ok and hand_result.bbox is not None:
                ok, reason = check_palm_min_size(
                    hand_result.bbox, min_width=min_w, min_height=min_h)
                add("check_palm_size", "PASS" if ok else "FAIL", reason, level="image")
            else:
                add("check_palm_size", "SKIP",
                    "no single hand bbox (see check_palm_present)", level="image")

            # --- check_palm_brightness: mean V inside the hand bbox. Reuses the
            # shared brightness CORE (check_brightness) via the palm wrapper, so
            # face and palm judge exposure with identical logic over different
            # regions. SKIP when there is no single-hand bbox (same root cause as
            # size: present already FAILed), or when disabled in config. ---
            if not brightness_enabled:
                add("check_palm_brightness", "SKIP",
                    "palm.brightness.enabled=false", level="image")
            elif hand_result.ok and hand_result.bbox is not None:
                from qc.checks.check_brightness import check_brightness_palm
                ok, reason = check_brightness_palm(
                    path, hand_result.bbox,
                    dark_threshold=dark_th, bright_threshold=bright_th,
                    margin=bright_margin)
                add("check_palm_brightness", "PASS" if ok else "FAIL",
                    reason, level="image")
            else:
                add("check_palm_brightness", "SKIP",
                    "no single hand bbox (see check_palm_present)", level="image")

            # --- check_palm_spread: angular gap between adjacent fingers.
            # QUALITY check, enforced on the NEUTRAL (N) pose ONLY: RL/RR/PU/PD
            # tilt the hand out of the image plane and distort an in-plane
            # spread angle. SKIP on non-N poses and when disabled in config.
            # SKIP when there is no single-hand result (same root cause as size:
            # present already FAILed). Per the current assumption, a fully-
            # detected hand is expected; a non-measurable spread returns FAIL via
            # the check's own message. [DESIGN -> CONFIRM with อ.เหมียว] ---
            spread_cfg = palm_cfg.get("hand_pose", {}).get("spread", {})
            spread_enabled = spread_cfg.get("enabled", True)
            spread_poses = spread_cfg.get("eval_poses", ["N"])
            if not spread_enabled:
                add("check_palm_spread", "SKIP",
                    "palm.hand_pose.spread.enabled=false", level="image")
            elif pose is not None and pose not in spread_poses:
                add("check_palm_spread", "SKIP",
                    f"pose={pose} not in eval_poses={spread_poses} "
                    f"(spread enforced on neutral only)", level="image")
            elif hand_result.ok and hand_result.landmarks_norm is not None:
                from qc.checks.check_palm_spread import check_palm_spread
                ok, reason = check_palm_spread(
                    hand_result.landmarks_norm,
                    min_gap_inter_tip_deg=spread_cfg.get("min_gap_inter_tip_deg", 5.0),
                    min_gap_thumb_tip_deg=spread_cfg.get("min_gap_thumb_tip_deg", 20.0),
                    min_gap_inter_pip_deg=spread_cfg.get("min_gap_inter_pip_deg", 3.0),
                    min_gap_thumb_pip_deg=spread_cfg.get("min_gap_thumb_pip_deg", 12.0),
                    required_pairs=spread_cfg.get("required_pairs"),
                    frame_margin=spread_cfg.get("frame_margin", 0.02))
                # A non-measurable hand (MCP cropped, or both tip+pip cropped for
                # some finger) is NOT a defect -> SKIP, not FAIL. The check's
                # message starts with "unmeasurable:" in that case.
                if not ok and reason.startswith("unmeasurable:"):
                    add("check_palm_spread", "SKIP", reason, level="image")
                else:
                    add("check_palm_spread", "PASS" if ok else "FAIL",
                        reason, level="image")
            else:
                add("check_palm_spread", "SKIP",
                    "no normalized landmarks (see check_palm_present)", level="image")

            # --- check_palm_angle: now graded at PARTICIPANT/BATCH level, not
            # per-image. Absolute per-image angles are unreliable (a valid N
            # reads non-zero), so a rotated pose is graded RELATIVE to this
            # hand's own N (see run_palm_participant / check_palm_pose_delta).
            # Here we only (a) emit a deferred SKIP row, and (b) capture the raw
            # measured angle so the batch layer can compute the N-delta. The raw
            # angle is returned via `measured_angle` (None if unmeasurable). ---
            if hand_result.ok and hand_result.world_landmarks is not None:
                from qc.checks.check_palm_angle import calculate_palm_angles
                aok, ainfo = calculate_palm_angles(hand_result.world_landmarks)
                if aok:
                    measured_angle = {"roll": ainfo["roll"], "pitch": ainfo["pitch"]}
            add("check_palm_angle", "SKIP",
                "deferred to participant-level N-relative grading", level="image")

        except FileNotFoundError as e:
            # Model bundle missing -- don't crash the run. The metadata rows
            # already passed; mark the hand checks SKIP so the gap is explicit.
            logger.warning("PALM | hand model unavailable, hand checks skipped: %s", e)
            for cname in ("check_palm_present", "check_palm_size", "check_palm_brightness", "check_palm_spread", "check_palm_angle"):
                add(cname, "SKIP", "hand model bundle unavailable", level="image")
        except Exception as e:
            logger.warning("PALM | hand detection error, hand checks skipped: %s", e)
            for cname in ("check_palm_present", "check_palm_size", "check_palm_brightness", "check_palm_spread", "check_palm_angle"):
                add(cname, "SKIP", f"hand detection error: {e}", level="image")

    # timeline is [] -- a still has no per-frame timeline.
    # 4-tuple: (rows, timeline, hand_result, measured_angle). measured_angle is
    # the raw {"roll","pitch"} (or None) the participant-level pass needs for
    # N-relative grading. Existing callers that unpack 3 values should switch to
    # 4; run_palm_participant relies on the 4th.
    return rows, [], hand_result, measured_angle

def run_palm_participant(
    participant_id: str,
    image_paths: list,
    config: dict,
    *,
    progress=None,
    detect: bool = True,
):
    """Run palm QC for ONE participant across all their palm images, grading the
    angle check at PARTICIPANT level (each rotated pose relative to that hand's
    own N), which per-image grading cannot do.

    Flow:
      1. Per image: run the per-image pipeline (container/resolution/present/
         size/brightness/spread). The per-image angle row is a deferred SKIP;
         the raw measured angle is captured here for the batch pass.
      2. Group captured angles by hand (L/R).
      3. For each hand, take N as the baseline and grade RL/RR/PU/PD as deltas
         (check_palm_pose_delta). Replace each rotated pose's deferred angle row
         with the real PASS/FAIL verdict.

    Decisions (confirmed with the project):
      - N's own angle row: SKIP (N is the reference, not graded).
      - No N FILE for a hand            -> rotated poses FAIL ("no N reference").
      - N file present but undetectable -> rotated poses FAIL ("N unusable").
      - Rotated pose undetectable        -> that pose FAIL ("no hand; likely
                                            over-rotation").
      - Pose file entirely absent        -> SILENT (no row for that pose).
      - Axes that FAIL: roll, pitch only (no yaw pose in the spec).

    Args:
        participant_id: e.g. "004".
        image_paths: list of this participant's palm image paths (any subset of
            the 10 hand x pose files).
        config: loaded config.yml dict.
        progress: optional callback(row).
        detect: run the hand detector (required for angle/size/etc.).

    Returns:
        (rows, timelines) where rows is the combined list[CheckRow] across all
        images (with angle rows finalised) and timelines is a list of [] (one
        empty per image, kept for writer-shape parity).
    """
    import re as _re

    _PALM_RE = _re.compile(
        r"^(?P<vid>\d+)_palm_(?P<hand>[LR])_(?P<pose>N|RL|RR|PU|PD)\.jpg$",
        _re.IGNORECASE)

    palm_cfg = config.get("palm", {})
    angle_cfg = palm_cfg.get("angle", {})
    max_abs_angle = angle_cfg.get("max_abs_deg", 45)
    min_rotation = angle_cfg.get("min_rotation_deg", 10)
    off_axis_tol = angle_cfg.get("off_axis_tol_deg", 20)
    sign_overrides = angle_cfg.get("hand_sign_overrides", {}) or None

    from qc.checks.check_palm_angle import check_palm_pose_delta

    all_rows: list[CheckRow] = []
    all_timelines: list = []

    # angle_state[hand][pose] = {"row": CheckRow, "angle": {...}|None, "detected": bool}
    angle_state: dict = {"L": {}, "R": {}}

    # --- pass 1: per-image checks; stash each image's deferred angle row + raw angle
    for path in image_paths:
        fname = os.path.basename(path)
        m = _PALM_RE.match(fname)
        hand = m.group("hand").upper() if m else None
        pose = m.group("pose").upper() if m else None

        rows, timeline, hand_result, measured_angle = run_palm(
            path, participant_id, config,
            hand=hand, pose=pose, progress=None, detect=detect)

        # Find this image's deferred angle row so the batch pass can finalise it.
        angle_row = None
        for r in rows:
            if r.check_name == "check_palm_angle":
                angle_row = r
                break

        if hand in ("L", "R") and pose in ("N", "RL", "RR", "PU", "PD"):
            angle_state[hand][pose] = {
                "row": angle_row,
                "angle": measured_angle,                 # None if unmeasurable
                "detected": bool(hand_result and hand_result.ok),
            }

        # Emit the non-angle rows now (angle rows are finalised in pass 2).
        for r in rows:
            if r.check_name != "check_palm_angle":
                all_rows.append(r)
                if progress is not None:
                    progress(r)
        all_timelines.append(timeline)

    # --- pass 2: per hand, grade rotated poses against that hand's N
    rotated = ("RL", "RR", "PU", "PD")
    for hand in ("L", "R"):
        poses = angle_state[hand]
        n_entry = poses.get("N")

        # N row (if the N file exists): SKIP -- reference, not graded. We STILL
        # report N's raw measured roll/pitch so the reference value is visible
        # (it is the baseline every rotated pose is judged against).
        if n_entry and n_entry["row"] is not None:
            n_reason = "N is the reference (not graded)"
            raw = _fmt_raw_angle(n_entry.get("angle"))
            if raw:
                n_reason = f"{n_reason}; {raw}"
            n_row = _set_row(n_entry["row"], "SKIP", n_reason)
            _emit(all_rows, progress, n_row)

        # Determine the usable N baseline (file present AND detectable AND measured).
        n_ok = bool(n_entry and n_entry["detected"] and n_entry["angle"] is not None)
        n_angle = n_entry["angle"] if n_ok else None

        for pose in rotated:
            entry = poses.get(pose)
            if entry is None:
                continue  # pose file entirely absent -> SILENT (no row)

            row = entry["row"]
            if row is None:
                continue  # shouldn't happen (detect on), but stay safe

            if not n_ok:
                # No usable N reference -> the rotated pose cannot be graded.
                why = ("no N file for this hand" if n_entry is None
                       else "N present but unusable (undetected/unmeasurable)")
                row = _set_row(row, "FAIL", f"cannot grade {hand}/{pose}: {why}")
            elif not entry["detected"] or entry["angle"] is None:
                # Pose itself undetectable -> likely over-rotation -> FAIL.
                row = _set_row(row, "FAIL",
                               f"{hand}/{pose}: no hand to measure (likely over-rotation)")
            else:
                ok, reason = check_palm_pose_delta(
                    pose, hand, entry["angle"], n_angle,
                    max_abs_deg=max_abs_angle,
                    min_rotation_deg=min_rotation,
                    off_axis_tol_deg=off_axis_tol,
                    hand_sign_overrides=sign_overrides)
                # Report this pose's RAW roll/pitch IN ADDITION to the delta
                # verdict (the delta vs N is the grading; the raw value is the
                # extra context you asked for).
                raw = _fmt_raw_angle(entry.get("angle"))
                if raw:
                    reason = f"{reason} | {raw}"
                row = _set_row(row, "PASS" if ok else "FAIL", reason)

            _emit(all_rows, progress, row)

    return all_rows, all_timelines


def _fmt_raw_angle(angle):
    """Format a measured {"roll","pitch"} dict as 'raw roll=+12.3 pitch=-4.5',
    or '' when no angle was measured (unmeasurable / detection off)."""
    if not angle:
        return ""
    try:
        return (f"raw roll={float(angle['roll']):+.1f} "
                f"pitch={float(angle['pitch']):+.1f}")
    except (KeyError, TypeError, ValueError):
        return ""


def _set_row(row, status, reason):
    """Return a finalised copy of a deferred CheckRow with new status/reason.
    CheckRow is frozen (immutable), so we rebuild via dataclasses.replace rather
    than mutate in place."""
    import dataclasses
    return dataclasses.replace(row, status=status, reason=reason)


def _emit(all_rows, progress, row):
    all_rows.append(row)
    if progress is not None:
        progress(row)