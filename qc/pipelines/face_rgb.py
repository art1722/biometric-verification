"""Face RGB pipeline — the first complete QC pipeline.

For one `NNN_face_rgb.mp4` it runs, in order:

  Video-level (once, from metadata):
    - check_container   (.mp4, RGB, readable)
    - check_fps         (>= 5)
    - check_duration    (>= 40s)
    - check_resolution  (>= 180x180)

  Frame-level (per sampled frame):
    - face detected?            (get_lm)
    - head pose angles          (estimate_head_pose) -> classifies the frame
    - head fully visible?       (check_head_fully — ALL detected frames)
    Frontal frames only (label == front; on turn/mid frames these are SKIPped
    because the measurement domain is invalid — EAR breaks at profile, the
    bbox narrows at yaw, turning frames carry motion blur, and the brightness
    bbox is part background):
    - face size >= 180x180?     (check_face_min_size, consumes bbox)
    - eyes open?                (check_eye_status)
    - blur ok?                  (check_face_blur)
    - brightness ok?            (check_brightness_face)

  Sequence-level (whole timeline at once):
    - turn sequence             (check_turn_sequence_seg)

Every CheckRow carries a `level` ("video" / "sequence" / "frame") so the
report can be grouped from cheap whole-file checks down to per-frame detail.

Each check produces a CheckRow:
    volunteer_id, data_type, filename, check_name, status, reason
plus optional frame_index / measured / threshold for richer reports.

Efficiency notes (matter at 1,500-volunteer scale):
  - The face is detected ONCE per frame via get_lm; face_size and head_fully
    reuse those landmarks instead of re-detecting.
  - Detector-based checks (blur, brightness, pose) reuse ONE shared detector
    each, created in this pipeline, instead of building a new model per call.
  - Frames are consumed from a generator (iter_sampled_frames) so only one
    frame is in memory at a time.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Optional

import mediapipe as mp

from qc.utils.video import probe_video, iter_sampled_frames
from qc.checks.face_landmarker import create_face_landmarker, detect_face
from qc.checks.check_face_size import check_face_min_size
from qc.checks.check_head_fully import check_head_fully
from qc.checks.check_face_blur import check_face_blur
from qc.checks.check_brightness import check_brightness_face
from qc.checks.check_head_pose import estimate_head_pose
from qc.checks.check_eye import check_eye_status
from qc.checks.check_occlusion import check_occlusion
from qc.checks.check_turn_sequence_seg import (
    check_turn_sequence_seg, apply_gap_split_to_detection_rows,
)
from qc.checks._turn_common import TurnThresholds, classify_frame, FRONT
from qc.checks import check_metadata as md
from qc.schemas import CheckRow

logger = logging.getLogger(__name__)

DATA_TYPE = "face_rgb"

def _bool_to_status(ok: bool, *, fail="FAIL") -> str:
    return "PASS" if ok else fail


# Pull the single numeric value out of the brightness/blur message strings so it
# can be stored per-frame for the dashboard timelines. The checks already format
# these consistently: brightness messages always contain "brightness=NN" and a
# judged blur message always contains "sharpness=NN". Returns None when absent
# (e.g. a SKIPped blur frame has no sharpness number).
_BRIGHTNESS_RE = re.compile(r"brightness=(-?\d+(?:\.\d+)?)")
_SHARPNESS_RE = re.compile(r"sharpness=(-?\d+(?:\.\d+)?)")


def _extract_float(pattern, text):
    if not text:
        return None
    m = pattern.search(text)
    return float(m.group(1)) if m else None


def run_face_rgb(
    path: str,
    volunteer_id: str,
    config: dict,
    *,
    sample_fps: float | None = None,
    overlay: "Any | None" = None,
    progress=None,
    fail_fast: bool = True,
):
    """Run the full face_rgb pipeline on one video.

    Args:
        path: path to NNN_face_rgb.mp4
        volunteer_id: e.g. "001"
        config: the loaded config.yml dict (thresholds live here).
        sample_fps: how densely to sample frames for the per-frame checks.
        fail_fast: when True (batch default), a STRUCTURAL defect halts the
            pipeline early instead of processing the rest of the file — the
            fail-fast / circuit-breaker pattern. Three structural gates:
              1. any video-level metadata check FAILs (container/fps/duration/
                 resolution) -> stop before opening frames.
              2. the file opens but yields 0 frames -> emit a FAIL and stop.
              3. a frame contains MULTIPLE faces (a second person) -> emit a
                 FAIL and break the frame loop.
            Ratio-judged quality checks (size/eyes/brightness/blur) and the
            ratio-judged head_fully check NEVER fail-fast: a few bad frames are
            normal and an early break would lose the denominator.
            Set False for dashboard/overlay runs that need the FULL per-frame
            timeline for visualization — then every check still runs and the
            report aggregation makes the verdict instead.

    Returns:
        (rows, timeline) where:
          rows     = list[CheckRow] for every check on every level
          timeline = list[dict], ONE entry per sampled frame (gaps included):
                     {frame_index, timestamp_sec, face_detected, yaw, pitch, roll}
                     On a detection gap, face_detected=False and yaw/pitch/roll=None.
                     This is the data the turn-sequence check consumes: the None-yaw
                     gap frames are the signal for a deep (profile) turn whose peak
                     MediaPipe could not measure.
    """
    filename = os.path.basename(path)
    rows: list[CheckRow] = []

    # Per-frame check results for the overlay, populated only while overlay is
    # on. Maps check_name -> (status, reason) for the frame currently being
    # processed; reset at the top of each frame. None-overlay => stays unused.
    _frame_checks: dict[str, tuple] = {}

    def add(check_name, status, reason, frame_index=None, level="frame"):
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

        if overlay is not None and level == "frame":
            _frame_checks[check_name] = (status, reason)

        if progress is not None:
            progress(row)

    # ---- pull thresholds from config (spec is source of truth) ----
    face_cfg = config.get("face", {})
    meta_cfg = face_cfg.get("metadata", {})
    size_cfg = face_cfg.get("checks", {}).get("face_size", {})
    blur_cfg = face_cfg.get("checks", {}).get("blur", {})
    bright_cfg = face_cfg.get("checks", {}).get("brightness", {})
    hf_cfg = face_cfg.get("checks", {}).get("head_fully_visible", {})

    min_fps = meta_cfg.get("min_fps", 5)
    min_dur = meta_cfg.get("min_duration_sec", 40)
    min_w = face_cfg.get("size", {}).get("min_head_width_px", 180)
    min_h = face_cfg.get("size", {}).get("min_head_height_px", 180)
    
    # New blur score is Tenengrad/Sobel after normalization, not raw Laplacian.
    blur_th = blur_cfg.get("threshold", 35.0)
    blur_resize_px = blur_cfg.get("resize_px", 224)
    blur_crop_margin = blur_cfg.get("crop_margin", 0.15)
    blur_min_luma = blur_cfg.get("min_luma", 35.0)
    blur_min_contrast = blur_cfg.get("min_contrast", 4.0)
    
    blur_preprocess = os.getenv(
        "QC_BLUR_PREPROCESS",
        blur_cfg.get("preprocess", "clahe"),
    )

    blur_clahe_clip_limit = blur_cfg.get("clahe_clip_limit", 2.0)
    blur_clahe_tile_grid_size = blur_cfg.get("clahe_tile_grid_size", 8)

    blur_lide_d = blur_cfg.get("lide_d", 30)
    blur_lide_sigma_min = blur_cfg.get("lide_sigma_min", 30.0)
    
    dark_th = bright_cfg.get("dark_threshold", 35)
    bright_th = bright_cfg.get("bright_threshold", 200)
    diff_th = bright_cfg.get("diff_threshold", 20)
    margin = bright_cfg.get("margin", 0.1)
    hf_margin = hf_cfg.get("margin_px", 10)

    eye_cfg = face_cfg.get("checks", {}).get("eyes_open", {})
    blink_th = eye_cfg.get("blink_threshold", 0.5)

    # Occlusion (skin-colour method). Thresholds live under face.occlusion.skin;
    # the check builds a per-region ROI from the same landmarks and tests skin
    # presence in YCrCb. Read once here, reused for every frame.
    occ_cfg = face_cfg.get("occlusion", {}).get("skin", {})
    occ_required = occ_cfg.get("required_regions",
                               ["left_eye", "right_eye", "nose", "mouth"])
    occ_cr = (occ_cfg.get("cr_min", 133), occ_cfg.get("cr_max", 173))
    occ_cb = (occ_cfg.get("cb_min", 77), occ_cfg.get("cb_max", 127))
    occ_min_skin = occ_cfg.get("min_skin_ratio", 0.40)
    occ_roi_margin = occ_cfg.get("roi_margin", 0.15)
    
    # ---- video-level checks (once) ----
    meta = probe_video(path)
    
    # The turn-sequence check converts hold-seconds -> frame counts, so it needs
    # a concrete fps. When sampling natively (sample_fps is None = every frame),
    # the effective rate IS the source's native fps. Resolve it once here.
    effective_fps = sample_fps
    if effective_fps is None:
        effective_fps = meta.fps if (meta.fps and meta.fps > 0) else 30.0

    status, reason = md.check_container(meta, require_rgb=True)
    add("check_container", status, reason, level="video")

    status, reason = md.check_fps(meta, min_fps=min_fps)
    add("check_fps", status, reason, level="video")

    status, reason = md.check_duration(meta, min_duration_sec=min_dur)
    add("check_duration", status, reason, level="video")

    status, reason = md.check_resolution(meta, min_width=min_w, min_height=min_h)
    add("check_resolution", status, reason, level="video")

    # If the file isn't even readable, stop here — no frames to check.
    if not meta.readable:
        add("frame_checks", "SKIP", "video unreadable; frame checks skipped",
            level="video")
        return rows, []

    # ---- fail-fast GATE 1: video-level metadata ----
    # The four checks above describe the FILE, not any one frame. If the
    # container is wrong, the fps is too low, the clip is too short, or the
    # resolution is below spec, the file is structurally non-conforming and
    # no amount of frame analysis can change the verdict. Stop here rather
    # than spend model time on ~1,500-scale frame work for a doomed file.
    # (Skipped when fail_fast=False so dashboard/overlay runs still produce a
    # full per-frame timeline even on an off-spec file.)
    if fail_fast:
        VIDEO_GATE = ("check_container", "check_fps",
                      "check_duration", "check_resolution")
        gate_fail = next(
            (r for r in rows
             if r.check_name in VIDEO_GATE and r.status == "FAIL"),
            None,
        )
        if gate_fail is not None:
            add("frame_checks", "SKIP",
                f"fail-fast: {gate_fail.check_name} FAILed "
                f"({gate_fail.reason}); frame checks skipped",
                level="video")
            return rows, []

    # ---- per-frame checks ----
    # ONE timeline entry per sampled frame, gaps included. The turn-sequence
    # check reads this: a gap (face_detected=False, yaw=None) bracketed by
    # front-facing frames is the signature of a deep profile turn whose peak
    # MediaPipe could not measure.
    timeline: list[dict] = []
    frames_seen = 0
    
    # Set True if a fail-fast structural gate breaks the frame loop early
    # (currently: a multiple-faces frame). When aborted, the post-loop
    # sequence checks (gap split + turn sequence) are skipped because the
    # timeline is intentionally incomplete and the file already has a FAIL.
    aborted = False
    sequence_blocked_by_structural_fail = False
    
    # Positions of the check_face_detected rows for NO-FACE frames, keyed by
    # timeline index, so they can be re-statused (expected/unexpected gap)
    # once the whole timeline exists. Multiple-face frames are NOT tracked:
    # a second face is a different defect class (someone else in frame), not
    # a detector limitation, so its row keeps the conservative REVIEW.
    gap_row_positions: dict[int, int] = {}

    def add_frame(sf, *, face_detected, info=None, bbox=None):
        timeline.append({
            "frame_index": sf.frame_index,
            "timestamp_sec": sf.timestamp_sec,
            "face_detected": face_detected,
            "label_width": bbox[2] if bbox else None,
            "label_height": bbox[3] if bbox else None,
            "yaw": info["yaw"] if info else None,
            "pitch": info["pitch"] if info else None,
            "roll": info["roll"] if info else None,
            # Per-frame measured quality values for the dashboard timelines.
            # Filled in below as each check runs; stay None on no-face / SKIP /
            # non-frontal frames so the dashboard breaks the line at the gap.
            "brightness": None,
            "blink_left": None,
            "blink_right": None,
            "sharpness": None,
            # Per-region occlusion skin ratios (0.0-1.0), one column per region.
            # Filled by the occlusion check on frontal frames; None elsewhere so
            # the dashboard breaks each region's line at non-frontal / gap frames.
            "occ_forehead": None,
            "occ_left_eye": None,
            "occ_right_eye": None,
            "occ_nose": None,
            "occ_mouth": None,
            "occ_left_cheek": None,
            "occ_right_cheek": None,
        })


    # Create the detectors ONCE and reuse them for every frame. After the
    # Tasks-API migration there are two shared detectors:
    #   - face_landmarker : the NEW Tasks-API model. ONE inference per frame
    #     yields landmarks (face_size / head_fully / head pose) AND 52
    #     blendshapes (eyeBlinkLeft/Right -> eyes-open). Blendshapes do not
    #     exist on the legacy solutions API, which is why eyes-open was migrated.
    #   - face_det : legacy FaceDetection, still used by blur and brightness.
    #     Those two genuinely need a face-region box from this model (not just
    #     landmarks), so it is kept rather than folded into the landmarker.
    # All FaceLandmarker params now come from ONE block,
    # config.models.face_landmarker, using key names that match the
    # create_face_landmarker() parameters they feed. (They were previously
    # split across models.mediapipe + models.face_landmarker, a leftover from
    # the pre-Tasks-API migration: the legacy `mediapipe` block held
    # max_num_faces / min_detection_confidence under names that no longer
    # matched anything in the Tasks API. Consolidated so one detector is
    # configured from one place.)
    fl_cfg = config.get("models", {}).get("face_landmarker", {})
    fd_cfg = config.get("models", {}).get("face_detection", {})

    face_landmarker = create_face_landmarker(
        model_path=fl_cfg.get("model_path", "models/face_landmarker.task"),
        num_faces=fl_cfg.get("num_faces", 10),
        min_face_detection_confidence=fl_cfg.get(
            "min_face_detection_confidence", 0.6),
        min_face_presence_confidence=fl_cfg.get("min_face_presence_confidence", 0.5),
        min_tracking_confidence=fl_cfg.get("min_tracking_confidence", 0.5),
    )
    face_det = mp.solutions.face_detection.FaceDetection(
        model_selection=fd_cfg.get("model_selection", 1),
        min_detection_confidence=fd_cfg.get("min_detection_confidence", 0.5),
    )

    # Thresholds for per-frame position classification (front / left / right /
    # up / down / mid / gap) — same config + same classify_frame the
    # turn-sequence check uses, so a frame is "front" by ONE definition only.
    turn_th = TurnThresholds.from_config(config)

    # Quality checks that are only geometrically valid on a frontal face:
    #   - face_size : at high yaw the bbox width narrows (profile is thinner)
    #   - eyes_open : EAR needs both eyes' landmarks; invalid at profile/pitch
    #   - blur      : turning frames carry motion blur by nature
    #   - brightness: at profile the face bbox is part background -> "backlight"
    # head_fully is NOT in this list: "full head down to neck, not cut by the
    # frame edge" must hold during turns too, and its landmarks (top-of-head,
    # chin) remain meaningful whenever the face is detected at all.
    FRONT_ONLY_CHECKS = ("check_face_size", "check_eyes_open",
                         "check_face_blur", "check_brightness",
                         "check_occlusion")

    try:
        # Native sampling (sample_fps=None) must keep every frame so the overlay
        # is a true 1:1 copy -> disable the cap. Down-sampling keeps it. The cap
        # is read from config.video.max_frames (default 6000); it only triggers
        # for unusually long videos, and when it does the downsample is now an
        # even stride (no gaps). Set config.video.max_frames to null to disable.
        _cap = config.get("video", {}).get("max_frames", 6000)
        _max_frames = None if sample_fps is None else _cap
        for sf in iter_sampled_frames(path, sample_fps=sample_fps,
                                      max_frames=_max_frames):
            frames_seen += 1
            img = sf.image
            cspace = sf.color_space  # "BGR" or "RGB" — pass through so colors are right
            if overlay is not None:
                _frame_checks.clear()

            # 1) face detection (once) -> landmarks + bbox + blendshapes.
            fr = detect_face(
                img, detector=face_landmarker, input_color_space=cspace)
            ok = fr.ok
            msg = fr.message
            landmarks = fr.landmarks_px          # pixel-space, get_lm-compatible
            bbox = fr.bbox
            blendshapes = fr.blendshapes
            norm_landmarks = fr.landmarks_norm   # raw normalized, for head pose
            if not ok:
                no_faces = msg.startswith("No faces")
                multiple_faces = msg.startswith("Multiple faces detected")

                if no_faces:
                    add("check_face_detected", "SKIP",
                        f"frame={sf.frame_index} {msg}", sf.frame_index)
                    gap_row_positions[len(timeline)] = len(rows) - 1
                else:
                    add("check_face_detected", "FAIL",
                        f"frame={sf.frame_index} {msg}", sf.frame_index)
                    
                add_frame(sf, face_detected=False)
                if overlay is not None:
                    overlay.add_frame(
                        img, cspace, sf.frame_index, sf.timestamp_sec,
                        face_detected=False, landmarks=None, bbox=None,
                        pose=None, label="no-face", checks=dict(_frame_checks))

                if multiple_faces:
                    sequence_blocked_by_structural_fail = True

                    if fail_fast:
                        aborted = True
                        break
                continue
            add("check_face_detected", "PASS",
                f"frame={sf.frame_index} face detected", sf.frame_index)

            # 2) head pose FIRST (moved up) — the timeline entry it produces is
            # what classifies this frame as front / turn / mid, and that label
            # decides which quality checks are valid to run below.
            ok, info = estimate_head_pose(
                landmarks=norm_landmarks, input_color_space=cspace)
            pose_ok = ok  # preserve: `ok` is reused by later checks below
            # Face was detected this frame (we got past get_lm), so face_detected
            # is True regardless of whether the pose estimate itself succeeded.
            # If pose failed, info stays None -> yaw/pitch/roll recorded as None.
            add_frame(sf, face_detected=True, info=info if ok else None, bbox=bbox)

            # 3) classify this frame from the entry just appended.
            label = classify_frame(timeline[-1], turn_th)

            # 4) checks valid on EVERY detected frame, frontal or not:
            #    full head-to-neck must be visible throughout the video.
            ok, msg = check_head_fully(landmarks, img.shape[0], img.shape[1], margin_px=hf_margin)
            add("check_head_fully", _bool_to_status(ok),
                f"frame={sf.frame_index} {msg}", sf.frame_index)

            # 5) frontal-only quality checks. On non-frontal frames these are
            #    measured in an invalid domain (see FRONT_ONLY_CHECKS note), so
            #    emit SKIP — the frame stays accounted for in the report, but a
            #    turning frame can no longer fail a quality check.
            if label != FRONT:
                for name in FRONT_ONLY_CHECKS:
                    add(name, "SKIP",
                        f"frame={sf.frame_index} non-frontal (label={label}); not judged",
                        sf.frame_index)
                if overlay is not None:
                    overlay.add_frame(
                        img, cspace, sf.frame_index, sf.timestamp_sec,
                        face_detected=True, landmarks=landmarks, bbox=bbox,
                        pose=(info if pose_ok else None), label=label,
                        checks=dict(_frame_checks))
                continue

            # face size (consumes bbox, no re-detection)
            ok, msg = check_face_min_size(bbox, min_width=min_w, min_height=min_h)
            add("check_face_size", _bool_to_status(ok),
                f"frame={sf.frame_index} {msg}", sf.frame_index)

            # occlusion (consumes the same landmarks; no re-detection, no model).
            # Builds a per-region ROI and tests skin presence in YCrCb — a
            # required region (eyes/nose/mouth) without skin is flagged occluded.
            ok, msg, occ_ratios = check_occlusion(
                img, landmarks, input_color_space=cspace,
                required_regions=occ_required,
                cr_bounds=occ_cr, cb_bounds=occ_cb,
                min_skin_ratio=occ_min_skin, roi_margin=occ_roi_margin,
                return_ratios=True)
            # Store each region's ratio on the timeline for the dashboard. Keys
            # mirror the add_frame slots ("occ_<region>"); a None ratio stays
            # None so the dashboard breaks that region's line at this frame.
            for _rg, _val in occ_ratios.items():
                timeline[-1][f"occ_{_rg}"] = _val
            add("check_occlusion", _bool_to_status(ok),
                f"frame={sf.frame_index} {msg}", sf.frame_index)

            # eyes open (consumes landmarks, no re-detection)
            # eyes open (consumes blendshapes from the same detection; no model)
            ok, msg, blink_left, blink_right = check_eye_status(
                blendshapes, blink_th, return_scores=True)
            timeline[-1]["blink_left"] = blink_left
            timeline[-1]["blink_right"] = blink_right
            add("check_eyes_open", _bool_to_status(ok),
                f"frame={sf.frame_index} {msg}", sf.frame_index)

            # brightness first. If exposure is bad, blur is not a reliable
            # independent judgment; do not double-fail the same root cause.
            bright_ok, bright_msg = check_brightness_face(
                img, dark_th, bright_th, margin,
                detector=face_det, input_color_space=cspace)
            timeline[-1]["brightness"] = _extract_float(_BRIGHTNESS_RE, bright_msg)
            add("check_brightness", _bool_to_status(bright_ok),
                f"frame={sf.frame_index} {bright_msg}", sf.frame_index)

            if not bright_ok:
                add("check_face_blur", "SKIP",
                    f"frame={sf.frame_index} brightness failed; blur not judged: {bright_msg}",
                    sf.frame_index)
            else:
                # Reuse the bbox from FaceLandmarker. This avoids a second face
                # detection whose crop may disagree with the rest of the frame checks.
                blur_ok, blur_msg = check_face_blur(
                    img,
                    blur_th,
                    bbox=bbox,
                    input_color_space=cspace,
                    resize_px=blur_resize_px,
                    crop_margin=blur_crop_margin,
                    min_luma=blur_min_luma,
                    min_contrast=blur_min_contrast,
                    preprocess=blur_preprocess,
                    clahe_clip_limit=blur_clahe_clip_limit,
                    clahe_tile_grid_size=blur_clahe_tile_grid_size,
                    lide_d=blur_lide_d,
                    lide_sigma_min=blur_lide_sigma_min,
                )
                blur_status = "SKIP" if blur_ok is None else _bool_to_status(blur_ok)
                timeline[-1]["sharpness"] = _extract_float(_SHARPNESS_RE, blur_msg)
                add("check_face_blur", blur_status,
                    f"frame={sf.frame_index} {blur_msg}", sf.frame_index)

            if overlay is not None:
                overlay.add_frame(
                    img, cspace, sf.frame_index, sf.timestamp_sec,
                    face_detected=True, landmarks=landmarks, bbox=bbox,
                    pose=(info if pose_ok else None), label=label,
                    checks=dict(_frame_checks))
    finally:
        # Always release the models, even if a frame raises mid-loop.
        face_landmarker.close()
        face_det.close()

    # ---- fail-fast GATE 2: zero frames ----
    # The file opened (meta.readable was True) but the decoder yielded NO
    # frames — a corrupt / empty / fully-dropped stream. Previously this line
    # was an unconditional PASS, so such a file slipped through with no FAIL
    # anywhere: every frame-level check aggregated over an empty group to SKIP
    # and the verdict was a silent pass. A delivered video with no readable
    # frames is non-conforming, so emit a FAIL and stop. This holds regardless
    # of fail_fast: zero frames is never a valid sample.
    if frames_seen == 0:
        add("frames_sampled", "FAIL",
            "video opened but yielded 0 frames (corrupt/empty stream)",
            level="video")
        return rows, timeline
    add("frames_sampled", "PASS", f"sampled {frames_seen} frames", level="video")

    # If a fail-fast gate broke the loop early, the timeline is intentionally
    # incomplete and the file already carries a FAIL. The gap split and turn
    # sequence both reason over the WHOLE timeline, so running them on a
    # truncated one would be meaningless (and could emit a misleading verdict).
    # Skip them; the structural FAIL already decides the file.
    if aborted or sequence_blocked_by_structural_fail:
        return rows, timeline

    # ---- expected/unexpected gap split for check_face_detected ----
    # Re-status the no-face rows now that the whole timeline exists:
    #   gap inside an absorbed turn -> expected   (default SKIP, not judged)
    #   gap anywhere else           -> unexpected (default FAIL, counts in ratio)
    # Statuses come from config face.checks.detection. Only meaningful for a
    # turn-protocol video, so it is gated on the same flag as the turn check;
    # with the turn check disabled, no-face rows keep the legacy REVIEW.
    turn_enabled = face_cfg.get("turn_sequence", {}).get("enabled", True)
    if turn_enabled and gap_row_positions:
        apply_gap_split_to_detection_rows(
            rows, timeline, gap_row_positions, config)

    # ---- turn-sequence check (consumes the whole timeline at once) ----
    # Runs only if enabled in config for this stream. Uses the SAME sample_fps
    # the timeline was built at, so hold-duration -> frame-count is correct.
    if turn_enabled:
        def emit_turn_row(check_name, status, reason, frame_index=None):
            return CheckRow(volunteer_id, DATA_TYPE, filename,
                            check_name, status, reason, frame_index,
                            level="sequence")
        turn_rows = check_turn_sequence_seg(
            timeline, config, sample_fps=effective_fps, emit_row=emit_turn_row)
        rows.extend(turn_rows)

    return rows, timeline