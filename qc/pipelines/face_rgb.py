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
    - brightness ok?            (check_lightpol)

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
from dataclasses import dataclass, asdict
from typing import Any, Optional

import mediapipe as mp

from qc.utils.video import probe_video, iter_sampled_frames
from qc.checks.face_landmarker import create_face_landmarker, detect_face
from qc.checks.check_face_size import check_face_min_size
from qc.checks.check_head_fully import check_head_fully
from qc.checks.check_face_blur import check_face_blur
from qc.checks.check_light_pollution import check_lightpol
from qc.checks.check_head_pose import estimate_head_pose
from qc.checks.check_eye import check_eye_status
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


def run_face_rgb(
    path: str,
    volunteer_id: str,
    config: dict,
    *,
    sample_fps: float | None = None,
    overlay: "Any | None" = None,
    progress=None,
):
    """Run the full face_rgb pipeline on one video.

    Args:
        path: path to NNN_face_rgb.mp4
        volunteer_id: e.g. "001"
        config: the loaded config.yml dict (thresholds live here).
        sample_fps: how densely to sample frames for the per-frame checks.

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
    
    dark_th = bright_cfg.get("dark_threshold", 35)
    bright_th = bright_cfg.get("bright_threshold", 200)
    diff_th = bright_cfg.get("diff_threshold", 20)
    margin = bright_cfg.get("margin", 0.1)
    hf_margin = hf_cfg.get("margin_px", 10)

    eye_cfg = face_cfg.get("checks", {}).get("eyes_open", {})
    blink_th = eye_cfg.get("blink_threshold", 0.5)
    
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

    # ---- per-frame checks ----
    # ONE timeline entry per sampled frame, gaps included. The turn-sequence
    # check reads this: a gap (face_detected=False, yaw=None) bracketed by
    # front-facing frames is the signature of a deep profile turn whose peak
    # MediaPipe could not measure.
    timeline: list[dict] = []
    frames_seen = 0
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
    # Landmarker params come from config.models.mediapipe (same keys as before);
    # the model bundle path from config.models.face_landmarker.model_path.
    mp_cfg = config.get("models", {}).get("mediapipe", {})
    fl_cfg = config.get("models", {}).get("face_landmarker", {})
    fd_cfg = config.get("models", {}).get("face_detection", {})

    face_landmarker = create_face_landmarker(
        model_path=fl_cfg.get("model_path", "models/face_landmarker.task"),
        num_faces=mp_cfg.get("max_num_faces", 10),
        min_face_detection_confidence=mp_cfg.get("min_detection_confidence", 0.6),
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
                         "check_face_blur", "check_brightness")

    try:
        # Native sampling (sample_fps=None) must keep every frame so the overlay
        # is a true 1:1 copy -> disable the 600-frame cap. Down-sampling keeps it.
        _max_frames = None if sample_fps is None else 600
        for sf in iter_sampled_frames(path, sample_fps=sample_fps,
                                      max_frames=_max_frames):
            frames_seen += 1
            img = sf.image
            cspace = sf.color_space  # "BGR" or "RGB" — pass through so colors are right
            if overlay is not None:
                _frame_checks.clear()

            # 1) face detection (once) -> landmarks + bbox + blendshapes.
            # ONE Tasks-API inference yields everything the frame needs.
            fr = detect_face(
                img, detector=face_landmarker, input_color_space=cspace)
            ok = fr.ok
            msg = fr.message
            landmarks = fr.landmarks_px          # pixel-space, get_lm-compatible
            bbox = fr.bbox
            blendshapes = fr.blendshapes
            norm_landmarks = fr.landmarks_norm   # raw normalized, for head pose
            if not ok:
                # No usable face this frame. Two distinct cases:
                #
                #   "No faces ..."  -> a detection gap. Whether acceptable depends
                #     on WHERE it sits, known only once the whole timeline exists.
                #     Start at SKIP (the expected-gap value) as a safe provisional;
                #     the gap split after the loop re-statuses it to SKIP (expected,
                #     inside a turn) or FAIL (unexpected, elsewhere). REVIEW is no
                #     longer used: with no human review capacity, every row must
                #     resolve to PASS / FAIL / SKIP.
                #
                #   "Multiple faces ..." -> a real defect (a second person in
                #     frame). The spec wants ONE clearly-visible subject, so this
                #     FAILs outright and is NOT tracked for the gap split.
                if msg.startswith("No faces"):
                    add("check_face_detected", "SKIP",
                        f"frame={sf.frame_index} {msg}", sf.frame_index)
                    gap_row_positions[len(timeline)] = len(rows) - 1
                else:
                    add("check_face_detected", "FAIL",
                        f"frame={sf.frame_index} {msg}", sf.frame_index)
                # Gap frame: still record it on the timeline so the turn-sequence
                # check can SEE the gap (face_detected=False, yaw=None) rather than
                # finding the frame simply absent.
                add_frame(sf, face_detected=False)
                if overlay is not None:
                    overlay.add_frame(
                        img, cspace, sf.frame_index, sf.timestamp_sec,
                        face_detected=False, landmarks=None, bbox=None,
                        pose=None, label="no-face", checks=dict(_frame_checks))
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

            # eyes open (consumes landmarks, no re-detection)
            # eyes open (consumes blendshapes from the same detection; no model)
            ok, msg = check_eye_status(blendshapes, blink_th)
            add("check_eyes_open", _bool_to_status(ok),
                f"frame={sf.frame_index} {msg}", sf.frame_index)

            # brightness first. If exposure is bad, blur is not a reliable
            # independent judgment; do not double-fail the same root cause.
            bright_ok, bright_msg = check_lightpol(
                img, dark_th, bright_th, margin,
                detector=face_det, input_color_space=cspace)
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
                )
                blur_status = "SKIP" if blur_ok is None else _bool_to_status(blur_ok)
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

    add("frames_sampled", "PASS", f"sampled {frames_seen} frames", level="video")

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