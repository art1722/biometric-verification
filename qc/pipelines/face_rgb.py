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
from qc.checks.get_landmarks import get_lm
from qc.checks.check_face_size import check_face_min_size
from qc.checks.check_head_fully import check_head_fully
from qc.checks.check_face_blur import check_face_blur
from qc.checks.check_light_pollution import check_lightpol
from qc.checks.check_head_pose import estimate_head_pose
from qc.checks.check_eye import check_eye_status
from qc.checks.check_turn_sequence_seg import check_turn_sequence_seg
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
    sample_fps: float = 1.0,
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

    def add(check_name, status, reason, frame_index=None, level="frame"):
        rows.append(CheckRow(volunteer_id, DATA_TYPE, filename,
                             check_name, status, reason, frame_index,
                             level=level))

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
    blur_th = blur_cfg.get("threshold", 90)
    dark_th = bright_cfg.get("dark_threshold", 35)
    bright_th = bright_cfg.get("bright_threshold", 200)
    diff_th = bright_cfg.get("diff_threshold", 20)
    margin = bright_cfg.get("margin", 0.1)
    hf_margin = hf_cfg.get("margin_px", 10)

    eye_cfg = face_cfg.get("checks", {}).get("eyes_open", {})
    ear_th = eye_cfg.get("ear_threshold", 0.37)
    
    # ---- video-level checks (once) ----
    meta = probe_video(path)

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

    def add_frame(sf, *, face_detected, info=None):
        timeline.append({
            "frame_index": sf.frame_index,
            "timestamp_sec": sf.timestamp_sec,
            "face_detected": face_detected,
            "yaw": info["yaw"] if info else None,
            "pitch": info["pitch"] if info else None,
            "roll": info["roll"] if info else None,
        })

    # Create the detectors ONCE and reuse them for every frame and every check
    # that needs them. Building a MediaPipe model is expensive; doing it per
    # call (the previous behavior) meant ~4 model setups per frame. Two shared
    # detectors cover all face checks:
    #   - face_mesh : landmarks/bbox (get_lm) and head pose (estimate_head_pose)
    #   - face_det  : blur and brightness (both use FaceDetection)
    # Settings match what each check created internally, so results are identical.
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,
        min_detection_confidence=0.5,
        refine_landmarks=True,
    )
    face_det = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)

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
        for sf in iter_sampled_frames(path, sample_fps=sample_fps):
            frames_seen += 1
            img = sf.image
            cspace = sf.color_space  # "BGR" or "RGB" — pass through so colors are right

            # 1) face detection (once) -> landmarks + bbox
            ok, msg, landmarks, bbox, _ = get_lm(
                img, detector=face_mesh, input_color_space=cspace)
            if not ok:
                # No usable face this frame. For a turn video, that's expected on
                # some frames (profile turns) — record as REVIEW, not FAIL.
                add("check_face_detected", "REVIEW",
                    f"frame={sf.frame_index} {msg}", sf.frame_index)
                # Gap frame: still record it on the timeline so the turn-sequence
                # check can SEE the gap (face_detected=False, yaw=None) rather than
                # finding the frame simply absent.
                add_frame(sf, face_detected=False)
                continue
            add("check_face_detected", "PASS",
                f"frame={sf.frame_index} face ok", sf.frame_index)

            # 2) head pose FIRST (moved up) — the timeline entry it produces is
            # what classifies this frame as front / turn / mid, and that label
            # decides which quality checks are valid to run below.
            ok, info = estimate_head_pose(
                img, detector=face_mesh, input_color_space=cspace)
            # Face was detected this frame (we got past get_lm), so face_detected
            # is True regardless of whether the pose estimate itself succeeded.
            # If pose failed, info stays None -> yaw/pitch/roll recorded as None.
            add_frame(sf, face_detected=True, info=info if ok else None)

            # 3) classify this frame from the entry just appended.
            label = classify_frame(timeline[-1], turn_th)

            # 4) checks valid on EVERY detected frame, frontal or not:
            #    full head-to-neck must be visible throughout the video.
            ok, msg = check_head_fully(landmarks, img.shape[0], margin_px=hf_margin)
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
                continue

            # face size (consumes bbox, no re-detection)
            ok, msg = check_face_min_size(bbox, min_width=min_w, min_height=min_h)
            add("check_face_size", _bool_to_status(ok),
                f"frame={sf.frame_index} {msg}", sf.frame_index)

            # eyes open (consumes landmarks, no re-detection)
            ok, msg = check_eye_status(landmarks, ear_th)
            add("check_eyes_open", _bool_to_status(ok),
                f"frame={sf.frame_index} {msg}", sf.frame_index)

            # blur (needs image + face-detection model)
            ok, msg = check_face_blur(
                img, blur_th, detector=face_det, input_color_space=cspace)
            add("check_face_blur", _bool_to_status(ok),
                f"frame={sf.frame_index} {msg}", sf.frame_index)

            # brightness
            ok, msg = check_lightpol(img, dark_th, bright_th, margin,
                                     detector=face_det, input_color_space=cspace)
            add("check_brightness", _bool_to_status(ok),
                f"frame={sf.frame_index} {msg}", sf.frame_index)
    finally:
        # Always release the models, even if a frame raises mid-loop.
        face_mesh.close()
        face_det.close()

    add("frames_sampled", "PASS", f"sampled {frames_seen} frames", level="video")

    # ---- turn-sequence check (consumes the whole timeline at once) ----
    # Runs only if enabled in config for this stream. Uses the SAME sample_fps
    # the timeline was built at, so hold-duration -> frame-count is correct.
    if face_cfg.get("turn_sequence", {}).get("enabled", True):
        def emit_turn_row(check_name, status, reason, frame_index=None):
            return CheckRow(volunteer_id, DATA_TYPE, filename,
                            check_name, status, reason, frame_index,
                            level="sequence")
        turn_rows = check_turn_sequence_seg(
            timeline, config, sample_fps=sample_fps, emit_row=emit_turn_row)
        rows.extend(turn_rows)

    return rows, timeline