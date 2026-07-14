"""Gait (walk) pipeline — MVP slice.

Scope of THIS version
---------------------
The full gait spec (§4) has many requirements: metadata (fps/res/duration),
full-body visibility every frame, walk-direction (F: toward+away, S: left+right),
brightness, plus SKIP-only environmental items. This MVP implements EXACTLY ONE
graded check so the pipeline + runner + wiring exist end-to-end and can be grown
one check at a time:

    check_body_height  (spec: person height >= 1/2 image height)

Everything else is deliberately omitted for now (not stubbed as SKIP rows yet) so
the MVP output is a single, easy-to-verify row per video.

Where the height is measured (researcher decision, 2026-07-09)
--------------------------------------------------------------
On the FIRST frame only -- the point where the walker is farthest from the
camera and therefore appears smallest. If they clear the half-frame bar at their
smallest, they clear it throughout. This runs on BOTH camera videos (_F and _S)
independently; each is one call to run_walk with its own `view`.

Shape / conventions
-------------------
Mirrors run_palm (qc/pipelines/palm.py):
  - same add() helper building CheckRow, same (rows, timeline) return so the
    runner and report code treat gait exactly like face/palm,
  - DATA_TYPE carries the view ("walk_F"/"walk_S") so the summary row's
    data_type column distinguishes the two cameras (the router/config key the
    file already maps to). level="video" because a walk file IS a video
    container (unlike palm's "image").

The pose detector is built ONCE by the caller-facing run_walk and consumed by
the check -- the single-detect discipline the face/palm pipelines established.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from qc.utils.video import probe_video, iter_sampled_frames
from qc.checks.pose_landmarker import create_pose_landmarker, detect_pose
from qc.checks.check_body_height import check_body_height
from qc.checks.check_brightness import check_brightness_walk
from qc.checks.check_person_fully import check_person_fully
from qc.checks.check_face_blur import check_face_blur  # reused for person blur
from qc.checks.check_walk_direction import check_walk_direction
from qc.checks.yolo_detector import (
    create_yolo_detector, detect_objects, close_yolo_detector, PERSON_CLASS_ID,
)
from qc.checks.check_occlusion_yolo import check_occlusion_yolo, _boxes_overlap
from qc.checks import check_metadata as md
from qc.schemas import CheckRow

logger = logging.getLogger(__name__)


def _bool_to_status(ok: bool, *, fail: str = "FAIL") -> str:
    """PASS/FAIL from a check's success bool (mirror face_rgb._bool_to_status)."""
    return "PASS" if ok else fail


def run_walk(
    path: str,
    volunteer_id: str,
    config: dict,
    *,
    view: Optional[str] = None,
    detector: Any = None,
    yolo_detector: Any = None,
    progress=None,
    overlay: bool = True,
    out_root: str = "reports",
    sample_fps: Optional[float] = 5.0,
    fail_fast: bool = True,
):
    """Run the MVP gait pipeline (first-frame body-height only) on one walk video.

    Args:
        path: path to NNN_walk_F.mp4 or NNN_walk_S.mp4.
        volunteer_id: e.g. "001".
        config: the loaded config.yml dict (thresholds live under `walk:` and
            `models.pose_landmarker`).
        view: "F" or "S" parsed from the filename by the runner. Used only to
            label the data_type ("walk_F"/"walk_S") for now; later direction
            checks will branch on it. Falls back to parsing the filename if not
            given.
        detector: a PoseLandmarker from create_pose_landmarker. If None, this
            function builds one from config (convenient for a single-file run;
            a batch runner should build ONE and pass it in to avoid reloading
            the .task bundle per video).
        progress: optional callback(row) invoked as each CheckRow is emitted.
        overlay: write a per-frame debug .mp4 (skeleton + bbox on the frame,
            the height verdict in a strip below). ON by default; the runner's
            --no-overlay flag flips it off. The overlay is a REVIEW AID only:
            it iterates EVERY frame for drawing, but the graded verdict still
            comes from the first frame exactly as before -- grading is not
            changed, only a debug video is added.
        out_root: folder the overlay .mp4 is written under (default "reports").
            Ignored when overlay=False.
        sample_fps: frames sampled per source second FOR THE OVERLAY only.
            Default 5. None (or <=0) means every frame (native). The overlay is
            written at THIS SAME rate so the output video's duration matches the
            source exactly (N sampled frames played at sample_fps = original
            length), mirroring run_face's overlay_fps=sample_fps rule. Grading
            is unaffected -- the verdict always reads the first frame.

    Returns:
        (rows, timeline). timeline is [] in this MVP (no per-frame series yet).
    """
    filename = os.path.basename(path)
    view = view or _view_from_filename(filename)
    data_type = f"walk_{view}" if view else "walk"

    rows: list[CheckRow] = []

    def add(check_name, status, reason, frame_index=None, level="video"):
        row = CheckRow(
            volunteer_id,
            data_type,
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

    # ---- thresholds from config (spec is source of truth) ----
    walk_cfg = config.get("walk", {})
    vis_cfg = walk_cfg.get("visibility", {})
    min_ratio = vis_cfg.get("min_person_height_ratio", 0.5)

    bri_cfg = walk_cfg.get("brightness", {})
    bri_dark = bri_cfg.get("dark_threshold", 35.0)
    bri_bright = bri_cfg.get("bright_threshold", 200.0)
    bri_margin = bri_cfg.get("margin", 0.1)

    meta_cfg = walk_cfg.get("metadata", {})
    min_fps = meta_cfg.get("min_fps", 30)
    min_dur = meta_cfg.get("min_duration_sec", 15)
    min_w = meta_cfg.get("min_width_px", 1920)
    min_h = meta_cfg.get("min_height_px", 1080)

    blur_cfg = walk_cfg.get("blur", {})
    blur_threshold = blur_cfg.get("threshold", 35.0)

    occ_cfg = walk_cfg.get("occlusion", {})
    occ_enabled = occ_cfg.get("enabled", True)
    occ_conf = occ_cfg.get("conf", 0.35)
    occ_overlay_draw = occ_cfg.get("overlay_draw", "all")  # "all" | "overlap"

    dir_cfg = walk_cfg.get("direction", {})

    # Per-frame series (brightness per sampled frame) for the detail_header CSV
    # and the dashboard, mirroring the face pipeline's timeline.
    timeline: list[dict] = []

    models_cfg = config.get("models", {})
    pose_cfg = models_cfg.get("pose_landmarker", {})
    model_path = pose_cfg.get("model_path", "models/pose_landmarker.task")

    # ---- build the detector once if the caller did not supply one ----
    own_detector = False
    if detector is None:
        detector = create_pose_landmarker(
            model_path,
            min_pose_detection_confidence=pose_cfg.get(
                "min_pose_detection_confidence", 0.5),
            min_pose_presence_confidence=pose_cfg.get(
                "min_pose_presence_confidence", 0.5),
            min_tracking_confidence=pose_cfg.get(
                "min_tracking_confidence", 0.5),
        )
        own_detector = True

    # ---- build the YOLO (COCO) detector once, for the occlusion check ----
    # Separate model from the pose detector, so it is built and closed on its own
    # handle, mirroring the pose lifecycle. If occlusion is disabled in config, or
    # ultralytics is not installed, we do NOT crash the run: occlusion rows become
    # SKIP (see _run_frame_pass), so the rest of the walk QC still completes.
    own_yolo = False
    occ_active = bool(occ_enabled)
    if occ_active and yolo_detector is None:
        try:
            yolo_detector = create_yolo_detector(config)
            own_yolo = True
        except ImportError as e:
            logger.warning("occlusion check disabled: %s", e)
            occ_active = False

    def _cleanup():
        # Close BOTH detectors we own, in one place, so every return path frees
        # the pose model and (if we built it) the YOLO model. A caller-injected
        # detector is left open (own_* is False) since the caller still needs it.
        _maybe_close(detector, own_detector)
        if own_yolo:
            close_yolo_detector(yolo_detector)

    # ---- probe metadata (once, no pixels beyond the first frame) ----
    # decode_first_frame=True so probe_video populates channel_count and is_color
    # from a decoded frame -- check_container needs them to verify RGB. (The old
    # walk MVP used decode_first_frame=False, which left channel_count=None and
    # made check_container always FAIL "channel count unknown". Face uses the
    # default True for exactly this reason.)
    meta = probe_video(path, decode_first_frame=True)

    # ---- video-level metadata checks (mirror face_rgb; shared functions) ----
    # These describe the FILE, not any frame: container (.mp4 + RGB), fps>=30,
    # duration>=15s, resolution>=1920x1080 (walk spec, from config). They run on
    # the same VideoMetadata probe_video returns. Emitted as level="video" rows
    # (frame_index empty) exactly like the face pipeline, so they flow through
    # report.py into detail/result/overall/summary with no report changes.
    #
    # An unreadable file cannot be metadata-checked at all -> all four SKIP with
    # the reason, and (below) the height/brightness/overlay work is skipped too.
    # A metadata FAIL on a readable file (e.g. fps<30) is RECORDED but does NOT
    # stop the pipeline: walk writes an overlay by default, and face's own gate
    # is disabled whenever an overlay is produced (fail_fast=False), so the
    # faithful mirror is record-and-continue -> the reviewer still gets a full
    # overlay + per-frame rows on an off-spec clip.
    if not meta.readable:
        for cname in ("check_container", "check_fps",
                      "check_duration", "check_resolution"):
            add(cname, "SKIP", f"video not readable: {meta.reason}",
                level="video")
        add("check_body_height", "SKIP",
            f"video not readable: {meta.reason}", level="video")
        _cleanup()
        return rows, []

    status, reason = md.check_container(meta, require_rgb=True)
    add("check_container", status, reason, level="video")

    status, reason = md.check_fps(meta, min_fps=min_fps)
    add("check_fps", status, reason, level="video")

    status, reason = md.check_duration(meta, min_duration_sec=min_dur)
    add("check_duration", status, reason, level="video")

    status, reason = md.check_resolution(meta, min_width=min_w, min_height=min_h)
    add("check_resolution", status, reason, level="video")

    # ---- fail-fast GATE 1: video-level metadata (mirror face_rgb GATE 1) ----
    # The four checks above describe the FILE, not any frame. If any FAILs the
    # file is structurally non-conforming and no frame analysis can change the
    # verdict, so stop before the (expensive, 1,500-scale) frame pass. Skipped
    # when fail_fast=False (overlay/dashboard runs) so an off-spec file still
    # gets a full per-frame timeline + overlay.
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
            add("check_body_height", "SKIP",
                f"fail-fast: {gate_fail.check_name} FAILed; height not measured",
                level="video")
            _cleanup()
            return rows, []
    try:
        for sf in iter_sampled_frames(path, sample_fps=None, max_frames=1,
                                      include_first_frame=True,
                                      include_last_frame=False):
            first_frame = sf
            break
    except Exception as e:  # keep the no-crash contract
        add("check_body_height", "SKIP",
            f"could not read first frame: {e}", level="video")
        _cleanup()
        return rows, []

    if first_frame is None:
        add("check_body_height", "SKIP", "no frames decoded", level="video")
        _cleanup()
        return rows, []

    # ---- pose on that one frame, then the height check ----
    # SampledFrame.image is BGR by default (color_space says so); pass that
    # through to detect_pose which converts to RGB internally.
    pose = detect_pose(first_frame.image, detector=detector,
                       input_color_space=first_frame.color_space)

    if not pose.ok:
        # No usable pose on the farthest frame -> cannot measure height. FAIL
        # (not SKIP): a walk clip whose first frame has no detectable full body
        # is a real defect for this check, not an inapplicable one.
        height_status = "FAIL"
        height_reason = f"frame={first_frame.frame_index} no pose: {pose.message}"
    else:
        ok, msg = check_body_height(pose.landmarks_norm, min_ratio=min_ratio)
        height_status = _bool_to_status(ok)
        height_reason = f"frame={first_frame.frame_index} {msg}"
    add("check_body_height", height_status, height_reason,
        frame_index=first_frame.frame_index, level="video")

    # ---- per-frame pass: brightness (+ timeline) and, if enabled, the overlay ----
    # ONE detect_pose per sampled frame feeds BOTH the brightness check and the
    # overlay drawing, so we never detect twice. Brightness emits a frame-level
    # CheckRow per frame; report.py aggregates them by the config fail-ratio
    # (walk.brightness.frame_fail_ratio) into the video-level verdict. The
    # overlay (if on) draws the same frame with the frame's brightness verdict
    # in the strip. Height stays a first-frame video-level row, untouched above.
    aborted, frames_seen = _run_frame_pass(
        path, volunteer_id, filename, data_type, detector, add, timeline,
        overlay=overlay, out_root=out_root, sample_fps=sample_fps, meta=meta,
        bri_dark=bri_dark, bri_bright=bri_bright, bri_margin=bri_margin,
        blur_threshold=blur_threshold,
        height_status=height_status, height_reason=height_reason,
        fail_fast=fail_fast,
        yolo_detector=yolo_detector, occ_active=occ_active, occ_conf=occ_conf,
        occ_overlay_draw=occ_overlay_draw,
    )

    # ---- fail-fast GATE 2: zero frames (mirror face_rgb GATE 2) ----
    # The file opened and metadata passed, but the frame pass yielded nothing
    # (corrupt/empty stream). That is a structural FAIL and there is no timeline
    # to run check_pose over, so record it and stop. Unlike face this runs
    # regardless of fail_fast: zero frames is never a valid sample.
    if frames_seen == 0:
        add("frames_sampled", "FAIL",
            "video opened but yielded 0 frames (corrupt/empty stream)",
            level="video")
        _cleanup()
        return rows, timeline
    add("frames_sampled", "PASS", f"sampled {frames_seen} frames", level="video")

    # If a fail-fast gate broke the loop early, the timeline is intentionally
    # incomplete and the file already carries a FAIL (multiple-persons). The
    # walk-direction check reasons over the WHOLE timeline, so running it on a
    # truncated one would be meaningless. Skip it -- the structural FAIL already
    # decides the file. Mirrors face_rgb skipping turn-sequence on abort.
    if aborted:
        _cleanup()
        return rows, timeline

    # ---- check_pose: sequence-level walk-direction verdict ----
    # Reasons over the whole timeline (F: toward+away via body_scale; S:
    # left+right via centroid_x). One row per video, level="sequence", mirroring
    # face's check_turn_sequence. Reported as "check_pose" per the reviewer.
    view = view or _view_from_filename(filename)
    dir_ok, dir_msg = check_walk_direction(
        timeline, view,
        smooth_window=dir_cfg.get("smooth_window", 5),
        min_run=dir_cfg.get("min_run", 3),
        eps=dir_cfg.get("eps", 0.002),
        min_frames=dir_cfg.get("min_frames", 8),
    )
    add("check_pose", _bool_to_status(dir_ok), dir_msg, level="sequence")

    _cleanup()
    return rows, timeline


def _run_frame_pass(path, volunteer_id, filename, data_type, detector, add,
                    timeline, *, overlay, out_root, sample_fps, meta,
                    bri_dark, bri_bright, bri_margin, blur_threshold,
                    height_status, height_reason, fail_fast=True,
                    yolo_detector=None, occ_active=False, occ_conf=0.35,
                    occ_overlay_draw="all"):
    """One pass over the sampled frames: pose once per frame, feeding BOTH the
    per-frame brightness check and (if enabled) the overlay video.

    Returns (aborted, frames_seen):
      aborted     True if a fail-fast structural gate (multiple persons in a
                  frame) broke the loop early. The caller then skips the
                  sequence-level check_pose (the timeline is truncated), exactly
                  as face_rgb skips its turn-sequence check on an aborted file.
      frames_seen number of frames actually processed (for the 0-frame gate).

    Emits one frame-level `check_brightness` CheckRow per sampled frame (which
    report.py aggregates by walk.brightness.frame_fail_ratio into the video
    verdict) and appends a timeline entry per frame (brightness value + body
    box dims) for the detail_header CSV / dashboard.

    Sampling / duration (overlay): frames are sampled at `sample_fps` (default
    5; None/<=0 = every frame) and the overlay is WRITTEN at that same rate, so
    its duration matches the source (run_face's overlay_fps=sample_fps rule).

    The overlay strip shows BOTH the video-level height verdict (constant across
    frames) and THIS frame's brightness verdict, so the debug video and the CSV
    rows always agree. Height is NOT re-graded here.

    Overlay writing is best-effort: an overlay failure is logged and swallowed so
    it never breaks the grade. The brightness rows are emitted regardless.
    """
    native_fps = meta.fps if getattr(meta, "fps", None) else 30.0
    if sample_fps is not None and sample_fps > 0:
        overlay_fps = float(sample_fps)
        iter_fps = float(sample_fps)
    else:
        overlay_fps = native_fps
        iter_fps = None

    writer = None
    if overlay:
        try:
            from qc.utils.walk_overlay import WalkOverlayWriter
            # Output name: walk_<id>_<view>_overlay.mp4 (e.g. walk_002_F_overlay.mp4),
            # NOT <stem>_overlay. View parsed from the source filename; falls back
            # to the stem if the pattern is unexpected so we never crash on naming.
            view = _view_from_filename(filename)
            if volunteer_id and view:
                base = f"walk_{volunteer_id}_{view}_overlay.mp4"
            else:
                base = f"{os.path.splitext(filename)[0]}_overlay.mp4"
            os.makedirs(out_root, exist_ok=True)
            out_path = os.path.join(out_root, base)
            # ONE canvas width for the whole clip, decided up front from the
            # video's own width (meta), clamped to the overlay's band. Passing
            # it in means every frame is composed at the same width -- a walk
            # video's frames are all one size, but locking it here also guards
            # against any per-frame drift and satisfies the constant-size the
            # VideoWriter needs. None -> the writer falls back to the first
            # frame's width.
            canvas_width = getattr(meta, "width", None)
            writer = WalkOverlayWriter(out_path, fps=overlay_fps,
                                       volunteer_id=volunteer_id,
                                       filename=filename,
                                       canvas_width=canvas_width)
        except Exception as e:  # pragma: no cover - overlay is best-effort
            logger.warning("walk overlay writer init failed for %s: %s",
                           filename, e)
            writer = None

    aborted = False
    frames_seen = 0
    try:
        for sf in iter_sampled_frames(path, sample_fps=iter_fps,
                                      include_first_frame=True,
                                      include_last_frame=True,
                                      max_frames=None):
            frames_seen += 1
            p = detect_pose(sf.image, detector=detector,
                            input_color_space=sf.color_space)
            frame_h, frame_w = sf.image.shape[:2]

            # detect_pose returns ok=False for BOTH "No poses detected" and
            # "Multiple poses detected: N". These are DIFFERENT cases (mirror
            # face_rgb, which splits "No faces" from "Multiple faces detected"):
            #   no person   -> routine per-frame FAIL (person walked off / not
            #                  yet in view); ratio-judged, does NOT break.
            #   multiple    -> STRUCTURAL defect (a second person in frame). The
            #                  gait protocol films ONE walker, so this fails the
            #                  whole video and, under fail_fast, breaks the loop
            #                  (the 100% denominator then makes it ratio-exempt,
            #                  same mechanism as face's multiple-faces gate).
            multiple_persons = (not p.ok) and \
                str(p.message).startswith("Multiple poses detected")

            # Each per-frame check captures its (status, reason) into a local so
            # the SAME verdict feeds both the report row (add) and the overlay
            # strip -- the two can never disagree. Reason strings carry the
            # `frame=N` prefix so the overlay reads identically to the CSV row.

            # ---- check_person_detected (frame-level; mirror check_face_detected) ----
            # Did the pose detector find EXACTLY ONE person this frame? PASS if
            # yes, FAIL otherwise (no person, or multiple). This is the gate the
            # other per-frame person checks depend on.
            if p.ok:
                pd_status = "PASS"
                pd_reason = f"frame={sf.frame_index} person detected"
            else:
                pd_status = "FAIL"
                pd_reason = f"frame={sf.frame_index} no person: {p.message}"
            add("check_person_detected", pd_status, pd_reason,
                frame_index=sf.frame_index, level="frame")

            # ---- check_person_fully (frame-level; 8 key landmarks in frame) ----
            if p.ok:
                pf_ok, pf_msg = check_person_fully(
                    p.landmarks_px, frame_w, frame_h)
                pf_status = _bool_to_status(pf_ok)
                pf_reason = f"frame={sf.frame_index} {pf_msg}"
            else:
                pf_status = "FAIL"
                pf_reason = f"frame={sf.frame_index} no person: {p.message}"
            add("check_person_fully", pf_status, pf_reason,
                frame_index=sf.frame_index, level="frame")

            # ---- check_person_blur (frame-level; REUSES check_face_blur on the
            # body bbox -- the scorer is region-agnostic). None -> SKIP (too
            # dark / too low contrast to judge), same mapping face uses. ----
            if p.ok and p.bbox is not None:
                blur_res, blur_msg = check_face_blur(
                    sf.image, threshold=blur_threshold, bbox=p.bbox,
                    input_color_space=sf.color_space)
                if blur_res is None:
                    blur_status = "SKIP"
                else:
                    blur_status = _bool_to_status(blur_res)
                blur_reason = f"frame={sf.frame_index} {blur_msg}"
            else:
                blur_status = "FAIL"
                blur_reason = f"frame={sf.frame_index} no person: {p.message}"
            add("check_person_blur", blur_status, blur_reason,
                frame_index=sf.frame_index, level="frame")

            # ---- brightness (frame-level, INVERTED verdict inside the check) ----
            if p.ok:
                b_ok, b_msg = check_brightness_walk(
                    sf.image, p.bbox,
                    dark_threshold=bri_dark, bright_threshold=bri_bright,
                    margin=bri_margin, input_color_space=sf.color_space)
                b_status = _bool_to_status(b_ok)
            else:
                # No body this frame -> brightness cannot be measured. FAIL the
                # frame (a walk frame with no detectable body is a real defect),
                # so a clip that loses the person is not silently rewarded.
                b_status, b_msg = "FAIL", f"no pose: {p.message}"

            add("check_brightness", b_status, b_msg,
                frame_index=sf.frame_index, level="frame")

            # ---- check_occlusion (frame-level; COCO YOLO obstruction gate) ----
            # Spec ("ไม่มีสิ่งกีดขวาง"): a foreign object fails the frame only if
            # it OBSTRUCTS the walker, approximated by overlapping the person box.
            # Person box: prefer the MediaPipe pose bbox (p.bbox); if pose failed
            # this frame, fall back to YOLO's own person detection (class 0). If
            # NEITHER finds a person, person_bbox stays None and the check FAILs
            # any foreign object (can't clear it). If YOLO is inactive the row is
            # SKIP so the CSV/overlay still shows the check.
            if occ_active and yolo_detector is not None:
                dets = detect_objects(yolo_detector, sf.image, conf=occ_conf)

                # person box: MediaPipe first, else the most confident YOLO person
                person_bbox = p.bbox if p.ok and p.bbox is not None else None
                if person_bbox is None:
                    yolo_persons = [d for d in dets
                                    if d.cls_id == PERSON_CLASS_ID]
                    if yolo_persons:
                        person_bbox = max(
                            yolo_persons, key=lambda d: d.conf).bbox

                occ_ok, occ_msg = check_occlusion_yolo(
                    dets, conf=occ_conf, person_bbox=person_bbox)
                occ_status = _bool_to_status(occ_ok)
                occ_reason = f"frame={sf.frame_index} {occ_msg}"

                # Build the object boxes to DRAW on the overlay (visualization
                # only; does not affect PASS/FAIL). "all" = every non-person
                # object at/above conf; "overlap" = only those overlapping the
                # person box. Each entry: (label, conf, bbox).
                foreign = [d for d in dets
                           if d.conf >= occ_conf and d.cls_id != PERSON_CLASS_ID]
                if occ_overlay_draw == "overlap" and person_bbox is not None:
                    foreign = [d for d in foreign
                               if _boxes_overlap(d.bbox, person_bbox)]
                object_boxes = [(d.cls_name, d.conf, d.bbox) for d in foreign]
            else:
                occ_status = "SKIP"
                occ_reason = f"frame={sf.frame_index} occlusion detector inactive"
                object_boxes = []
            add("check_occlusion", occ_status, occ_reason,
                frame_index=sf.frame_index, level="frame")

            # ---- timeline entry (brightness + direction signals) ----
            bval = _brightness_value(b_msg)
            bw = p.bbox[2] if p.ok and p.bbox else None
            bh = p.bbox[3] if p.ok and p.bbox else None
            # Direction signals (normalized so they are resolution-independent):
            #   body_scale = bbox height / frame height  (grows toward camera) -> F
            #   centroid_x = bbox center x / frame width  (traverses L<->R)     -> S
            body_scale = (bh / frame_h) if bh else None
            centroid_x = ((p.bbox[0] + p.bbox[2] / 2.0) / frame_w
                          if p.ok and p.bbox else None)
            timeline.append({
                "frame_index": sf.frame_index,
                "timestamp_sec": sf.timestamp_sec,
                "brightness": bval,
                "body_width": bw,
                "body_height_px": bh,
                "body_scale": body_scale,
                "centroid_x": centroid_x,
            })

            # ---- overlay frame (all per-frame verdicts in the strip) ----
            # Pass every check so the strip mirrors the report. check_body_height
            # is the video-level first-frame verdict (constant across frames);
            # the rest are this frame's. The overlay orders them by the shared
            # the walk order (REPORT_CHECK_ORDER_WALK), so the strip reads in the
            # same order as the CSV.
            if writer is not None:
                writer.add_frame(
                    sf.image, sf.color_space, sf.frame_index, sf.timestamp_sec,
                    pose_detected=p.ok,
                    landmarks_px=p.landmarks_px if p.ok else None,
                    bbox=p.bbox if p.ok else None,
                    checks={
                        "check_body_height": (height_status, height_reason),
                        "check_person_detected": (pd_status, pd_reason),
                        "check_person_fully": (pf_status, pf_reason),
                        "check_brightness": (b_status, b_msg),
                        "check_person_blur": (blur_status, blur_reason),
                        "check_occlusion": (occ_status, occ_reason),
                    },
                    object_boxes=object_boxes,
                )

            # ---- fail-fast GATE 3: multiple persons (mirror face_rgb GATE 3) ----
            # A second person is a structural defect, not a ratio-judged
            # transient. The FAIL row + overlay frame for THIS frame are already
            # emitted above; break so we don't spend model time on a doomed file.
            # The caller skips check_pose on an aborted (truncated) timeline. Only
            # under fail_fast: with the overlay on (fail_fast=False) we keep going
            # so the reviewer still gets the full overlay (same tradeoff as face).
            if multiple_persons and fail_fast:
                aborted = True
                break
    except Exception as e:  # pragma: no cover
        logger.warning("walk frame pass failed for %s: %s", filename, e)
    finally:
        if writer is not None:
            writer.close()

    return aborted, frames_seen


def _brightness_value(msg):
    """Pull the integer brightness out of a check_brightness_walk message, or
    None if absent (e.g. a 'no pose' frame). Kept local so the pipeline does not
    depend on the check's regex."""
    import re
    m = re.search(r"brightness=(\d+)", msg)
    return int(m.group(1)) if m else None


def _view_from_filename(filename: str) -> Optional[str]:
    """Pull the F/S view out of NNN_walk_F.mp4 / NNN_walk_S.mp4. None if absent."""
    base = os.path.basename(filename)
    if "_walk_F" in base:
        return "F"
    if "_walk_S" in base:
        return "S"
    return None


def _maybe_close(detector: Any, own: bool) -> None:
    """Close the detector only if run_walk created it (don't close a shared one
    the caller still needs). PoseLandmarker exposes .close(); guard in case a
    future detector type does not."""
    if own and hasattr(detector, "close"):
        try:
            detector.close()
        except Exception:  # pragma: no cover - cleanup best-effort
            pass