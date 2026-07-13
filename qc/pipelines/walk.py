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
independently; each is one call to run_gait with its own `view`.

Shape / conventions
-------------------
Mirrors run_palm (qc/pipelines/palm.py):
  - same add() helper building CheckRow, same (rows, timeline) return so the
    runner and report code treat gait exactly like face/palm,
  - DATA_TYPE carries the view ("walk_F"/"walk_S") so the summary row's
    data_type column distinguishes the two cameras (the router/config key the
    file already maps to). level="video" because a walk file IS a video
    container (unlike palm's "image").

The pose detector is built ONCE by the caller-facing run_gait and consumed by
the check -- the single-detect discipline the face/palm pipelines established.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from qc.utils.video import probe_video, iter_sampled_frames
from qc.checks.pose_landmarker import create_pose_landmarker, detect_pose
from qc.checks.check_body_height import check_body_height
from qc.schemas import CheckRow

logger = logging.getLogger(__name__)


def _bool_to_status(ok: bool, *, fail: str = "FAIL") -> str:
    """PASS/FAIL from a check's success bool (mirror face_rgb._bool_to_status)."""
    return "PASS" if ok else fail


def run_gait(
    path: str,
    volunteer_id: str,
    config: dict,
    *,
    view: Optional[str] = None,
    detector: Any = None,
    progress=None,
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

    # ---- grab the FIRST frame (farthest-from-camera point) ----
    # probe first so an unreadable/empty video fails cleanly rather than raising
    # a raw exception out of the frame iterator.
    meta = probe_video(path, decode_first_frame=False)
    if not meta.readable:
        add("check_body_height", "SKIP",
            f"video not readable: {meta.reason}", level="video")
        _maybe_close(detector, own_detector)
        return rows, []

    first_frame = None
    try:
        for sf in iter_sampled_frames(path, sample_fps=None, max_frames=1,
                                      include_first_frame=True,
                                      include_last_frame=False):
            first_frame = sf
            break
    except Exception as e:  # keep the no-crash contract
        add("check_body_height", "SKIP",
            f"could not read first frame: {e}", level="video")
        _maybe_close(detector, own_detector)
        return rows, []

    if first_frame is None:
        add("check_body_height", "SKIP", "no frames decoded", level="video")
        _maybe_close(detector, own_detector)
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
        add("check_body_height", "FAIL",
            f"frame={first_frame.frame_index} no pose: {pose.message}",
            frame_index=first_frame.frame_index, level="video")
        _maybe_close(detector, own_detector)
        return rows, []

    ok, msg = check_body_height(pose.landmarks_norm, min_ratio=min_ratio)
    add("check_body_height", _bool_to_status(ok),
        f"frame={first_frame.frame_index} {msg}",
        frame_index=first_frame.frame_index, level="video")

    _maybe_close(detector, own_detector)
    return rows, []


def _view_from_filename(filename: str) -> Optional[str]:
    """Pull the F/S view out of NNN_walk_F.mp4 / NNN_walk_S.mp4. None if absent."""
    base = os.path.basename(filename)
    if "_walk_F" in base:
        return "F"
    if "_walk_S" in base:
        return "S"
    return None


def _maybe_close(detector: Any, own: bool) -> None:
    """Close the detector only if run_gait created it (don't close a shared one
    the caller still needs). PoseLandmarker exposes .close(); guard in case a
    future detector type does not."""
    if own and hasattr(detector, "close"):
        try:
            detector.close()
        except Exception:  # pragma: no cover - cleanup best-effort
            pass