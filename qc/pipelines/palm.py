"""Palm pipeline — MINIMAL skeleton (container + resolution only).

This is the palm-side mirror of face_rgb.py, deliberately reduced to the two
image-metadata checks that have a direct face analogue:

    Image-level (once, from metadata):
      - check_container   (.jpg, decodable, 3-channel color)
      - check_resolution  (>= 200x200, the palm spec minimum)

Everything else the palm spec requires (hand detected, palm size from bbox,
angle, finger spread, veins, jewelry) is intentionally OUT of this skeleton --
those land in later phases. Keeping this first slice tiny proves tWhe
file/CSV/report plumbing end-to-end before any model runs.

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

    # ---- optional hand detection (only when an overlay is requested) ----
    # The minimal slice is metadata-only, so detection is OFF by default. When
    # `detect=True` (the runner sets this for --overlay), run the shared
    # HandLandmarker once so the overlay has landmarks + bbox to draw. This does
    # NOT yet emit palm check rows (present/size/angle are later phases); it
    # only produces a HandResult for visualization.
    hand_result = None
    if detect:
        try:
            from qc.checks.hand_landmarker import (
                create_hand_landmarker, detect_hand,
            )
            models_cfg = config.get("models", {})
            hl_cfg = models_cfg.get("hand_landmarker", {})
            hands_cfg = models_cfg.get("hands", {})
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
        except FileNotFoundError as e:
            # Model bundle missing — don't crash the metadata run; just skip the
            # overlay's landmark layer. The .jpg will still be written with the
            # bbox/landmarks simply absent.
            logger.warning("PALM | hand model unavailable, overlay skipped: %s", e)
        except Exception as e:
            logger.warning("PALM | hand detection error, overlay degraded: %s", e)

    # timeline is [] -- a still has no per-frame timeline.
    return rows, [], hand_result
