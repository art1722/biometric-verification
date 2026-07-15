"""live.py — run QC on ONE uploaded file, dispatched by modality.

POST /checks/file accepts any single file. router.route() parses the name and decides:
  - face_rgb            -> run_face_rgb, return the verdict          (runnable)
  - walk_F / walk_S     -> run_walk, return the verdict             (runnable)
  - palm_* /            -> raise NotImplementedModality              (501)
    other face streams
  - unrecognised name   -> raise UnrecognisedFile                    (422)

face_rgb and walk each grade a single file end-to-end, so both run here. Palm is
DELIBERATELY 501 on this endpoint: a palm image's angle verdict is only
meaningful across all five poses (run_palm_participant), which one image cannot
provide — palm is batch-only (/checks/batch, /checks/uploads). To add another
single-file modality, branch in check_one() and list its key in
router.RUNNABLE_MODALITIES.
"""

from __future__ import annotations

import os
import tempfile

from . import router


class UnrecognisedFile(ValueError):
    """Filename matches no pattern in config (-> 422)."""


class NotImplementedModality(Exception):
    """Filename is valid for a modality with no pipeline yet (-> 501)."""
    def __init__(self, data_key: str):
        self.data_key = data_key
        super().__init__(f"no QC pipeline for modality '{data_key}' yet")


def _run_face_rgb(tmp_path: str, volunteer_id: str, config: dict) -> dict:
    from qc.pipelines.face_rgb import run_face_rgb
    from qc.utils.report import build_overall_record, summarize_rows_by_check

    rows, timeline = run_face_rgb(
        tmp_path, volunteer_id, config,
        sample_fps=1.0, overlay=None, progress=None, fail_fast=False,
    )
    overall = build_overall_record(rows, timeline, config=config)
    agg = (config or {}).get("report", {}).get("aggregation", {})
    checks = summarize_rows_by_check(rows, agg)
    return {
        "overall_status": overall["final_status"],
        "failed_checks": overall["failed_checks"],
        "checks": checks,
    }


def _run_walk(tmp_path: str, volunteer_id: str, view: str,
              config: dict) -> dict:
    """Grade ONE walk video (_F or _S) and return the same verdict shape as
    _run_face_rgb. run_walk grades a single clip end-to-end, so unlike palm
    there is no cross-file dependency — a lone walk file is a complete unit.

    fail_fast=False so every frame is processed and the verdict reflects the
    whole clip (mirrors the face live path). overlay=False: /checks/file is a
    synchronous verdict call, not a review run, so we skip writing a debug .mp4.
    view ("F"/"S") comes from the data_key so the direction check can branch.
    """
    from qc.pipelines.walk import run_walk
    from qc.utils.report import build_overall_record, summarize_rows_by_check

    rows, timeline = run_walk(
        tmp_path, volunteer_id, config,
        view=view, overlay=False, progress=None, fail_fast=False,
    )
    overall = build_overall_record(rows, timeline, config=config)
    agg = (config or {}).get("report", {}).get("aggregation", {})
    checks = summarize_rows_by_check(rows, agg)
    return {
        "overall_status": overall["final_status"],
        "failed_checks": overall["failed_checks"],
        "checks": checks,
    }


def check_one(filename: str, raw_bytes: bytes, config: dict,
              config_path: str) -> dict:
    """Dispatch one uploaded file to the right pipeline and return a verdict.

    Raises UnrecognisedFile (422) or NotImplementedModality (501) before doing
    any heavy work, so bad/unsupported uploads are cheap to reject.
    """
    decision = router.route(filename, config_path)
    outcome = decision["outcome"]

    if outcome == router.UNRECOGNISED:
        raise UnrecognisedFile(
            f"filename '{filename}' matches no known modality pattern"
        )
    if outcome == router.MATCH_NOT_IMPLEMENTED:
        raise NotImplementedModality(decision["data_key"])

    # outcome == runnable
    data_key = decision["data_key"]
    volunteer_id = decision["volunteer_id"]

    tmp_dir = tempfile.mkdtemp(prefix="qc_live_")
    tmp_path = os.path.join(tmp_dir, os.path.basename(filename))
    try:
        with open(tmp_path, "wb") as f:
            f.write(raw_bytes)

        if data_key == "face_rgb":
            result = _run_face_rgb(tmp_path, volunteer_id, config)
        elif data_key in ("walk_F", "walk_S"):
            # data_key is "walk_F"/"walk_S"; the suffix IS the view.
            view = data_key.split("_", 1)[1]  # "F" or "S"
            result = _run_walk(tmp_path, volunteer_id, view, config)
        else:
            # Recognised modality with no single-file pipeline (palm) -> 501.
            raise NotImplementedModality(data_key)

        return {
            "volunteer_id": volunteer_id,
            "data_key": data_key,
            "filename": os.path.basename(filename),
            **result,
        }
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass