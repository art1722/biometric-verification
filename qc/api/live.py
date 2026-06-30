"""live.py — run QC on ONE uploaded file, dispatched by modality.

POST /checks accepts any single file. router.route() parses the name and decides:
  - face_rgb            -> run_face_rgb, return the verdict          (runnable)
  - palm_* / walk_* /   -> raise NotImplementedModality              (501)
    other face streams
  - unrecognised name   -> raise UnrecognisedFile                    (422)

Only face_rgb has a pipeline today. When palm/walk pipelines exist, add a branch
in check_one() and list the key in router.RUNNABLE_MODALITIES.
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
        else:
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