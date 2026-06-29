"""live.py — run QC on ONE uploaded video and return the verdict synchronously.

This is the 'check a clip right after recording' path. One video is fast (a few
seconds), so unlike the batch it's fine to run inside the request and return the
result directly — no job machinery needed.

Filename policy
---------------
The uploaded file MUST be named exactly <id>_face_rgb.mp4 (lowercase), the same
strict rule run_folder.py and validate_filenames.py enforce. The volunteer_id is
parsed from that name. A wrong name is rejected (the caller gets a 400) rather
than guessed — consistent with the rest of the system, and it means the verdict
is always tied to a real id.
"""

from __future__ import annotations

import os
import re
import tempfile

# Same strict pattern as run_folder.FACE_RGB_RE.
FACE_RGB_RE = re.compile(r"^(?P<vid>\d+)_face_rgb\.mp4$")


class BadFilename(ValueError):
    """Raised when the upload isn't a strict <id>_face_rgb.mp4 name."""


def parse_volunteer_id(filename: str) -> str:
    m = FACE_RGB_RE.match(os.path.basename(filename or ""))
    if not m:
        raise BadFilename(
            f"filename must be '<id>_face_rgb.mp4' (got '{filename}')"
        )
    return m.group("vid")


def check_one(filename: str, raw_bytes: bytes, config: dict) -> dict:
    """Run the face_rgb pipeline on the uploaded bytes and return a JSON-safe
    verdict. Saves to a temp file (run_face_rgb needs a path), runs QC, then
    deletes the temp file regardless of outcome."""
    from qc.pipelines.face_rgb import run_face_rgb
    from qc.utils.report import build_overall_record, summarize_rows_by_check

    vid = parse_volunteer_id(filename)

    tmp_dir = tempfile.mkdtemp(prefix="qc_live_")
    tmp_path = os.path.join(tmp_dir, os.path.basename(filename))
    try:
        with open(tmp_path, "wb") as f:
            f.write(raw_bytes)

        rows, timeline = run_face_rgb(
            tmp_path, vid, config,
            sample_fps=1.0, overlay=None, progress=None, fail_fast=False,
        )

        overall = build_overall_record(rows, timeline, config=config)
        agg = (config or {}).get("report", {}).get("aggregation", {})
        checks = summarize_rows_by_check(rows, agg)

        return {
            "volunteer_id": vid,
            "filename": os.path.basename(filename),
            "overall_status": overall["final_status"],
            "failed_checks": overall["failed_checks"],
            "checks": checks,
        }
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass