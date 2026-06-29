"""store.py — the ONLY place that knows how reports are stored on disk.

Reads the report CSVs that run_folder.py writes and returns plain dicts/lists
that FastAPI serialises to JSON. Nothing else in the API opens a file directly,
so if the storage layout ever changes, only this file changes.

Paths are read from environment variables so the same code runs locally, in a
container, or anywhere else without edits:

    REPORTS_DIR   where run_folder wrote its output   (default "reports")
    DATA_DIR      where the input videos live          (default "data")
    CONFIG_PATH   the config.yml                        (default "config.yml")

On-disk layout produced by run_folder.py
-----------------------------------------
    <REPORTS_DIR>/
        face_summary.csv              # batch: one row per volunteer
        <id>/
            face_<id>_overall.csv     # one overall PASS/FAIL row
            face_<id>_result.csv      # one row per check (summary)
            face_<id>_detail.csv      # one row per individual check result
"""

from __future__ import annotations

import csv
import os
from typing import Optional


def reports_dir() -> str:
    return os.environ.get("REPORTS_DIR", "reports")


def data_dir() -> str:
    return os.environ.get("DATA_DIR", "data")


def config_path() -> str:
    return os.environ.get("CONFIG_PATH", "config.yml")


def summary_csv_path() -> str:
    return os.path.join(reports_dir(), "face_summary.csv")


def _read_csv(path: str) -> list[dict]:
    """Read a CSV into a list of dicts. Missing file -> []. utf-8-sig strips the
    BOM the pipeline writes, so the first column isn't '\\ufeffvolunteer_id'."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def list_volunteer_ids() -> list[str]:
    """Every volunteer id that has a report folder with an overall CSV, sorted."""
    root = reports_dir()
    if not os.path.isdir(root):
        return []
    ids = []
    for name in os.listdir(root):
        folder = os.path.join(root, name)
        if os.path.isdir(folder) and os.path.exists(
            os.path.join(folder, f"face_{name}_overall.csv")
        ):
            ids.append(name)
    return sorted(ids)


def read_batch_summary(status: Optional[str] = None) -> list[dict]:
    """face_summary.csv as a list of dicts. If `status` is given (e.g. "FAIL"),
    return only rows whose overall_status matches (case-insensitive) — this is
    the 'problems only' view a reviewer wants instead of scrolling 30,000 rows."""
    rows = _read_csv(summary_csv_path())
    if status:
        want = status.strip().upper()
        rows = [r for r in rows if (r.get("overall_status") or "").upper() == want]
    return rows


def read_overall(volunteer_id: str) -> Optional[dict]:
    """The single overall PASS/FAIL record for one volunteer, or None."""
    rows = _read_csv(
        os.path.join(reports_dir(), volunteer_id, f"face_{volunteer_id}_overall.csv")
    )
    return rows[0] if rows else None


def read_checks(volunteer_id: str) -> list[dict]:
    """The per-check summary rows (face_<id>_result.csv) for one volunteer."""
    return _read_csv(
        os.path.join(reports_dir(), volunteer_id, f"face_{volunteer_id}_result.csv")
    )