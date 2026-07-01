"""store.py — the ONLY place that knows how reports are stored on disk.

Everything the API serves about RESULTS is read here and returned as plain
dicts/lists that FastAPI serialises to JSON. Nothing else in the API opens a
report file directly, so if the storage layout changes, only this file changes.

Paths come from environment variables so the same code runs locally, in a
container, or anywhere else without edits:

    REPORTS_DIR   where run_folder wrote its output   (default "reports")
    DATA_DIR      where the input data lives            (default "data")
    CONFIG_PATH   the config.yml                        (default "config.yml")

Results source of truth: all_summary.json
------------------------------------------
run_folder writes a cross-modal roll-up as BOTH all_summary.csv and
all_summary.json. Per the project meeting, the API's /results endpoints read
all_summary.json, so face + palm (+ future walk) all surface through ONE
modality-agnostic source with no per-modality branching here.

IMPORTANT — this roll-up is FAIL/ERROR ONLY by design: files that PASS every
check are NOT written to all_summary. So /results shows the problem set (the
reviewer's worklist), not every processed file. If PASS files are needed later,
change how all_summary is WRITTEN (qc.utils.report.AllSummaryWriter) — the API
here stays the same.

all_summary.json shape (one object per problem FILE):
    {
      "data_type": "face_rgb" | "palm" | "walk_*",
      "volunteer_id": "001",
      "filename": "001_face_rgb.mp4",
      "overall_status": "FAIL" | "ERROR",
      "failures": [ {"check_name": "...", "reason": "..."}, ... ],  # [] on ERROR
      "error": "..."                                                # "" on FAIL
    }

Per-volunteer detail CSVs (face_<id>_*.csv, palm_<id>_*.csv) still exist on disk
for humans; the API no longer needs them for /results.
"""

from __future__ import annotations

import json
import os
from typing import Optional


def reports_dir() -> str:
    return os.environ.get("REPORTS_DIR", "reports")


def data_dir() -> str:
    return os.environ.get("DATA_DIR", "data")


def config_path() -> str:
    return os.environ.get("CONFIG_PATH", "config.yml")


# --- summary paths (still used by jobs.py for progress counting) ---

def summary_csv_path() -> str:
    return os.path.join(reports_dir(), "face_summary.csv")


def palm_summary_csv_path() -> str:
    return os.path.join(reports_dir(), "palm_summary.csv")


def all_summary_json_path() -> str:
    """The cross-modal FAIL/ERROR roll-up the /results endpoints read."""
    return os.path.join(reports_dir(), "all_summary.json")


# --- the one reader every /results endpoint builds on ---

def _read_all_summary() -> list[dict]:
    """all_summary.json as a list of per-file problem records. Missing/unreadable
    file -> [] (nothing has been flagged yet, or no batch has run)."""
    path = all_summary_json_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


# --- endpoints' data ---

def list_volunteer_ids() -> list[str]:
    """Every volunteer id that has at least one FAIL/ERROR file in the roll-up,
    sorted. (PASS-only volunteers are absent by design — see module docstring.)"""
    ids = {
        (rec.get("volunteer_id") or "").strip()
        for rec in _read_all_summary()
    }
    ids.discard("")
    return sorted(ids)


def read_batch_summary(status: Optional[str] = None) -> list[dict]:
    """The cross-modal problem list as flat per-CHECK rows, so a reviewer can
    scan/pivot it the way the old face_summary.csv allowed.

    One row per failed check (ERROR files get one row carrying the error text),
    across ALL modalities. If `status` is given (e.g. "FAIL"/"ERROR"), only rows
    whose overall_status matches (case-insensitive) are returned.

    Row shape mirrors the old CSV columns so existing consumers keep working:
      data_type, volunteer_id, filename, overall_status, check_name, reason
    """
    want = status.strip().upper() if status else None
    rows: list[dict] = []
    for rec in _read_all_summary():
        st = (rec.get("overall_status") or "").upper()
        if want and st != want:
            continue
        base = {
            "data_type": rec.get("data_type") or "",
            "volunteer_id": rec.get("volunteer_id") or "",
            "filename": rec.get("filename") or "",
            "overall_status": st,
        }
        failures = rec.get("failures") or []
        if st == "ERROR":
            rows.append({**base, "check_name": "", "reason": rec.get("error", "")})
        elif failures:
            for fl in failures:
                rows.append({
                    **base,
                    "check_name": fl.get("check_name", ""),
                    "reason": fl.get("reason", ""),
                })
        else:
            # FAIL with no detail (shouldn't happen) -> one placeholder row.
            rows.append({**base, "check_name": "", "reason": ""})
    return rows


def _volunteer_records(volunteer_id: str) -> list[dict]:
    """All problem FILES for one volunteer, across modalities."""
    vid = str(volunteer_id).strip()
    return [
        rec for rec in _read_all_summary()
        if (rec.get("volunteer_id") or "").strip() == vid
    ]


def read_overall(volunteer_id: str) -> Optional[dict]:
    """One volunteer's overall verdict, aggregated across their problem files.

    Returns None if the volunteer has NO FAIL/ERROR files in the roll-up (either
    they passed everything, or they were never processed — the roll-up cannot
    tell these apart, since PASS files are not recorded). ERROR outranks FAIL.
    """
    recs = _volunteer_records(volunteer_id)
    if not recs:
        return None

    statuses = {(r.get("overall_status") or "").upper() for r in recs}
    final = "ERROR" if "ERROR" in statuses else "FAIL"

    # Which files (and their modalities) tripped, for a quick overview.
    problem_files = sorted(
        {(r.get("data_type") or "", r.get("filename") or "") for r in recs}
    )
    modalities = sorted({dt for dt, _fn in problem_files if dt})

    return {
        "volunteer_id": str(volunteer_id).strip(),
        "final_status": final,
        "modalities": modalities,
        "problem_file_count": len(problem_files),
        "problem_files": [fn for _dt, fn in problem_files],
    }


def read_checks(volunteer_id: str) -> list[dict]:
    """One volunteer's failed checks across ALL modalities, one row per check.

    Shape: data_type, filename, check_name, reason (+ overall_status). ERROR
    files contribute one row with the error text in `reason`.
    """
    out: list[dict] = []
    for rec in _volunteer_records(volunteer_id):
        st = (rec.get("overall_status") or "").upper()
        base = {
            "data_type": rec.get("data_type") or "",
            "filename": rec.get("filename") or "",
            "overall_status": st,
        }
        if st == "ERROR":
            out.append({**base, "check_name": "", "reason": rec.get("error", "")})
            continue
        for fl in (rec.get("failures") or []):
            out.append({
                **base,
                "check_name": fl.get("check_name", ""),
                "reason": fl.get("reason", ""),
            })
    return out