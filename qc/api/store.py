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

IMPORTANT — this roll-up records EVERY processed file, one object per file, with
overall_status PASS or FAIL (media-level verdict is PASS/FAIL only). So /results
returns the full set by default; callers narrow to the problem set with
?status=FAIL (or ?status=PASS). Filtering is an API/frontend concern, done in
read_batch_summary / the endpoints below — the file on disk always holds all
statuses.

A file that CRASHED mid-grading (corrupt media, a library fault, a bug) is
recorded as FAIL too, with the exception text surfaced as a synthetic
"processing_error" check inside failures[] — so a crash reads like any other
failure, no separate status to special-case. (A legacy "ERROR" status existed
for crashes; the read paths below still tolerate it defensively, but nothing
writes it anymore.)

all_summary.json shape (one object per PROCESSED file):
    {
      "data_type": "face_rgb" | "palm" | "walk_*",
      "volunteer_id": "001",
      "filename": "001_face_rgb.mp4",
      "overall_status": "PASS" | "FAIL",
      "failures": [ {"check_name": "...", "reason": "..."}, ... ],  # [] on PASS
      "error": ""   # legacy field, empty now (crash text lives in failures[])
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


def all_summary_json_path(reports: Optional[str] = None) -> str:
    """The cross-modal roll-up the /results endpoints read. Defaults to the
    shared reports dir; pass `reports` to read a specific upload's dir."""
    return os.path.join(reports or reports_dir(), "all_summary.json")


# --- the one reader every /results endpoint builds on ---

def _read_all_summary(reports: Optional[str] = None) -> list[dict]:
    """all_summary.json as a list of per-file records. Missing/unreadable
    file -> [] (nothing processed yet, or no batch has run). `reports` selects
    which reports dir to read (an upload's own dir vs the shared default)."""
    path = all_summary_json_path(reports)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


# --- endpoints' data ---

def list_volunteer_ids(reports: Optional[str] = None) -> list[str]:
    """Every volunteer id present in the roll-up, sorted. Now that all_summary
    records PASS media too, this is every processed volunteer (not just those
    with a problem)."""
    ids = {
        (rec.get("volunteer_id") or "").strip()
        for rec in _read_all_summary(reports)
    }
    ids.discard("")
    return sorted(ids)


def read_batch_summary(status: Optional[str] = None,
                       reports: Optional[str] = None) -> list[dict]:
    """The cross-modal result list as flat per-CHECK rows, so a reviewer can
    scan/pivot it the way the old face_summary.csv allowed.

    One row per failed check; a PASS media contributes one placeholder row
    (blank check_name/reason).
    If `status` is given (e.g. "FAIL"/"PASS"/"SKIP"), only rows whose
    overall_status matches (case-insensitive) are returned — so ?status=FAIL
    yields just the problem set.

    Row shape mirrors the old CSV columns so existing consumers keep working:
      data_type, volunteer_id, filename, overall_status, check_name, reason
    """
    want = status.strip().upper() if status else None
    rows: list[dict] = []
    for rec in _read_all_summary(reports):
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
        if failures:
            for fl in failures:
                rows.append({
                    **base,
                    "check_name": fl.get("check_name", ""),
                    "reason": fl.get("reason", ""),
                })
        else:
            # PASS media (or a FAIL with no detail) -> one placeholder row.
            rows.append({**base, "check_name": "", "reason": ""})
    return rows


def _volunteer_records(volunteer_id: str,
                       reports: Optional[str] = None) -> list[dict]:
    """All result FILES for one volunteer, across modalities (PASS + FAIL)."""
    vid = str(volunteer_id).strip()
    return [
        rec for rec in _read_all_summary(reports)
        if (rec.get("volunteer_id") or "").strip() == vid
    ]


def read_overall(volunteer_id: str,
                 reports: Optional[str] = None) -> Optional[dict]:
    """One volunteer's overall verdict, aggregated across their media files.

    Returns None only if the volunteer is absent from the roll-up entirely (not
    processed). FAIL outranks PASS. Now that PASS media are
    recorded, a volunteer who passed everything returns final_status=PASS
    instead of None.
    """
    recs = _volunteer_records(volunteer_id, reports)
    if not recs:
        return None

    statuses = {(r.get("overall_status") or "").upper() for r in recs}
    if "FAIL" in statuses:
        final = "FAIL"
    else:
        final = "PASS"

    # All this volunteer's files + which ones are problems, for a quick overview.
    all_files = sorted(
        {(r.get("data_type") or "", r.get("filename") or "") for r in recs}
    )
    problem_files = sorted(
        {(r.get("data_type") or "", r.get("filename") or "") for r in recs
         if (r.get("overall_status") or "").upper() == "FAIL"}
    )
    modalities = sorted({dt for dt, _fn in all_files if dt})

    return {
        "volunteer_id": str(volunteer_id).strip(),
        "final_status": final,
        "modalities": modalities,
        "file_count": len(all_files),
        "problem_file_count": len(problem_files),
        "problem_files": [fn for _dt, fn in problem_files],
    }


def read_checks(volunteer_id: str,
                reports: Optional[str] = None) -> list[dict]:
    """One volunteer's checks across ALL modalities, one row per failed check.

    Shape: data_type, filename, check_name, reason (+ overall_status). A PASS
    media contributes one placeholder row (blank check_name/reason). A crashed
    file surfaces its text via the synthetic processing_error check.
    """
    out: list[dict] = []
    for rec in _volunteer_records(volunteer_id, reports):
        st = (rec.get("overall_status") or "").upper()
        base = {
            "data_type": rec.get("data_type") or "",
            "filename": rec.get("filename") or "",
            "overall_status": st,
        }
        failures = rec.get("failures") or []
        if not failures:
            # PASS media -> one clean placeholder row so it's visible in detail.
            out.append({**base, "check_name": "", "reason": ""})
            continue
        for fl in failures:
            out.append({
                **base,
                "check_name": fl.get("check_name", ""),
                "reason": fl.get("reason", ""),
            })
    return out