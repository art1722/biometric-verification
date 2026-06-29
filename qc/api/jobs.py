"""jobs.py — start a batch QC run as a SUBPROCESS and track its progress.

Design
------
POST /jobs must return immediately, but run_folder.py over ~1,500 volunteers
takes a long time. So we DON'T run it inside the request. Instead:

    1. launch run_folder.py as a separate OS process (subprocess.Popen),
    2. return a job_id straight away,
    3. let the frontend poll GET /jobs/{id} for status + progress.

Why a subprocess (not a thread): QC is CPU-heavy (MediaPipe/OpenCV). A separate
process is scheduled independently by the OS, so the API stays responsive while
the batch runs, and a crash in QC cannot take the API down. It also runs the
EXISTING run_folder.py unchanged — no refactor of tested code.

THE ONE SWAPPABLE PIECE
-----------------------
`_launch()` is the only function that knows HOW the work is started. Today it
runs `python run_folder.py ...` locally. If this ever needs to run on an HPC
scheduler instead, only `_launch()` changes (e.g. build an sbatch command and
return its job id) — the tracking, progress, and endpoints stay identical.

Progress
--------
run_folder.py streams one row per finished video to face_summary.csv and flushes
after each. So progress = (rows currently in that CSV) / (videos discovered).
No need to parse the subprocess's stdout.

State
-----
Jobs are tracked in an in-memory dict. This is fine for one researcher's machine
running a few batches. It is NOT persistent — if the API process restarts, the
job registry is lost (the reports on disk survive). A database/file store would
be the next step if persistence is needed.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import threading
import time
import uuid
from typing import Optional

from . import store

# job_id -> job dict. Guarded by _lock because the poll endpoint reads while the
# launcher writes.
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _count_videos(data_dir: str) -> int:
    """How many *_face_rgb.mp4 files run_folder will process — the denominator
    for the progress bar. Mirrors run_folder's strict match without importing it
    (keeps this module independent of the CLI's globals)."""
    import re

    rgb = re.compile(r"^(\d+)_face_rgb\.mp4$")
    n = 0
    if os.path.isdir(data_dir):
        for _, _, files in os.walk(data_dir):
            n += sum(1 for name in files if rgb.match(name))
    return n


def _count_summary_rows(summary_path: str) -> int:
    """How many videos run_folder has finished so far = data rows in the summary
    CSV (minus the header). Missing file -> 0 (not started writing yet)."""
    if not os.path.exists(summary_path):
        return 0
    try:
        with open(summary_path, "r", encoding="utf-8-sig", newline="") as f:
            return max(0, sum(1 for _ in csv.reader(f)) - 1)
    except OSError:
        return 0


def _launch(data_dir: str, reports_dir: str, config_path: str,
            extra_args: Optional[list[str]] = None) -> subprocess.Popen:
    """Start the batch and return the running process.

    *** This is the ONLY environment-specific function. ***
    Local version: run run_folder.py with the current Python interpreter.
    To target a scheduler later, replace the command built here.
    """
    cmd = [
        sys.executable, "run_folder.py", data_dir,
        "--out-root", reports_dir,
        "--config", config_path,
    ]
    if extra_args:
        cmd += extra_args
    # Line-buffered text so a future stdout reader works; we don't rely on it for
    # progress, but capturing it means a crash's traceback is retrievable.
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _watch(job_id: str, proc: subprocess.Popen) -> None:
    """Background thread: wait for the process to exit, then finalise status.
    (The thread only WAITS — it does no QC work, so the GIL is not a factor.)"""
    out, _ = proc.communicate()  # blocks until the subprocess finishes
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["return_code"] = proc.returncode
        job["finished_at"] = time.time()
        # run_folder exits 0 only if nothing FAILed/ERRORed, non-zero otherwise.
        # A non-zero code here does NOT mean the run crashed — it can mean "ran
        # fine, some videos failed QC". So success = the process actually ran to
        # completion (we reached here), and we record the code for the caller.
        job["status"] = "done" if proc.returncode is not None else "failed"
        # Keep only the tail of the log so a huge run doesn't bloat memory.
        if out:
            job["log_tail"] = "".join(out.splitlines(keepends=True)[-40:])


def start_job(extra_args: Optional[list[str]] = None) -> dict:
    """Launch a batch QC run. Returns the new job's public record immediately."""
    job_id = uuid.uuid4().hex[:12]
    data = store.data_dir()
    reports = store.reports_dir()
    config = store.config_path()
    total = _count_videos(data)

    proc = _launch(data, reports, config, extra_args)

    record = {
        "job_id": job_id,
        "status": "running",
        "total": total,
        "started_at": time.time(),
        "finished_at": None,
        "return_code": None,
        "_summary_path": store.summary_csv_path(),
    }
    with _lock:
        _jobs[job_id] = record

    threading.Thread(target=_watch, args=(job_id, proc), daemon=True).start()
    return public_view(record)


def public_view(job: dict) -> dict:
    """The JSON-safe shape returned to clients (drops private '_' fields, adds
    live progress)."""
    done = _count_summary_rows(job["_summary_path"])
    total = job.get("total", 0)
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "done": done,
        "total": total,
        "progress": round(done / total, 3) if total else None,
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "return_code": job["return_code"],
    }


def get_job(job_id: str) -> Optional[dict]:
    with _lock:
        job = _jobs.get(job_id)
        return public_view(job) if job else None


def list_jobs() -> list[dict]:
    with _lock:
        return [public_view(j) for j in _jobs.values()]