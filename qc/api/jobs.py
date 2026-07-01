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


def _count_face_videos(data_dir: str) -> int:
    """How many *_face_rgb.mp4 files run_folder will process. Mirrors
    run_folder's strict match without importing it (keeps this module
    independent of the CLI's globals)."""
    import re

    rgb = re.compile(r"^(\d+)_face_rgb\.mp4$")
    n = 0
    if os.path.isdir(data_dir):
        for _, _, files in os.walk(data_dir):
            n += sum(1 for name in files if rgb.match(name))
    return n


def _count_palm_images(data_dir: str) -> int:
    """How many strict palm image files run_folder --palm will process. The
    batch grades PER PARTICIPANT but emits one summary row PER IMAGE, so the
    progress UNIT is the image (matches the palm_summary.csv grain). Mirrors
    run_folder.PALM_RE without importing it."""
    import re

    palm = re.compile(r"^(\d+)_palm_[LR]_(N|RL|RR|PU|PD)\.jpg$")
    n = 0
    if os.path.isdir(data_dir):
        for _, _, files in os.walk(data_dir):
            n += sum(1 for name in files if palm.match(name))
    return n


def _count_expected(data_dir: str, want_face: bool, want_palm: bool) -> int:
    """Total UNITS the run will process = face videos (if face is on) + palm
    images (if --palm is set). Both denominators and the 'done' numerator use
    the same unit (one file), so progress stays in [0, 1]."""
    total = 0
    if want_face:
        total += _count_face_videos(data_dir)
    if want_palm:
        total += _count_palm_images(data_dir)
    return total


def _count_distinct_filenames(summary_path: str) -> int:
    """How many DISTINCT files a summary CSV has finished so far.

    The summary CSVs (face_summary.csv / palm_summary.csv) have ONE ROW PER
    FAILED CHECK — a file that fails N checks writes N rows, a PASS file writes
    1. So a raw row count over-counts multi-failure files and can exceed the
    file total. Counting DISTINCT values of the 'filename' column gives one
    unit per finished file, so done never exceeds total.

    Missing file -> 0 (the run hasn't started writing it yet).
    """
    if not os.path.exists(summary_path):
        return 0
    names = set()
    try:
        with open(summary_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("filename") or "").strip()
                if name:
                    names.add(name)
    except OSError:
        return 0
    return len(names)


def _count_done(job: dict) -> int:
    """Distinct finished files across every summary CSV this job writes (face
    and/or palm), so the numerator matches the combined unit denominator."""
    done = 0
    for path in job.get("_summary_paths", []):
        done += _count_distinct_filenames(path)
    return done


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
        job["status"] = "completed" if proc.returncode is not None else "failed"
        # Keep only the tail of the log so a huge run doesn't bloat memory.
        if out:
            job["log_tail"] = "".join(out.splitlines(keepends=True)[-40:])


def start_job(extra_args: Optional[list[str]] = None, *,
              run_palm: bool = False, run_face: bool = True,
              data_dir: Optional[str] = None,
              reports_dir: Optional[str] = None,
              fresh_all: bool = False) -> dict:
    """Launch a batch QC run. Returns the new job's public record immediately.

    run_face / run_palm decide which modalities the batch processes AND which
    summary CSVs the progress counter watches, so 'total' and 'done' agree on
    the unit set. When run_palm is True, '--palm' is added to the subprocess
    args; run_face=False adds '--no-face'.

    data_dir / reports_dir override where the batch READS input and WRITES
    output. Both default to the shared server dirs (store.data_dir() /
    store.reports_dir()), so the normal batch is unchanged. An UPLOAD run passes
    its own temp dirs here so each upload is isolated from the shared data/ and
    from other uploads.

    fresh_all truncates the all_summary roll-up before writing (adds
    --fresh-all) instead of appending. An upload run sets this True so its
    all_summary reflects ONLY that upload — otherwise uploads would pile onto
    the shared cross-modal roll-up. The normal batch leaves it False (append),
    preserving the face-then-palm accumulation behaviour.
    """
    job_id = uuid.uuid4().hex[:12]
    data = data_dir if data_dir is not None else store.data_dir()
    reports = reports_dir if reports_dir is not None else store.reports_dir()
    config = store.config_path()

    args = list(extra_args or [])
    if run_palm and "--palm" not in args:
        args.append("--palm")
    if not run_face and "--no-face" not in args:
        args.append("--no-face")
    if fresh_all and "--fresh-all" not in args:
        args.append("--fresh-all")

    total = _count_expected(data, want_face=run_face, want_palm=run_palm)

    # Which summary CSVs to sum 'done' from — must match the modalities run AND
    # the reports dir this job writes to (an upload writes to its own reports
    # dir, so the progress counter must watch THOSE CSVs, not the shared ones).
    summary_paths = []
    if run_face:
        summary_paths.append(os.path.join(reports, "face_summary.csv"))
    if run_palm:
        summary_paths.append(os.path.join(reports, "palm_summary.csv"))

    proc = _launch(data, reports, config, args)

    record = {
        "job_id": job_id,
        "status": "running",
        "total": total,
        "started_at": time.time(),
        "finished_at": None,
        "return_code": None,
        "_summary_paths": summary_paths,
        # Where this job wrote its output. For an upload run this is the upload's
        # own reports dir; the client reads all_summary from here.
        "reports_dir": reports,
    }
    with _lock:
        _jobs[job_id] = record

    threading.Thread(target=_watch, args=(job_id, proc), daemon=True).start()
    return public_view(record)


def public_view(job: dict) -> dict:
    """The JSON-safe shape returned to clients (drops private '_' fields, adds
    live progress)."""
    done = _count_done(job)
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
        "reports_dir": job.get("reports_dir"),
    }


def get_job(job_id: str) -> Optional[dict]:
    with _lock:
        job = _jobs.get(job_id)
        return public_view(job) if job else None


def list_jobs() -> list[dict]:
    with _lock:
        return [public_view(j) for j in _jobs.values()]


def reports_dir_for(job_id: str) -> Optional[str]:
    """The reports dir a given job wrote to (its own dir for an upload run, the
    shared dir otherwise). None if the job_id is unknown — lets the API 404."""
    with _lock:
        job = _jobs.get(job_id)
        return job.get("reports_dir") if job else None