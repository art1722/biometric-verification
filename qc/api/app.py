"""app.py — the FastAPI application: all endpoints in one place.

Groups
------
  Batch trigger (job-based):
    POST /jobs                start a run_folder batch over local data/, async
    GET  /jobs                list jobs
    GET  /jobs/{job_id}       poll one job's status + progress

  Read results (instant, JSON):
    GET  /results             all volunteers; ?status=FAIL for problems only
    GET  /results/{id}        one volunteer: overall + per-check detail
    GET  /volunteers          ids that have reports

  Live single video:
    POST /check               upload one <id>_face_rgb.mp4, run QC, return verdict

  Utility:
    GET  /health              liveness + which dirs are in use

Run (from repo root):  uvicorn main:app --reload   ->  http://localhost:8000/docs
"""

from __future__ import annotations

import functools

from fastapi import FastAPI, HTTPException, UploadFile, File, Query

from . import store, jobs, live

app = FastAPI(
    title="Biometric QC API",
    description="Trigger QC batches, read results, and check single videos.",
    version="0.3.0",
)


@functools.lru_cache(maxsize=1)
def _load_config() -> dict:
    """Load config.yml once (cached). Used by the live single-video check."""
    import yaml

    with open(store.config_path(), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "reports_dir": store.reports_dir(),
        "data_dir": store.data_dir(),
    }


# ---- batch trigger (job-based) ----

@app.post("/jobs")
def create_job():
    """Start a batch QC run over the local data/ folder. Returns a job_id
    immediately; poll GET /jobs/{job_id} for progress."""
    return jobs.start_job()


@app.get("/jobs")
def get_jobs():
    return {"jobs": jobs.list_jobs()}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job '{job_id}'")
    return job


# ---- read results ----

@app.get("/results")
def results(status: str | None = Query(default=None,
            description="filter by overall status, e.g. FAIL")):
    rows = store.read_batch_summary(status=status)
    return {"count": len(rows), "status_filter": status, "results": rows}


@app.get("/results/{volunteer_id}")
def result_for_volunteer(volunteer_id: str):
    overall = store.read_overall(volunteer_id)
    if overall is None:
        raise HTTPException(
            status_code=404,
            detail=f"No QC report for volunteer '{volunteer_id}'",
        )
    return {
        "volunteer_id": volunteer_id,
        "overall": overall,
        "checks": store.read_checks(volunteer_id),
    }


@app.get("/volunteers")
def volunteers():
    ids = store.list_volunteer_ids()
    return {"count": len(ids), "volunteer_ids": ids}


# ---- live single-video check ----

@app.post("/check")
async def check(file: UploadFile = File(...)):
    """Upload one <id>_face_rgb.mp4 and get its QC verdict synchronously."""
    # Validate the filename FIRST — reject a bad upload cheaply, before reading
    # the body or loading config.
    try:
        live.parse_volunteer_id(file.filename)
    except live.BadFilename as e:
        raise HTTPException(status_code=400, detail=str(e))

    raw = await file.read()
    return live.check_one(file.filename, raw, _load_config())