"""app.py — the FastAPI application: all endpoints in one place.

Groups
------
  Jobs (batch QC over local data/):
    POST /jobs                start a run_folder batch; returns 202 + job_id
    GET  /jobs                list jobs
    GET  /jobs/{job_id}       poll one job's status + progress

  Results (read stored reports as JSON):
    GET  /results             all volunteers; ?status=FAIL for problems only
    GET  /results/{id}        one volunteer: overall + per-check detail
    GET  /volunteers          ids that have reports

  Live single file:
    POST /checks              upload one file; routed by filename to its pipeline

  Utility:
    GET  /health              liveness + which dirs are in use

Run (from repo root):  uvicorn main:app --reload   ->  http://localhost:8000/docs
"""

from __future__ import annotations

import functools

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, status
from fastapi.responses import JSONResponse

from . import store, jobs, live

app = FastAPI(
    title="Biometric QC API",
    description="Trigger QC batches, read results, and check single files.",
    version="0.4.0",
)


@functools.lru_cache(maxsize=1)
def _load_config() -> dict:
    import yaml
    with open(store.config_path(), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@app.get("/health", tags=["Utility"])
def health():
    return {
        "status": "ok",
        "reports_dir": store.reports_dir(),
        "data_dir": store.data_dir(),
    }


# ---- Jobs ----

@app.post("/jobs", status_code=status.HTTP_202_ACCEPTED, tags=["Jobs"])
def create_job(response_obj=None):
    """Start a batch QC run over the local data/ folder. Returns 202 Accepted
    with a job_id immediately; poll GET /jobs/{job_id} for progress."""
    job = jobs.start_job()
    # 202 + a Location header pointing at the new job resource.
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=job,
        headers={"Location": f"/jobs/{job['job_id']}"},
    )


@app.get("/jobs", tags=["Jobs"])
def get_jobs(status_filter: str | None = Query(default=None, alias="status",
             description="filter by status, e.g. running | completed | failed")):
    js = jobs.list_jobs()
    if status_filter:
        want = status_filter.strip().lower()
        js = [j for j in js if (j.get("status") or "").lower() == want]
    return {"jobs": js}


@app.get("/jobs/{job_id}", tags=["Jobs"])
def get_job(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job '{job_id}'")
    return job


# ---- Results ----

@app.get("/results", tags=["Results"])
def results(status_filter: str | None = Query(default=None, alias="status",
            description="filter by overall status, e.g. FAIL")):
    rows = store.read_batch_summary(status=status_filter)
    return {"count": len(rows), "status_filter": status_filter, "results": rows}


@app.get("/results/{volunteer_id}", tags=["Results"])
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


@app.get("/volunteers", tags=["Results"])
def volunteers():
    ids = store.list_volunteer_ids()
    return {"count": len(ids), "volunteer_ids": ids}


# ---- Live single file ----

@app.post("/checks", tags=["Live"])
async def create_check(file: UploadFile = File(...)):
    """Upload one file. Its name is parsed to decide the modality:
      - face_rgb            -> run QC, return the verdict        (200)
      - palm_* / walk_* /   -> not built yet                     (501)
        other face streams
      - unrecognised name   -> rejected                          (422)
    """
    raw = await file.read()
    try:
        return live.check_one(file.filename, raw, _load_config(),
                              store.config_path())
    except live.UnrecognisedFile as e:
        raise HTTPException(status_code=422, detail=str(e))
    except live.NotImplementedModality as e:
        raise HTTPException(status_code=501, detail=str(e))