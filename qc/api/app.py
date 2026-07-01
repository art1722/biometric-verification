"""app.py — the FastAPI application: all endpoints in one place.

Endpoint namespace (v0.5.0)
---------------------------
Everything that RUNS a check now lives under a single /checks/* namespace, with
the grain in the path. Reading stored results stays under /results (a separate
concern from running checks).

  Checks (run QC):
    POST /checks/file          one uploaded file, SYNC — returns the verdict (200)
    POST /checks/batch         start a batch over local data/; ASYNC — 202 + job_id
    GET  /checks/batch         list batch runs
    GET  /checks/batch/{id}    poll one batch run's status + progress

  Results (read stored reports as JSON):
    GET  /results              all volunteers; ?status=FAIL for problems only
    GET  /results/{id}         one volunteer: overall + per-check detail
    GET  /volunteers           ids that have reports

  Utility:
    GET  /health               liveness + which dirs are in use

Why the split (sync vs async)
-----------------------------
POST /checks/file runs inline and returns the verdict in the response body — the
caller waits. POST /checks/batch cannot: a run over ~1,500 volunteers is far too
long for one request, so it returns 202 + a job_id immediately and the caller
POLLs GET /checks/batch/{id}. The returned object still carries job_id/status/
progress — it is conceptually a "job", only the URL changed.

Backwards compatibility
-----------------------
The previous paths (POST/GET /jobs, /jobs/{id}, POST /checks) are kept as
DEPRECATED aliases that forward to the same handlers, so an existing client
(e.g. พี่ยอ's frontend) does not break. They can be removed once every caller
has moved to /checks/*.

Run (from repo root):  uvicorn main:app --reload   ->  http://localhost:8000/docs
"""

from __future__ import annotations

import functools

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, status
from fastapi.responses import JSONResponse

from . import store, jobs, live

app = FastAPI(
    title="Biometric QC API",
    description="Run QC (single file or batch), read results, check single files.",
    version="0.5.0",
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


# ---------------------------------------------------------------------------
# Checks — batch (async)
# ---------------------------------------------------------------------------

def _start_batch(run_palm: bool = True, run_face: bool = True):
    """Shared handler: launch a batch run and return 202 + Location.

    Defaults to BOTH modalities. run_palm/run_face select which the batch
    processes. Behavior is otherwise unchanged from the old POST /jobs — it runs
    run_folder over the server's local data/ dir. (Zip upload is a later step;
    this does not change what the batch reads.)
    """
    job = jobs.start_job(run_palm=run_palm, run_face=run_face)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=job,
        headers={"Location": f"/checks/batch/{job['job_id']}"},
    )


def _list_batches(status_filter: str | None):
    js = jobs.list_jobs()
    if status_filter:
        want = status_filter.strip().lower()
        js = [j for j in js if (j.get("status") or "").lower() == want]
    return {"jobs": js}


def _get_batch(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No batch run '{job_id}'")
    return job


@app.post("/checks/batch", status_code=status.HTTP_202_ACCEPTED, tags=["Checks"])
def create_batch(
    run_face: bool = Query(default=True,
                           description="run the face video batch "
                                       "(set false to run palm only)"),
    run_palm: bool = Query(default=True,
                           description="run the palm image batch "
                                       "(set false to run face only)"),
):
    """Start a batch QC run over the local data/ folder.

    By default runs BOTH modalities (face videos + palm images) — the researcher
    uploads everything, so everything is checked. Narrow with run_face=false or
    run_palm=false. Returns 202 Accepted with a job_id immediately; poll
    GET /checks/batch/{job_id} for progress.
    """
    return _start_batch(run_palm=run_palm, run_face=run_face)


@app.get("/checks/batch", tags=["Checks"])
def list_batches(status_filter: str | None = Query(
        default=None, alias="status",
        description="filter by status, e.g. running | completed | failed")):
    return _list_batches(status_filter)


@app.get("/checks/batch/{job_id}", tags=["Checks"])
def get_batch(job_id: str):
    return _get_batch(job_id)


# ---------------------------------------------------------------------------
# Checks — single file (sync)
# ---------------------------------------------------------------------------

async def _check_one_file(file: UploadFile):
    raw = await file.read()
    try:
        return live.check_one(file.filename, raw, _load_config(),
                              store.config_path())
    except live.UnrecognisedFile as e:
        raise HTTPException(status_code=422, detail=str(e))
    except live.NotImplementedModality as e:
        raise HTTPException(status_code=501, detail=str(e))


@app.post("/checks/file", tags=["Checks"])
async def create_check_file(file: UploadFile = File(...)):
    """Upload one file. Its name is parsed to decide the modality:
      - face_rgb            -> run QC, return the verdict        (200)
      - palm_* / walk_* /   -> not built yet                     (501)
        other face streams
      - unrecognised name   -> rejected                          (422)
    """
    return await _check_one_file(file)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Deprecated aliases (old paths) — forward to the same handlers.
# Remove once all callers have migrated to /checks/*.
# ---------------------------------------------------------------------------

@app.post("/jobs", status_code=status.HTTP_202_ACCEPTED,
          tags=["Deprecated"], deprecated=True)
def create_job_deprecated():
    """DEPRECATED — use POST /checks/batch.

    Pinned to the ORIGINAL face-only behavior so existing callers don't suddenly
    start running palm. New callers should use POST /checks/batch (both by
    default).
    """
    return _start_batch(run_face=True, run_palm=False)


@app.get("/jobs", tags=["Deprecated"], deprecated=True)
def get_jobs_deprecated(status_filter: str | None = Query(
        default=None, alias="status")):
    """DEPRECATED — use GET /checks/batch."""
    return _list_batches(status_filter)


@app.get("/jobs/{job_id}", tags=["Deprecated"], deprecated=True)
def get_job_deprecated(job_id: str):
    """DEPRECATED — use GET /checks/batch/{job_id}."""
    return _get_batch(job_id)


@app.post("/checks", tags=["Deprecated"], deprecated=True)
async def create_check_deprecated(file: UploadFile = File(...)):
    """DEPRECATED — use POST /checks/file."""
    return await _check_one_file(file)