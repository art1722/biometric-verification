"""app.py — the FastAPI application: all endpoints in one place.

Endpoint namespace (v0.5.0)
---------------------------
Everything that RUNS a check now lives under a single /checks/* namespace, with
the grain in the path. Reading stored results stays under /results (a separate
concern from running checks).

  Checks (run QC):
    POST /checks/file          one uploaded file, SYNC — returns the status (200)
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
POST /checks/file runs inline and returns the status in the response body — the
caller waits. POST /checks/batch cannot: a run over ~1,500 volunteers is far too
long for one request, so it returns 202 + a job_id immediately and the caller
POLLs GET /checks/batch/{id}. The returned object still carries job_id/status/
progress — it is conceptually a "job", only the URL changed.

Run (from repo root):  uvicorn main:app --reload   ->  http://localhost:8000/docs
"""

from __future__ import annotations

import functools
import io
import os
import zipfile

from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, status
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse

from . import store, jobs, live, uploads

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

def _start_batch(run_face: bool = True, run_palm: bool = True,
                 run_walk: bool = True, sample_fps: float | None = None):
    """Shared handler: launch a batch run and return 202 + Location.

    Defaults to ALL THREE modalities (face + palm + walk), matching run_folder
    where everything runs unless turned off. The run_* flags select which the
    batch processes; sample_fps optionally overrides the sampling rate. It runs
    run_folder over the server's local data/ dir.
    """
    job = jobs.start_job(run_face=run_face, run_palm=run_palm,
                         run_walk=run_walk, sample_fps=sample_fps)
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
                           description="run the face video batch"),
    run_palm: bool = Query(default=True,
                           description="run the palm image batch"),
    run_walk: bool = Query(default=True,
                           description="run the walk/gait video batch"),
    sample_fps: float | None = Query(
        default=None,
        description="frames sampled per source second. Omit to use the batch "
                    "default (1.0). 0 or negative = native (every frame)."),
):
    """Start a batch QC run over the local data/ folder.

    By default runs ALL THREE modalities (face videos + palm images + walk
    videos) — the researcher uploads everything, so everything is checked. Turn
    any off with run_face=false / run_palm=false / run_walk=false. sample_fps
    optionally overrides the frame-sampling rate. Returns 202 Accepted with a
    job_id immediately; poll GET /checks/batch/{job_id} for progress.
    """
    return _start_batch(run_face=run_face, run_palm=run_palm,
                        run_walk=run_walk, sample_fps=sample_fps)


@app.get("/checks/batch", tags=["Checks"])
def list_batches(status_filter: str | None = Query(
        default=None, alias="status",
        description="filter by status, e.g. running | completed | failed")):
    return _list_batches(status_filter)


@app.get("/checks/batch/{job_id}", tags=["Checks"])
def get_batch(job_id: str):
    return _get_batch(job_id)


# ---------------------------------------------------------------------------
# Checks — uploaded batch (async): upload files -> QC the whole set
# ---------------------------------------------------------------------------

@app.post("/checks/uploads", status_code=status.HTTP_202_ACCEPTED,
          tags=["Checks"])
async def create_uploads_batch(
    files: List[UploadFile] = File(
        ..., description="Either ONE .zip of the data folder, or MANY files "
                         "from a folder picker (each keeps its relative path)."),
    run_face: bool = Query(default=True,
                           description="run the face video batch"),
    run_palm: bool = Query(default=True,
                           description="run the palm image batch"),
    run_walk: bool = Query(default=True,
                           description="run the walk/gait video batch"),
    sample_fps: float | None = Query(
        default=None,
        description="frames sampled per source second. Omit to use the batch "
                    "default (1.0). 0 or negative = native (every frame)."),
):
    """Upload a whole data set and QC it in one shot.

    This is POST /checks/batch with an upload layer on top: instead of running
    over the server's standing data/ dir, it unpacks the upload into an isolated
    temp dir and runs the SAME batch over that. Every batch option (run_face /
    run_palm / run_walk / sample_fps) applies identically.

    Accepts the folder in whichever shape the client sends it:
      - ONE .zip                -> unpacked server-side
      - MANY files (folder pick) -> written into a temp tree, subpaths preserved

    The upload is unpacked into its OWN isolated temp dir (never the shared
    data/), then the batch runs over it and writes to that upload's own reports
    dir with a FRESH all_summary (this upload only — the default now, since the
    batch overwrites all_summary unless --append). Returns 202 + a job_id
    immediately; poll GET /checks/batch/{job_id}, then read the results from
    GET /results (the job's reports_dir is also returned for reference).

    Errors:
      - empty upload / corrupt zip / unsafe path -> 422
    """
    # Read every uploaded file into (name, bytes). UploadFile.filename carries
    # the relative path for folder-picker uploads (webkitRelativePath), which
    # unpack_uploads preserves under the temp root.
    pairs = []
    for f in files:
        raw = await f.read()
        pairs.append((f.filename or "", raw))

    try:
        data_dir, n_files = uploads.unpack_uploads(pairs)
    except uploads.BadUpload as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Each upload writes into a reports dir NEXT TO its data dir, so its output
    # is isolated from the shared reports/ and from other uploads.
    reports_dir = data_dir.rstrip("/\\") + "_reports"

    # No append= here: the batch overwrites all_summary by default, so each
    # upload's roll-up is naturally isolated to that upload (was fresh_all=True
    # under the old flag model).
    job = jobs.start_job(
        run_face=run_face, run_palm=run_palm, run_walk=run_walk,
        sample_fps=sample_fps,
        data_dir=data_dir, reports_dir=reports_dir,
    )
    job["uploaded_files"] = n_files
    # Tell the caller exactly where to poll and where to fetch all_summary once
    # this async run finishes. all_summary does NOT exist yet at 202 time (the
    # batch is still running), so we return the URLs rather than the file: poll
    # batch_url until status=completed, then GET results_url / download_url.
    jid = job["job_id"]
    job["batch_url"] = f"/checks/batch/{jid}"
    job["results_url"] = f"/results?job_id={jid}"
    job["download_url"] = f"/results/download?job_id={jid}&format=json"
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=job,
        headers={"Location": f"/checks/batch/{jid}"},
    )


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
      - face_rgb            -> run QC, return the status         (200)
      - palm_* / walk_* /   -> not built yet                     (501)
        other face streams
      - unrecognised name   -> rejected                          (422)
    """
    return await _check_one_file(file)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _reports_for_job(job_id: str | None) -> str | None:
    """Resolve which reports dir a /results call should read.

    job_id given   -> that job's own reports dir (an upload's isolated dir).
                      Unknown job_id -> 404.
    job_id omitted -> None, so store.* falls back to the shared reports dir
                      (the pre-upload behaviour, unchanged).
    """
    if not job_id:
        return None
    reports = jobs.reports_dir_for(job_id)
    if reports is None:
        raise HTTPException(status_code=404, detail=f"No batch run '{job_id}'")
    return reports


@app.get("/results", tags=["Results"])
def results(
    status_filter: str | None = Query(
        default=None, alias="status",
        description="filter by overall status, e.g. FAIL (omit for all: "
                    "PASS + FAIL)"),
    job_id: str | None = Query(
        default=None,
        description="read THIS upload's results (its own reports dir). Omit to "
                    "read the shared reports dir."),
):
    """Cross-modal results as flat per-check rows.

    all_summary now records every processed media file, so with no status filter
    this returns PASS + FAIL. Pass status=FAIL for just the problem set.
    Pass job_id to read a specific upload's results.
    """
    reports = _reports_for_job(job_id)
    rows = store.read_batch_summary(status=status_filter, reports=reports)
    return {
        "count": len(rows),
        "status_filter": status_filter,
        "job_id": job_id,
        "results": rows,
    }


#: Files a reviewer needs for the default ("summary") download. Both are written
#: at the reports ROOT by run_folder.py: all_summary.* is the cross-modal
#: verdict roll-up, filenames.* is the completeness/naming report. They answer
#: two different questions ("did the files pass?" vs "did they send the right
#: files?"), so the default hands over both rather than making the client make
#: two calls and zip them itself.
_SUMMARY_MEMBERS = ("all_summary", "filenames")

#: Overlay media (face_001_overlay.mp4, palm_002_L_N_overlay.jpg, ...) are debug
#: visualisations, not deliverables. Across ~1,500 volunteers they dominate the
#: size of reports/ — streaming them through one HTTP response would time out —
#: so scope=full drops them unless include_overlays=true is passed explicitly.
_OVERLAY_MARKER = "_overlay."


def _zip_of(paths_with_arcnames) -> io.BytesIO:
    """Build an in-memory zip from (absolute_path, name_inside_zip) pairs.

    In-memory (not a temp file on disk) because these bundles are small in the
    summary case and this keeps the endpoint free of cleanup logic. scope=full
    can be large; see the size note on the endpoint.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in paths_with_arcnames:
            zf.write(src, arcname)
    buf.seek(0)
    return buf


def _collect_full_reports(base: str, include_overlays: bool):
    """Walk a reports dir, yielding (path, arcname) for everything to ship.

    Keeps the on-disk layout inside the zip (root CSVs at the top, per-volunteer
    files under <volunteer_id>/) so the extracted folder looks exactly like the
    reports/ folder a CLI run produces.
    """
    for dirpath, _dirnames, filenames in os.walk(base):
        for fn in filenames:
            if not include_overlays and _OVERLAY_MARKER in fn:
                continue
            src = os.path.join(dirpath, fn)
            yield src, os.path.relpath(src, base)


@app.get("/results/download", tags=["Results"])
def download_results(
    scope: str = Query(
        default="summary",
        pattern="^(summary|full|all_summary|filenames)$",
        description="summary = zip of all_summary + filenames (default); "
                    "full = zip of the whole reports folder; "
                    "all_summary / filenames = that single file only"),
    fmt: str = Query(default="csv", alias="format", pattern="^(json|csv)$",
                     description="csv or json (ignored when scope=full)"),
    include_overlays: bool = Query(
        default=False,
        description="scope=full only: also include *_overlay.* debug media. "
                    "Off by default — overlays dominate the folder size"),
    job_id: str | None = Query(
        default=None, description="download THIS upload's results"),
):
    """Download stored results.

    Four scopes, one endpoint:

      scope=summary (DEFAULT) -> reports_summary.zip
          all_summary.<fmt> + filenames.<fmt>. The reviewer's deliverable: the
          verdicts plus the completeness report. Files missing from disk are
          skipped rather than 404-ing the whole bundle, so a run with filename
          validation disabled (--no-filenames) still yields a usable zip; a 404
          is raised only when NEITHER file exists.

      scope=full -> reports_full.zip
          The entire reports folder, per-volunteer subfolders included, laid out
          exactly as on disk. Excludes *_overlay.* unless include_overlays=true.
          SIZE WARNING: with overlays on and a large batch this is many GB in a
          single response and will likely time out; prefer fetching per-
          volunteer files, or run the CLI and read reports/ directly.

      scope=all_summary / scope=filenames -> that one file
          Unchanged single-file behaviour (all_summary was the old default), so
          existing clients keep working.

    fmt selects .csv or .json for the single-file and summary scopes; it does
    not apply to scope=full, which ships whatever is on disk. Pass job_id to
    read an upload's own isolated reports dir instead of the shared one.
    """
    reports = _reports_for_job(job_id)
    base = reports or store.reports_dir()

    if not os.path.isdir(base):
        raise HTTPException(
            status_code=404,
            detail=f"reports dir not found: {base}")

    # ---- scope=full: everything on disk, overlays optional ----
    if scope == "full":
        members = list(_collect_full_reports(base, include_overlays))
        if not members:
            raise HTTPException(
                status_code=404,
                detail="reports dir is empty (has the batch finished?)")
        buf = _zip_of(members)
        return StreamingResponse(
            buf, media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="reports_full.zip"'
            })

    ext = "json" if fmt == "json" else "csv"

    # ---- single-file scopes: unchanged behaviour ----
    if scope in _SUMMARY_MEMBERS:
        fname = f"{scope}.{ext}"
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            raise HTTPException(
                status_code=404,
                detail=f"{fname} not found (has the batch finished writing?)")
        media = "application/json" if ext == "json" else "text/csv"
        return FileResponse(path, media_type=media, filename=fname)

    # ---- scope=summary (default): both files in one zip ----
    members = []
    for stem in _SUMMARY_MEMBERS:
        fname = f"{stem}.{ext}"
        path = os.path.join(base, fname)
        if os.path.exists(path):
            members.append((path, fname))

    if not members:
        raise HTTPException(
            status_code=404,
            detail=f"neither all_summary.{ext} nor filenames.{ext} found "
                   f"(has the batch finished writing?)")

    buf = _zip_of(members)
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="reports_summary.zip"'
        })


@app.get("/results/{volunteer_id}", tags=["Results"])
def result_for_volunteer(
    volunteer_id: str,
    job_id: str | None = Query(
        default=None, description="read from THIS upload's results"),
):
    reports = _reports_for_job(job_id)
    overall = store.read_overall(volunteer_id, reports=reports)
    if overall is None:
        raise HTTPException(
            status_code=404,
            detail=f"No QC report for volunteer '{volunteer_id}'",
        )
    return {
        "volunteer_id": volunteer_id,
        "overall": overall,
        "checks": store.read_checks(volunteer_id, reports=reports),
    }


@app.get("/volunteers", tags=["Results"])
def volunteers(job_id: str | None = Query(
        default=None, description="list volunteers in THIS upload's results")):
    reports = _reports_for_job(job_id)
    ids = store.list_volunteer_ids(reports=reports)
    return {"count": len(ids), "volunteer_ids": ids}