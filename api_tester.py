"""api_tester.py — a minimal Streamlit frontend to exercise the QC FastAPI.

This is NOT the results dashboard (dashboard.py). It drives the LIVE API end to
end, the way a real client would:

    1. POST /checks/uploads   (upload ONE .zip of the data folder)  -> 202 + job
    2. poll GET /checks/batch/{job_id}   until status == "completed"
    3. GET /results?job_id=...            preview all rows
       GET /results?job_id=...&status=FAIL  the FAIL-only set (shown here)
    4. GET /results/download?job_id=...&format=csv   download button

Run the API first (from the repo root):
    uvicorn main:app --reload            # -> http://localhost:8000

Then run this app:
    streamlit run api_tester.py

The API base URL is configurable in the sidebar (default http://localhost:8000)
so you can point it at a remote deployment too.
"""

from __future__ import annotations

import io
import time

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="QC API tester", layout="wide")

DEFAULT_API = "http://localhost:8000"
POLL_SECONDS = 2.0            # gap between status polls
POLL_MAX_TRIES = 600         # ~20 min ceiling at 2s; batch of 1,500 is long


# ---------------------------------------------------------------------------
# Thin API client — one function per endpoint we touch. Every call returns
# (ok, payload_or_error) so the UI never has to try/except inline.
# ---------------------------------------------------------------------------

def _url(base, path):
    return base.rstrip("/") + path


def api_health(base):
    try:
        r = requests.get(_url(base, "/health"), timeout=5)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, str(e)


def api_upload_zip(base, filename, data, *, run_face, run_palm, run_walk):
    """POST /checks/uploads with ONE zip. Returns (ok, job_dict|error)."""
    params = {
        "run_face": str(run_face).lower(),
        "run_palm": str(run_palm).lower(),
        "run_walk": str(run_walk).lower(),
    }
    files = {"files": (filename, data, "application/zip")}
    try:
        r = requests.post(_url(base, "/checks/uploads"),
                          params=params, files=files, timeout=120)
        if r.status_code == 422:
            return False, f"422 rejected: {r.json().get('detail', r.text)}"
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, str(e)


def api_get_job(base, job_id):
    try:
        r = requests.get(_url(base, f"/checks/batch/{job_id}"), timeout=10)
        if r.status_code == 404:
            return False, f"no batch run '{job_id}'"
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, str(e)


def api_results(base, job_id, *, status=None):
    params = {"job_id": job_id}
    if status:
        params["status"] = status
    try:
        r = requests.get(_url(base, "/results"), params=params, timeout=30)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, str(e)


def api_download_csv(base, job_id):
    """GET /results/download?format=csv -> (ok, bytes|error)."""
    params = {"job_id": job_id, "format": "csv"}
    try:
        r = requests.get(_url(base, "/results/download"),
                         params=params, timeout=60)
        if r.status_code == 404:
            return False, r.json().get("detail", r.text)
        r.raise_for_status()
        return True, r.content
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("QC API tester")
st.caption(
    "Uploads a zip to the FastAPI, waits for the batch job to finish, then "
    "previews and downloads the result. This calls the live API — it does not "
    "run the pipeline directly."
)

with st.sidebar:
    st.subheader("API")
    api_base = st.text_input("Base URL", value=DEFAULT_API)
    if st.button("Check health"):
        ok, payload = api_health(api_base)
        if ok:
            st.success("API is up")
            st.json(payload)
        else:
            st.error(f"API unreachable: {payload}")

    st.subheader("Modalities to run")
    run_face = st.checkbox("Face", value=True)
    run_palm = st.checkbox("Palm", value=True)
    run_walk = st.checkbox("Walk", value=True)

# Session state carries the job across reruns (Streamlit reruns top-to-bottom on
# every interaction, so the job_id must live in state, not a local).
if "job" not in st.session_state:
    st.session_state.job = None

# ---- Step 1: upload ----
st.subheader("1 · Upload a data zip")
up = st.file_uploader(
    "Zip of the data folder (the same layout run_folder expects)",
    type=["zip"],
)

col_a, col_b = st.columns([1, 3])
with col_a:
    start = st.button("Upload & run", type="primary", disabled=up is None)

if start and up is not None:
    if not (run_face or run_palm or run_walk):
        st.error("Select at least one modality to run.")
    else:
        with st.spinner("Uploading zip and starting the batch…"):
            ok, payload = api_upload_zip(
                api_base, up.name, up.getvalue(),
                run_face=run_face, run_palm=run_palm, run_walk=run_walk,
            )
        if not ok:
            st.error(f"Upload failed: {payload}")
        else:
            st.session_state.job = payload
            st.success(
                f"Started job {payload.get('job_id')} "
                f"({payload.get('uploaded_files', '?')} files uploaded)."
            )

job = st.session_state.job

# ---- Step 2: poll until completed ----
if job:
    job_id = job.get("job_id")
    st.subheader("2 · Job status")
    st.caption(f"job_id: `{job_id}`")

    status_box = st.empty()
    prog_box = st.empty()

    # Fetch current status once per rerun; if still running, offer a blocking
    # poll (spinner) OR a manual refresh, so the user is never stuck if the API
    # is slow. We poll in a bounded loop rather than recursing.
    ok, current = api_get_job(api_base, job_id)
    if not ok:
        st.error(f"Could not read job: {current}")
    else:
        cur_status = (current.get("status") or "").lower()
        status_box.info(f"Status: **{cur_status or 'unknown'}**")

        if cur_status not in ("completed", "failed"):
            if st.button("Poll until complete"):
                with st.spinner("Polling the batch until it finishes…"):
                    for _ in range(POLL_MAX_TRIES):
                        ok, current = api_get_job(api_base, job_id)
                        if not ok:
                            status_box.error(f"Poll error: {current}")
                            break
                        cur_status = (current.get("status") or "").lower()
                        status_box.info(f"Status: **{cur_status}**")
                        prog = current.get("progress")
                        if prog:
                            prog_box.write(prog)
                        if cur_status in ("completed", "failed"):
                            break
                        time.sleep(POLL_SECONDS)
                # persist the last-seen job so the results section renders
                st.session_state.job = {**job, **current}
                job = st.session_state.job
            st.caption(
                "Batches over many volunteers take a while. You can also click "
                "again to re-poll."
            )

        if cur_status == "failed":
            st.error("The batch job FAILED. Check the API logs.")
        elif cur_status == "completed":
            st.success("Job completed.")

    # ---- Step 3: results (only once completed) ----
    if (current.get("status") or "").lower() == "completed":
        st.subheader("3 · Results")

        # FAIL-only, per the request: query ?status=FAIL and show those rows.
        ok_fail, fail_payload = api_results(api_base, job_id, status="FAIL")
        ok_all, all_payload = api_results(api_base, job_id)

        if ok_all:
            total = all_payload.get("count", 0)
            n_fail = fail_payload.get("count", 0) if ok_fail else "?"
            st.caption(f"{total} total rows · {n_fail} FAIL rows")

        st.markdown("**FAIL rows**")
        if ok_fail:
            fails = fail_payload.get("results", [])
            if fails:
                df = pd.DataFrame(fails)
                cols = [c for c in ["volunteer_id", "data_type", "filename",
                                    "overall_status", "check_name", "reason"]
                        if c in df.columns]
                st.dataframe(df[cols] if cols else df,
                             use_container_width=True, hide_index=True)
            else:
                st.success("No FAIL rows — everything passed.")
        else:
            st.error(f"Could not read FAIL results: {fail_payload}")

        # Full preview (collapsed) so the reviewer can see PASS rows too.
        with st.expander("Preview all rows (PASS + FAIL)"):
            if ok_all:
                allrows = all_payload.get("results", [])
                if allrows:
                    st.dataframe(pd.DataFrame(allrows),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("No rows returned.")
            else:
                st.error(f"Could not read results: {all_payload}")

        # ---- Step 4: download the result CSV ----
        st.subheader("4 · Download")
        ok_csv, csv_bytes = api_download_csv(api_base, job_id)
        if ok_csv:
            st.download_button(
                "Download result CSV (all_summary.csv)",
                data=csv_bytes,
                file_name=f"all_summary_{job_id}.csv",
                mime="text/csv",
            )
        else:
            st.warning(f"CSV not available yet: {csv_bytes}")
