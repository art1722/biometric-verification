"""Streamlit per-volunteer QC inspector for the face_rgb pipeline.

Reads a reports/ folder produced by run_folder.py (or run_face.py). Each
volunteer has a sub-folder reports/<id>/ containing:
    face_<id>_overall.csv         one OVERALL verdict row
    face_<id>_result.csv          one row per check (aggregated verdict)
    face_<id>_detail.csv          one row per check, per frame
    face_<id>_detail_header.csv   one row per frame (pose: yaw/pitch/roll)

Two modes:
  - Single   : pick one volunteer, see verdict + per-check table + per-frame
               pose timeline + raw detail.
  - Compare  : pick TWO volunteers; their per-check verdicts are aligned in
               one table so differences stand out, with pose ranges side by side.

Run:
    pip install streamlit pandas
    streamlit run dashboard.py
    streamlit run dashboard.py -- --reports reports
"""
import argparse
import glob
import os
import sys

import pandas as pd
import streamlit as st

STATUS_COLORS = {
    "PASS": "#2E7D5B", "FAIL": "#C0392B",
    "SKIP": "#B08D2E", "REVIEW": "#7A5AA0", "ERROR": "#5A6473",
}


def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    argv = sys.argv[1:]
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    args, _ = ap.parse_known_args(argv)
    return args


def list_volunteers(reports_dir):
    """Volunteer ids = sub-folders that contain a face_<id>_overall.csv."""
    ids = []
    for entry in sorted(os.listdir(reports_dir)):
        sub = os.path.join(reports_dir, entry)
        if os.path.isdir(sub) and glob.glob(os.path.join(sub, "*_overall.csv")):
            ids.append(entry)
    return ids


def _find(reports_dir, vid, suffix):
    """Locate a per-volunteer CSV by suffix, tolerant of naming."""
    sub = os.path.join(reports_dir, vid)
    hits = glob.glob(os.path.join(sub, f"*{suffix}.csv"))
    return hits[0] if hits else None


@st.cache_data(show_spinner=False)
def load_csv(path):
    if not path or not os.path.exists(path):
        return None
    return pd.read_csv(path, dtype={"volunteer_id": str})


def load_volunteer(reports_dir, vid):
    return {
        "overall": load_csv(_find(reports_dir, vid, "_overall")),
        "result": load_csv(_find(reports_dir, vid, "_result")),
        "detail": load_csv(_find(reports_dir, vid, "_detail")),
        "header": load_csv(_find(reports_dir, vid, "_detail_header")),
    }


def status_pill(status):
    c = STATUS_COLORS.get(str(status).upper(), "#888")
    return (f'<span style="background:{c};color:#fff;padding:3px 12px;'
            f'border-radius:999px;font-weight:600;font-size:0.85rem">'
            f'{status}</span>')


def overall_verdict(data):
    """Pull the single OVERALL status/reason from the overall CSV."""
    o = data["overall"]
    if o is None or o.empty:
        return None, None, {}
    row = o.iloc[0].to_dict()
    return row.get("final_status", "—"), row.get("reason", ""), row


def render_single(reports_dir, vid):
    data = load_volunteer(reports_dir, vid)
    status, reason, orow = overall_verdict(data)
    if status is None:
        st.warning(f"No overall result found for {vid}.")
        return

    st.markdown(f"### Volunteer {vid} &nbsp; {status_pill(status)}",
                unsafe_allow_html=True)
    st.caption(reason or "")

    # quick facts from the overall row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames sampled", orow.get("frames_sampled", "—"))
    c2.metric("Detection gaps", orow.get("detection_gaps", "—"))
    yaw = f'{orow.get("yaw_min","")} … {orow.get("yaw_max","")}'
    pit = f'{orow.get("pitch_min","")} … {orow.get("pitch_max","")}'
    c3.metric("Yaw range (°)", yaw)
    c4.metric("Pitch range (°)", pit)

    # per-check verdicts
    st.subheader("Checks")
    res = data["result"]
    if res is not None and not res.empty:
        cols = [c for c in ["check_level", "check_name", "final_status",
                            "pass", "fail", "skip", "reason"] if c in res.columns]
        styled = res[cols].rename(columns={"final_status": "status"})
        st.dataframe(_style_status(styled, "status"),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No per-check result CSV for this volunteer.")

    # pose timeline
    hdr = data["header"]
    if hdr is not None and not hdr.empty and "yaw" in hdr.columns:
        st.subheader("Head pose over time")
        st.caption("yaw swings left↔right, pitch swings down↔up as the head "
                   "turns through the protocol.")
        plot = hdr.copy()
        for c in ["time", "yaw", "pitch", "roll"]:
            if c in plot.columns:
                plot[c] = pd.to_numeric(plot[c], errors="coerce")
        idx = "time" if "time" in plot.columns else "frame_index"
        pose_cols = [c for c in ["yaw", "pitch", "roll"] if c in plot.columns]
        st.line_chart(plot.set_index(idx)[pose_cols], height=300)

    # raw detail (collapsed)
    det = data["detail"]
    if det is not None and not det.empty:
        with st.expander("Raw per-frame detail"):
            st.dataframe(det, use_container_width=True, hide_index=True)


def _style_status(df, col):
    """Color the status cell green/red/amber."""
    def color(v):
        c = STATUS_COLORS.get(str(v).upper())
        return f"background-color:{c};color:#fff" if c else ""
    return df.style.map(color, subset=[col])


def render_compare(reports_dir, a, b):
    da, db = load_volunteer(reports_dir, a), load_volunteer(reports_dir, b)
    sa, ra, oa = overall_verdict(da)
    sb, rb, ob = overall_verdict(db)

    h1, h2 = st.columns(2)
    h1.markdown(f"### {a} &nbsp; {status_pill(sa)}", unsafe_allow_html=True)
    h1.caption(ra or "")
    h2.markdown(f"### {b} &nbsp; {status_pill(sb)}", unsafe_allow_html=True)
    h2.caption(rb or "")

    # side-by-side quick facts
    st.subheader("Summary")
    facts = pd.DataFrame({
        "metric": ["frames_sampled", "detection_gaps",
                   "yaw_min", "yaw_max", "pitch_min", "pitch_max"],
        a: [oa.get(k, "") for k in ["frames_sampled", "detection_gaps",
                                    "yaw_min", "yaw_max", "pitch_min", "pitch_max"]],
        b: [ob.get(k, "") for k in ["frames_sampled", "detection_gaps",
                                    "yaw_min", "yaw_max", "pitch_min", "pitch_max"]],
    })
    st.dataframe(facts, use_container_width=True, hide_index=True)

    # aligned per-check verdicts: one row per check, a status vs b status
    st.subheader("Per-check comparison")
    st.caption("Checks where the two volunteers differ are the interesting "
               "rows. ✓ = same verdict, ✗ = differs.")
    ca = _result_status_map(da["result"])
    cb = _result_status_map(db["result"])
    all_checks = sorted(set(ca) | set(cb))
    rows = []
    for chk in all_checks:
        va, vb = ca.get(chk, "—"), cb.get(chk, "—")
        rows.append({"check": chk, a: va, b: vb,
                     "same?": "✓" if va == vb else "✗"})
    cmp_df = pd.DataFrame(rows)
    st.dataframe(_style_status(_style_status(cmp_df, a), b),
                 use_container_width=True, hide_index=True)

    differing = cmp_df[cmp_df["same?"] == "✗"]
    if not differing.empty:
        st.markdown("**Differences:** " +
                    ", ".join(f"`{c}`" for c in differing["check"]))
    else:
        st.success("Both volunteers have identical check verdicts.")

    # overlaid pose ranges
    st.subheader("Pose range")
    ha, hb = da["header"], db["header"]
    if ha is not None and hb is not None and "yaw" in ha.columns:
        def span(h, c):
            v = pd.to_numeric(h[c], errors="coerce")
            return v.max() - v.min()
        span_df = pd.DataFrame({
            "axis": ["yaw range", "pitch range"],
            a: [span(ha, "yaw"), span(ha, "pitch")],
            b: [span(hb, "yaw"), span(hb, "pitch")],
        }).set_index("axis")
        st.bar_chart(span_df, height=260)


def _result_status_map(res):
    """check_name -> final_status from a result CSV."""
    if res is None or res.empty:
        return {}
    if "check_name" not in res.columns or "final_status" not in res.columns:
        return {}
    return dict(zip(res["check_name"], res["final_status"]))


def main():
    args = parse_cli()
    st.set_page_config(page_title="QC Inspector", page_icon="🔎", layout="wide")
    st.markdown("<style>.block-container{max-width:1250px;padding-top:2rem}</style>",
                unsafe_allow_html=True)

    st.title("QC Inspector")
    st.caption(f"Reading per-volunteer results from `{args.reports}/`")

    if not os.path.isdir(args.reports):
        st.error(f"Reports folder not found: `{args.reports}`")
        st.info("Run a batch first (`python run_folder.py data`), or pass "
                "`-- --reports path/to/reports`.")
        st.stop()

    ids = list_volunteers(args.reports)
    if not ids:
        st.warning(f"No volunteer folders with an *_overall.csv under "
                   f"`{args.reports}/`.")
        st.stop()

    mode = st.sidebar.radio("Mode", ["Single", "Compare"])
    st.sidebar.caption(f"{len(ids)} volunteers found")

    if mode == "Single":
        vid = st.sidebar.selectbox("Volunteer", ids)
        render_single(args.reports, vid)
    else:
        a = st.sidebar.selectbox("Volunteer A", ids, index=0)
        b = st.sidebar.selectbox("Volunteer B", ids,
                                 index=min(1, len(ids) - 1))
        if a == b:
            st.info("Pick two different volunteers to compare.")
        else:
            render_compare(args.reports, a, b)


if __name__ == "__main__":
    main()