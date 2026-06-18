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
import json
import os
import sys

import pandas as pd
import streamlit as st
import altair as alt
import yaml

STATUS_COLORS = {
    "PASS": "#2E7D5B",
    "FAIL": "#C0392B",
    "SKIP": "#B08D2E",
    "REVIEW": "#7A5AA0",
    "ERROR": "#5A6473",

    # filename validation statuses
    "COMPLETE": "#2E7D5B",
    "INCOMPLETE": "#C0392B",
    "UNRECOGNISED": "#C0392B",
    "NO_REPORT": "#5A6473",
}

REPORT_CHECK_ORDER = {
    # 1) Video/file-level checks
    "check_container": 10,
    "check_fps": 20,
    "check_duration": 30,
    "check_resolution": 40,
    "frames_sampled": 50,
    "frame_checks": 60,

    # 2) Face evidence / landmark availability
    "check_face_detected": 100,
    "check_head_fully": 110,

    # 3) Frontal-frame quality checks
    "check_face_size": 200,
    "check_eyes_open": 210,
    "check_brightness": 220,
    "check_face_blur": 230,

    # 4) Turn-protocol checks
    "check_turn_left": 300,
    "check_turn_right": 310,
    "check_turn_down": 320,
    "check_turn_up": 330,
    "check_turn_sequence": 340,
}


def ordered_checks(*status_maps):
    """Return check names in the same researcher-facing order as result CSV."""
    checks = set()
    for m in status_maps:
        checks |= set(m)

    return sorted(
        checks,
        key=lambda c: (REPORT_CHECK_ORDER.get(c, 9999), c),
    )
    


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

@st.cache_data(show_spinner=False)
def load_config(config_path="config.yml"):
    """Load config.yml for dashboard-only visualization settings."""
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def render_pose_threshold_chart(df, *, x_col, y_col, title, threshold, height=260):
    """Render one pose axis with positive/negative turn threshold lines.

    Uses turn threshold only:
    - yaw: +/- side_yaw_tolerance_deg
    - pitch: +/- tilt_pitch_tolerance_deg

    Does not plot roll.
    """
    if x_col not in df.columns or y_col not in df.columns:
        st.info(f"No {y_col} data available.")
        return

    plot = df[[x_col, y_col]].copy()
    plot[x_col] = pd.to_numeric(plot[x_col], errors="coerce")
    plot[y_col] = pd.to_numeric(plot[y_col], errors="coerce")
    plot = plot.dropna(subset=[x_col, y_col])

    if plot.empty:
        st.info(f"No valid {y_col} values to plot.")
        return

    # Keep threshold domain visible even if the measured values are small.
    y_abs = max(
        float(plot[y_col].abs().max()),
        float(abs(threshold)),
    )
    y_limit = max(10.0, y_abs * 1.15)

    line = (
        alt.Chart(plot)
        .mark_line()
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_col),
            y=alt.Y(
                f"{y_col}:Q",
                title=f"{y_col} (deg)",
                scale=alt.Scale(domain=[-y_limit, y_limit]),
            ),
            tooltip=[
                alt.Tooltip(f"{x_col}:Q", title=x_col),
                alt.Tooltip(f"{y_col}:Q", title=f"{y_col}°", format=".1f"),
            ],
        )
    )

    threshold_df = pd.DataFrame({
        y_col: [threshold, -threshold],
        "label": [
            f"+turn threshold ({threshold:g}°)",
            f"-turn threshold ({threshold:g}°)",
        ],
    })

    rules = (
        alt.Chart(threshold_df)
        .mark_rule(strokeDash=[6, 4])
        .encode(
            y=f"{y_col}:Q",
            tooltip=[
                alt.Tooltip("label:N", title="threshold"),
                alt.Tooltip(f"{y_col}:Q", title="value", format=".1f"),
            ],
        )
    )

    labels = (
        alt.Chart(threshold_df)
        .mark_text(
            align="left",
            dx=6,
            dy=-6,
        )
        .encode(
            x=alt.value(5),
            y=f"{y_col}:Q",
            text="label:N",
        )
    )

    chart = (
        (line + rules + labels)
        .properties(title=title, height=height)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

def get_turn_thresholds(config):
    """Return turn-floor thresholds, not front-zone thresholds."""
    tol = (
        config.get("face", {})
        .get("turn_sequence", {})
        .get("tolerance", {})
        or {}
    )

    yaw_turn = float(tol.get("side_yaw_tolerance_deg", 30))
    pitch_turn = float(tol.get("tilt_pitch_tolerance_deg", 15))

    return yaw_turn, pitch_turn

def load_volunteer(reports_dir, vid):
    return {
        "overall": load_csv(_find(reports_dir, vid, "_overall")),
        "result": load_csv(_find(reports_dir, vid, "_result")),
        "detail": load_csv(_find(reports_dir, vid, "_detail")),
        "header": load_csv(_find(reports_dir, vid, "_detail_header")),
    }
    
@st.cache_data(show_spinner=False)
def load_filename_report(reports_dir):
    """Load validate_filenames.py output from reports/filenames.json or CSV.

    Prefer JSON because missing/duplicates are preserved as lists.
    Fall back to CSV if JSON is absent.
    """
    json_path = os.path.join(reports_dir, "filenames.json")
    csv_path = os.path.join(reports_dir, "filenames.csv")

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return {"kind": "json", "data": json.load(f)}

    if os.path.exists(csv_path):
        return {
            "kind": "csv",
            "data": pd.read_csv(csv_path, dtype={"volunteer": str}),
        }

    return None


def filename_issues_for_volunteer(reports_dir, vid):
    """Return filename-completeness issues for one volunteer."""
    report = load_filename_report(reports_dir)

    empty = {
        "status": "NO_REPORT",
        "missing": [],
        "duplicates": [],
        "unrecognised": [],
        "found_count": "",
    }

    if report is None:
        return empty

    if report["kind"] == "json":
        data = report["data"]

        volunteers = data.get("volunteers", [])
        item = next(
            (v for v in volunteers if str(v.get("volunteer", "")) == str(vid)),
            None,
        )

        unrecognised = [
            u.get("path", "")
            for u in data.get("unrecognised_files", [])
            if str(u.get("volunteer_guess", "")) == str(vid)
        ]

        if item is None:
            if unrecognised:
                return {
                    "status": "INCOMPLETE",
                    "missing": [],
                    "duplicates": [],
                    "unrecognised": unrecognised,
                    "found_count": "",
                }
            return empty

        return {
            "status": item.get("status", "INCOMPLETE"),
            "missing": item.get("missing", []) or [],
            "duplicates": item.get("duplicates", []) or [],
            "unrecognised": unrecognised,
            "found_count": item.get("found_count", ""),
        }

    # CSV fallback: long/tidy format
    df = report["data"]
    if df is None or df.empty or "volunteer" not in df.columns:
        return empty

    d = df[df["volunteer"].astype(str).eq(str(vid))].copy()
    if d.empty:
        return empty

    for c in ["status", "issue_type", "item", "path"]:
        if c in d.columns:
            d[c] = d[c].fillna("").astype(str)

    missing = d.loc[d["issue_type"].eq("missing"), "item"].tolist()
    duplicate_rows = d[d["issue_type"].eq("duplicate")]
    unrecognised = d.loc[d["issue_type"].eq("unrecognised"), "path"].tolist()

    duplicates = []
    if not duplicate_rows.empty:
        for item, g in duplicate_rows.groupby("item"):
            duplicates.append({
                "item": item,
                "paths": g["path"].tolist(),
            })

    status = "COMPLETE"
    if missing or duplicates or unrecognised:
        status = "INCOMPLETE"

    return {
        "status": status,
        "missing": missing,
        "duplicates": duplicates,
        "unrecognised": unrecognised,
        "found_count": "",
    }


def render_filename_summary(reports_dir, vid, compact=False):
    """Render filename validation result for the selected volunteer."""
    info = filename_issues_for_volunteer(reports_dir, vid)
    status = info["status"]

    st.markdown(
        f"**Volunteer file delivery** &nbsp; {status_pill(status)}",
        unsafe_allow_html=True,
    )

    if status == "NO_REPORT":
        st.caption(
            "No filename validation report found. "
            "Run `python validate_filenames.py data --out reports/filenames.csv`."
        )
        return

    missing = info["missing"]
    duplicates = info["duplicates"]
    unrecognised = info["unrecognised"]

    if not missing and not duplicates and not unrecognised:
        found = info.get("found_count", "")
        suffix = f" Found files: {found}." if found != "" else ""
        st.caption("All required files are present; no duplicates or unrecognised files." + suffix)
        return

    if missing:
        st.error("Missing files: " + ", ".join(f"`{m}`" for m in missing))

    if duplicates:
        dup_names = [d.get("item", "") for d in duplicates]
        st.warning("Duplicate items: " + ", ".join(f"`{d}`" for d in dup_names))

    if unrecognised:
        st.warning(f"Unrecognised files attributed to this volunteer: {len(unrecognised)}")

    if not compact:
        with st.expander("Filename issue details"):
            if duplicates:
                st.markdown("**Duplicates**")
                for d in duplicates:
                    st.write(d.get("item", ""))
                    for p in d.get("paths", []):
                        st.code(p)

            if unrecognised:
                st.markdown("**Unrecognised files**")
                for p in unrecognised:
                    st.code(p)

def render_face_qc_summary(status, reason):
    """Render the per-face-rgb QC result separately from file delivery."""
    st.markdown(
        f"**Face RGB QC** &nbsp; {status_pill(status)}",
        unsafe_allow_html=True,
    )

    if reason:
        st.caption(reason)
    else:
        st.caption("No face RGB QC reason available.")


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

    st.markdown(f"### Volunteer {vid}", unsafe_allow_html=True)
    render_filename_summary(reports_dir, vid)

    st.markdown("")
    render_face_qc_summary(status, reason)

    st.divider()
    
    st.subheader("Face RGB Result")


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
    if hdr is not None and not hdr.empty and {"yaw", "pitch"}.issubset(hdr.columns):
        st.subheader("Head pose over time")
        st.caption(
            "Yaw and pitch are shown separately. Dashed lines are turn thresholds "
            "from config.yml, not front-zone thresholds."
        )

        cfg = load_config("config.yml")
        yaw_turn_th, pitch_turn_th = get_turn_thresholds(cfg)

        plot = hdr.copy()
        for c in ["time", "frame_index", "yaw", "pitch"]:
            if c in plot.columns:
                plot[c] = pd.to_numeric(plot[c], errors="coerce")

        idx = "time" if "time" in plot.columns else "frame_index"

        render_pose_threshold_chart(
            plot,
            x_col=idx,
            y_col="yaw",
            title="Yaw over time",
            threshold=yaw_turn_th,
            height=260,
        )

        render_pose_threshold_chart(
            plot,
            x_col=idx,
            y_col="pitch",
            title="Pitch over time",
            threshold=pitch_turn_th,
            height=260,
        )

    # raw detail (collapsed)
    det = data["detail"]
    if det is not None and not det.empty:
        with st.expander("Raw per-frame detail"):
            st.dataframe(det, use_container_width=True, hide_index=True)


def _style_status(df, cols):
    """Color one or more status columns."""
    if isinstance(cols, str):
        cols = [cols]

    existing_cols = [c for c in cols if c in df.columns]

    def color(v):
        c = STATUS_COLORS.get(str(v).upper())
        return f"background-color:{c};color:#fff" if c else ""

    if not existing_cols:
        return df

    return df.style.map(color, subset=existing_cols)


def render_compare(reports_dir, a, b):
    da, db = load_volunteer(reports_dir, a), load_volunteer(reports_dir, b)
    sa, ra, oa = overall_verdict(da)
    sb, rb, ob = overall_verdict(db)

    h1, h2 = st.columns(2)
    h1.markdown(f"### {a}", unsafe_allow_html=True)
    with h1:
        render_filename_summary(reports_dir, a, compact=True)
        st.markdown("")
        render_face_qc_summary(sa, ra)

    h2.markdown(f"### {b}", unsafe_allow_html=True)
    with h2:
        render_filename_summary(reports_dir, b, compact=True)
        st.markdown("")
        render_face_qc_summary(sb, rb)


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

    rows = []
    for chk in ordered_checks(ca, cb):
        va, vb = ca.get(chk, "—"), cb.get(chk, "—")
        rows.append({
            "check": chk,
            a: va,
            b: vb,
            "same?": "✓" if va == vb else "✗",
        })

    cmp_df = pd.DataFrame(rows)
    st.dataframe(
        _style_status(cmp_df, [a, b]),
        use_container_width=True,
        hide_index=True,
    )

    differing = cmp_df[cmp_df["same?"] == "✗"]
    if not differing.empty:
        st.markdown("**Differences:** " +
                    ", ".join(f"`{c}`" for c in differing["check"]))
    else:
        st.success("Both volunteers have identical check verdicts.")

    # overlaid pose ranges
    # st.subheader("Pose range")
    # ha, hb = da["header"], db["header"]
    # if (
    #     ha is not None and hb is not None
    #     and {"yaw", "pitch"}.issubset(ha.columns)
    #     and {"yaw", "pitch"}.issubset(hb.columns)
    # ):
    #     def span(h, c):
    #         v = pd.to_numeric(h[c], errors="coerce")
    #         return v.max() - v.min()
    #     span_df = pd.DataFrame({
    #         "axis": ["yaw range", "pitch range"],
    #         a: [span(ha, "yaw"), span(ha, "pitch")],
    #         b: [span(hb, "yaw"), span(hb, "pitch")],
    #     }).set_index("axis")
    #     st.bar_chart(span_df, height=260)


def _result_status_map(res):
    """check_name -> final_status from a result CSV."""
    if res is None or res.empty:
        return {}
    if "check_name" not in res.columns or "final_status" not in res.columns:
        return {}
    return dict(zip(res["check_name"], res["final_status"]))


def main():
    args = parse_cli()
    st.set_page_config(page_title="Biometric verification", page_icon="", layout="wide")
    st.markdown("<style>.block-container{max-width:1250px;padding-top:2rem}</style>",
                unsafe_allow_html=True)

    st.title("Biometric verification")
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