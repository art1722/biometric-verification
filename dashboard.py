"""Streamlit per-volunteer QC inspector for the face_rgb pipeline.

Reads a reports/ folder produced by run_folder.py (or run_face.py). Each
volunteer has a sub-folder reports/<id>/ containing:
    face_<id>_overall.csv         one OVERALL verdict row
    face_<id>_result.csv          one row per check (aggregated verdict)
    face_<id>_detail.csv          one row per check, per frame
    face_<id>_detail_header.csv   one row per frame (pose: yaw/pitch/roll)

Modes:
  - Single   : pick one volunteer, see verdict + per-check table + per-frame
               pose timeline + raw detail.
  - Compare  : pick TWO volunteers; their per-check verdicts are aligned in
               one table so differences stand out, with pose ranges side by side.
  - Summaries: read one modality's customer-facing summary at the reports root
               (face_summary.csv / palm_summary.csv / walk_summary.csv), one
               row per failed check, with a failures-only toggle.
  - All failures: read all_summary.csv / all_summary.json — the cross-modal
               FAIL/ERROR roll-up — showing each failed check + its reason,
               with CSV and JSON download.

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

    # palm per-image grid: a slot with no graded row
    "MISSING": "#5A6473",
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

    # Drop rows that cannot be placed on the time axis (bad x), but KEEP rows
    # whose y is NaN — those are no-face frames. Removing them would make the
    # line bridge straight across the gap, which falsely reads as "face was
    # detected and the pose stayed constant". By keeping the NaN y, Altair
    # breaks the line at the gap instead. The gap itself is then made explicit
    # with the gray no-face markers below.
    plot = plot.dropna(subset=[x_col])

    if plot.empty or plot[y_col].notna().sum() == 0:
        st.info(f"No valid {y_col} values to plot.")
        return

    # No-face frames: x is valid but y (yaw/pitch) is missing.
    noface = plot[plot[y_col].isna()].copy()

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

    # Point layer ONLY for isolated detected frames. A line connects adjacent
    # valid points; a single detected frame whose immediate neighbours (in time
    # order) are both no-face/NaN has nothing to connect to, so the line alone
    # renders it invisible (e.g. one frame at the peak of a turn). We draw a dot
    # for exactly those frames — same colour as the line — so they stay visible
    # without cluttering normal runs, which the line already shows.
    plot = plot.sort_values(x_col)
    y_isna = plot[y_col].isna()
    prev_isna = y_isna.shift(1, fill_value=True)   # treat off-the-edge as a gap
    next_isna = y_isna.shift(-1, fill_value=True)
    isolated = plot[(~y_isna) & prev_isna & next_isna]

    points = (
        alt.Chart(isolated)
        .mark_point(size=18, filled=True, opacity=1.0)
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_col),
            y=alt.Y(f"{y_col}:Q", scale=alt.Scale(domain=[-y_limit, y_limit])),
            color=alt.value("#4c78a8"),   # Altair/Vega default blue — matches the line
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

    layers = [line, points, rules, labels]

    # No-face overlay: draw a gray tick near the bottom of the chart for every
    # frame where the face was not detected. This makes the gaps in the pose
    # line explicit ("no face here") instead of leaving an ambiguous blank that
    # could be misread as missing data or a constant pose.
    if not noface.empty:
        noface = noface.assign(_marker=-y_limit, _status="no face detected")
        noface_marks = (
            alt.Chart(noface)
            .mark_tick(color="gray", opacity=0.6, thickness=2, size=12)
            .encode(
                x=alt.X(f"{x_col}:Q", title=x_col),
                y=alt.Y("_marker:Q", scale=alt.Scale(domain=[-y_limit, y_limit])),
                tooltip=[
                    alt.Tooltip(f"{x_col}:Q", title=x_col),
                    alt.Tooltip("_status:N", title="status"),
                ],
            )
        )
        layers.append(noface_marks)

    chart = (
        alt.layer(*layers)
        .properties(title=title, height=height)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

def render_metric_threshold_chart(
    df, *, x_col, y_col, title, y_label,
    thresholds, line_color, height=260,
):
    """Render one measured metric over time with config-driven cutoff line(s).

    Generic single-series timeline used for brightness and sharpness. Unlike
    the pose chart this does NOT mirror the threshold about zero — these metrics
    are one-sided (e.g. sharpness has a single floor). Pass one or more cutoffs
    as a list of (value, label) so brightness can show both a dark floor and a
    bright ceiling.

    line_color is passed explicitly so each metric is visually distinct from
    the yaw/pitch charts (which use Vega's default blue).
    """
    if x_col not in df.columns or y_col not in df.columns:
        st.info(f"No {y_col} data available.")
        return

    plot = df[[x_col, y_col]].copy()
    plot[x_col] = pd.to_numeric(plot[x_col], errors="coerce")
    plot[y_col] = pd.to_numeric(plot[y_col], errors="coerce")

    # Keep no-value frames (NaN y) so the line breaks at gaps instead of bridging
    # across them — same rationale as the pose chart. Only drop bad-x rows.
    plot = plot.dropna(subset=[x_col])

    if plot.empty or plot[y_col].notna().sum() == 0:
        st.info(f"No valid {y_col} values to plot.")
        return

    plot = plot.sort_values(x_col)

    # Y domain: span the data and every cutoff, with headroom.
    cut_vals = [float(v) for v, _ in thresholds]
    y_vals = plot[y_col].dropna().tolist() + cut_vals
    y_lo = min(y_vals)
    y_hi = max(y_vals)
    pad = max(1.0, (y_hi - y_lo) * 0.1)
    domain = [y_lo - pad, y_hi + pad]

    # Pin the x-domain to the data range. The dashed threshold rule has no x
    # encoding, so under .interactive() its unbounded x-extent would otherwise
    # stretch the axis past the data. A small pad keeps end samples off the edge.
    x_vals = plot[x_col].dropna()
    x_min = float(x_vals.min())
    x_max = float(x_vals.max())
    x_pad = max(0.5, (x_max - x_min) * 0.02)
    x_domain = [x_min - x_pad, x_max + x_pad]

    def _x(**kw):
        return alt.X(
            f"{x_col}:Q", title=x_col,
            scale=alt.Scale(domain=x_domain, nice=False), **kw,
        )

    line = (
        alt.Chart(plot)
        .mark_line()
        .encode(
            x=_x(),
            y=alt.Y(
                f"{y_col}:Q",
                title=y_label,
                scale=alt.Scale(domain=domain),
            ),
            color=alt.value(line_color),
            tooltip=[
                alt.Tooltip(f"{x_col}:Q", title=x_col),
                alt.Tooltip(f"{y_col}:Q", title=y_label, format=".1f"),
            ],
        )
    )

    # Isolated detected frames (both time-neighbours are gaps) get a dot so a
    # lone sample is not invisible — same treatment as the pose chart.
    y_isna = plot[y_col].isna()
    prev_isna = y_isna.shift(1, fill_value=True)
    next_isna = y_isna.shift(-1, fill_value=True)
    isolated = plot[(~y_isna) & prev_isna & next_isna]

    points = (
        alt.Chart(isolated)
        .mark_point(size=18, filled=True, opacity=1.0)
        .encode(
            x=_x(),
            y=alt.Y(f"{y_col}:Q", scale=alt.Scale(domain=domain)),
            color=alt.value(line_color),
            tooltip=[
                alt.Tooltip(f"{x_col}:Q", title=x_col),
                alt.Tooltip(f"{y_col}:Q", title=y_label, format=".1f"),
            ],
        )
    )

    threshold_df = pd.DataFrame({
        "_cut": cut_vals,
        "label": [lab for _, lab in thresholds],
    })

    rules = (
        alt.Chart(threshold_df)
        .mark_rule(strokeDash=[6, 4], color="#888888")
        .encode(
            y="_cut:Q",
            tooltip=[
                alt.Tooltip("label:N", title="cutoff"),
                alt.Tooltip("_cut:Q", title="value", format=".1f"),
            ],
        )
    )

    labels = (
        alt.Chart(threshold_df)
        .mark_text(align="left", dx=6, dy=-6, color="#888888")
        .encode(
            x=alt.value(5),
            y="_cut:Q",
            text="label:N",
        )
    )

    chart = (
        alt.layer(line, points, rules, labels)
        .properties(title=title, height=height)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

def render_eyes_chart(
    df, *, x_col, left_col, right_col, threshold,
    title="Eye blink score over time", height=260,
):
    """Plot left and right blink scores on ONE chart, each its own colour.

    Blink score is in [0, 1]; HIGH = eye closed. The dashed cutoff is the
    config blink_threshold: a line crossing AT/ABOVE it means that eye was
    judged closed on that frame.
    """
    have = [c for c in (left_col, right_col) if c in df.columns]
    if x_col not in df.columns or not have:
        st.info("No eye blink data available.")
        return

    keep = [x_col] + have
    plot = df[keep].copy()
    for c in keep:
        plot[c] = pd.to_numeric(plot[c], errors="coerce")
    plot = plot.dropna(subset=[x_col]).sort_values(x_col)

    if plot.empty or plot[have].notna().sum().sum() == 0:
        st.info("No valid eye blink values to plot.")
        return

    # Long form so a single colour encoding draws both eyes with a legend.
    rename = {left_col: "left eye", right_col: "right eye"}
    long = plot.melt(
        id_vars=[x_col],
        value_vars=have,
        var_name="eye",
        value_name="blink",
    )
    long["eye"] = long["eye"].map(rename).fillna(long["eye"])

    # Fixed colours, distinct from yaw/pitch blue and from brightness/sharpness.
    eye_scale = alt.Scale(
        domain=["left eye", "right eye"],
        range=["#9467bd", "#2ca02c"],   # purple, green
    )

    # Blink score is bounded [0,1]; keep the cutoff visible with headroom.
    y_hi = max(1.0, float(long["blink"].max() or 0), float(threshold)) * 1.05

    # Pin the x-domain to the actual data range. Without this, the dashed
    # threshold rule (which has no x encoding, so its x-extent is unbounded)
    # unions with the data under .interactive() and stretches the axis far past
    # the data (e.g. -10..75 for 0..48s of frames). A small pad keeps the first
    # and last samples off the very edge.
    x_min = float(long[x_col].min())
    x_max = float(long[x_col].max())
    x_pad = max(0.5, (x_max - x_min) * 0.02)
    x_domain = [x_min - x_pad, x_max + x_pad]
    x_enc = alt.X(
        f"{x_col}:Q",
        title=x_col,
        scale=alt.Scale(domain=x_domain, nice=False),
    )

    # --- legend on/off toggle (same approach as the occlusion chart) ---
    # Clicking a legend entry isolates that eye; non-selected fades. Altair 5
    # uses selection_point + add_params, Altair 4 selection_single +
    # add_selection; if neither exists, render without the toggle (no crash).
    eye_sel = None
    _use_opacity = False
    try:
        eye_sel = alt.selection_point(fields=["eye"], bind="legend")
        _use_opacity = True
    except AttributeError:
        try:
            eye_sel = alt.selection_single(fields=["eye"], bind="legend")
            _use_opacity = True
        except Exception:
            eye_sel = None
            _use_opacity = False

    _eye_enc = dict(
        x=x_enc,
        y=alt.Y(
            "blink:Q",
            title="blink score (high = closed)",
            scale=alt.Scale(domain=[0, y_hi]),
        ),
        color=alt.Color("eye:N", scale=eye_scale, title="eye"),
        tooltip=[
            alt.Tooltip(f"{x_col}:Q", title=x_col),
            alt.Tooltip("eye:N", title="eye"),
            alt.Tooltip("blink:Q", title="blink", format=".2f"),
        ],
    )
    if _use_opacity:
        _eye_enc["opacity"] = alt.condition(
            eye_sel, alt.value(1.0), alt.value(0.12))

    line = alt.Chart(long).mark_line().encode(**_eye_enc)
    if eye_sel is not None:
        if hasattr(line, "add_params"):
            line = line.add_params(eye_sel)
        elif hasattr(line, "add_selection"):
            line = line.add_selection(eye_sel)

    threshold_df = pd.DataFrame({
        "_cut": [float(threshold)],
        "label": [f"closed cutoff > {threshold:g}"],
    })

    rules = (
        alt.Chart(threshold_df)
        .mark_rule(strokeDash=[6, 4], color="#888888")
        .encode(
            y="_cut:Q",
            tooltip=[
                alt.Tooltip("label:N", title="cutoff"),
                alt.Tooltip("_cut:Q", title="value", format=".2f"),
            ],
        )
    )

    labels = (
        alt.Chart(threshold_df)
        .mark_text(align="left", dx=6, dy=-6, color="#888888")
        .encode(x=alt.value(5), y="_cut:Q", text="label:N")
    )

    chart = (
        alt.layer(line, rules, labels)
        .properties(title=title, height=height)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


# Per-region occlusion skin-ratio columns written by the pipeline (face_rgb.py).
# Order here is the legend order. Cheeks/forehead included so all measured
# regions can be toggled, even ones not in required_regions.
_OCC_REGION_COLS = [
    ("occ_forehead", "forehead"),
    ("occ_left_eye", "left eye"),
    ("occ_right_eye", "right eye"),
    ("occ_nose", "nose"),
    ("occ_mouth", "mouth"),
    ("occ_left_cheek", "left cheek"),
    ("occ_right_cheek", "right cheek"),
]


def render_occlusion_chart(df, *, x_col, threshold, title="Occlusion skin ratio over time", height=300):
    """Plot each face region's skin ratio over time on ONE chart.

    Each region is its own line. A clickable legend toggles individual regions
    on/off (click a legend entry to isolate it; shift-click to multi-select).
    The dashed cutoff is the config min_skin_ratio: a region's line crossing
    BELOW it means that region was judged occluded on that frame.

    Skin ratio is bounded [0, 1]. Gaps (non-frontal / no-face / skipped frames)
    break each line rather than bridging, same as the other quality charts.
    """
    have = [(col, lab) for col, lab in _OCC_REGION_COLS if col in df.columns]
    if x_col not in df.columns or not have:
        st.info("No occlusion data available.")
        return

    have_cols = [col for col, _ in have]
    keep = [x_col] + have_cols
    plot = df[keep].copy()
    for c in keep:
        plot[c] = pd.to_numeric(plot[c], errors="coerce")
    plot = plot.dropna(subset=[x_col]).sort_values(x_col)

    if plot.empty or plot[have_cols].notna().sum().sum() == 0:
        st.info("No valid occlusion values to plot.")
        return

    # Long form so one colour encoding draws every region with a legend.
    rename = {col: lab for col, lab in have}
    long = plot.melt(
        id_vars=[x_col],
        value_vars=have_cols,
        var_name="region",
        value_name="skin",
    )
    long["region"] = long["region"].map(rename).fillna(long["region"])

    # Stable colour per region (distinct ramp from yaw/pitch/brightness/eyes).
    region_domain = [lab for _, lab in have]
    region_range = [
        "#1f77b4", "#9467bd", "#2ca02c", "#d62728",
        "#ff7f0e", "#17becf", "#8c564b",
    ][:len(region_domain)]
    region_scale = alt.Scale(domain=region_domain, range=region_range)

    # Skin ratio is [0,1]; keep the cutoff visible with a little headroom.
    y_hi = max(1.0, float(long["skin"].max() or 0), float(threshold)) * 1.02

    x_min = float(long[x_col].min())
    x_max = float(long[x_col].max())
    x_pad = max(0.5, (x_max - x_min) * 0.02)
    x_domain = [x_min - x_pad, x_max + x_pad]
    x_enc = alt.X(f"{x_col}:Q", title=x_col,
                  scale=alt.Scale(domain=x_domain, nice=False))

    # --- the legend on/off toggle ---
    # A point selection bound to the legend: clicking a legend entry selects
    # that region; the opacity encoding fades everything not selected. With
    # nothing selected (default) all lines show at full opacity.
    #
    # Altair 5 uses selection_point + add_params; Altair 4 used selection_single
    # + add_selection. Detect which is available so the chart works on either,
    # and if neither is (very old Altair), fall back to a plain multi-line chart
    # with no toggle rather than crashing.
    region_sel = None
    _use_opacity = False
    try:
        region_sel = alt.selection_point(fields=["region"], bind="legend")
        _use_opacity = True
    except AttributeError:
        try:
            region_sel = alt.selection_single(fields=["region"], bind="legend")
            _use_opacity = True
        except Exception:
            region_sel = None
            _use_opacity = False

    base_enc = dict(
        x=x_enc,
        y=alt.Y("skin:Q", title="skin ratio (low = occluded)",
                scale=alt.Scale(domain=[0, y_hi])),
        color=alt.Color("region:N", scale=region_scale, title="region"),
        tooltip=[
            alt.Tooltip(f"{x_col}:Q", title=x_col),
            alt.Tooltip("region:N", title="region"),
            alt.Tooltip("skin:Q", title="skin ratio", format=".2f"),
        ],
    )
    if _use_opacity:
        base_enc["opacity"] = alt.condition(
            region_sel, alt.value(1.0), alt.value(0.12))

    line = alt.Chart(long).mark_line().encode(**base_enc)
    if region_sel is not None:
        # add_params (Altair 5) or add_selection (Altair 4), whichever exists.
        if hasattr(line, "add_params"):
            line = line.add_params(region_sel)
        elif hasattr(line, "add_selection"):
            line = line.add_selection(region_sel)

    threshold_df = pd.DataFrame({
        "_cut": [float(threshold)],
        "label": [f"occluded < {threshold:g}"],
    })

    rules = (
        alt.Chart(threshold_df)
        .mark_rule(strokeDash=[6, 4], color="#888888")
        .encode(
            y="_cut:Q",
            tooltip=[
                alt.Tooltip("label:N", title="cutoff"),
                alt.Tooltip("_cut:Q", title="value", format=".2f"),
            ],
        )
    )

    labels = (
        alt.Chart(threshold_df)
        .mark_text(align="left", dx=6, dy=-6, color="#888888")
        .encode(x=alt.value(5), y="_cut:Q", text="label:N")
    )

    chart = (
        alt.layer(line, rules, labels)
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


def _face_checks_cfg(config):
    return (
        config.get("face", {})
        .get("checks", {})
        or {}
    )


def get_brightness_thresholds(config):
    """Return (dark_threshold, bright_threshold) from config.

    These are the cutoffs check_lightpol uses: a frame is too dark below the
    dark threshold and too bright above the bright threshold.
    """
    b = _face_checks_cfg(config).get("brightness", {}) or {}
    dark = float(b.get("dark_threshold", 35))
    bright = float(b.get("bright_threshold", 200))
    return dark, bright


def get_blink_threshold(config):
    """Return the eyes-open blink cutoff from config.

    Blink score >= this -> eye counts as CLOSED (so the line crossing ABOVE the
    dashed cutoff means that eye failed the open check on that frame).
    """
    e = _face_checks_cfg(config).get("eyes_open", {}) or {}
    return float(e.get("blink_threshold", 0.5))


def get_blur_threshold(config):
    """Return the sharpness (Tenengrad) cutoff from config.

    Sharpness < this -> frame judged blurry (line BELOW the dashed cutoff).
    """
    b = _face_checks_cfg(config).get("blur", {}) or {}
    return float(b.get("threshold", 50.0))


def get_occlusion_min_skin(config):
    """Return the occlusion min_skin_ratio cutoff from config.

    A region's skin ratio < this -> that region judged occluded (line crossing
    BELOW the dashed cutoff means that region failed on that frame). Lives under
    face.occlusion.skin (NOT face.checks), so it is read directly here.
    """
    face = config.get("face", {}) if isinstance(config, dict) else {}
    occ = (face.get("occlusion", {}) or {}).get("skin", {}) or {}
    return float(occ.get("min_skin_ratio", 0.40))

def load_volunteer(reports_dir, vid):
    return {
        "overall": load_csv(_find(reports_dir, vid, "_overall")),
        "result": load_csv(_find(reports_dir, vid, "_result")),
        "detail": load_csv(_find(reports_dir, vid, "_detail")),
        "header": load_csv(_find(reports_dir, vid, "_detail_header")),
        # Palm is graded PER IMAGE: palm_<id>_overall.csv has ONE row per palm
        # file (data_type = pose key, e.g. palm_L_N), each with its own PASS/FAIL.
        # _find(..., "_overall") already matches face_*_overall; disambiguate by
        # the palm_ prefix so we load the palm file specifically.
        "palm_overall": load_csv(_find_palm_overall(reports_dir, vid)),
        # Palm per-(image,check) rows: palm_<id>_detail.csv. We use it to pull
        # just the check_palm_angle rows for the Palm tab.
        "palm_detail": load_csv(_find_palm_detail(reports_dir, vid)),
        # Walk writes ONE set of files PER VIEW (F and S), named
        # walk_<id>_walk_<view>_<kind>.csv. Collect both views into dicts keyed
        # by "F"/"S" so the Walk tab can show each camera.
        "walk_overall": _load_walk_by_view(reports_dir, vid, "overall"),
        "walk_result": _load_walk_by_view(reports_dir, vid, "result"),
        "walk_detail": _load_walk_by_view(reports_dir, vid, "detail"),
    }


def _load_walk_by_view(reports_dir, vid, kind):
    """Load walk_<id>_walk_<view>_<kind>.csv for each view -> {"F": df, "S": df}.

    kind is "overall" | "result" | "detail". Missing views are simply absent
    from the returned dict. Returns {} if none found.

    We match on "_walk_<view>_<kind>.csv" so we do NOT accidentally pick up the
    detail_header file when kind="detail" (that file ends _detail_header.csv).
    """
    sub = os.path.join(reports_dir, vid)
    out = {}
    for view in ("F", "S"):
        hits = glob.glob(os.path.join(sub, f"walk_*_walk_{view}_{kind}.csv"))
        # Exclude detail_header when asking for detail (glob for _detail matches
        # _detail_header too, since the suffix appears mid-name).
        if kind == "detail":
            hits = [h for h in hits if not h.endswith("_detail_header.csv")]
        if hits:
            out[view] = load_csv(hits[0])
    return out


def _find_palm_detail(reports_dir, vid):
    """Locate palm_<id>_detail.csv specifically (not face_/walk_ detail)."""
    sub = os.path.join(reports_dir, vid)
    hits = glob.glob(os.path.join(sub, "palm_*_detail.csv"))
    return hits[0] if hits else None


def _find_palm_overall(reports_dir, vid):
    """Locate palm_<id>_overall.csv specifically (not face_/walk_).

    _find("_overall") globs *_overall.csv and returns the first hit, which may be
    the face file. Palm needs its OWN file, so match the palm_ prefix explicitly.
    """
    sub = os.path.join(reports_dir, vid)
    hits = glob.glob(os.path.join(sub, "palm_*_overall.csv"))
    return hits[0] if hits else None
    
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


def expected_palm_keys(config):
    """The 10 expected palm file keys, in config order (palm.pattern_keys).

    Falls back to the L/R x N/RL/RR/PU/PD product if pattern_keys is absent, so
    the grid still renders on an older config.
    """
    palm = (config or {}).get("palm", {})
    keys = palm.get("pattern_keys")
    if keys:
        return list(keys)
    hands = palm.get("hands", ["L", "R"])
    poses = palm.get("poses", ["N", "RL", "RR", "PU", "PD"])
    return [f"palm_{h}_{p}" for h in hands for p in poses]


def _palm_pose_key(filename, expected_keys):
    """Extract the pose key (e.g. palm_L_N) from a palm filename.

    The overall/detail CSVs carry the pose in the FILENAME (096_palm_L_N.jpg),
    not in data_type (which is just "palm"). We match against the known expected
    keys so we never mis-parse an odd filename into a bogus pose. Returns None if
    no expected key is found in the name.
    """
    name = str(filename)
    for key in expected_keys:
        if key in name:
            return key
    return None


def render_palm_qc_summary(palm_df, config):
    """Per-image PASS/FAIL grid for all 10 palm files of one volunteer.

    palm_df is palm_<id>_overall.csv: ONE row per palm image, keyed by data_type
    (e.g. palm_L_N). We show EVERY expected pose (config.palm.pattern_keys), so a
    file with no row is surfaced as MISSING rather than silently dropped -- the
    reviewer sees all 10 slots at a glance, present or not. Overall palm status
    = worst of the present rows (FAIL if any file FAILs), with missing files
    flagged separately so a missing file does not read as a QC FAIL.
    """
    expected = expected_palm_keys(config)

    # Map present rows to their pose key. IMPORTANT: the real overall CSV writes
    # data_type="palm" for EVERY row (the pose identity lives in the FILENAME,
    # e.g. 096_palm_L_N.jpg), so we key on the filename first and only fall back
    # to data_type if a future CSV populates it with the pose key directly.
    present = {}
    if palm_df is not None and not palm_df.empty:
        for _, row in palm_df.iterrows():
            key = _palm_pose_key(row.get("filename", ""), expected)
            if key is None:
                dt = str(row.get("data_type", ""))
                key = dt if dt in expected else None
            if key is None:
                continue  # unrecognised row -> skip, don't invent a "palm" slot
            present[key] = {
                "status": str(row.get("final_status", "—")),
                "reason": str(row.get("reason", "")),
                "filename": str(row.get("filename", "")),
            }

    # Overall palm verdict from the PRESENT files only (missing handled apart).
    present_statuses = [v["status"].upper() for v in present.values()]
    if not present_statuses:
        palm_status = "INCOMPLETE"
    elif "FAIL" in present_statuses or "ERROR" in present_statuses:
        palm_status = "FAIL"
    elif "REVIEW" in present_statuses:
        palm_status = "REVIEW"
    else:
        palm_status = "PASS"

    st.markdown(
        f"**Palm QC** &nbsp; {status_pill(palm_status)}",
        unsafe_allow_html=True,
    )

    if palm_df is None:
        st.caption(
            "No palm overall CSV for this volunteer. "
            "Run `python run_palm.py data` (or `run_folder.py data`)."
        )
        return

    n_present = len(present_statuses)
    n_missing = len([k for k in expected if k not in present])
    st.caption(
        f"{n_present}/{len(expected)} palm files graded"
        + (f" · {n_missing} missing" if n_missing else "")
    )

    # One grid row per EXPECTED pose (present -> its verdict; absent -> MISSING).
    # We list ONLY the expected poses, so a row whose pose we could not identify
    # never leaks in as a stray slot.
    grid = []
    for key in expected:
        info = present.get(key)
        if info is None:
            grid.append({"palm file": key, "status": "MISSING", "reason": ""})
        else:
            grid.append({
                "palm file": key,
                "status": info["status"],
                "reason": info["reason"],
            })

    grid_df = pd.DataFrame(grid, columns=["palm file", "status", "reason"])
    n_rows = len(grid_df)
    table_height = 38 + n_rows * 35 + 3
    st.dataframe(
        _style_status(grid_df, "status"),
        use_container_width=True, hide_index=True, height=table_height,
    )


def render_palm_detail(palm_detail_df, config, *, check_name="check_palm_angle"):
    """Palm-angle detail table for one volunteer.

    Reads palm_<id>_detail.csv (one row per (image, check)) and shows ONLY the
    `check_palm_angle` rows, on the columns the reviewer cares about:
        volunteer_id, data_type, filename, check_name, status, reason
    One row per palm image that has an angle verdict, ordered by config pose
    order (palm.pattern_keys) so L/R and N/RL/RR/PU/PD read in a stable order.
    """
    if palm_detail_df is None or palm_detail_df.empty:
        st.info(
            "No palm detail CSV for this volunteer. "
            "Run `python run_palm.py data` (or `run_folder.py data`)."
        )
        return

    df = palm_detail_df
    if "check_name" not in df.columns:
        st.info("Palm detail CSV has no check_name column.")
        return

    angle = df[df["check_name"] == check_name].copy()
    if angle.empty:
        st.info(f"No {check_name} rows in this volunteer's palm detail.")
        return

    # Order rows by the config pose order. The pose lives in the FILENAME (the
    # data_type column is just "palm"), so derive the key from the filename.
    expected = expected_palm_keys(config)
    order = {k: i for i, k in enumerate(expected)}
    if "filename" in angle.columns:
        angle["_ord"] = angle["filename"].map(
            lambda fn: order.get(_palm_pose_key(fn, expected), len(order)))
        angle = angle.sort_values("_ord").drop(columns="_ord")

    cols = [c for c in ["volunteer_id", "data_type", "filename",
                        "check_name", "status", "reason"] if c in angle.columns]
    view = angle[cols]

    n_rows = len(view)
    table_height = 38 + n_rows * 35 + 3
    st.dataframe(
        _style_status(view, "status"),
        use_container_width=True, hide_index=True, height=table_height,
    )


def _walk_overall_verdict(overall_df):
    """(status, reason) from a single walk overall CSV (one OVERALL row)."""
    if overall_df is None or overall_df.empty:
        return None, None
    row = overall_df.iloc[0].to_dict()
    return row.get("final_status", "—"), row.get("reason", "")


def _walk_failed_checks(result_df):
    """List of check_name that FAILed, from a walk result CSV (per-check rows)."""
    if result_df is None or result_df.empty:
        return []
    if "final_status" not in result_df.columns or "check_name" not in result_df.columns:
        return []
    failed = result_df[result_df["final_status"].astype(str).str.upper() == "FAIL"]
    return [str(c) for c in failed["check_name"].tolist()]


def render_walk_qc_summary(walk_overall, walk_result):
    """Top-of-page Walk summary: a combined Walk QC pill, then per-view (F/S)
    verdicts with their failed checks.

    walk_overall / walk_result are {"F": df, "S": df} dicts (one entry per view
    actually present). The combined Walk QC status is the worst of the present
    views (FAIL if either camera fails), mirroring the Palm QC header. Under it,
    each view shows its own OVERALL pill and, when it did not PASS, the list of
    checks that FAILed (from that view's result CSV).
    """
    if not walk_overall:
        st.markdown(
            f"**Walk QC** &nbsp; {status_pill('INCOMPLETE')}",
            unsafe_allow_html=True,
        )
        st.caption(
            "No walk overall CSV for this volunteer. "
            "Run `python run_walk.py data` (or `run_folder.py data`)."
        )
        return

    # Combined verdict from the present views (worst wins).
    view_statuses = []
    for view in ("F", "S"):
        odf = walk_overall.get(view)
        if odf is not None:
            s, _ = _walk_overall_verdict(odf)
            view_statuses.append(str(s).upper())

    if "FAIL" in view_statuses or "ERROR" in view_statuses:
        combined = "FAIL"
    elif "REVIEW" in view_statuses:
        combined = "REVIEW"
    elif view_statuses:
        combined = "PASS"
    else:
        combined = "INCOMPLETE"

    st.markdown(
        f"**Walk QC** &nbsp; {status_pill(combined)}",
        unsafe_allow_html=True,
    )

    for view in ("F", "S"):
        odf = walk_overall.get(view)
        if odf is None:
            st.markdown(
                f"**Walk {view}** &nbsp; {status_pill('MISSING')}",
                unsafe_allow_html=True,
            )
            st.caption(f"No walk_{view} overall CSV for this volunteer.")
            continue

        status, reason = _walk_overall_verdict(odf)
        st.markdown(
            f"**Walk {view}** &nbsp; {status_pill(status)}",
            unsafe_allow_html=True,
        )

        failed = _walk_failed_checks(walk_result.get(view))
        if failed:
            st.caption("failed checks: " + ", ".join(failed))
        elif reason:
            st.caption(reason)


def render_walk_detail(walk_detail):
    """Part 2 of the Walk tab: detail results per view.

    walk_detail is {"F": df, "S": df}. Each detail CSV is one row per
    (frame, check). We show the reviewer-facing columns per view, F first.
    """
    if not walk_detail:
        st.info(
            "No walk detail CSV for this volunteer. "
            "Run `python run_walk.py data` (or `run_folder.py data`)."
        )
        return

    for view in ("F", "S"):
        det = walk_detail.get(view)
        if det is None or det.empty:
            continue
        st.markdown(f"**Walk {view} detail**")
        cols = [c for c in ["frame_index", "time", "check_level",
                            "check_name", "status", "reason"] if c in det.columns]
        view_df = det[cols] if cols else det
        # Detail is long (frames x checks); cap height and let it scroll.
        st.dataframe(
            _style_status(view_df, "status"),
            use_container_width=True, hide_index=True, height=400,
        )


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

    st.markdown("")
    render_palm_qc_summary(data.get("palm_overall"), load_config("config.yml"))

    st.markdown("")
    render_walk_qc_summary(data.get("walk_overall", {}),
                           data.get("walk_result", {}))

    st.divider()

    # Detailed results split into per-modality tabs. Face RGB keeps its full
    # existing view; Palm shows the check_palm_angle rows per image.
    face_tab, palm_tab, walk_tab = st.tabs(["Face RGB", "Palm angle", "Walk"])

    with face_tab:
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
            # Show every check row without an inner scrollbar. Streamlit caps a
            # dataframe at ~400px and makes it scroll; instead size the height to
            # the row count (≈35px/row + 38px header + a small pad) so the whole
            # table is visible at once however many checks there are.
            n_rows = len(styled)
            table_height = 38 + n_rows * 35 + 3
            st.dataframe(_style_status(styled, "status"),
                         use_container_width=True, hide_index=True,
                         height=table_height)
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

        # quality timelines (brightness, eyes-open, sharpness)
        if hdr is not None and not hdr.empty:
            cfg = load_config("config.yml")
            idx = "time" if "time" in hdr.columns else "frame_index"

            qplot = hdr.copy()
            for c in [idx, "brightness", "blink_left", "blink_right", "sharpness"]:
                if c in qplot.columns:
                    qplot[c] = pd.to_numeric(qplot[c], errors="coerce")
            # Occlusion per-region columns (written by face_rgb.py) coerced too.
            _occ_cols = [col for col, _ in _OCC_REGION_COLS if col in qplot.columns]
            for c in _occ_cols:
                qplot[c] = pd.to_numeric(qplot[c], errors="coerce")

            has_bright = "brightness" in qplot.columns and qplot["brightness"].notna().any()
            has_eyes = (
                {"blink_left", "blink_right"} & set(qplot.columns)
                and qplot[[c for c in ("blink_left", "blink_right") if c in qplot.columns]]
                .notna().any().any()
            )
            has_sharp = "sharpness" in qplot.columns and qplot["sharpness"].notna().any()
            has_occ = bool(_occ_cols) and qplot[_occ_cols].notna().any().any()

            if has_bright or has_eyes or has_sharp or has_occ:
                st.subheader("Frame quality over time")
                st.caption(
                    "Brightness, eye blink, sharpness, and per-region occlusion skin "
                    "ratio on sampled frontal frames. Dashed lines are the pass/fail "
                    "cutoffs read from config.yml. Click a legend entry on the "
                    "occlusion chart to isolate a region. Gaps are non-frontal / "
                    "no-face / skipped frames."
                )

            if has_bright:
                dark_th, bright_th = get_brightness_thresholds(cfg)
                render_metric_threshold_chart(
                    qplot,
                    x_col=idx,
                    y_col="brightness",
                    title="Brightness over time",
                    y_label="face brightness (mean V)",
                    thresholds=[
                        (dark_th, f"too dark < {dark_th:g}"),
                        (bright_th, f"too bright > {bright_th:g}"),
                    ],
                    line_color="#e6a817",   # amber — distinct from yaw/pitch blue
                    height=260,
                )

            if has_eyes:
                blink_th = get_blink_threshold(cfg)
                render_eyes_chart(
                    qplot,
                    x_col=idx,
                    left_col="blink_left",
                    right_col="blink_right",
                    threshold=blink_th,
                    title="Eye blink score over time (left vs right)",
                    height=260,
                )

            if has_sharp:
                blur_th = get_blur_threshold(cfg)
                render_metric_threshold_chart(
                    qplot,
                    x_col=idx,
                    y_col="sharpness",
                    title="Sharpness over time",
                    y_label="Tenengrad sharpness",
                    thresholds=[(blur_th, f"blurry < {blur_th:g}")],
                    line_color="#17a2a2",   # teal — distinct from the others
                    height=260,
                )

            if has_occ:
                occ_th = get_occlusion_min_skin(cfg)
                render_occlusion_chart(
                    qplot,
                    x_col=idx,
                    threshold=occ_th,
                    title="Occlusion skin ratio over time (per region)",
                    height=300,
                )

        # raw detail (collapsed)
        det = data["detail"]
        if det is not None and not det.empty:
            with st.expander("Raw per-frame detail"):
                st.dataframe(det, use_container_width=True, hide_index=True)

    with palm_tab:
        st.subheader("Palm angle per image")
        st.caption(
            "check_palm_angle verdict for each palm file, read from "
            "palm_<id>_detail.csv."
        )
        render_palm_detail(data.get("palm_detail"), load_config("config.yml"))

    with walk_tab:
        st.subheader("Walk detail")
        st.caption(
            "Per-frame, per-check verdicts read from "
            "walk_<id>_walk_<view>_detail.csv."
        )
        render_walk_detail(data.get("walk_detail", {}))


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


# ---------------------------------------------------------------------------
# Customer-facing summaries (the per-check CSVs run_folder.py writes at the
# reports ROOT, not inside a volunteer folder):
#   <reports>/face_summary.csv | palm_summary.csv | walk_summary.csv
#       one row per FAILED check; a clean video gets one PASS row.
#   <reports>/all_summary.csv  + all_summary.json
#       cross-modal FAIL/ERROR roll-up only.
# These views read those files directly — no per-volunteer folders involved.
# ---------------------------------------------------------------------------

# The per-modality summaries the dashboard knows how to open. Label -> filename.
MODALITY_SUMMARIES = {
    "Face": "face_summary.csv",
    "Palm": "palm_summary.csv",
    "Walk": "walk_summary.csv",
}


@st.cache_data(show_spinner=False)
def load_summary_csv(path):
    """Load a per-check summary CSV. Missing file -> None (not built yet)."""
    if not path or not os.path.exists(path):
        return None
    return pd.read_csv(path, dtype={"volunteer_id": str})


@st.cache_data(show_spinner=False)
def load_all_summary_json(path):
    """Load all_summary.json (list of problem-video objects). Missing -> None."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fail_only(df):
    """Keep only the FAIL/ERROR rows of a per-check summary (drop PASS rows)."""
    if df is None or df.empty or "overall_status" not in df.columns:
        return df
    keep = df["overall_status"].astype(str).str.upper().isin({"FAIL", "ERROR"})
    return df[keep]


def render_summaries(reports_dir):
    """One per-modality summary at a time: face / palm / walk.

    Shows the per-check rows (one row per failed check). A FAIL-only toggle
    hides the PASS placeholder rows so a reviewer sees just the problems.
    """
    st.subheader("Modality summary")

    which = st.radio("Modality", list(MODALITY_SUMMARIES), horizontal=True)
    path = os.path.join(reports_dir, MODALITY_SUMMARIES[which])
    df = load_summary_csv(path)

    if df is None:
        st.info(f"No `{MODALITY_SUMMARIES[which]}` under `{reports_dir}/` yet. "
                f"This modality's pipeline may not have run (palm/walk are not "
                f"built yet).")
        return
    if df.empty:
        st.warning(f"`{MODALITY_SUMMARIES[which]}` is empty.")
        return

    fail_only = st.checkbox("Show failures only", value=False)
    view = _fail_only(df) if fail_only else df

    # Headline counts: distinct videos by verdict (not row count, since a
    # failing video spans several rows).
    if "volunteer_id" in df.columns and "overall_status" in df.columns:
        per_video = df.drop_duplicates(subset=["volunteer_id"])
        n_total = per_video["volunteer_id"].nunique()
        vc = per_video["overall_status"].astype(str).str.upper().value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Videos", n_total)
        c2.metric("PASS", int(vc.get("PASS", 0)))
        c3.metric("FAIL", int(vc.get("FAIL", 0)))
        c4.metric("ERROR", int(vc.get("ERROR", 0)))

    st.caption(f"{len(view)} row(s)"
               + (" — failures only" if fail_only else ""))
    # use_container_width=True (not width="stretch", which the installed
    # Streamlit version rejects with a TypeError on a string width).
    st.dataframe(
        _style_status(view, "overall_status"),
        use_container_width=True, hide_index=True,
    )

    # Let the reviewer pull the exact file they're looking at.
    st.download_button(
        f"Download {MODALITY_SUMMARIES[which]}",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=MODALITY_SUMMARIES[which],
        mime="text/csv",
    )


def render_all_failures(reports_dir):
    """The cross-modal all_summary: FAIL/ERROR cases across every modality.

    Reads all_summary.csv for the flat per-check table (fail check + reason),
    and offers the nested all_summary.json for download / per-video drill-in.
    """
    st.subheader("All failures (every modality)")

    csv_path = os.path.join(reports_dir, "all_summary.csv")
    json_path = os.path.join(reports_dir, "all_summary.json")
    df = load_summary_csv(csv_path)

    if df is None:
        st.info("No `all_summary.csv` yet. Run a batch first "
                "(`python run_folder.py data`).")
        return
    if df.empty:
        st.success("No failures recorded — all_summary is empty.")
        return

    # Optional modality filter (the data_type column distinguishes them).
    if "data_type" in df.columns:
        mods = ["(all)"] + sorted(df["data_type"].dropna().unique().tolist())
        pick = st.selectbox("Modality", mods, index=0)
        if pick != "(all)":
            df = df[df["data_type"] == pick]

    # Counts: distinct problem videos, and how many checks failed in total.
    n_videos = (df.drop_duplicates(subset=["data_type", "volunteer_id"])
                  .shape[0]) if "volunteer_id" in df.columns else len(df)
    n_failed_checks = int(
        df["check_name"].astype(str).str.len().gt(0).sum()
    ) if "check_name" in df.columns else len(df)
    c1, c2 = st.columns(2)
    c1.metric("Problem videos", n_videos)
    c2.metric("Failed checks", n_failed_checks)

    st.caption("One row per failed check (fail check + reason). ERROR videos "
               "carry the error text as the reason.")
    # use_container_width=True — see render_summaries.
    st.dataframe(
        _style_status(df, "overall_status"),
        use_container_width=True, hide_index=True,
    )

    # Downloads: the flat CSV and the nested JSON, as written on disk.
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download all_summary.csv",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="all_summary.csv",
            mime="text/csv",
        )
    with d2:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                st.download_button(
                    "Download all_summary.json",
                    data=f.read().encode("utf-8"),
                    file_name="all_summary.json",
                    mime="application/json",
                )


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

    mode = st.sidebar.radio(
        "Mode", ["Single", "Compare", "Summaries", "All failures"]
    )

    # The summary views read root-level CSVs (face_summary.csv, all_summary.*)
    # and do NOT need per-volunteer folders, so handle them before the
    # volunteer-folder check that Single/Compare rely on.
    if mode == "Summaries":
        render_summaries(args.reports)
        return
    if mode == "All failures":
        render_all_failures(args.reports)
        return

    ids = list_volunteers(args.reports)
    if not ids:
        st.warning(f"No volunteer folders with an *_overall.csv under "
                   f"`{args.reports}/`.")
        st.stop()

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