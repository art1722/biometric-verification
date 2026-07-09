"""Interactive 3D palm-angle debug visualizer.

This module is intentionally debug-only. The normal QC pipeline should not need
Plotly unless the runner is called with --angle-3d.

IMPORTANT (2026-07-09): the palm-angle math now runs on MediaPipe NORMALIZED
landmarks (HandResult.landmarks_norm, wrist-origin z, smaller z = closer), NOT
world_landmarks. So this visualizer must be fed the SAME normalized landmarks,
and it computes/plots everything in that same normalized space -- otherwise the
plane, the normal arrow, and the roll/pitch title disagree with the CSV that the
pipeline reports. The caller (run_palm.py) passes landmarks_norm.

The parameter is still named `world_landmarks` for backwards-compat with older
callers, but any list of objects exposing .x/.y/.z works; correctness requires
the caller to pass the SAME landmarks the angle CSV was computed from
(landmarks_norm).

What it shows:
  - all 21 landmarks (normalized space),
  - hand skeleton,
  - the five palm-reference landmarks: 0, 5, 9, 13, 17,
  - the fitted reference palm plane (visual aid),
  - the oriented palm normal used for roll/pitch,
  - camera axes: x right, y down, z toward camera (normalized-z convention).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import plotly.graph_objects as go

from qc.checks.check_palm_angle import PLANE_LANDMARK_IDXS, calculate_palm_angles
import json

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def landmarks_to_np(landmarks: Any) -> np.ndarray:
    """Convert a MediaPipe landmark list to an Nx3 float array.

    Source-agnostic: reads .x/.y/.z, so it accepts either normalized landmarks
    (what the angle math and this visualizer now use) or world landmarks. The
    caller decides which; for consistency with the reported CSV, pass the
    NORMALIZED landmarks.
    """
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float64,
    )


def make_plane_patch(plane_pts: np.ndarray, scale: float = 1.35) -> np.ndarray:
    """Return four 3D corners of a small plane patch spanning the fitted palm plane."""
    center = plane_pts.mean(axis=0)
    centered = plane_pts - center

    _, _, vt = np.linalg.svd(centered)
    u = vt[0]
    v = vt[1]

    radius = float(np.max(np.linalg.norm(centered, axis=1))) * scale
    if radius <= 0:
        radius = 0.03

    return np.array([
        center - radius * u - radius * v,
        center + radius * u - radius * v,
        center + radius * u + radius * v,
        center - radius * u + radius * v,
    ])


# Front-on camera looking from the +y side.
# This makes +y ("y down") protrude out of the screen toward the viewer.
# Keep up=-z so x still appears to the right in the initial Plotly view.
NORMAL_POSITION_CAMERA = dict(
    # Almost front-on from +y.
    # Tiny x/z offsets avoid Plotly choosing the opposite 180° screen orientation
    # after user orbit/reset.
    eye=dict(x=0.001, y=+2.2, z=0.001),
    up=dict(x=0.0, y=0.0, z=-1.0),
    center=dict(x=0.0, y=0.0, z=0.0),
)



def _subdivide_plane(corners: np.ndarray, n: int = 24):
    """Triangulate a quad plane into an n x n grid so a per-vertex colour
    gradient interpolates SMOOTHLY across it (a 2-triangle quad would only
    shade the 4 corners). Returns (verts Nx3, i, j, k, t) where t in [0,1] is
    the bilinear param along corner0->corner1 edge -- unused here but handy.

    corners order (from make_plane_patch): c0, c1, c2, c3 counter-clockwise,
    with edge c0->c1 = +u and edge c0->c3 = +v.
    """
    c0, c1, c2, c3 = corners
    u_vec = c1 - c0
    v_vec = c3 - c0

    us = np.linspace(0.0, 1.0, n + 1)
    vs = np.linspace(0.0, 1.0, n + 1)
    verts = []
    for vv in vs:
        for uu in us:
            verts.append(c0 + uu * u_vec + vv * v_vec)
    verts = np.array(verts, dtype=np.float64)

    stride = n + 1
    i_idx, j_idx, k_idx = [], [], []
    for row in range(n):
        for col in range(n):
            a = row * stride + col
            b = a + 1
            c = a + stride
            d = c + 1
            # two triangles per cell: (a,b,d) and (a,d,c)
            i_idx += [a, a]
            j_idx += [b, d]
            k_idx += [d, c]
    return verts, i_idx, j_idx, k_idx


def _add_line(fig: go.Figure, p0, p1, *, name: str, width: int = 5, showlegend: bool = False):
    fig.add_trace(go.Scatter3d(
        x=[p0[0], p1[0]],
        y=[p0[1], p1[1]],
        z=[p0[2], p1[2]],
        mode="lines",
        line=dict(width=width),
        name=name,
        showlegend=showlegend,
    ))


def _add_camera_axes(fig: go.Figure, origin: np.ndarray, length: float):
    """Draw the coordinate convention used by the palm-angle code."""
    axes = [
        (np.array([length, 0.0, 0.0]), "x right"),
        (np.array([0.0, length, 0.0]), "y down"),
        (np.array([0.0, 0.0, length]), "z toward camera"),
    ]
    for vec, label in axes:
        end = origin + vec
        _add_line(fig, origin, end, name=label, width=7, showlegend=True)
        label_pos = origin + vec * 1.25
        fig.add_trace(go.Scatter3d(
            x=[label_pos[0]], y=[label_pos[1]], z=[label_pos[2]],
            mode="text",
            text=[label],
            textposition="top center",
            showlegend=False,
        ))


def build_palm_angle_figure(
    world_landmarks: Any,
    angle_info: Optional[dict] = None,
    *,
    title: str = "Palm angle debug",
) -> go.Figure:
    """Build an interactive Plotly figure for one detected hand.

    Args:
        world_landmarks: the landmark list to PLOT. Now expected to be the
            NORMALIZED landmarks (HandResult.landmarks_norm) so the drawn plane,
            normal, and roll/pitch title match the pipeline's reported CSV.
            (Name kept for backwards-compat; any .x/.y/.z list works.)
        angle_info: optional dict from calculate_palm_angles(). If omitted, this
            function recomputes it FROM THE SAME landmarks passed here, so the
            drawing and the numbers always agree. Pass the pipeline's angle_info
            when you have it to guarantee identical numbers to the CSV.
        title: title shown at the top of the HTML figure.

    Returns:
        plotly.graph_objects.Figure.

    Raises:
        ValueError when angle calculation fails or landmarks are missing.

    Note on normalized space: MediaPipe normalized z has a much smaller range
    than x/y (it is a relative depth), so with aspectmode="data" the palm plane
    reads as a thin sheet -- that is faithful, not a bug. The tilt is still
    visible via the plane's depth-gradient shading and the normal cone.
    """
    if world_landmarks is None:
        raise ValueError("no landmarks available")

    if angle_info is None:
        # Recompute from the SAME landmarks we are about to plot, so the arrow
        # and title can never disagree with the drawing. No handedness is passed
        # here (debug view): the roll MAGNITUDE and the geometry are correct; the
        # roll SIGN shown may be the left-hand-style sign. For an exact CSV match
        # (correct sign), pass the pipeline's angle_info in.
        ok, angle_info = calculate_palm_angles(world_landmarks)
        if not ok:
            raise ValueError(angle_info.get("error", "could not compute palm angle"))

    pts = landmarks_to_np(world_landmarks)
    if pts.shape[0] <= max(PLANE_LANDMARK_IDXS):
        raise ValueError(f"expected 21 landmarks, got {pts.shape[0]}")

    plane_pts = pts[list(PLANE_LANDMARK_IDXS)]
    plane = make_plane_patch(plane_pts)

    normal = np.array(angle_info["normal"], dtype=np.float64)
    roll = float(angle_info["roll"])
    pitch = float(angle_info["pitch"])

    center = plane_pts.mean(axis=0)
    hand_span = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    arrow_len = max(hand_span * 0.6, 0.03)

    fig = go.Figure()

    # All 21 world landmarks.
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers+text",
        text=[str(i) for i in range(len(pts))],
        textposition="top center",
        marker=dict(size=4),
        name="21 world landmarks",
    ))

    # Hand skeleton.
    for a, b in HAND_CONNECTIONS:
        _add_line(fig, pts[a], pts[b], name="hand skeleton", width=4)

    # The landmarks used by the angle/reference geometry.
    fig.add_trace(go.Scatter3d(
        x=plane_pts[:, 0],
        y=plane_pts[:, 1],
        z=plane_pts[:, 2],
        mode="markers+text",
        text=[str(i) for i in PLANE_LANDMARK_IDXS],
        textposition="bottom center",
        marker=dict(size=8),
        name="angle-reference landmarks",
    ))

    # Palm plane patch -- subdivided so the purple gradient interpolates
    # smoothly. Shade by SIGNED HEIGHT ALONG THE PLANE NORMAL: every vertex of a
    # flat plane has height ~0 by construction, so instead we project each
    # vertex onto the normal RELATIVE TO the palm centroid of all 21 points.
    # This makes the pale->dark ramp read as a depth cue (which side of the palm
    # tilts toward vs away from the camera along the fitted normal).
    verts, pi, pj, pk = _subdivide_plane(plane, n=24)
    # height of each plane vertex along the oriented normal, vs the hand centroid
    hand_centroid = pts.mean(axis=0)
    intensity = (verts - hand_centroid) @ normal   # signed metres along normal
    # Pale -> dark purple ramp (light lavender to deep violet).
    purple_scale = [
        [0.0, "rgb(243, 235, 250)"],   # very pale lavender
        [0.5, "rgb(179, 136, 220)"],   # mid purple
        [1.0, "rgb(90, 40, 140)"],     # deep violet
    ]
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=pi, j=pj, k=pk,
        intensity=intensity,
        intensitymode="vertex",
        colorscale=purple_scale,
        showscale=False,
        opacity=0.45,
        flatshading=False,
        name="reference palm plane (visual aid)",
        hoverinfo="skip",
    ))

    # Plane normal as a 3D cone/quiver arrow.
    fig.add_trace(go.Cone(
        x=[center[0]],
        y=[center[1]],
        z=[center[2]],
        u=[normal[0] * arrow_len],
        v=[normal[1] * arrow_len],
        w=[normal[2] * arrow_len],
        sizemode="absolute",
        sizeref=arrow_len * 0.35,
        anchor="tail",
        name="plane normal cone",
        showlegend=True,
        visible="legendonly",
        showscale=False,   # hide meaningless green/pink cone colorbar
        opacity=0.75,
    ))


    _add_camera_axes(fig, center, arrow_len * 0.9)

    fig.update_layout(
        title=f"{title}<br>roll={roll:+.1f}°, pitch={pitch:+.1f}°",
        scene=dict(
            xaxis_title="x right",
            yaxis_title="y down",
            zaxis_title="z toward camera",
            aspectmode="data",
            camera=NORMAL_POSITION_CAMERA,
        ),
        margin=dict(l=0, r=0, b=0, t=70),
    )
    return fig


def save_palm_angle_debug_html(
    world_landmarks: Any,
    out_path: str,
    angle_info: Optional[dict] = None,
    *,
    title: str = "Palm angle debug",
) -> str:
    """Write one self-contained interactive HTML debug file and return its path."""
    fig = build_palm_angle_figure(world_landmarks, angle_info, title=title)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    return out_path


def plot_palm_angle_debug(
    world_landmarks: Any,
    angle_info: Optional[dict] = None,
    *,
    title: str = "Palm angle debug",
):
    """Open the interactive figure in the default browser for manual debugging."""
    fig = build_palm_angle_figure(world_landmarks, angle_info, title=title)
    fig.show()
    return fig


# ---------------------------------------------------------------------------
# Multi-tab HTML: one file, one clickable tab per hand/pose
# ---------------------------------------------------------------------------

_TABS_CSS = """
:root { --accent:#5a288c; --accent-pale:#f3ebfa; --border:#d9c8ec; }
* { box-sizing:border-box; }
body { margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
       color:#222; background:#fff; }
.tabbar { display:flex; flex-wrap:wrap; gap:4px; padding:10px 12px 0;
          border-bottom:2px solid var(--border); position:sticky; top:0;
          background:#fff; z-index:5; }
.tabbtn { border:1px solid var(--border); border-bottom:none;
          background:var(--accent-pale); color:var(--accent);
          padding:7px 14px; border-radius:8px 8px 0 0; cursor:pointer;
          font-size:14px; font-weight:600; }
.tabbtn:hover { background:#e7d7f6; }
.tabbtn.active { background:var(--accent); color:#fff; }
.tabbtn .sub { font-weight:400; opacity:.85; margin-left:6px; font-size:12px; }

:root { --accent:#5a288c; --accent-pale:#f3ebfa; --border:#d9c8ec; }
* { box-sizing:border-box; }
body { margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
       color:#222; background:#fff; }
.tabbar { display:flex; flex-wrap:wrap; gap:4px; padding:10px 12px 0;
          border-bottom:2px solid var(--border); position:sticky; top:0;
          background:#fff; z-index:5; }
.tabbtn { border:1px solid var(--border); border-bottom:none;
          background:var(--accent-pale); color:var(--accent);
          padding:7px 14px; border-radius:8px 8px 0 0; cursor:pointer;
          font-size:14px; font-weight:600; }
.tabbtn:hover { background:#e7d7f6; }
.tabbtn.active { background:var(--accent); color:#fff; }
.tabbtn .sub { font-weight:400; opacity:.85; margin-left:6px; font-size:12px; }

.resetbtn { margin-left:auto; border:1px solid var(--border);
            background:#fff; color:var(--accent); cursor:pointer;
            padding:7px 12px; border-radius:8px 8px 0 0;
            font-size:13px; font-weight:600; }
.resetbtn:hover { background:var(--accent-pale); }

.panel { display:none; padding:0; }
.panel.active { display:block; }
.plotwrap { width:100%; height:82vh; }
.hint { font-size:12px; color:#666; padding:6px 14px; }

.panel { display:none; padding:0; }
.panel.active { display:block; }
.plotwrap { width:100%; height:82vh; }
.hint { font-size:12px; color:#666; padding:6px 14px; }
"""

_TABS_JS = """
function cloneNormalCamera(){
  return JSON.parse(JSON.stringify(NORMAL_CAMERA));
}

function getPlotInPanel(panel){
  if (!panel) return null;
  return panel.querySelector('.js-plotly-plot') || panel.querySelector('.plotly-graph-div');
}

function cameraRelayoutUpdate(){
  const c = cloneNormalCamera();

  return {
    'scene.camera.eye': c.eye,
    'scene.camera.up': c.up,
    'scene.camera.center': c.center
  };
}

function resetOneCamera(gd){
  if (!window.Plotly || !gd) return;

  Plotly.Plots.resize(gd);
  window.requestAnimationFrame(function(){
    Plotly.relayout(gd, cameraRelayoutUpdate()).then(function(){
      Plotly.Plots.resize(gd);
    });
  });
}

function showTab(idx){
  document.querySelectorAll('.tabbtn').forEach((b,i)=>
     b.classList.toggle('active', i===idx));
  document.querySelectorAll('.panel').forEach((p,i)=>
     p.classList.toggle('active', i===idx));

  const panel = document.querySelectorAll('.panel')[idx];
  const gd = getPlotInPanel(panel);

  if (gd && window.Plotly) {
    Plotly.Plots.resize(gd);

    if (panel.dataset.needsCameraReset === "1") {
      setTimeout(function(){
        resetOneCamera(gd);
        panel.dataset.needsCameraReset = "0";
      }, 80);
    }
  }
}

function resetAllCameras(){
  if (!window.Plotly || typeof NORMAL_CAMERA === 'undefined') {
    console.warn("Plotly or NORMAL_CAMERA is missing");
    return;
  }

  const panels = document.querySelectorAll('.panel');

  panels.forEach(function(panel){
    panel.dataset.needsCameraReset = "1";
  });

  const activePanel = document.querySelector('.panel.active');
  const gd = getPlotInPanel(activePanel);

  if (gd) {
    resetOneCamera(gd);
    activePanel.dataset.needsCameraReset = "0";
  }
}

document.addEventListener('DOMContentLoaded', function(){
  document.querySelectorAll('.panel').forEach(function(panel){
    panel.dataset.needsCameraReset = "0";
  });
  showTab(0);
});
"""


def save_palm_angle_debug_tabs_html(
    entries: list,
    out_path: str,
    *,
    page_title: str = "Palm angle 3D debug",
) -> str:
    """Write ONE self-contained HTML file with a clickable tab per entry.

    Args:
        entries: list of dicts, each:
            {
              "label": str,                    # short tab label, e.g. "L / N"
              "world_landmarks": <21 landmarks>,
              "angle_info": dict | None,       # from calculate_palm_angles; if
                                               # None it is computed here
              "title": str | None,             # per-figure title (optional)
            }
            Entries whose angle cannot be computed are skipped with a note tab.
        out_path: output .html path.
        page_title: browser tab / page heading.

    Returns:
        out_path.

    Notes:
        - Only the FIRST embedded figure includes plotly.js (via CDN); the rest
          are injected as bare <div>s that reuse the already-loaded library.
          This keeps the file small even with 10 hands.
        - Each figure opens in the front-on NORMAL_POSITION_CAMERA view.
    """
    import plotly.io as pio

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    panels_html = []
    tabs_html = []
    first = True
    for idx, e in enumerate(entries):
        label = e.get("label", f"tab {idx+1}")
        title = e.get("title") or label
        wl = e.get("world_landmarks")
        ainfo = e.get("angle_info")

        # Build the figure; on failure emit a note panel instead of crashing.
        try:
            fig = build_palm_angle_figure(wl, ainfo, title=title)
            roll = fig.layout.title.text  # already contains roll/pitch line
            include_js = "cdn" if first else False
            div = pio.to_html(
                fig, include_plotlyjs=include_js, full_html=False,
                default_height="82vh", config={"displaylogo": False},
            )
            body = f'<div class="plotwrap">{div}</div>'
            sub = ""
        except Exception as ex:
            body = (f'<div class="hint">Could not render this hand: '
                    f'{type(ex).__name__}: {ex}</div>')
            sub = "n/a"

        active = " active" if first else ""
        tabs_html.append(
            f'<button class="tabbtn{active}" onclick="showTab({idx})">'
            f'{label}<span class="sub">{sub}</span></button>'
        )
        panels_html.append(f'<div class="panel{active}">{body}</div>')
        first = False

    if not entries:
        tabs_html.append('<button class="tabbtn active">no hands</button>')
        panels_html.append('<div class="panel active"><div class="hint">'
                            'No detectable hands were provided.</div></div>')

    normal_camera_js = json.dumps(NORMAL_POSITION_CAMERA)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{page_title}</title>
<style>{_TABS_CSS}</style>
</head>
<body>
<div class="tabbar">{''.join(tabs_html)}<button class="resetbtn" onclick="resetAllCameras()">Reset view</button></div>
<div class="hint">Front-on start view (y-down toward you). Drag to orbit; double-click to reset.</div>
{''.join(panels_html)}
<script>
const NORMAL_CAMERA = {normal_camera_js};
{_TABS_JS}
</script>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path