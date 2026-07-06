"""Interactive 3D palm-angle debug visualizer.

This module is intentionally debug-only. The normal QC pipeline should not need
Plotly unless the runner is called with --angle-3d.

What it shows:
  - all 21 MediaPipe world landmarks,
  - hand skeleton,
  - the five v3 palm-plane landmarks: 0, 5, 9, 13, 17,
  - the fitted least-squares palm plane,
  - the oriented palm normal used for roll/pitch,
  - camera/world axes: x right, y down, z toward camera.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import plotly.graph_objects as go

from qc.checks.check_palm_angle import PLANE_LANDMARK_IDXS, calculate_palm_angles

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def landmarks_to_np(world_landmarks: Any) -> np.ndarray:
    """Convert MediaPipe world landmarks to an Nx3 float array."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in world_landmarks],
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
        fig.add_trace(go.Scatter3d(
            x=[end[0]], y=[end[1]], z=[end[2]],
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
        world_landmarks: MediaPipe HandResult.world_landmarks, length 21.
        angle_info: optional dict from calculate_palm_angles(). If omitted, this
            function calculates it from world_landmarks.
        title: title shown at the top of the HTML figure.

    Returns:
        plotly.graph_objects.Figure.

    Raises:
        ValueError when angle calculation fails or world landmarks are missing.
    """
    if world_landmarks is None:
        raise ValueError("no world landmarks available")

    if angle_info is None:
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

    # The five landmarks used by the plane fit.
    fig.add_trace(go.Scatter3d(
        x=plane_pts[:, 0],
        y=plane_pts[:, 1],
        z=plane_pts[:, 2],
        mode="markers+text",
        text=[str(i) for i in PLANE_LANDMARK_IDXS],
        textposition="bottom center",
        marker=dict(size=8),
        name="plane-fit landmarks",
    ))

    # Palm plane patch.
    fig.add_trace(go.Mesh3d(
        x=plane[:, 0],
        y=plane[:, 1],
        z=plane[:, 2],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        opacity=0.28,
        name="least-squares palm plane",
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
        sizeref=arrow_len,
        anchor="tail",
        name="oriented plane normal",
    ))

    _add_camera_axes(fig, center, arrow_len * 0.9)

    fig.update_layout(
        title=f"{title}<br>roll={roll:+.1f}°, pitch={pitch:+.1f}°",
        scene=dict(
            xaxis_title="x right",
            yaxis_title="y down",
            zaxis_title="z toward camera",
            aspectmode="data",
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
