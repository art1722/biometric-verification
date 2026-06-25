"""Palm angle check -- wrist roll/pitch within the spec's +/-45 deg tolerance.

Spec requirement (source of truth)
----------------------------------
Each hand is captured at 5 wrist angles: N (neutral), RL (roll-left),
RR (roll-right), PU (pitch-up), PD (pitch-down), with roll and pitch within
+/-45 degrees. This check measures the hand's roll and pitch from landmarks and
flags any image whose |roll| or |pitch| exceeds the configured maximum.

GATING -- this check is OFF by default
--------------------------------------
config: palm.angle.check_angle_enabled (default false). Per the project plan,
angle is a stretch / later check: the spec's per-pose angles (RL/RR/PU/PD) are
DELIBERATE captures, so a large roll/pitch is often CORRECT for that pose, not a
defect. The pipeline therefore decides whether to run this at all; the function
itself stays pure and always computes a value, but `check_angle_enabled` lets
the caller emit SKIP instead of a PASS/FAIL row until per-pose expected angles
are calibrated with the researcher (อ.เหมียว).

Coordinate space -- WORLD landmarks, not image landmarks
--------------------------------------------------------
This mirrors check_head_pose's lesson about coordinate space, taken one step
further. head_pose uses MediaPipe's NORMALIZED (0..1) face landmarks because
that is the space the researcher's notebook calibrated in. For the HAND, the
Tasks-API HandLandmarker also exposes WORLD landmarks (metric 3D, origin at the
hand centre), and hand_landmarker.py's own docstring states world coords are
"the cleanest signal for the roll/pitch angle check (image-space coords distort
with perspective)". So this check consumes HandResult.world_landmarks. Image/
pixel landmarks scale x by width and y by height and bend under perspective,
which would corrupt an angle; world coords do not.

Method (geometry, no model)
---------------------------
Build a hand reference frame from three stable keypoints -- WRIST (0),
INDEX_FINGER_MCP (5), PINKY_MCP (17) -- the base of the palm, which is rigid
relative to the wrist (finger curl does not move these much):

  - palm "up" axis    : wrist  -> midpoint(index_mcp, pinky_mcp)
                        (points from wrist toward the fingers, along the palm)
  - palm "across" axis : index_mcp -> pinky_mcp
                        (spans the knuckle line, left-right across the palm)
  - palm normal        : across x up   (out of the palm surface)

Then, in MediaPipe world axes (x right, y down, z toward camera):
  - roll  = rotation of the across-axis about the camera's view direction,
            i.e. how tilted the knuckle line is in the image plane:
            atan2(across.y, across.x)
  - pitch = how much the palm's up-axis tips toward/away from the camera
            (out of the frontal plane): atan2(up.z, up.y)

These are intentionally simple, monotonic readings (not a full solvePnP),
matching head_pose's "geometric, empirically-read" style. They give a stable,
signed magnitude suitable for a +/-45 threshold; exact per-pose calibration is a
[CONFIRM] item, which is why the check ships gated off.

Returns -- two-layer, like check_head_pose
------------------------------------------
calculate_palm_angles(world_landmarks) -> (success, info_dict)
    info_dict = {"roll": float, "pitch": float}  (degrees), or an "error" key.
check_palm_angle(world_landmarks, ...) -> (success, message)
    the pipeline-facing wrapper: applies the +/-max thresholds and formats a
    report message, mirroring the (success, message) contract every other palm
    check returns.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Base-of-palm keypoints (rigid relative to the wrist; finger curl barely moves
# them). Indices are the standard MediaPipe HandLandmark order, matching the
# HandLandmark enum in hand_landmarker.py.
_WRIST = 0
_INDEX_MCP = 5
_PINKY_MCP = 17


def _vec(a, b):
    """b - a for two world-landmark objects (.x/.y/.z, metres)."""
    return (b.x - a.x, b.y - a.y, b.z - a.z)


def _cross(u, v):
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _midpoint(a, b):
    """Synthetic landmark-like point at the midpoint of a and b."""
    class _P:
        __slots__ = ("x", "y", "z")
    p = _P()
    p.x = (a.x + b.x) / 2.0
    p.y = (a.y + b.y) / 2.0
    p.z = (a.z + b.z) / 2.0
    return p


def calculate_palm_angles(world_landmarks: Any):
    """Geometric roll/pitch (degrees) from WORLD hand landmarks.

    Args:
        world_landmarks: HandResult.world_landmarks -- an indexable sequence of
            21 landmark objects with .x/.y/.z in metres (origin at hand centre).

    Returns:
        (success, info_dict). On success info_dict has float "roll" and "pitch"
        in degrees. On failure (missing/short landmarks) success is False and
        info_dict has an "error" string. Never raises on bad input -- keeps the
        no-crash contract the other checks follow.
    """
    if world_landmarks is None:
        return (False, {"error": "No world landmarks provided"})

    try:
        n = len(world_landmarks)
    except TypeError:
        return (False, {"error": "world_landmarks is not indexable"})

    if n <= _PINKY_MCP:
        return (False, {"error": f"Expected 21 landmarks, got {n}"})

    try:
        wrist = world_landmarks[_WRIST]
        index_mcp = world_landmarks[_INDEX_MCP]
        pinky_mcp = world_landmarks[_PINKY_MCP]

        knuckle_mid = _midpoint(index_mcp, pinky_mcp)

        up = _vec(wrist, knuckle_mid)       # wrist -> fingers, along the palm
        across = _vec(index_mcp, pinky_mcp)  # knuckle line, across the palm
        _ = _cross(across, up)               # palm normal (reserved for future use)

        # roll: tilt of the knuckle line in the image plane (x right, y down).
        roll = math.degrees(math.atan2(across[1], across[0]))
        # Fold into [-90, 90]: the knuckle line is undirected (index<->pinky),
        # so a 180-degree flip is the same physical tilt. This keeps "level" near 0.
        if roll > 90:
            roll -= 180
        elif roll < -90:
            roll += 180

        # pitch: how far the palm's up-axis tips out of the frontal plane
        # toward/away from the camera (z toward camera, y down).
        pitch = math.degrees(math.atan2(up[2], up[1]))
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180

        return (True, {"roll": float(roll), "pitch": float(pitch)})

    except Exception as e:  # never crash a batch on one odd frame
        logger.debug("PALM_ANGLE | error: %s", e)
        return (False, {"error": f"Error computing palm angle: {e}"})


def check_palm_angle(
    world_landmarks: Any,
    *,
    max_abs_roll_deg: float = 45.0,
    max_abs_pitch_deg: float = 45.0,
):
    """Pipeline-facing wrapper: are |roll| and |pitch| within tolerance?

    Mirrors the (success, message) shape of check_palm_min_size and the face
    checks, so the palm pipeline appends this row identically. The caller is
    responsible for the OFF-by-default gate: only run this (vs emit SKIP) when
    config palm.angle.check_angle_enabled is true.

    Args:
        world_landmarks: HandResult.world_landmarks (21 metric 3D points).
        max_abs_roll_deg: spec palm.angle.max_abs_roll_deg (default 45).
        max_abs_pitch_deg: spec palm.angle.max_abs_pitch_deg (default 45).

    Returns:
        (success, message). Message reports measured roll/pitch and which bound
        (if any) was exceeded, e.g.
            "roll=12.3 pitch=-8.0 within +/-45/+/-45"
            "roll=58.1 > 45 (pitch=-3.2 ok)"
    """
    ok, info = calculate_palm_angles(world_landmarks)
    if not ok:
        # Could not measure -> report the reason; the pipeline decides whether a
        # non-measurable angle is FAIL or SKIP (consistent with how a no-hand
        # frame is handled upstream by detect_hand).
        return (False, info.get("error", "Could not compute palm angle"))

    roll = info["roll"]
    pitch = info["pitch"]
    roll_ok = abs(roll) <= max_abs_roll_deg
    pitch_ok = abs(pitch) <= max_abs_pitch_deg

    if roll_ok and pitch_ok:
        return (
            True,
            f"roll={roll:.1f} pitch={pitch:.1f} "
            f"within +/-{max_abs_roll_deg:g}/+/-{max_abs_pitch_deg:g}",
        )

    reasons = []
    if not roll_ok:
        reasons.append(f"roll={roll:.1f} > {max_abs_roll_deg:g}")
    else:
        reasons.append(f"roll={roll:.1f} ok")
    if not pitch_ok:
        reasons.append(f"pitch={pitch:.1f} > {max_abs_pitch_deg:g}")
    else:
        reasons.append(f"pitch={pitch:.1f} ok")
    return (False, " ".join(reasons))