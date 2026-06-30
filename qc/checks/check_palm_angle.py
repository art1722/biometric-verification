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


def _normalize(v):
    """Unit vector of a 3-tuple, or None if it has zero length."""
    mag = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if mag == 0.0:
        return None
    return (v[0] / mag, v[1] / mag, v[2] / mag)


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

    Method -- palm NORMAL via cross product, then decoupled angles
    --------------------------------------------------------------
    Adapted from the researcher's face-angle reference (eye/mouth/vertical ->
    face normal -> yaw/pitch). The earlier version of this function read roll
    from `across` (x,y only) and pitch from `up` (y,z only), each DROPPING a
    whole axis -- so a hand rotated on both axes at once had its two readings
    BLEED into each other (a pure roll shifted the pitch number and vice versa).

    This version builds ONE palm normal = normalize(cross(across, up)) and reads
    both angles from a consistent frame:

      - pitch: how far the palm normal tips up/down out of the frontal plane,
               measured against its FULL horizontal projection:
                   pitch = atan2(normal.y, sqrt(normal.x^2 + normal.z^2))
               The sqrt(x^2+z^2) denominator is the decoupling fix: pitch stays
               correct even when the palm is also rolled (the reference's trick).
      - roll : in-plane tilt of the knuckle line. Kept as the angle of `across`
               in the image plane, atan2(across.y, across.x), which is the
               rotation about the camera view direction (independent of pitch).

    NOTE on signs: switching to the normal improves the MAGNITUDE/decoupling of
    pitch, but the SIGN per pose still depends on handedness + axis orientation,
    so the L/R sign-calibration step (check_palm_pose / hand_sign_overrides) is
    unchanged and still required.

    Args:
        world_landmarks: HandResult.world_landmarks -- an indexable sequence of
            21 landmark objects with .x/.y/.z in metres (origin at hand centre).

    Returns:
        (success, info_dict). On success info_dict has float "roll" and "pitch"
        in degrees (and "normal" as a 3-tuple for debugging/overlay). On failure
        success is False and info_dict has an "error" string. Never raises.
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

        up = _vec(wrist, knuckle_mid)        # wrist -> fingers, along the palm
        across = _vec(index_mcp, pinky_mcp)  # knuckle line, across the palm

        # Palm normal = across x up, normalized. This is the single consistent
        # frame both angles are read from (the reference's cross-vector method).
        normal = _normalize(_cross(across, up))
        if normal is None:
            return (False, {"error": "degenerate palm geometry (zero-length normal)"})

        nx, ny, nz = normal

        # pitch: normal's elevation out of the frontal plane, measured against
        # the FULL horizontal projection sqrt(nx^2+nz^2) -> decoupled from roll.
        pitch = math.degrees(math.atan2(ny, math.sqrt(nx * nx + nz * nz)))

        # roll: in-plane tilt of the knuckle line about the view direction.
        roll = math.degrees(math.atan2(across[1], across[0]))
        # Fold into [-90, 90]: the knuckle line is undirected (index<->pinky),
        # so a 180-degree flip is the same physical tilt; keeps "level" near 0.
        if roll > 90:
            roll -= 180
        elif roll < -90:
            roll += 180

        return (True, {"roll": float(roll), "pitch": float(pitch),
                       "normal": (float(nx), float(ny), float(nz))})

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

# ---------------------------------------------------------------------------
# Per-pose, per-hand directional validation (check_palm_pose)
# ---------------------------------------------------------------------------
# The symmetric check_palm_angle above only enforces |roll|,|pitch| <= 45 -- it
# CANNOT tell a correctly-rolled RL (negative roll) from a wrong-way RL (positive
# roll), so it is unsuitable for validating the 5 deliberate poses. This function
# adds the directional rule the spec actually states (doc lines 36-47):
#
#   pose  spec wording                              expected band (deg)
#   N     no rotation, parallel to camera           roll in [-N_tol, +N_tol],
#                                                    pitch in [-N_tol, +N_tol]
#   RL    roll not exceeding -45                     roll in [-45, -min_rot]
#   RR    roll not exceeding +45                     roll in [+min_rot, +45]
#   PU    pitch not exceeding -45                    pitch in [-45, -min_rot]
#   PD    pitch not exceeding +45                    pitch in [+min_rot, +45]
#
# min_rot is the "must actually rotate" floor (a deliberate pose with ~0 roll is
# a defect -> FAIL). For the rotated poses the OTHER axis should stay near zero
# (an RL should not also be pitched), enforced by an off-axis tolerance.
#
# !!! HANDEDNESS / SIGN CALIBRATION -- READ BEFORE TRUSTING THE SIGNS !!!
# The spec defines RL/RR relative to the VOLUNTEER's hand, and the motion is
# MIRRORED between left and right hands (L-RL tilts the pinky edge toward the
# sensor; R-RL tilts the THUMB edge). The roll SIGN that calculate_palm_angles
# produces for a "correct RL" therefore MAY DIFFER between L and R, because the
# across-axis (index_mcp -> pinky_mcp) flips direction with handedness. The signs
# encoded in _POSE_BANDS below are the SPEC's nominal signs and are a STARTING
# HYPOTHESIS ONLY. They MUST be calibrated: run one known-correct image per
# (hand, pose) through calculate_palm_angles, observe the actual sign, and flip
# the affected _POSE_BANDS / config rows if the code's convention disagrees.
# Until that calibration is signed off (อ.เหมียว), treat FAILs from this check as
# advisory.  [CONFIRM]

# Axis each pose acts on.
_POSE_AXIS = {"N": None, "RL": "roll", "RR": "roll", "PU": "pitch", "PD": "pitch"}

# Nominal expected SIGN per pose (spec). +1 expects positive, -1 negative.
# These are the calibration-pending hypothesis (see warning above).
_POSE_SIGN = {"RL": -1, "RR": +1, "PU": -1, "PD": +1}


def _band_for_pose(
    hand: str,
    pose: str,
    *,
    max_abs_deg: float,
    min_rotation_deg: float,
    neutral_tol_deg: float,
    hand_sign_overrides: Optional[dict] = None,
):
    """Return (axis, lo, hi) the active-axis angle must fall within for this
    (hand, pose), or (None, None, None) for N (handled separately).

    hand_sign_overrides lets config flip the expected sign for a specific
    (hand, pose) after calibration, e.g. {"R": {"RL": +1}} if a right-hand RL
    measures positive in this code's convention. Without overrides the nominal
    _POSE_SIGN is used.
    """
    if pose == "N":
        return (None, None, None)
    axis = _POSE_AXIS.get(pose)
    if axis is None:
        return (None, None, None)

    sign = _POSE_SIGN.get(pose, +1)
    if hand_sign_overrides:
        ov = hand_sign_overrides.get(hand, {})
        if pose in ov:
            sign = ov[pose]

    if sign < 0:
        return (axis, -max_abs_deg, -min_rotation_deg)   # e.g. [-45, -10]
    return (axis, +min_rotation_deg, +max_abs_deg)        # e.g. [+10, +45]


def check_palm_pose(
    world_landmarks: Any,
    hand: str,
    pose: str,
    *,
    max_abs_deg: float = 45.0,
    min_rotation_deg: float = 10.0,
    neutral_tol_deg: float = 10.0,
    off_axis_tol_deg: float = 20.0,
    hand_sign_overrides: Optional[dict] = None,
):
    """Validate a palm image against the DIRECTIONAL per-pose spec.

    Args:
        world_landmarks: HandResult.world_landmarks (21 metric 3D points).
        hand: "L" or "R" (parsed from filename). Selects the sign convention.
        pose: "N" | "RL" | "RR" | "PU" | "PD".
        max_abs_deg: upper magnitude bound (spec: 45).
        min_rotation_deg: a deliberate pose must rotate AT LEAST this much, else
            it is treated as not-actually-posed -> FAIL.
        neutral_tol_deg: for N, both roll and pitch must be within +/- this.
        off_axis_tol_deg: for a rotated pose, the OTHER axis must stay within
            +/- this (an RL should not also be strongly pitched).
        hand_sign_overrides: optional config-supplied sign flips per (hand,pose),
            applied AFTER calibration. See _band_for_pose.

    Returns:
        (success, message). Mirrors the (success, message) contract. A
        non-measurable angle returns (False, error) and the pipeline decides
        SKIP vs FAIL (same as the other angle path).
    """
    ok, info = calculate_palm_angles(world_landmarks)
    if not ok:
        return (False, info.get("error", "Could not compute palm angle"))

    roll = info["roll"]
    pitch = info["pitch"]
    pose = (pose or "").upper()
    hand = (hand or "").upper()

    # --- Neutral: both axes near zero ---
    if pose == "N":
        roll_ok = abs(roll) <= neutral_tol_deg
        pitch_ok = abs(pitch) <= neutral_tol_deg
        if roll_ok and pitch_ok:
            return (True, f"N ok: roll={roll:.1f} pitch={pitch:.1f} "
                          f"within +/-{neutral_tol_deg:g}")
        bad = []
        if not roll_ok:
            bad.append(f"roll={roll:.1f} > +/-{neutral_tol_deg:g}")
        if not pitch_ok:
            bad.append(f"pitch={pitch:.1f} > +/-{neutral_tol_deg:g}")
        return (False, f"N not neutral: {', '.join(bad)}")

    axis, lo, hi = _band_for_pose(
        hand, pose,
        max_abs_deg=max_abs_deg, min_rotation_deg=min_rotation_deg,
        neutral_tol_deg=neutral_tol_deg, hand_sign_overrides=hand_sign_overrides)

    if axis is None:
        return (False, f"unknown pose '{pose}' (expected N/RL/RR/PU/PD)")

    active = roll if axis == "roll" else pitch
    other = pitch if axis == "roll" else roll
    other_name = "pitch" if axis == "roll" else "roll"

    in_band = lo <= active <= hi
    off_axis_ok = abs(other) <= off_axis_tol_deg

    if in_band and off_axis_ok:
        return (True,
                f"{hand}/{pose} ok: {axis}={active:.1f} in [{lo:g},{hi:g}]; "
                f"{other_name}={other:.1f} within +/-{off_axis_tol_deg:g}")

    reasons = []
    if not in_band:
        # Distinguish wrong-direction from out-of-range for a clear report.
        if (lo < 0 and active > 0) or (lo > 0 and active < 0):
            reasons.append(f"{axis}={active:.1f} WRONG DIRECTION (expected [{lo:g},{hi:g}])")
        elif abs(active) < min_rotation_deg:
            reasons.append(f"{axis}={active:.1f} not rotated enough (need |{axis}|>={min_rotation_deg:g})")
        else:
            reasons.append(f"{axis}={active:.1f} out of [{lo:g},{hi:g}]")
    if not off_axis_ok:
        reasons.append(f"{other_name}={other:.1f} off-axis > +/-{off_axis_tol_deg:g}")
    return (False, f"{hand}/{pose} bad: {'; '.join(reasons)}")


# ---------------------------------------------------------------------------
# Batch-relative (N-baseline) pose validation: check_palm_pose_delta
# ---------------------------------------------------------------------------
# Absolute per-image angles fail in practice: a real hand's `up`/`across` axes
# are never perfectly vertical/horizontal, so even a valid NEUTRAL reads a
# non-zero roll/pitch (observed ~11 deg on PASS images). That offset is a per-
# person, per-capture BASELINE, not a defect. The fix is to grade each rotated
# pose RELATIVE TO THAT HAND'S OWN N: the baseline is present in both the N frame
# and the pose frame, so subtracting it cancels the offset.
#
#     delta_roll  = pose_roll  - N_roll
#     delta_pitch = pose_pitch - N_pitch
#
# The same directional bands then apply to the DELTA (not the absolute angle):
#     RL: delta_roll  in [-45, -min_rotation]   RR: delta_roll  in [+min_rotation, +45]
#     PU: delta_pitch in [-45, -min_rotation]   PD: delta_pitch in [+min_rotation, +45]
# Only roll and pitch FAIL (spec has no yaw pose). N itself is not graded here --
# it is the reference (the caller emits SKIP for N).
#
# Sign note: deltas cancel the per-person baseline, which makes signs MORE
# consistent across people, but the direction of a "correct" delta still depends
# on handedness; hand_sign_overrides applies exactly as in check_palm_pose.


def check_palm_pose_delta(
    pose: str,
    hand: str,
    pose_angles: dict,
    n_angles: dict,
    *,
    max_abs_deg: float = 45.0,
    min_rotation_deg: float = 10.0,
    off_axis_tol_deg: float = 20.0,
    hand_sign_overrides: Optional[dict] = None,
):
    """Validate a rotated pose against this hand's OWN N baseline (delta space).

    Args:
        pose: "RL" | "RR" | "PU" | "PD" (N should not be passed -- it is the
            reference and is not graded; caller emits SKIP for N).
        hand: "L" | "R" -- selects the sign convention.
        pose_angles: {"roll":..,"pitch":..} measured for THIS pose image.
        n_angles:    {"roll":..,"pitch":..} measured for this hand's N image.
        max_abs_deg, min_rotation_deg, off_axis_tol_deg, hand_sign_overrides:
            same meaning as check_palm_pose, applied to the DELTA.

    Returns:
        (success, message). Mirrors the (success, message) contract.
    """
    pose = (pose or "").upper()
    hand = (hand or "").upper()

    if pose == "N":
        return (True, "N is the reference (not graded)")

    axis = _POSE_AXIS.get(pose)
    if axis is None:
        return (False, f"unknown pose '{pose}' (expected RL/RR/PU/PD)")

    try:
        d_roll = float(pose_angles["roll"]) - float(n_angles["roll"])
        d_pitch = float(pose_angles["pitch"]) - float(n_angles["pitch"])
    except (KeyError, TypeError, ValueError) as e:
        return (False, f"missing angle data for delta: {e}")

    _, lo, hi = _band_for_pose(
        hand, pose,
        max_abs_deg=max_abs_deg, min_rotation_deg=min_rotation_deg,
        neutral_tol_deg=0.0, hand_sign_overrides=hand_sign_overrides)

    active = d_roll if axis == "roll" else d_pitch
    other = d_pitch if axis == "roll" else d_roll
    other_name = "d_pitch" if axis == "roll" else "d_roll"
    active_name = "d_roll" if axis == "roll" else "d_pitch"

    in_band = lo <= active <= hi
    off_axis_ok = abs(other) <= off_axis_tol_deg

    if in_band and off_axis_ok:
        return (True,
                f"{hand}/{pose} ok: {active_name}={active:+.1f} in [{lo:g},{hi:g}]; "
                f"{other_name}={other:+.1f} within +/-{off_axis_tol_deg:g} "
                f"(vs N)")

    reasons = []
    if not in_band:
        if (lo < 0 and active > 0) or (lo > 0 and active < 0):
            reasons.append(f"{active_name}={active:+.1f} WRONG DIRECTION (expected [{lo:g},{hi:g}])")
        elif abs(active) < min_rotation_deg:
            reasons.append(f"{active_name}={active:+.1f} not rotated enough vs N (need |delta|>={min_rotation_deg:g})")
        else:
            reasons.append(f"{active_name}={active:+.1f} out of [{lo:g},{hi:g}]")
    if not off_axis_ok:
        reasons.append(f"{other_name}={other:+.1f} off-axis > +/-{off_axis_tol_deg:g}")
    return (False, f"{hand}/{pose} bad (vs N): {'; '.join(reasons)}")