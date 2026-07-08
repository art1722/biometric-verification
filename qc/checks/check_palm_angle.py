"""Palm angle check -- spec-aligned depth-wise wrist roll/pitch.

Spec meaning
------------
Palm roll in the data specification is NOT image-plane rotation. It is the
out-of-plane wrist rotation where one side of the palm moves closer to the
sensor:

  - RL/RR: thumb-side vs pinky-side palm edge moves toward the sensor.
  - PU/PD: wrist-side vs finger-side palm region moves toward the sensor.

This module therefore measures depth-wise tilt from MediaPipe 3D hand landmarks:

  - roll  from the z-depth difference between thumb/index side and pinky side.
  - pitch from the z-depth difference between wrist side and upper/finger side.

MediaPipe's z convention is model-estimated, not calibrated physical depth. The
pipeline still grades each rotated pose relative to the same hand's neutral (N)
image, and keeps the raw +/-45 degree cap from the spec.

Coordinate/sign convention used here
------------------------------------
MediaPipe hand landmarks use a camera-like z value where the side with the
smaller z is closer to the camera/sensor. The returned signs are normalized to
the filename/spec pose labels:

  - RL should be negative roll.
  - RR should be positive roll.
  - PU should be negative pitch (wrist side closer).
  - PD should be positive pitch (finger side closer).

Because the spec defines RL/RR differently for left and right hands, roll sign
uses the FILENAME hand (L/R), not MediaPipe's mirrored handedness label.

Returns
-------
calculate_palm_angles(world_landmarks, handedness="L"|"R") -> (ok, info)
    info = {"roll": float, "pitch": float, "normal": (x,y,z), ...}
check_palm_angle(...) -> (ok, message)
    symmetric absolute wrapper for +/- roll/pitch caps.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Base-of-palm keypoints (rigid relative to the wrist; finger curl barely moves
# them). Indices are the standard MediaPipe HandLandmark order, matching the
# HandLandmark enum in hand_landmarker.py (PINKY_MCP is 17; 13 is RING).
_WRIST = 0
_INDEX_MCP = 5
_MIDDLE_MCP = 9
_RING_MCP = 13
_PINKY_MCP = 17

# Base landmarks used for the palm angle/reference geometry. The name is kept
# for backwards compatibility with overlay/debug modules that import it.
PLANE_LANDMARK_IDXS = (_WRIST, _INDEX_MCP, _MIDDLE_MCP, _RING_MCP, _PINKY_MCP)


def calculate_palm_angles(world_landmarks: Any, *, handedness: Any = None):
    """Return spec-aligned depth-wise palm roll/pitch in degrees.

    This intentionally replaces the old local-normal formula
    ``atan2(normal_x, -normal_y)``. That old formula reads the palm normal's
    projected direction in the image plane; the spec's roll is depth-wise: which
    side of the palm is closer to the sensor.

    Landmarks used:
      - thumb/index side: mean(THUMB_CMC=1, INDEX_MCP=5)
      - pinky side:      mean(RING_MCP=13, PINKY_MCP=17)
      - wrist side:      WRIST=0
      - upper palm side: mean(INDEX_MCP=5, MIDDLE_MCP=9, RING_MCP=13, PINKY_MCP=17)

    Sign convention after normalization:
      - Left hand:  pinky closer -> negative roll; thumb/index closer -> positive.
      - Right hand: thumb/index closer -> negative roll; pinky closer -> positive.
      - Wrist closer -> negative pitch (PU); upper/finger side closer -> positive (PD).

    `handedness` should come from the filename (..._palm_L_... / ..._palm_R_...),
    not from MediaPipe's mirrored handedness label. If handedness is omitted, the
    roll magnitude is still useful for absolute caps, but the roll sign is
    assumed left-hand style and should not be used for RL/RR grading.
    """
    if world_landmarks is None:
        return (False, {"error": "No world landmarks provided"})

    try:
        n = len(world_landmarks)
    except TypeError:
        return (False, {"error": "world_landmarks is not indexable"})

    required = (_WRIST, 1, _INDEX_MCP, _MIDDLE_MCP, _RING_MCP, _PINKY_MCP)
    if n <= max(required):
        return (False, {"error": f"Expected 21 landmarks, got {n}"})

    def _pt(idx: int) -> np.ndarray:
        lm = world_landmarks[idx]
        return np.array([float(lm.x), float(lm.y), float(lm.z)], dtype=np.float64)

    def _unit(v: np.ndarray, name: str):
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            raise ValueError(f"degenerate palm geometry (zero-length {name})")
        return v / norm

    try:
        wrist = _pt(_WRIST)
        thumb_side = np.mean([_pt(1), _pt(_INDEX_MCP)], axis=0)
        pinky_side = np.mean([_pt(_RING_MCP), _pt(_PINKY_MCP)], axis=0)
        upper_palm = np.mean(
            [_pt(_INDEX_MCP), _pt(_MIDDLE_MCP), _pt(_RING_MCP), _pt(_PINKY_MCP)],
            axis=0,
        )

        # Across-palm vector: thumb/index edge -> pinky edge.
        side_vec = pinky_side - thumb_side
        side_xy = math.hypot(float(side_vec[0]), float(side_vec[1]))
        if side_xy < 1e-9:
            return (False, {"error": "degenerate palm geometry (zero-width palm side axis)"})

        # Up-palm vector: wrist side -> upper/finger side.
        up_vec = upper_palm - wrist
        up_xy = math.hypot(float(up_vec[0]), float(up_vec[1]))
        if up_xy < 1e-9:
            return (False, {"error": "degenerate palm geometry (zero-length wrist-to-upper-palm axis)"})

        # Raw side depth: negative means pinky side has smaller z => closer.
        raw_side_depth_deg = math.degrees(math.atan2(float(side_vec[2]), side_xy))

        # Spec-normalized roll sign. With smaller z = closer:
        #   Left:  pinky closer -> raw negative -> RL negative.
        #   Right: thumb closer  -> raw positive -> RL negative, so flip sign.
        hs = str(handedness or "").strip().upper()
        is_right = hs in ("R", "RIGHT")
        roll = -raw_side_depth_deg if is_right else raw_side_depth_deg

        # Raw vertical depth: positive means upper palm has larger z, so wrist is
        # closer. Spec PU should be negative, hence the leading minus.
        raw_up_depth_deg = math.degrees(math.atan2(float(up_vec[2]), up_xy))
        pitch = -raw_up_depth_deg

        # A visual/debug normal only. The grading uses the explicit depth-axis
        # angles above, not this normal's x/y projection.
        side_axis = _unit(side_vec, "side axis")
        up_axis = _unit(up_vec, "up axis")
        normal = np.cross(side_axis, up_axis)
        normal = _unit(normal, "debug normal")

        return (True, {
            "roll": float(roll),
            "pitch": float(pitch),
            "normal": (float(normal[0]), float(normal[1]), float(normal[2])),
            "raw_side_depth_deg": float(raw_side_depth_deg),
            "raw_up_depth_deg": float(raw_up_depth_deg),
            "handedness_used": "R" if is_right else "L_or_default",
        })

    except Exception as e:  # never crash a batch on one odd image
        logger.debug("PALM_ANGLE | error: %s", e)
        return (False, {"error": f"Error computing palm angle: {e}"})

def check_palm_angle(
    world_landmarks: Any,
    *,
    handedness: Any = None,
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
    ok, info = calculate_palm_angles(world_landmarks, handedness=handedness)
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
# !!! HANDEDNESS / SIGN CALIBRATION -- STATUS !!!
# The spec defines RL/RR relative to the VOLUNTEER (L-RL tilts the pinky edge
# toward the sensor; R-RL tilts the THUMB edge). Both describe the SAME body
# direction (for RL the palm normal sweeps toward the volunteer's left), so
# the measured roll sign is expected to be IDENTICAL for L and R hands under
# one capture geometry; what flips it is IMAGE MIRRORING, not handedness.
# CALIBRATED 2026-07-07 on a real LEFT-hand, palm-side, MIRRORED front-camera
# set (see _POSE_SIGN below). Still open before full sign-off (อ.เหมียว):
#   (a) verify the RIGHT hand reproduces the same signs (prediction: R/RL
#       should ALSO measure a NEGATIVE d_roll under mirrored capture),
#   (b) confirm whether the PRODUCTION rig saves mirrored or unmirrored
#       images -- if unmirrored, RL/RR signs flip back to {+1, -1},
#   (c) keep directional FAILs advisory/REVIEW until (a)+(b) are confirmed.
# [CONFIRM]

# Axis each pose acts on.
_POSE_AXIS = {"N": None, "RL": "roll", "RR": "roll", "PU": "pitch", "PD": "pitch"}

# Expected SIGN per pose. +1 expects a positive delta, -1 negative.
# This follows the specification labels after calculate_palm_angles normalizes
# depth signs by filename hand: RL=-, RR=+, PU=-, PD=+. If the production rig's
# z/mirroring convention is shown to invert a pose, override it from config via
# palm.angle.hand_sign_overrides.
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
    pose = (pose or "").upper()
    hand = (hand or "").upper()
    ok, info = calculate_palm_angles(world_landmarks, handedness=hand)
    if not ok:
        return (False, info.get("error", "Could not compute palm angle"))

    roll = info["roll"]
    pitch = info["pitch"]

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
    enforce_abs_cap: bool = True,
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
        enforce_abs_cap: also FAIL when the RAW |roll| or |pitch| exceeds
            max_abs_deg (the spec's absolute +/-45 bound), independent of the
            delta verdict. Default True. [SPEC]

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
        p_roll = float(pose_angles["roll"])
        p_pitch = float(pose_angles["pitch"])
        d_roll = p_roll - float(n_angles["roll"])
        d_pitch = p_pitch - float(n_angles["pitch"])
    except (KeyError, TypeError, ValueError) as e:
        return (False, f"missing angle data for delta: {e}")

    # ABSOLUTE spec cap, independent of the delta: the spec's "+/-45" bounds the
    # RAW angle of the capture, not only its change vs N. Researchers asked for
    # the absolute value to be graded and reported too. [SPEC]
    abs_reasons = []
    if enforce_abs_cap:
        if abs(p_roll) > max_abs_deg:
            abs_reasons.append(
                f"|raw roll|={abs(p_roll):.1f} > {max_abs_deg:g} [SPEC]")
        if abs(p_pitch) > max_abs_deg:
            abs_reasons.append(
                f"|raw pitch|={abs(p_pitch):.1f} > {max_abs_deg:g} [SPEC]")

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

    if in_band and off_axis_ok and not abs_reasons:
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
    reasons.extend(abs_reasons)
    return (False, f"{hand}/{pose} bad (vs N): {'; '.join(reasons)}")


def check_palm_n_reference(
    n_angles: Optional[dict],
    *,
    n_reference_max_deg: float = 15.0,
):
    """Grade the N (neutral) image ABSOLUTELY.

    Researchers expect a valid N to read roll ~ 0 AND pitch ~ 0 -- and N is the
    baseline every rotated pose's delta is judged against, so a tilted N is
    both a capture defect in its own right (re-capture it) and a bias on every
    delta for that hand.

    Args:
        n_angles: {"roll":..,"pitch":..} measured for the N image (or None).
        n_reference_max_deg: |roll| and |pitch| must both be <= this.
            config: palm.angle.n_reference_max_deg  [ASSUMPTION -> CONFIRM].

    Returns:
        (success, message) -- the standard contract.
    """
    if not n_angles:
        return (False, "N reference unmeasurable (no angle)")
    try:
        roll = float(n_angles["roll"])
        pitch = float(n_angles["pitch"])
    except (KeyError, TypeError, ValueError) as e:
        return (False, f"N reference missing angle data: {e}")

    roll_ok = abs(roll) <= n_reference_max_deg
    pitch_ok = abs(pitch) <= n_reference_max_deg
    if roll_ok and pitch_ok:
        return (True,
                f"N ok (reference): roll={roll:+.1f} pitch={pitch:+.1f} "
                f"within +/-{n_reference_max_deg:g}")
    bad = []
    if not roll_ok:
        bad.append(f"roll={roll:+.1f} > +/-{n_reference_max_deg:g}")
    if not pitch_ok:
        bad.append(f"pitch={pitch:+.1f} > +/-{n_reference_max_deg:g}")
    return (False, "N not neutral (re-capture N): " + ", ".join(bad))


def calculate_palm_axis_tilts(world_landmarks):
    """DEBUG cross-check for calculate_palm_angles -- NOT used for grading.

    Returns the raw depth tilts before handedness/sign normalization:
      across_tilt_raw: thumb/index side -> pinky side depth tilt.
      up_tilt_raw:     wrist side -> upper-palm side depth tilt.

    calculate_palm_angles converts these into spec signs:
      roll  = across_tilt_raw for L, -across_tilt_raw for R.
      pitch = -up_tilt_raw.
    """
    try:
        p = lambda i: (float(world_landmarks[i].x),
                       float(world_landmarks[i].y),
                       float(world_landmarks[i].z))
        def mean(indices):
            arr = np.array([p(i) for i in indices], dtype=np.float64)
            return arr.mean(axis=0)
        thumb_side = mean([1, _INDEX_MCP])
        pinky_side = mean([_RING_MCP, _PINKY_MCP])
        upper_palm = mean([_INDEX_MCP, _MIDDLE_MCP, _RING_MCP, _PINKY_MCP])
        wrist = np.array(p(_WRIST), dtype=np.float64)
        side_vec = pinky_side - thumb_side
        up_vec = upper_palm - wrist
        tilt = lambda v: math.degrees(math.atan2(float(v[2]), math.hypot(float(v[0]), float(v[1]))))
        return (True, {"across_tilt_raw": tilt(side_vec), "up_tilt_raw": tilt(up_vec)})
    except Exception as e:
        return (False, {"error": f"axis tilt failed: {e}"})

