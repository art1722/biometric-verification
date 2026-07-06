"""Tests for the v3 palm angle math (5-point plane fit, oriented normal,
bounded roll/pitch) plus the N-reference gate and the delta absolute cap.

All geometry is SYNTHETIC (rotated planes built with numpy), so these tests
need no MediaPipe model bundle, no cv2, and no real image -- they verify the
math contract that broke in v2:

  v2 bugs reproduced-then-fixed here:
    - _PINKY_MCP was 13 (RING_FINGER_MCP), not 17          -> plane uses 17
    - pitch = atan2(ny, nz) wrapped past +/-90 (read +138) -> bounded formula
    - cross-product normal flipped sign on back-of-hand    -> oriented normal
"""
import math
import types

import numpy as np
import pytest

from qc.checks.check_palm_angle import (
    PLANE_LANDMARK_IDXS,
    calculate_palm_angles,
    check_palm_n_reference,
    check_palm_pose_delta,
)


# ---------------------------------------------------------------------------
# Synthetic hand construction
# ---------------------------------------------------------------------------
# A flat palm facing the camera lives in the z=0 plane (MediaPipe world axes:
# x right, y down, z toward camera -> its normal is (0, 0, 1)). The 5 base-of-
# palm points span x and y; the other 16 landmarks are irrelevant to the plane
# fit but must exist so len(...) == 21.
_BASE = {
    0:  (0.000,  0.080, 0.0),   # wrist (below the knuckles: +y is down)
    5:  (-0.040, 0.000, 0.0),   # index MCP
    9:  (-0.010, -0.008, 0.0),  # middle MCP
    13: (0.020,  0.000, 0.0),   # ring MCP
    17: (0.045,  0.010, 0.0),   # pinky MCP
}


def _rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)],
                     [0, 1, 0],
                     [-math.sin(a), 0, math.cos(a)]])


def _rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0],
                     [0, math.cos(a), -math.sin(a)],
                     [0, math.sin(a), math.cos(a)]])


def _make_hand(rot=None, noise_idx=None, noise=(0.0, 0.0, 0.0)):
    """Build 21 landmark-like objects; the 5 plane points optionally rotated."""
    lms = []
    for i in range(21):
        p = np.array(_BASE.get(i, (0.001 * i, 0.002 * i, 0.0)))
        if rot is not None:
            p = rot @ p
        if noise_idx is not None and i == noise_idx:
            p = p + np.array(noise)
        lms.append(types.SimpleNamespace(x=float(p[0]), y=float(p[1]),
                                         z=float(p[2])))
    return lms


# ---------------------------------------------------------------------------
# calculate_palm_angles
# ---------------------------------------------------------------------------

def test_flat_palm_reads_zero():
    ok, info = calculate_palm_angles(_make_hand())
    assert ok
    assert abs(info["roll"]) < 1e-6
    assert abs(info["pitch"]) < 1e-6


def test_pure_side_tilt_reads_roll_only():
    # Ry(30) sends the normal (0,0,1) -> (sin30, 0, cos30): roll=+30, pitch=0.
    ok, info = calculate_palm_angles(_make_hand(rot=_rot_y(30)))
    assert ok
    assert info["roll"] == pytest.approx(30.0, abs=1e-6)
    assert info["pitch"] == pytest.approx(0.0, abs=1e-6)


def test_pure_vertical_tilt_reads_pitch_only():
    # Rx(25) sends the normal (0,0,1) -> (0, -sin25, cos25): pitch=-25, roll=0.
    ok, info = calculate_palm_angles(_make_hand(rot=_rot_x(25)))
    assert ok
    assert info["pitch"] == pytest.approx(-25.0, abs=1e-6)
    assert info["roll"] == pytest.approx(0.0, abs=1e-6)


def test_combined_tilt_stays_bounded_and_decoupled():
    ok, info = calculate_palm_angles(_make_hand(rot=_rot_x(25) @ _rot_y(30)))
    assert ok
    # Expected values from the same spherical decomposition of the known
    # rotated normal n = Rx(25) @ Ry(30) @ (0,0,1).
    n = _rot_x(25) @ _rot_y(30) @ np.array([0.0, 0.0, 1.0])
    exp_roll = math.degrees(math.atan2(n[0], n[2]))
    exp_pitch = math.degrees(math.atan2(n[1], math.hypot(n[0], n[2])))
    assert info["roll"] == pytest.approx(exp_roll, abs=1e-6)
    assert info["pitch"] == pytest.approx(exp_pitch, abs=1e-6)
    assert abs(info["roll"]) <= 90 and abs(info["pitch"]) <= 90


def test_never_wraps_past_90():
    # v2's atan2(ny, nz) read +138 deg on a real capture. Whatever the plane
    # orientation, v3 must stay bounded.
    for rot in (_rot_x(120), _rot_x(160), _rot_y(150) @ _rot_x(100)):
        ok, info = calculate_palm_angles(_make_hand(rot=rot))
        assert ok
        assert abs(info["roll"]) <= 90.0
        assert abs(info["pitch"]) <= 90.0


def test_back_of_hand_reads_same_as_palm_side():
    # Ry(180) flips the plane over (dorsal side toward camera). With the
    # oriented normal, a flat back-of-hand must read the same as a flat palm.
    ok, info = calculate_palm_angles(_make_hand(rot=_rot_y(180)))
    assert ok
    assert abs(info["roll"]) < 1e-6
    assert abs(info["pitch"]) < 1e-6


def test_one_noisy_landmark_shifts_angle_only_slightly():
    # Plane fit over 5 points: 4 mm of z-noise on ONE MCP must not swing the
    # reading by tens of degrees (the 3-point cross product would).
    ok0, base = calculate_palm_angles(_make_hand())
    ok1, noisy = calculate_palm_angles(
        _make_hand(noise_idx=13, noise=(0.0, 0.0, 0.004)))
    assert ok0 and ok1
    assert abs(noisy["roll"] - base["roll"]) < 8.0
    assert abs(noisy["pitch"] - base["pitch"]) < 8.0


def test_degenerate_collinear_points_error_not_crash():
    lms = _make_hand()
    for k, i in enumerate(PLANE_LANDMARK_IDXS):   # squash onto a line
        lms[i].x, lms[i].y, lms[i].z = float(k) * 0.01, 0.0, 0.0
    ok, info = calculate_palm_angles(lms)
    assert not ok
    assert "degenerate" in info["error"]


def test_short_landmark_list_errors():
    ok, info = calculate_palm_angles(_make_hand()[:10])
    assert not ok


# ---------------------------------------------------------------------------
# check_palm_n_reference
# ---------------------------------------------------------------------------

def test_n_reference_pass_and_fail():
    ok, msg = check_palm_n_reference({"roll": 3.0, "pitch": -4.0},
                                     n_reference_max_deg=15.0)
    assert ok and "N ok" in msg

    ok, msg = check_palm_n_reference({"roll": -45.5, "pitch": -31.0},
                                     n_reference_max_deg=15.0)
    assert not ok
    assert "re-capture N" in msg

    ok, _ = check_palm_n_reference(None)
    assert not ok


# ---------------------------------------------------------------------------
# check_palm_pose_delta: absolute spec cap
# ---------------------------------------------------------------------------

def test_delta_pass_within_band_and_cap():
    ok, msg = check_palm_pose_delta(
        "RR", "L", {"roll": +30.0, "pitch": 2.0}, {"roll": 0.0, "pitch": 0.0},
        max_abs_deg=45, min_rotation_deg=10, off_axis_tol_deg=45)
    assert ok, msg


def test_delta_fails_when_raw_exceeds_spec_cap():
    # Delta (+30) is inside the RR band, but the RAW roll (+50) violates the
    # spec's absolute +/-45 -> must FAIL with the [SPEC] reason.
    ok, msg = check_palm_pose_delta(
        "RR", "L", {"roll": +50.0, "pitch": 2.0}, {"roll": +20.0, "pitch": 0.0},
        max_abs_deg=45, min_rotation_deg=10, off_axis_tol_deg=45)
    assert not ok
    assert "raw roll" in msg and "[SPEC]" in msg


def test_delta_cap_can_be_disabled():
    ok, _ = check_palm_pose_delta(
        "RR", "L", {"roll": +50.0, "pitch": 2.0}, {"roll": +20.0, "pitch": 0.0},
        max_abs_deg=45, min_rotation_deg=10, off_axis_tol_deg=45,
        enforce_abs_cap=False)
    assert ok
