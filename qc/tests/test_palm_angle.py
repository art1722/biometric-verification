"""Tests for spec-aligned depth-wise palm angle math.

The spec's roll is depth-wise wrist rotation: one palm edge moves closer to the
sensor. It is not the x/y image-plane orientation of the palm normal.
"""
import types

import pytest

from qc.checks.check_palm_angle import (
    calculate_palm_angles,
    check_palm_pose_absolute,
)


def _make_hand(*, thumb_z=0.0, pinky_z=0.0, wrist_z=0.0, upper_z=0.0):
    """Build 21 landmark-like objects for deterministic depth-tilt tests.

    Smaller z is interpreted as closer to the camera/sensor. The x/y coordinates
    form a stable open palm; only z changes across test cases.
    """
    base = {
        0:  (0.000,  0.080, wrist_z),  # wrist
        1:  (-0.065, 0.030, thumb_z),  # thumb/index side support
        5:  (-0.040, 0.000, thumb_z if thumb_z != 0.0 else upper_z),
        9:  (-0.005, -0.006, upper_z),
        13: (0.030,  0.000, pinky_z if pinky_z != 0.0 else upper_z),
        17: (0.065,  0.012, pinky_z),
    }
    lms = []
    for i in range(21):
        x, y, z = base.get(i, (0.001 * i, 0.002 * i, 0.0))
        lms.append(types.SimpleNamespace(x=x, y=y, z=z))
    return lms


# ---------------------------------------------------------------------------
# calculate_palm_angles -- spec depth signs
# ---------------------------------------------------------------------------

def test_returns_roll_pitch_and_normal():
    ok, info = calculate_palm_angles(_make_hand(), handedness="L")
    assert ok
    assert "roll" in info and "pitch" in info and "normal" in info
    assert isinstance(info["roll"], float) and isinstance(info["pitch"], float)
    assert len(info["normal"]) == 3


def test_left_roll_follows_spec_edge_depth():
    # Left RL: pinky side closer -> negative roll.
    _, pinky_closer = calculate_palm_angles(
        _make_hand(thumb_z=+0.02, pinky_z=-0.02), handedness="L")
    # Left RR: thumb/index side closer -> positive roll.
    _, thumb_closer = calculate_palm_angles(
        _make_hand(thumb_z=-0.02, pinky_z=+0.02), handedness="L")
    assert pinky_closer["roll"] < -10.0
    assert thumb_closer["roll"] > +10.0


def test_right_roll_follows_spec_edge_depth():
    # Right RL: thumb/index side closer -> negative roll.
    _, thumb_closer = calculate_palm_angles(
        _make_hand(thumb_z=-0.02, pinky_z=+0.02), handedness="R")
    # Right RR: pinky side closer -> positive roll.
    _, pinky_closer = calculate_palm_angles(
        _make_hand(thumb_z=+0.02, pinky_z=-0.02), handedness="R")
    assert thumb_closer["roll"] < -10.0
    assert pinky_closer["roll"] > +10.0


def test_pitch_follows_spec_depth_direction():
    # PU: wrist side closer -> negative pitch.
    _, wrist_closer = calculate_palm_angles(
        _make_hand(wrist_z=-0.02, upper_z=+0.02), handedness="L")
    # PD: upper/finger side closer -> positive pitch.
    _, upper_closer = calculate_palm_angles(
        _make_hand(wrist_z=+0.02, upper_z=-0.02), handedness="L")
    assert wrist_closer["pitch"] < -10.0
    assert upper_closer["pitch"] > +10.0


def test_pitch_is_handedness_independent():
    l_ok, l = calculate_palm_angles(_make_hand(wrist_z=-0.02, upper_z=+0.02), handedness="L")
    r_ok, r = calculate_palm_angles(_make_hand(wrist_z=-0.02, upper_z=+0.02), handedness="R")
    assert l_ok and r_ok
    assert l["pitch"] == pytest.approx(r["pitch"], abs=1e-6)


# ---------------------------------------------------------------------------
# calculate_palm_angles -- failure handling (never raises)
# ---------------------------------------------------------------------------

def test_none_landmarks_error():
    ok, info = calculate_palm_angles(None, handedness="L")
    assert not ok and "error" in info


def test_short_landmark_list_errors():
    ok, info = calculate_palm_angles(_make_hand()[:10], handedness="L")
    assert not ok and "error" in info


def test_degenerate_side_axis_error():
    lms = _make_hand()
    # Collapse thumb/index side and pinky side into the same x/y position.
    for idx in (1, 5, 13, 17):
        lms[idx].x = 0.0
        lms[idx].y = 0.0
    ok, info = calculate_palm_angles(lms, handedness="L")
    assert not ok
    assert "degenerate" in info["error"]


# ---------------------------------------------------------------------------
# check_palm_pose_absolute: raw-angle band grading (no N baseline, no delta)
# ---------------------------------------------------------------------------

def test_n_pass_and_fail():
    # N passes when both axes are within the neutral tolerance.
    ok, msg = check_palm_pose_absolute(
        "N", "L", {"roll": 3.0, "pitch": -4.0}, neutral_tol_deg=10.0)
    assert ok and "N ok" in msg and "raw roll" in msg

    # N fails when an axis exceeds the tolerance; raw values are reported.
    ok, msg = check_palm_pose_absolute(
        "N", "L", {"roll": -45.5, "pitch": -3.0}, neutral_tol_deg=10.0)
    assert not ok and "not neutral" in msg and "roll=-45.5" in msg


def test_rotated_pose_passes_within_band():
    # RL (positive-roll convention): in band -> PASS, other axis reported.
    ok, msg = check_palm_pose_absolute(
        "RR", "L", {"roll": -30.0, "pitch": 2.0},
        max_abs_deg=45, min_rotation_deg=10)
    assert ok, msg
    assert "raw roll=-30.0 within [-45,-10]" in msg
    assert "pitch=+2.0" in msg  # other axis reported, not gating


def test_pitch_pose_signs_match_spec():
    ok, msg = check_palm_pose_absolute(
        "PU", "L", {"roll": 0.0, "pitch": -25.0},
        max_abs_deg=45, min_rotation_deg=10)
    assert ok, msg
    ok, msg = check_palm_pose_absolute(
        "PD", "L", {"roll": 0.0, "pitch": +25.0},
        max_abs_deg=45, min_rotation_deg=10)
    assert ok, msg


def test_rotated_pose_fails_report_raw():
    # Not rotated enough -> FAIL, raw value + band reported.
    ok, msg = check_palm_pose_absolute(
        "RL", "L", {"roll": +6.1, "pitch": -4.8},
        max_abs_deg=45, min_rotation_deg=10)
    assert not ok
    assert "raw roll=+6.1 not in [10,45]" in msg and "pitch=-4.8" in msg

    # Over the +/-45 cap -> FAIL.
    ok, msg = check_palm_pose_absolute(
        "RL", "L", {"roll": +52.0, "pitch": 2.0},
        max_abs_deg=45, min_rotation_deg=10)
    assert not ok and "not in [10,45]" in msg


def test_off_axis_does_not_gate():
    # A large "other" axis must NOT fail the pose (other axis is report-only).
    ok, msg = check_palm_pose_absolute(
        "RL", "L", {"roll": +40.0, "pitch": +38.0},
        max_abs_deg=45, min_rotation_deg=10)
    assert ok, msg
    assert "pitch=+38.0" in msg