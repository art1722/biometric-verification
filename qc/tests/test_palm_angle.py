"""Tests for spec-aligned depth-wise palm angle math.

The spec's roll is depth-wise wrist rotation: one palm edge moves closer to the
sensor. It is not the x/y image-plane orientation of the palm normal.
"""
import types

import pytest

from qc.checks.check_palm_angle import (
    calculate_palm_angles,
    check_palm_n_reference,
    check_palm_pose_delta,
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


def test_roll_flips_sign_when_edge_depth_reverses():
    # The researcher's method is sign-consistent: swapping which palm edge is
    # closer must flip the roll sign, and the magnitude should be meaningful.
    # (We assert the RELATIONSHIP, not a fixed polarity: the exact polarity was
    # calibrated on the real rig CSV, not this synthetic fixture, whose z sign is
    # not guaranteed to match live MediaPipe normalized z.)
    _, pinky_closer = calculate_palm_angles(
        _make_hand(thumb_z=+0.02, pinky_z=-0.02), handedness="L")
    _, thumb_closer = calculate_palm_angles(
        _make_hand(thumb_z=-0.02, pinky_z=+0.02), handedness="L")
    assert pinky_closer["roll"] * thumb_closer["roll"] < 0        # opposite signs
    assert abs(pinky_closer["roll"]) > 10.0
    assert abs(thumb_closer["roll"]) > 10.0


def test_right_hand_roll_is_negated_vs_left():
    # The researcher's Right branch negates raw roll relative to Left for the
    # SAME depth pattern. Verify that relationship holds.
    _, left = calculate_palm_angles(
        _make_hand(thumb_z=+0.02, pinky_z=-0.02), handedness="L")
    _, right = calculate_palm_angles(
        _make_hand(thumb_z=+0.02, pinky_z=-0.02), handedness="R")
    assert left["roll"] == pytest.approx(-right["roll"], abs=1e-6)


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
# check_palm_n_reference
# ---------------------------------------------------------------------------

def test_n_reference_pass_and_fail():
    ok, msg = check_palm_n_reference({"roll": 3.0, "pitch": -4.0},
                                     n_reference_max_deg=15.0)
    assert ok and "N ok" in msg

    ok, msg = check_palm_n_reference({"roll": -45.5, "pitch": -31.0},
                                     n_reference_max_deg=15.0)
    assert not ok and "re-capture N" in msg

    ok, _ = check_palm_n_reference(None)
    assert not ok


# ---------------------------------------------------------------------------
# check_palm_pose_delta: direction bands + absolute spec cap
# ---------------------------------------------------------------------------

def test_delta_pass_within_band_and_cap():
    # New convention: RR expects NEGATIVE roll (calibrated to researcher's method).
    ok, msg = check_palm_pose_delta(
        "RR", "L", {"roll": -30.0, "pitch": 2.0}, {"roll": 0.0, "pitch": 0.0},
        max_abs_deg=45, min_rotation_deg=10, off_axis_tol_deg=45)
    assert ok, msg


def test_pitch_delta_signs_match_spec():
    ok, msg = check_palm_pose_delta(
        "PU", "L", {"roll": 0.0, "pitch": -25.0}, {"roll": 0.0, "pitch": 0.0},
        max_abs_deg=45, min_rotation_deg=10, off_axis_tol_deg=45)
    assert ok, msg

    ok, msg = check_palm_pose_delta(
        "PD", "L", {"roll": 0.0, "pitch": +25.0}, {"roll": 0.0, "pitch": 0.0},
        max_abs_deg=45, min_rotation_deg=10, off_axis_tol_deg=45)
    assert ok, msg


def test_delta_fails_when_raw_exceeds_spec_cap():
    # Delta (-30) is inside the RR band, but the RAW roll (-50) violates the
    # spec's absolute +/-45 -> must FAIL with the [SPEC] reason.
    ok, msg = check_palm_pose_delta(
        "RR", "L", {"roll": -50.0, "pitch": 2.0}, {"roll": -20.0, "pitch": 0.0},
        max_abs_deg=45, min_rotation_deg=10, off_axis_tol_deg=45)
    assert not ok
    assert "raw roll" in msg and "[SPEC]" in msg


def test_delta_cap_can_be_disabled():
    ok, _ = check_palm_pose_delta(
        "RR", "L", {"roll": -50.0, "pitch": 2.0}, {"roll": -20.0, "pitch": 0.0},
        max_abs_deg=45, min_rotation_deg=10, off_axis_tol_deg=45,
        enforce_abs_cap=False)
    assert ok