"""Tests for the expected/unexpected gap split (check_face_detected re-status).

Run:  python qc/tests/test_gap_split.py
No pytest dependency — plain asserts + a tiny runner, same as
test_turn_sequence.py, so it runs anywhere (including LANTA) with no models.

Covers:
  split_gap_frames                  — which gaps a turn explains
  apply_gap_split_to_detection_rows — the row re-status + config policy
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from qc.schemas import CheckRow
from qc.checks.check_turn_sequence_seg import (
    split_gap_frames, apply_gap_split_to_detection_rows,
)

CONFIG = {
    "face": {
        "turn_sequence": {
            "tolerance": {"side_yaw_tolerance_deg": 30, "tilt_pitch_tolerance_deg": 15,
                          "front_zone_yaw_deg": 15, "front_zone_pitch_deg": 15},
        },
        "checks": {"detection": {"expected_gap_status": "SKIP",
                                 "unexpected_gap_status": "FAIL"}},
    }
}


def F(i, yaw=0.0, pitch=0.0, det=True):
    return {"frame_index": i, "timestamp_sec": float(i), "face_detected": det,
            "yaw": (None if not det else yaw),
            "pitch": (None if not det else pitch), "roll": 0.0}


def gap(i):
    return F(i, det=False)


def timeline(*specs):
    """specs: sequence of ('front'|'left'|'right'|'up'|'down'|'gap', n)."""
    kw = {"front": {}, "left": {"yaw": -50}, "right": {"yaw": 50},
          "down": {"pitch": -40}, "up": {"pitch": 40}}
    t, i = [], 0
    for label, n in specs:
        for _ in range(n):
            t.append(gap(i) if label == "gap" else F(i, **kw[label]))
            i += 1
    return t


def test_gap_inside_turn_is_expected():
    # front(5) left(2) gap(3) left(1) front(2): gap bracketed by left on both sides
    t = timeline(("front", 5), ("left", 2), ("gap", 3), ("left", 1), ("front", 2))
    expected, unexpected = split_gap_frames(t, CONFIG)
    assert set(expected) == {7, 8, 9}, expected
    assert all(d == "left" for d in expected.values()), expected
    assert unexpected == [], unexpected


def test_gap_with_one_turn_shoulder_is_expected():
    # left -> gap -> front: one adjacent over-floor frame is enough evidence
    t = timeline(("front", 5), ("left", 2), ("gap", 2), ("front", 2))
    expected, unexpected = split_gap_frames(t, CONFIG)
    assert set(expected) == {7, 8}, expected
    assert unexpected == [], unexpected


def test_trailing_gap_after_turn_is_expected():
    # video ends stuck in a gap right after a turn: per-frame it's explained
    # (the structure check still fails the video for the missing sequence)
    t = timeline(("front", 5), ("left", 2), ("gap", 4))
    expected, unexpected = split_gap_frames(t, CONFIG)
    assert set(expected) == {7, 8, 9, 10}, expected
    assert unexpected == [], unexpected


def test_gap_between_fronts_is_unexpected():
    # front -> gap -> front: no turn evidence anywhere near the gap
    t = timeline(("front", 5), ("gap", 3), ("front", 4))
    expected, unexpected = split_gap_frames(t, CONFIG)
    assert expected == {}, expected
    assert unexpected == [5, 6, 7], unexpected


def test_ambiguous_gap_between_two_directions_is_unexpected():
    # left -> gap -> right: two different turn neighbours, cannot attribute
    t = timeline(("front", 5), ("left", 2), ("gap", 2), ("right", 2), ("front", 2))
    expected, unexpected = split_gap_frames(t, CONFIG)
    assert expected == {}, expected
    assert unexpected == [7, 8], unexpected


def test_leading_gap_before_front_is_unexpected():
    t = timeline(("gap", 2), ("front", 5))
    expected, unexpected = split_gap_frames(t, CONFIG)
    assert expected == {}, expected
    assert unexpected == [0, 1], unexpected


def _det_row(i, status="REVIEW"):
    return CheckRow("001", "face_rgb", "001_face_rgb.mp4",
                    "check_face_detected", status,
                    f"frame={i} No faces detected", i)


def test_apply_split_restatuses_rows():
    # front(3) left(1) gap(2) front(3) gap(1) front(2)
    #   gaps 4,5 -> expected (left shoulder); gap 9 -> unexpected (front..front)
    t = timeline(("front", 3), ("left", 1), ("gap", 2), ("front", 3),
                 ("gap", 1), ("front", 2))
    rows, gap_pos = [], {}
    for idx, e in enumerate(t):
        if not e["face_detected"]:
            rows.append(_det_row(idx))
            gap_pos[idx] = len(rows) - 1
        else:
            rows.append(CheckRow("001", "face_rgb", "001_face_rgb.mp4",
                                 "check_face_detected", "PASS",
                                 f"frame={idx} face ok", idx))
    apply_gap_split_to_detection_rows(rows, t, gap_pos, CONFIG)
    assert rows[4].status == "SKIP" and "expected gap (peak of left turn)" in rows[4].reason
    assert rows[5].status == "SKIP"
    assert rows[9].status == "FAIL" and "unexpected gap" in rows[9].reason
    # detected frames untouched
    assert rows[0].status == "PASS" and rows[6].status == "PASS"


def test_policy_comes_from_config():
    t = timeline(("front", 3), ("left", 1), ("gap", 1), ("front", 3))
    rows = [_det_row(4)]
    cfg = {"face": {"turn_sequence": CONFIG["face"]["turn_sequence"],
                    "checks": {"detection": {"expected_gap_status": "PASS",
                                             "unexpected_gap_status": "REVIEW"}}}}
    apply_gap_split_to_detection_rows(rows, t, {4: 0}, cfg)
    assert rows[0].status == "PASS", rows[0]


def test_invalid_policy_falls_back_to_defaults():
    t = timeline(("front", 3), ("gap", 1), ("front", 3))
    rows = [_det_row(3)]
    cfg = {"face": {"turn_sequence": CONFIG["face"]["turn_sequence"],
                    "checks": {"detection": {"unexpected_gap_status": "BANANA"}}}}
    apply_gap_split_to_detection_rows(rows, t, {3: 0}, cfg)
    assert rows[0].status == "FAIL", rows[0]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()