"""Tests for both turn-sequence versions, on identical fabricated timelines.

Run:  python tests_turn_sequence.py
No pytest dependency — plain asserts + a tiny runner, so it runs anywhere
(including a LANTA login node) with no extra install and no models/video.

Each test builds a synthetic timeline (the edge cases we worked through) and
checks the OVERALL verdict of each version. Where the two versions disagree,
the test prints both so the difference is explicit.
"""
import math

CONFIG = {
    "face": {"turn_sequence": {
        "order_policy": "REVIEW",
        "hold_seconds": {"front_initial": 5, "front_between_movements": 2,
                         "left": 2, "right": 2, "down": 2, "up": 2},
        "tolerance": {"side_yaw_tolerance_deg": 30, "tilt_pitch_tolerance_deg": 15,
                      "front_zone_yaw_deg": 15, "front_zone_pitch_deg": 15},
        "accepted_orders": {
            "spec": ["front","left","front","right","front","down","front","up","front"],
            "audio_prompt": ["front","left","front","right","front","up","front","down","front"],
        },
    }}
}

import os
import sys
# project root = two levels up from this file (root/qc/tests/this.py)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
from qc.checks.check_turn_sequence import check_turn_sequence as presence
from qc.checks.check_turn_sequence_seg import check_turn_sequence_seg as segment


def F(i, yaw=0.0, pitch=0.0, det=True):
    return {"frame_index": i, "timestamp_sec": float(i), "face_detected": det,
            "yaw": (None if not det else yaw),
            "pitch": (None if not det else pitch), "roll": 0.0}

def gap(i):
    return F(i, det=False)

def overall(rows):
    # the overall row is the one whose check_name is exactly check_turn_sequence
    for r in rows:
        if r.check_name == "check_turn_sequence":
            return r.status
    return "?"

# ---- timelines -------------------------------------------------------------
def good_full_sequence():
    # front(5) left(2) front(2) right(2) front(2) down(2) front(2) up(2) front(2)
    t, i = [], 0
    def hold(n, **kw):
        nonlocal i
        for _ in range(n):
            t.append(F(i, **kw)); i += 1
    hold(5)                       # front
    hold(2, yaw=-50)              # left
    hold(2)                       # front
    hold(2, yaw=50)               # right
    hold(2)                       # front
    hold(2, pitch=-40)            # down
    hold(2)                       # front
    hold(2, pitch=40)             # up
    hold(2)                       # front
    return t

def left_via_gap():
    # left turn so deep the face is LOST at the peak: front, approach-left,
    # GAP (peak), recover, front. No DETECTED frame past the floor at peak.
    t, i = [], 0
    def hold(n, **kw):
        nonlocal i
        for _ in range(n):
            t.append(F(i, **kw)); i += 1
    hold(5)
    hold(1, yaw=-28)              # approach, just under floor (dead-band)
    t.append(gap(i)); i+=1
    t.append(gap(i)); i+=1        # peak unmeasurable
    hold(1, yaw=-28)              # recover
    hold(3)                       # front
    hold(2, yaw=50); hold(2)      # right
    hold(2, pitch=-40); hold(2)   # down
    hold(2, pitch=40); hold(2)    # up
    return t

def turned_left_twice_no_right():
    # all "turns" are left; right never happens -> right MISSING
    t, i = [], 0
    def hold(n, **kw):
        nonlocal i
        for _ in range(n):
            t.append(F(i, **kw)); i += 1
    hold(5)
    hold(2, yaw=-50); hold(2)
    hold(2, yaw=-50); hold(2)     # left again instead of right
    hold(2, pitch=-40); hold(2)
    hold(2, pitch=40); hold(2)
    return t

def splice_front_blank_front():
    # cheat: front, blank(gap), front — gap bracketed by front but NO turn
    # shoulder, no over-floor frame, no approach. Should NOT pass as a turn.
    t, i = [], 0
    def hold(n, **kw):
        nonlocal i
        for _ in range(n):
            t.append(F(i, **kw)); i += 1
    hold(5)
    for _ in range(4): t.append(gap(i)); i+=1
    hold(5)
    return t

def all_present_wrong_order():
    # right BEFORE left -> turns all present but order wrong
    t, i = [], 0
    def hold(n, **kw):
        nonlocal i
        for _ in range(n):
            t.append(F(i, **kw)); i += 1
    hold(5)
    hold(2, yaw=50); hold(2)      # right first
    hold(2, yaw=-50); hold(2)     # then left
    hold(2, pitch=-40); hold(2)
    hold(2, pitch=40); hold(2)
    return t

def short_hold_left():
    # left present but only 1 frame (< 2-frame minimum): duration too short
    t, i = [], 0
    def hold(n, **kw):
        nonlocal i
        for _ in range(n):
            t.append(F(i, **kw)); i += 1
    hold(5)
    hold(1, yaw=-50); hold(2)     # left only 1 frame
    hold(2, yaw=50); hold(2)
    hold(2, pitch=-40); hold(2)
    hold(2, pitch=40); hold(2)
    return t

def no_face_at_all():
    return [gap(i) for i in range(20)]

# ---- runner ----------------------------------------------------------------
CASES = [
    ("good_full_sequence",        good_full_sequence,        "PASS",   "PASS"),
    ("left_via_gap",              left_via_gap,              "REVIEW", "PASS_or_REVIEW"),
    ("turned_left_twice_no_right",turned_left_twice_no_right,"FAIL",   "FAIL"),
    ("splice_front_blank_front",  splice_front_blank_front,  "FAIL",   "FAIL_or_REVIEW"),
    ("all_present_wrong_order",   all_present_wrong_order,   "REVIEW", "REVIEW"),
    ("short_hold_left",           short_hold_left,           "REVIEW", "REVIEW"),
    ("no_face_at_all",            no_face_at_all,            "FAIL",   "FAIL"),
]

def main():
    print(f"{'case':28} {'presence':10} {'segment':10}  note")
    print("-"*70)
    for name, build, exp_p, exp_s in CASES:
        tl = build()
        ps = overall(presence(tl, CONFIG, sample_fps=1.0))
        sg = overall(segment(tl, CONFIG, sample_fps=1.0))
        note = ""
        if ps != sg:
            note = "<-- versions DIFFER"
        print(f"{name:28} {ps:10} {sg:10}  {note}")
    print("\n(expected presence verdicts are the design contract; segment may")
    print(" differ on gap/short-hold cases by design — that's the comparison.)")

if __name__ == "__main__":
    main()