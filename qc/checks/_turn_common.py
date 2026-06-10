"""Shared helpers for the turn-sequence check (both presence-first and
segmentation-first versions import from here, so the per-frame classification
is identical and the two versions differ ONLY in how they USE it).

The turn-sequence check consumes the per-frame `timeline` produced by the
face_rgb pipeline. Each entry is one sampled frame:

    {
      "frame_index":   int,
      "timestamp_sec": float,
      "face_detected": bool,        # False on a detection gap (profile turn)
      "yaw":   float | None,        # None on a gap or pose failure
      "pitch": float | None,
      "roll":  float | None,
    }

Sign conventions (established from ground-truth frames of video 002):
    yaw < 0  -> head turned LEFT  (left side of face toward camera)
    yaw > 0  -> head turned RIGHT
    pitch < 0 -> looking DOWN
    pitch > 0 -> looking UP

Two threshold SETS, read from config (NOT hardcoded):
    turn FLOOR  (side_yaw_tolerance_deg / tilt_pitch_tolerance_deg):
        a frame counts as a turn in a direction only if the angle EXCEEDS this.
    front ZONE  (front_zone_yaw_deg / front_zone_pitch_deg):
        a frame counts as FRONT only if BOTH |yaw| and |pitch| stay UNDER this.
        Must be tighter than the turn floor; the band between them is a
        deliberate dead-zone that is neither front nor a confident turn.

A detection GAP (face_detected False / yaw None) is NOT "front" and NOT a
classified turn on its own; it is potential evidence of a deep turn whose peak
MediaPipe could not measure, and is interpreted by the bracketing logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# The five meaningful positions plus the two non-classifiable frame kinds.
FRONT = "front"
LEFT = "left"
RIGHT = "right"
DOWN = "down"
UP = "up"
GAP = "gap"          # face not detected (potential profile-turn peak)
NEUTRALish = "mid"   # detected, but in the dead-band: neither front nor a turn


@dataclass(frozen=True)
class TurnThresholds:
    """Threshold set, loaded from config.face.turn_sequence.tolerance."""
    yaw_turn_floor: float        # side_yaw_tolerance_deg
    pitch_turn_floor: float      # tilt_pitch_tolerance_deg
    front_zone_yaw: float        # front_zone_yaw_deg
    front_zone_pitch: float      # front_zone_pitch_deg

    @classmethod
    def from_config(cls, config: dict) -> "TurnThresholds":
        tol = (config.get("face", {})
                     .get("turn_sequence", {})
                     .get("tolerance", {}))
        return cls(
            yaw_turn_floor=float(tol.get("side_yaw_tolerance_deg", 30)),
            pitch_turn_floor=float(tol.get("tilt_pitch_tolerance_deg", 15)),
            front_zone_yaw=float(tol.get("front_zone_yaw_deg", 15)),
            front_zone_pitch=float(tol.get("front_zone_pitch_deg", 15)),
        )


def classify_frame(entry: dict, th: TurnThresholds) -> str:
    """Map ONE timeline entry to a position label.

    Returns one of: FRONT, LEFT, RIGHT, DOWN, UP, GAP, NEUTRALish.
    This is the single source of per-frame meaning shared by both versions.
    """
    if not entry.get("face_detected"):
        return GAP

    yaw = entry.get("yaw")
    pitch = entry.get("pitch")
    if yaw is None or pitch is None:
        # Detected but pose failed — treat like a gap (no usable angle).
        return GAP

    # FRONT: neutral on BOTH axes (tight zone).
    if abs(yaw) < th.front_zone_yaw and abs(pitch) < th.front_zone_pitch:
        return FRONT

    # Turn classification: pick the axis that is most strongly past its floor.
    # A frame is rarely both a big yaw and a big pitch in this protocol (turns
    # are done one axis at a time), but if it is, the larger normalized
    # excursion wins, so the label reflects the dominant motion.
    yaw_excess = (abs(yaw) - th.yaw_turn_floor)
    pitch_excess = (abs(pitch) - th.pitch_turn_floor)

    yaw_is_turn = yaw_excess >= 0
    pitch_is_turn = pitch_excess >= 0

    if yaw_is_turn and (not pitch_is_turn or yaw_excess >= pitch_excess):
        return LEFT if yaw < 0 else RIGHT
    if pitch_is_turn:
        return DOWN if pitch < 0 else UP

    # Detected, but in the dead-band on both axes: neither front nor a turn.
    return NEUTRALish


def is_front(label: str) -> bool:
    return label == FRONT


def is_turn(label: str) -> bool:
    return label in (LEFT, RIGHT, DOWN, UP)


# ---- the model-fallback SEAM (Phase 2) -------------------------------------
# Positive evidence that a real profile turn happened inside a detection gap,
# for the case where no DETECTED frame reached the turn floor (the head turned
# so far the face mesh failed at the peak). Phase 1: not available -> None.
#
# When implemented, this will run a full-range head-pose model
# (6DRepNet360 / WHENet) on the gap frames and return the inferred direction
# (LEFT/RIGHT/DOWN/UP) if it reads a profile-range angle, else None.
def confirm_gap_turn(timeline: list, gap_start_idx: int, gap_end_idx: int,
                     expected_direction: Optional[str] = None) -> Optional[str]:
    """Phase 1 stub. Returns None = 'cannot positively confirm'.

    A gap that can only be confirmed by this (no detected over-floor frame and
    this returns None) should be scored REVIEW, never silently PASSed.
    """
    return None  # TODO Phase 2: head-pose model on gap frames