"""Palm handedness check -- does the detected hand match the filename hand?

Purpose
-------
Participants sometimes submit the WRONG hand under a filename: e.g. a physical
RIGHT hand saved as ..._palm_L_... (or a LEFT hand saved as ..._palm_R_...).
This check compares MediaPipe's detected handedness against the hand encoded in
the filename (L/R) and flags a mismatch.

The rule (confirmed 2026-07-09 on participant 099 via the pipeline overlay:
099_palm_L_N -> MediaPipe "Left"):

    PASS if  mediapipe_label == filename_hand   (L <-> Left, R <-> Right)
    FAIL if  they differ

Convention flag
---------------
MediaPipe assigns Left/Right assuming a mirrored/selfie image. Whether a given
rig saves mirrored or un-mirrored frames can invert the expected relation, so
the comparison is driven by `expected_relation`:

    "same"     -> L expects "Left",  R expects "Right"   (099's pipeline, DEFAULT)
    "opposite" -> L expects "Right", R expects "Left"

If a future batch/rig is shown to invert the label, flip this via config
(palm.handedness.expected_relation) instead of editing code. [CONFIRM per rig]

Confidence gate (mismatch only)
-------------------------------
MediaPipe's handedness is unreliable on palm-facing / tilted poses (the same
physical LEFT hand was labelled "Left" on the N pose but "Right" on a PU tilt,
both at low confidence ~0.6-0.7). So:
  - a MATCHING label always PASSes (agreement is trusted),
  - a MISMATCH only FAILs when MediaPipe is CONFIDENT (score >= min_confidence);
    a low-confidence mismatch SKIPs rather than fail a possibly-correct file.

Because even the neutral pose can be shaky, the pipeline runs this check on the
NEUTRAL (N) pose ONLY (like check_palm_spread) and SKIPs the rotated poses.

Mirrors the other palm checks: a pure (result, message) function that consumes
the single HandResult from detect_hand -- it does NOT re-run the model.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Normalized forms: filename uses L/R; MediaPipe uses Left/Right.
_FILENAME_TO_FULL = {"L": "Left", "R": "Right"}
_FULL = {"LEFT": "Left", "RIGHT": "Right"}


def _expected_label(filename_hand: str, relation: str) -> Optional[str]:
    """Return the MediaPipe label a CORRECT file should produce, or None if the
    filename hand is unknown."""
    full = _FILENAME_TO_FULL.get((filename_hand or "").upper())
    if full is None:
        return None
    if relation == "opposite":
        return "Right" if full == "Left" else "Left"
    return full  # "same"


def check_palm_handedness(
    handedness: Optional[str],
    filename_hand: Optional[str],
    *,
    handedness_score: Optional[float] = None,
    expected_relation: str = "same",
    min_confidence: float = 0.7,
):
    """Compare MediaPipe handedness against the filename hand.

    Grading logic (agreed 2026-07-09):
      - labels MATCH                       -> PASS (trust agreement, any score)
      - labels MISMATCH, score <  min_conf -> SKIP (disagrees but unsure; do
                                               not punish -- MediaPipe handedness
                                               is flaky on palm-facing poses)
      - labels MISMATCH, score >= min_conf -> FAIL (disagrees AND confident ->
                                               likely a real wrong-hand submission)

    NOTE: this check is only meaningful on the NEUTRAL (N) pose. Rotated/tilted
    palm poses confuse MediaPipe's left/right classifier (observed: the same
    physical left hand labelled "Left" on N but "Right" on a PU tilt). The
    pipeline restricts this check to N and SKIPs the other poses.

    Args:
        handedness: MediaPipe's label ("Left"/"Right"), or None if unavailable.
        filename_hand: "L" or "R" from the filename.
        handedness_score: MediaPipe's confidence for the label (0..1), or None.
        expected_relation: "same" (L<->Left, DEFAULT) or "opposite" (L<->Right).
        min_confidence: a MISMATCH only FAILs at/above this confidence; below it
            the mismatch SKIPs. A matching label always PASSes regardless.

    Returns:
        (result, message) where result is:
          True  -> PASS   (labels match)
          False -> FAIL   (confident mismatch -> likely wrong hand)
          None  -> SKIP   (no label / unknown filename hand / low-confidence
                   mismatch). The pipeline maps None -> SKIP.
        The message ALWAYS reports the raw MediaPipe label + score.
    """
    fh = (filename_hand or "").upper()
    score_txt = f"{handedness_score:.2f}" if handedness_score is not None else "n/a"

    # Unknown filename hand -> cannot compare.
    if fh not in ("L", "R"):
        return (None, f"handedness not graded: unknown filename hand "
                      f"'{filename_hand}' (mediapipe={handedness}, "
                      f"score={score_txt})")

    # No MediaPipe label -> cannot compare (no hand / label missing).
    mp = _FULL.get((handedness or "").upper())
    if mp is None:
        return (None, f"handedness not graded: no MediaPipe label "
                      f"(filename={fh}, mediapipe={handedness}, score={score_txt})")

    expected = _expected_label(fh, expected_relation)

    # Agreement -> PASS regardless of confidence.
    if mp == expected:
        return (True, f"handedness ok: filename={fh} matches mediapipe={mp} "
                      f"(score={score_txt}, relation={expected_relation})")

    # Mismatch: only FAIL if MediaPipe is CONFIDENT; otherwise SKIP. MediaPipe
    # handedness is unreliable on palm-facing poses, so a low-confidence
    # disagreement is not trustworthy enough to fail a (possibly correct) file.
    if handedness_score is not None and handedness_score < min_confidence:
        return (None, f"handedness mismatch but low-confidence (skipped): "
                      f"filename={fh} expects {expected}, got {mp} "
                      f"score={score_txt} < {min_confidence:g} "
                      f"(relation={expected_relation})")

    return (False, f"handedness MISMATCH: filename={fh} expects "
                   f"mediapipe={expected} but got {mp} "
                   f"(score={score_txt} >= {min_confidence:g}, "
                   f"relation={expected_relation}) -- likely wrong hand submitted")