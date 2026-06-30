"""Finger-spread check -- are the fingers spread naturally apart (not closed)?

Spec requirement (source of truth)
----------------------------------
The palm spec (doc lines 33, 36-47) requires the hand "in a spread pose with
all five fingers naturally apart, NOT pressed together to the point of blocking
light." This check measures the angular gap between adjacent fingers and FAILS
the image if any required gap is too small (fingers closed / touching).

Scope -- N pose only (caller's job)
-----------------------------------
This runs on the NEUTRAL (N) pose only. RL/RR/PU/PD are deliberate wrist
rotations where the hand is tilted out of the image plane, so an in-plane
spread angle distorts and would false-fail. The pipeline gates this check to
the N pose (palm.hand_pose.spread.eval_poses, default ["N"]); the function
itself stays pure and always computes a value. This mirrors the front-frame-only
scoping used for face occlusion. [DESIGN -> CONFIRM with อ.เหมียว]

Coordinate space -- NORMALIZED image landmarks, NOT world landmarks
-------------------------------------------------------------------
check_palm_angle uses WORLD landmarks because roll/pitch are OUT-of-plane
rotations that image coords distort. Spread is the opposite: it is an IN-plane
angular quantity, and for the N pose (palm flat and parallel to the sensor) the
image plane IS the palm plane. So we read the 2D (x, y) of the normalized image
landmarks (HandResult.landmarks_norm) and measure planar angles.

Cropped fingertips -> TIP vs PIP source selection
-------------------------------------------------
MediaPipe Hand Landmarker is a REGRESSION model: it predicts ALL 21 landmarks
whether or not they are in view. A fingertip cropped out of the top of the frame
is NOT dropped -- it is EXTRAPOLATED from the model's hand prior, and its
normalized coords can fall OUTSIDE [0, 1] (e.g. y < 0). So a "tip was detected"
flag is meaningless; the tip may be a hallucinated point. We therefore decide
which landmark to trust GEOMETRICALLY, by whether it lies safely inside the
frame, not by any detection flag.

Each finger has three points we care about:
    finger : MCP (origin)        -> PIP (lower joint)     -> TIP (end)
    index  : 5                   -> 6                      -> 8
    middle : 9                   -> 10                     -> 12
    ring   : 13                  -> 14                     -> 16
    pinky  : 17                  -> 18                     -> 20
    thumb  : 2 (MCP)             -> 3 (IP, pip-equivalent) -> 4 (TIP)

Source selection is HAND-WIDE and CONSISTENT (not per-finger), to avoid mixing
vectors of different lengths/curvature across a single gap (a tip-vector and a
pip-vector of the same finger point in slightly different directions because the
finger curves, so a mixed gap would not be comparable to a pure one under a
single threshold). The rule:

    1. The MCP (vector ORIGIN) of every required finger must be in-frame, else
       that finger is unusable.
    2. If EVERY required finger also has its TIP in-frame -> source = "tip"
       (full angular separation; the preferred, most sensitive source).
    3. Else if EVERY required finger has its PIP in-frame -> source = "pip"
       (degraded but real; used when any tip is cropped/extrapolated).
    4. Else -> cannot measure -> the check returns a non-measurable result and
       the pipeline emits SKIP.

Because the two sources have different natural magnitudes (tip-tip gaps fan
wider than pip-pip gaps), thresholds are SOURCE-AWARE: min_gap_*_tip vs
min_gap_*_pip. Calibrate each on real samples.

Method (geometry, no model)
---------------------------
Each finger is a vector from its MCP to the chosen end point (TIP or PIP). The
gap between two adjacent fingers is the unsigned angle between their vectors.
Four adjacent gaps are measured: thumb-index, index-middle, middle-ring,
ring-pinky. A gap PASSES if it is >= the source's floor for that pair; the
check FAILS if any REQUIRED gap is below its floor (fingers closed).

Contract (mirrors the other palm checks)
-----------------------------------------
- Consumes landmarks_norm (the normalized .x/.y/.z list from detect_hand). No
  model is run here.
- Returns (success: bool, message: str). The message embeds the source used and
  every measured gap, e.g.
  "spread ok; source=tip; thumb-index=37.8 index-middle=11.3 ... (min ...)".
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Standard MediaPipe HandLandmark indices (mirrors HandLandmark in
# hand_landmarker.py): (MCP, PIP, TIP) per finger. Thumb uses IP(3) as its
# pip-equivalent and MCP(2) as its origin.
_FINGERS: Dict[str, Tuple[int, int, int]] = {
    # name      MCP  PIP  TIP
    "thumb":  (2,   3,   4),
    "index":  (5,   6,   8),
    "middle": (9,   10,  12),
    "ring":   (13,  14,  16),
    "pinky":  (17,  18,  20),
}

# Adjacent pairs whose angular gap is measured, in spatial order across the hand.
_ADJACENT_PAIRS: List[Tuple[str, str]] = [
    ("thumb", "index"),
    ("index", "middle"),
    ("middle", "ring"),
    ("ring", "pinky"),
]

_MAX_IDX = 20  # highest landmark index we touch; quick length guard.

# --- Default thresholds (degrees), SOURCE-AWARE. All overridable via config. --
# tip-tip gaps fan wider than pip-pip gaps, so each source has its own floors.
# The thumb-index gap is naturally much wider than the inter-finger gaps, so it
# gets its own larger floor within each source. STARTING values -- calibrate on
# real spread/closed samples with อ.เหมียว.
_DEFAULT_MIN_GAP_INTER_TIP = 5.0    # index-middle/middle-ring/ring-pinky, tip source
_DEFAULT_MIN_GAP_THUMB_TIP = 20.0   # thumb-index, tip source
_DEFAULT_MIN_GAP_INTER_PIP = 3.0    # inter-finger, pip source (PIPs fan less)
_DEFAULT_MIN_GAP_THUMB_PIP = 12.0   # thumb-index, pip source

# Which gaps FAIL the frame if too small.
_DEFAULT_REQUIRED_PAIRS = ["thumb-index", "index-middle", "middle-ring", "ring-pinky"]

# Fingers whose geometry must be usable for the check to run. Thumb included
# because thumb-index is a required gap by default.
_REQUIRED_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

_DEFAULT_FRAME_MARGIN = 0.02  # a point must sit inside [margin, 1-margin] to count


def _pt(landmarks: Sequence, i: int) -> Optional[Tuple[float, float]]:
    """(x, y) of a normalized landmark (.x/.y or [0]/[1]), or None."""
    try:
        p = landmarks[i]
    except (IndexError, TypeError):
        return None
    try:
        return (float(p.x), float(p.y))
    except AttributeError:
        try:
            return (float(p[0]), float(p[1]))
        except (IndexError, TypeError, ValueError):
            return None


def _in_frame(pt: Optional[Tuple[float, float]], margin: float) -> bool:
    """True if a normalized point lies safely inside the frame.

    MediaPipe extrapolates out-of-view landmarks and can return coords outside
    [0, 1], so a point at/over the edge (or beyond it) is treated as NOT in
    frame -- it is likely hallucinated, not measured.
    """
    if pt is None:
        return False
    x, y = pt
    return (margin < x < 1.0 - margin) and (margin < y < 1.0 - margin)


def _angle_between(u: Tuple[float, float], v: Tuple[float, float]) -> float:
    """Unsigned angle (degrees) between two 2D vectors, in [0, 180]."""
    dot = u[0] * v[0] + u[1] * v[1]
    nu = math.hypot(u[0], u[1])
    nv = math.hypot(v[0], v[1])
    if nu == 0.0 or nv == 0.0:
        return 0.0
    c = max(-1.0, min(1.0, dot / (nu * nv)))
    return math.degrees(math.acos(c))


def _select_source(
    landmarks: Sequence,
    required_fingers: Sequence[str],
    margin: float,
) -> Tuple[Optional[str], str]:
    """Decide a single hand-wide vector source: "tip", "pip", or None.

    Returns (source, detail). source is None when the hand cannot be measured
    (some required MCP, or both TIP and PIP of some finger, is out of frame);
    detail is a human-readable reason/breakdown for the message.
    """
    # 1. Every required finger's MCP (the vector origin) must be in-frame.
    bad_mcp = []
    for name in required_fingers:
        mcp_i = _FINGERS[name][0]
        if not _in_frame(_pt(landmarks, mcp_i), margin):
            bad_mcp.append(name)
    if bad_mcp:
        return (None, f"MCP out of frame: {', '.join(bad_mcp)}")

    # 2. Can we use TIPS for ALL required fingers?
    tips_ok = all(
        _in_frame(_pt(landmarks, _FINGERS[name][2]), margin)
        for name in required_fingers
    )
    if tips_ok:
        return ("tip", "all required tips in frame")

    # 3. Else, can we use PIPS for ALL required fingers?
    pips_ok = all(
        _in_frame(_pt(landmarks, _FINGERS[name][1]), margin)
        for name in required_fingers
    )
    if pips_ok:
        cropped = [
            name for name in required_fingers
            if not _in_frame(_pt(landmarks, _FINGERS[name][2]), margin)
        ]
        return ("pip", f"tip cropped ({', '.join(cropped)}) -> using PIP")

    # 4. Neither source is fully usable.
    bad = [
        name for name in required_fingers
        if not _in_frame(_pt(landmarks, _FINGERS[name][1]), margin)
    ]
    return (None, f"PIP also out of frame: {', '.join(bad)}")


def calculate_finger_gaps(
    landmarks_norm: Any,
    *,
    required_fingers: Optional[Sequence[str]] = None,
    frame_margin: float = _DEFAULT_FRAME_MARGIN,
):
    """Angular gaps (degrees) between adjacent fingers, with source selection.

    Returns:
        (success, info). On success info = {"source": "tip"|"pip",
        "gaps": {"a-b": deg, ...}, "detail": str}. On failure success is False
        and info has an "error" key (and a "detail"). Never raises on bad input.
    """
    if landmarks_norm is None:
        return (False, {"error": "No landmarks provided"})
    try:
        n = len(landmarks_norm)
    except TypeError:
        return (False, {"error": "landmarks_norm is not indexable"})
    if n <= _MAX_IDX:
        return (False, {"error": f"Expected 21 landmarks, got {n}"})

    req_fingers = list(required_fingers) if required_fingers else list(_REQUIRED_FINGERS)

    source, detail = _select_source(landmarks_norm, req_fingers, frame_margin)
    if source is None:
        return (False, {"error": f"unmeasurable: {detail}", "detail": detail})

    end_idx = 2 if source == "tip" else 1  # TIP=index 2, PIP=index 1 in the tuple

    vecs: Dict[str, Optional[Tuple[float, float]]] = {}
    for name in _FINGERS:
        mcp = _pt(landmarks_norm, _FINGERS[name][0])
        end = _pt(landmarks_norm, _FINGERS[name][end_idx])
        if mcp is None or end is None:
            vecs[name] = None
            continue
        v = (end[0] - mcp[0], end[1] - mcp[1])
        vecs[name] = None if (v[0] == 0.0 and v[1] == 0.0) else v

    gaps: Dict[str, float] = {}
    for a, b in _ADJACENT_PAIRS:
        if vecs[a] is None or vecs[b] is None:
            continue  # degenerate; leave this gap unmeasured
        gaps[f"{a}-{b}"] = _angle_between(vecs[a], vecs[b])  # type: ignore[arg-type]

    return (True, {"source": source, "gaps": gaps, "detail": detail})


def check_palm_spread(
    landmarks_norm: Any,
    *,
    min_gap_inter_tip_deg: float = _DEFAULT_MIN_GAP_INTER_TIP,
    min_gap_thumb_tip_deg: float = _DEFAULT_MIN_GAP_THUMB_TIP,
    min_gap_inter_pip_deg: float = _DEFAULT_MIN_GAP_INTER_PIP,
    min_gap_thumb_pip_deg: float = _DEFAULT_MIN_GAP_THUMB_PIP,
    required_pairs: Optional[Sequence[str]] = None,
    frame_margin: float = _DEFAULT_FRAME_MARGIN,
) -> Tuple[bool, str]:
    """Are the fingers spread naturally apart (every required gap wide enough)?

    Mirrors the (success, message) contract of the other palm checks. The caller
    is responsible for the N-pose gate (run only on neutral; SKIP otherwise).

    Source selection is hand-wide (tips if all in frame, else PIPs), so all gaps
    share one source and thresholds stay comparable. Thresholds are source-aware.

    Returns:
        (success, message). A non-measurable hand (MCP cropped, or both tip and
        pip cropped for some finger) returns (False, "unmeasurable: ...") -- the
        pipeline should treat that as SKIP, not a defect.
    """
    ok, info = calculate_finger_gaps(
        landmarks_norm, frame_margin=frame_margin)
    if not ok:
        return (False, info.get("error", "Could not compute finger spread"))

    source: str = info["source"]
    gaps: Dict[str, float] = info["gaps"]
    detail: str = info.get("detail", "")
    req = list(required_pairs) if required_pairs else list(_DEFAULT_REQUIRED_PAIRS)

    if source == "tip":
        thumb_floor, inter_floor = min_gap_thumb_tip_deg, min_gap_inter_tip_deg
    else:
        thumb_floor, inter_floor = min_gap_thumb_pip_deg, min_gap_inter_pip_deg

    def _floor(pair: str) -> float:
        return thumb_floor if pair == "thumb-index" else inter_floor

    closed: List[str] = []
    for pair in req:
        g = gaps.get(pair)
        # A required pair we could not measure (degenerate vector) is treated as
        # closed -- we cannot confirm the fingers are apart.
        if g is None or g < _floor(pair):
            closed.append(pair)

    all_gaps_str = " ".join(f"{p}={gaps[p]:.1f}" for p in gaps)
    thr = f"(source={source} min thumb={thumb_floor:g} inter={inter_floor:g})"

    if closed:
        bad = ", ".join(
            f"{p}({'NA' if gaps.get(p) is None else f'{gaps[p]:.1f}'}<{_floor(p):g})"
            for p in closed
        )
        return (False, f"closed: {bad}; source={source}; {all_gaps_str} {thr}")

    return (True, f"spread ok; source={source}; {all_gaps_str} {thr}")