"""Occlusion / foreign-object check using a COCO YOLO detector.

Spec: for every frame, only ONE COCO class is allowed in the scene: 0=person
(the walker). Any of the other 79 classes (bicycle, car, chair, backpack, cell
phone, ... toothbrush) counts as a foreign object occluding / contaminating the
capture, so the frame FAILs.

This module is the PURE decision half: it takes the already-computed detections
(from qc.checks.yolo_detector.detect_objects) plus a confidence gate and returns
(success, message), the same (bool, str) contract every other check uses. It does
NOT run the model, so it has no ultralytics dependency and is trivial to test.

Confidence gate
---------------
Low-confidence ghost detections (a 0.15 "toothbrush" on a noisy frame) would
false-FAIL almost every clip, so a foreign object only counts when its
confidence is >= conf. Default 0.35 [DESIGN]; confirm the value with อ.เหมียว.
The person class is never a failure reason, at any confidence.

Aggregation (done by report.py, not here)
-----------------------------------------
This returns a per-FRAME verdict. report.py aggregates the frame verdicts into
the video-level result by the configured fail-ratio (walk.occlusion.
frame_fail_ratio, default 0.20), exactly like brightness and blur: a video FAILs
check_occlusion only if MORE than 20% of judged frames have a foreign object, so
a single spurious frame does not sink an otherwise clean clip.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from qc.checks.yolo_detector import Detection, PERSON_CLASS_ID

logger = logging.getLogger(__name__)

DEFAULT_CONF = 0.35


def _boxes_overlap(a, b) -> bool:
    """True if two (x, y, w, h) pixel boxes overlap at all (intersection > 0).

    'Any overlap' per spec: even a 1px intersection counts as กีดขวาง. Uses a
    strict > so boxes that merely touch at an edge (zero-area intersection) do
    not count.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    return (ix2 - ix) > 0 and (iy2 - iy) > 0


def check_occlusion_yolo(
    detections: Optional[Iterable[Detection]],
    *,
    conf: float = DEFAULT_CONF,
    person_bbox: Optional[tuple] = None,
    allowed_class_ids: Iterable[int] = (PERSON_CLASS_ID,),
    max_names_in_message: int = 5,
):
    """Frame-level occlusion verdict from COCO detections, using person overlap.

    Spec ("ไม่มีสิ่งกีดขวาง" = no obstruction of passage): a foreign object only
    counts if it actually obstructs the walker, approximated by its 2D box
    overlapping the person's box. An object off to the side (no overlap) is NOT
    an obstruction -> it does not fail the frame.

    Args:
        detections: iterable of Detection for this frame (may be empty/None).
        conf: only detections with confidence >= conf are considered.
        person_bbox: the walker's (x, y, w, h) in PIXELS. The caller supplies the
            MediaPipe pose bbox, falling back to YOLO's own person detection when
            pose failed. If None, NO person box is available this frame:
              - any foreign object present -> FAIL ("can't clear": with no person
                box we cannot prove the object is off the path).
              - no foreign object -> PASS.
        allowed_class_ids: COCO ids that never count as occlusion (default person).
        max_names_in_message: cap distinct classes listed in the message.

    Returns:
        (success, message).
          PASS: no foreign object, or foreign objects exist but none overlaps the
                person box.
          FAIL: at least one foreign object overlaps the person box; OR a foreign
                object exists and there is no person box to clear it against. The
                message lists the offending classes with confidence.
    """
    allowed = set(allowed_class_ids)

    if not detections:
        return (True, "no objects detected")

    # Gather foreign objects (non-allowed class, at/above conf) with their box.
    # Keep the highest-confidence sighting per class for the message, but decide
    # PASS/FAIL on the actual overlapping instances.
    foreign = [d for d in detections
               if d.conf >= conf and d.cls_id not in allowed]

    if not foreign:
        return (True, "only person present")

    # No person box this frame -> we cannot check overlap. Per spec decision, an
    # object we cannot clear against a person is treated as obstructing -> FAIL.
    if person_bbox is None:
        best = _rank_message(foreign, max_names_in_message)
        return (False, "foreign objects (no person box to clear): " + best)

    # Person box present -> FAIL only the objects that overlap it.
    overlapping = [d for d in foreign if _boxes_overlap(d.bbox, person_bbox)]

    if not overlapping:
        return (True, f"{len(foreign)} object(s) present but none overlaps person")

    return (False, "obstructing objects: " + _rank_message(overlapping,
                                                           max_names_in_message))


def _rank_message(dets, max_names_in_message: int) -> str:
    """Build 'name (conf), ...' listing highest-confidence sighting per class."""
    best: dict = {}
    for d in dets:
        prev = best.get(d.cls_name)
        if prev is None or d.conf > prev:
            best[d.cls_name] = d.conf
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    shown = ranked[:max_names_in_message]
    parts = [f"{name} ({c:.2f})" for name, c in shown]
    extra = len(ranked) - len(shown)
    if extra > 0:
        parts.append(f"+{extra} more")
    return ", ".join(parts)