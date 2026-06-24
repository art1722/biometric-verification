"""Palm overlay — annotate ONE still image and save ONE .jpg.

The palm-side mirror of qc/utils/overlay.py, but deliberately simpler. The face
overlay is a VIDEO writer (OverlayWriter streams frame-by-frame to .mp4) because
face QC runs on a video. Palm QC runs on a single still, so this is a one-shot:
draw on the image, write a .jpg. No video container, no per-frame loop.

What it draws (only what detect_hand actually provides)
------------------------------------------------------
  - 21 hand landmarks as dots (HandResult.landmarks_px)
  - the hand skeleton (bones) connecting those landmarks, so the pose reads at
    a glance instead of as a dot cloud (HAND_CONNECTIONS below)
  - the bounding box (HandResult.bbox) with its WxH label  <-- yes, detect_hand
    returns a bbox (landmark extent + 10% margin), so it is drawn
  - a header stat panel: handedness + score, bbox WxH, landmark count, and any
    (check_name -> status) results passed in

Everything is guarded: if a field is None (no hand detected, no bbox), that part
is simply skipped and the rest still draws. "Write whatever is available."

Colour + text style copied from overlay.py so face and palm annotations look
consistent (light-on-near-black for legibility on any image).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import cv2
import numpy as np

# --- style, matched to qc/utils/overlay.py (BGR) ---
_GREEN = (90, 255, 90)
_RED = (90, 90, 255)
_AMBER = (60, 200, 255)
_GREY = (190, 190, 190)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_PANEL = (20, 20, 20)
_CYAN = (255, 220, 90)     # light cyan for the bone skeleton
_FONT = cv2.FONT_HERSHEY_SIMPLEX

_STATUS_COLOR = {"PASS": _GREEN, "FAIL": _RED, "SKIP": _AMBER, "REVIEW": _AMBER}

# Fingertip landmark indices (drawn larger, like face's _HEAD_FULLY_IDXS).
_TIP_IDXS = (4, 8, 12, 16, 20)

# Standard MediaPipe hand skeleton: pairs of landmark indices to connect.
# Thumb, index, middle, ring, pinky chains + the palm base across the knuckles.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # wrist -> pinky base (palm edge)
)


def _put(img, text, org, color, scale=0.5, thick=2):
    """Text with a thick black outline so it stays legible on any background."""
    cv2.putText(img, text, org, _FONT, scale, _BLACK, thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, thick, cv2.LINE_AA)


def _to_bgr(image: Any, color_space: str):
    """Resolve a path or array to a BGR array for drawing."""
    if isinstance(image, str):
        return cv2.imread(image)
    if color_space == "RGB":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image.copy()


def draw_palm_overlay(
    image: Any,
    result,
    *,
    color_space: str = "BGR",
    checks: Optional[dict] = None,
    out_path: Optional[str] = None,
):
    """Annotate one image with a HandResult and optionally save it.

    Args:
        image: file path (str) or a decoded array.
        result: a HandResult from detect_hand (carries landmarks_px / bbox /
            handedness / handedness_score). Any field may be None; whatever is
            present is drawn, the rest is skipped.
        color_space: "BGR" (OpenCV default) or "RGB" for the input array.
        checks: optional {check_name: status} to list in the header panel,
            e.g. {"check_palm_present": "PASS", "check_palm_size": "FAIL"}.
        out_path: if given, the annotated image is written here (.jpg/.png by
            extension) and the path is returned.

    Returns:
        The annotated BGR image array (and writes out_path if provided).
    """
    img = _to_bgr(image, color_space)
    if img is None:
        raise ValueError("could not load image for overlay")

    h, w = img.shape[:2]
    landmarks = getattr(result, "landmarks_px", None)
    bbox = getattr(result, "bbox", None)
    handedness = getattr(result, "handedness", None)
    hand_score = getattr(result, "handedness_score", None)

    # Sizes scale with the image so dots/lines/text are proportionate on a
    # small crop or a large photo (same idea as the face overlay's scaling).
    dim_min = min(w, h)
    lm_radius = max(2, min(8, int(round(dim_min * 0.006))))
    bone_thick = max(1, min(6, int(round(dim_min * 0.004))))
    box_thick = max(1, min(10, int(round(dim_min * 0.005))))
    fs = max(0.4, min(2.0, dim_min * 0.0012))

    # --- skeleton (bones) first, so dots sit on top ---
    if landmarks:
        for a, b in HAND_CONNECTIONS:
            if a < len(landmarks) and b < len(landmarks):
                ax, ay, _ = landmarks[a]
                bx2, by2, _ = landmarks[b]
                if 0 <= ax < w and 0 <= ay < h and 0 <= bx2 < w and 0 <= by2 < h:
                    cv2.line(img, (ax, ay), (bx2, by2), _CYAN, bone_thick, cv2.LINE_AA)

    # --- landmark dots (fingertips larger / red, like face's key idxs) ---
    if landmarks:
        for i, (px, py, _z) in enumerate(landmarks):
            if 0 <= px < w and 0 <= py < h:
                if i in _TIP_IDXS:
                    cv2.circle(img, (px, py), lm_radius + 2, _RED, -1, cv2.LINE_AA)
                else:
                    cv2.circle(img, (px, py), lm_radius, (210, 210, 210), -1, cv2.LINE_AA)

    # --- bounding box + WxH label (detect_hand DOES return a bbox) ---
    if bbox is not None:
        bx, by, bw, bh = bbox
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), _GREEN, box_thick)
        label = f"{bw}x{bh}"
        ly = max(int(fs * 30), by - 8)
        _put(img, label, (bx + 4, ly), _GREEN, scale=fs, thick=max(1, box_thick))

    # --- header stat panel (top-left) ---
    lines = []
    if handedness is not None:
        s = f" ({hand_score:.2f})" if hand_score is not None else ""
        lines.append((f"hand: {handedness}{s}", _WHITE))
    if bbox is not None:
        lines.append((f"bbox: {bbox[2]}x{bbox[3]} px", _WHITE))
    lines.append((f"landmarks: {len(landmarks) if landmarks else 0}/21", _WHITE))
    if not getattr(result, "ok", True):
        lines.append((f"detect: {getattr(result, 'message', 'no hand')}", _AMBER))
    if checks:
        for name, status in checks.items():
            lines.append((f"{name}: {status}",
                          _STATUS_COLOR.get(status, _GREY)))

    if lines:
        pad = int(fs * 12)
        line_h = int(fs * 34)
        panel_w = int(max(len(t) for t, _ in lines) * fs * 15) + pad * 2
        panel_h = line_h * len(lines) + pad
        ov = img.copy()
        cv2.rectangle(ov, (0, 0), (min(panel_w, w), panel_h), _PANEL, -1)
        cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)
        y = line_h
        for text, color in lines:
            _put(img, text, (pad, y), color, scale=fs, thick=max(1, int(fs * 2)))
            y += line_h

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, img)

    return img
