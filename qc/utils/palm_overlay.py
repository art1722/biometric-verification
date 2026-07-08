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
    panel_below: bool = True,
    draw_angle_vectors: bool = True,
    min_strip_width: int = 900,
):
    """Annotate one image with a HandResult and optionally save it.

    Args:
        image: file path (str) or a decoded array.
        result: a HandResult from detect_hand (carries landmarks_px / bbox /
            handedness / handedness_score). Any field may be None; whatever is
            present is drawn, the rest is skipped.
        color_space: "BGR" (OpenCV default) or "RGB" for the input array.
        checks: optional {check_name: (status, reason)} (or bare status) to
            list in the bottom check panel.
        out_path: if given, the annotated image is written here (.jpg/.png by
            extension) and the path is returned.
        panel_below: when True (DEFAULT), the per-check panel is drawn on a
            separate grey strip APPENDED BELOW the image, so it never covers the
            palm. When False, the panel is drawn ON TOP of the image (the old
            behaviour; the runner's --overlay-on-image flag sets this).
        draw_angle_vectors: when True (DEFAULT), draw the angle geometry on
            every pose including N: the palm "up" axis (wrist -> knuckle
            midpoint), the "across" axis (index_mcp -> pinky_mcp), orange rings
            on the FIVE angle-reference landmarks (PLANE_LANDMARK_IDXS: wrist + the
            four finger MCPs -- exactly the points v3's least-squares palm
            plane is fitted through), and the palm NORMAL projected into the
            image as an orange arrow from the plane points' centroid (its 2D
            length grows with how far the palm is tilted away from the camera;
            near-zero tilt is drawn as a small circle labelled "normal ->
            camera"). Roll/pitch from the depth-wise angle check are also added to the
            header panel.
        min_strip_width: minimum pixel width for the BELOW check strip. A
            portrait/narrow palm photo would otherwise starve the reason column
            and truncate reasons ("container ok (.jp..") even for a PASS. When
            the source image is narrower than this, the strip is drawn at this
            width and the photo is centre-padded to match so they stack cleanly.
            No effect on wide images or when panel_below is False.

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

    # v3 angle-reference angles: computed here from the WORLD landmarks (when the
    # HandResult carries them) so the overlay reports/draws EXACTLY what
    # check_palm_angle measures -- one source of truth, no re-derivation.
    angle_info = None
    _world_lms = getattr(result, "world_landmarks", None)
    if _world_lms is not None:
        try:
            from qc.checks.check_palm_angle import calculate_palm_angles
            _aok, _ainfo = calculate_palm_angles(
                _world_lms, handedness=getattr(result, "handedness", None))
            if _aok:
                angle_info = _ainfo
        except Exception:
            angle_info = None  # drawing must never break on a math error

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

    # --- angle geometry: draw what check_palm_angle (v3) measures, on every
    # pose including N. The v3 measurement fits a least-squares PLANE through
    # WRIST(0) + the four finger MCPs (5, 9, 13, 17) and reads roll/pitch from
    # the plane NORMAL. We draw, in PIXEL space (world coords aren't drawable):
    #   - orange rings on the five angle-reference landmarks,
    #   - the projected palm NORMAL as an orange arrow from their centroid
    #     (2D length ~ sin(tilt); a palm facing the camera has a ~zero arrow),
    #   - the legacy "up" / "across" axes, kept as orientation context.
    if draw_angle_vectors and landmarks and len(landmarks) > 17:
        try:
            import math as _math
            wx, wy, _ = landmarks[0]      # wrist
            ix, iy, _ = landmarks[5]      # index_mcp
            kx, ky, _ = landmarks[17]     # pinky_mcp
            mx, my = (ix + kx) // 2, (iy + ky) // 2   # knuckle midpoint
            vec_thick = max(2, bone_thick + 1)

            # tipLength in cv2.arrowedLine is a FRACTION OF THE LINE LENGTH, so
            # a short vector (the "across" knuckle line) would get a tiny,
            # near-invisible head while a long one (the "up" axis) looks fine.
            # Fix: target a FIXED arrowhead size in pixels (scaled to the image)
            # and convert to the per-line fraction = head_px / line_length. This
            # gives both arrows an equally visible head regardless of length.
            head_px = max(18.0, min(w, h) * 0.025)

            def _tip_frac(x0, y0, x1, y1):
                length = _math.hypot(x1 - x0, y1 - y0)
                if length < 1.0:
                    return 0.3  # degenerate; let OpenCV draw something
                return max(0.05, min(0.6, head_px / length))

            # "across" axis (knuckle line, index_mcp -> pinky_mcp) in magenta.
            cv2.arrowedLine(img, (ix, iy), (kx, ky), (255, 90, 220),
                            vec_thick, cv2.LINE_AA,
                            tipLength=_tip_frac(ix, iy, kx, ky))
            # "up" axis (wrist -> knuckle midpoint) in yellow.
            cv2.arrowedLine(img, (wx, wy), (mx, my), (60, 255, 255),
                            vec_thick, cv2.LINE_AA,
                            tipLength=_tip_frac(wx, wy, mx, my))
            # Small labels at the arrow heads.
            _put(img, "across", (kx + 6, ky), (255, 90, 220),
                 scale=fs * 0.7, thick=max(1, int(fs)))
            _put(img, "up", (mx + 6, my), (60, 255, 255),
                 scale=fs * 0.7, thick=max(1, int(fs)))

            # --- depth-angle landmarks + projected debug normal ---
            _ORANGE = (60, 160, 255)  # BGR
            try:
                from qc.checks.check_palm_angle import PLANE_LANDMARK_IDXS
            except Exception:
                PLANE_LANDMARK_IDXS = (0, 5, 9, 13, 17)

            plane_pts = [(landmarks[i][0], landmarks[i][1])
                         for i in PLANE_LANDMARK_IDXS if i < len(landmarks)]
            for (ppx, ppy) in plane_pts:
                if 0 <= ppx < w and 0 <= ppy < h:
                    cv2.circle(img, (ppx, ppy), lm_radius + 4, _ORANGE,
                               max(2, bone_thick), cv2.LINE_AA)

        except Exception:
            pass  # never let a drawing glitch break the overlay

    # --- bounding box + WxH label (detect_hand DOES return a bbox) ---
    if bbox is not None:
        bx, by, bw, bh = bbox
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), _GREEN, box_thick)
        label = f"{bw}x{bh}"
        ly = max(int(fs * 30), by - 8)
        _put(img, label, (bx + 4, ly), _GREEN, scale=fs, thick=max(1, box_thick))

    # --- header stat panel (top-left): hand / bbox / landmarks ONLY ---
    # The per-check results moved to the BOTTOM panel (face-overlay style), so
    # the header now carries just the detection facts.
    lines = []
    if handedness is not None:
        s = f" ({hand_score:.2f})" if hand_score is not None else ""
        lines.append((f"hand: {handedness}{s}", _WHITE))
    if bbox is not None:
        lines.append((f"bbox: {bbox[2]}x{bbox[3]} px", _WHITE))
    lines.append((f"landmarks: {len(landmarks) if landmarks else 0}/21", _WHITE))
    if angle_info is not None:
        lines.append((f"roll={angle_info['roll']:+.1f} "
                      f"pitch={angle_info['pitch']:+.1f} (depth tilt)", _WHITE))
    if not getattr(result, "ok", True):
        lines.append((f"detect: {getattr(result, 'message', 'no hand')}", _AMBER))

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

    # --- per-check panel ---
    # Default (panel_below=True): append a separate grey strip BELOW the image
    # and draw the checks there, so the panel never covers the palm. The image
    # canvas grows taller; the original photo is untouched above the strip.
    # panel_below=False: draw the panel ON TOP of the image (old behaviour;
    # the runner's --overlay-on-image flag).
    if checks:
        if panel_below:
            img = _append_check_strip(img, checks, w, fs,
                                      min_strip_width=min_strip_width)
        else:
            _draw_check_panel(img, checks, w, h, fs)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, img)

    return img


def _append_check_strip(img, checks: dict, img_w: int, base_fs: float,
                        min_strip_width: int = 900):
    """Return a NEW image: the original with a grey check-strip appended below.

    The strip is its own band of pixels (not drawn over the photo), tall enough
    to hold one row per check. Columns: STATUS | name | reason.

    Width handling (the portrait-image fix):
      The strip is drawn at strip_w = max(img_w, min_strip_width). If the source
      photo is narrower than min_strip_width (a portrait palm shot), the strip is
      widened to a readable size so the reason column is not starved and reasons
      stop truncating to "..". Because np.vstack needs equal widths, the photo is
      then centre-padded (black bars left/right) to strip_w. Wide photos are
      unaffected (strip_w == img_w, no padding).
    """
    if not checks:
        return img

    s = max(0.5, min(2.5, base_fs / 0.85))
    fs = 0.85 * s
    thick = max(1, int(round(1.25 * s)))
    line_h = int(40 * s)
    pad = int(16 * s)

    names = sorted(checks.keys())
    items = [(nm, *_normalize_check(checks[nm])) for nm in names]

    # Column geometry, measured from the actual rendered text.
    label_texts = [f"{st:5s} {nm}" for nm, st, _r in items]
    max_label_w = max(
        cv2.getTextSize(t, _FONT, fs, thick)[0][0]
        for t in label_texts
    )

    gap = int(30 * s)
    status_col_w = int(95 * s)
    name_col_w = max_label_w + gap

    status_x = pad
    name_x = status_x + status_col_w
    reason_x = pad + name_col_w

    reason_texts = [reason for _name, _status, reason in items if reason]
    max_reason_w = max(
        (
            cv2.getTextSize(reason, _FONT, fs, thick)[0][0]
            for reason in reason_texts
        ),
        default=0,
    )

    wanted_reason_w = max(int(420 * s), max_reason_w + int(20 * s))

    strip_w = max(
        img_w,
        int(min_strip_width),
        reason_x + wanted_reason_w + pad,
    )

    reason_col_w = max(0, strip_w - reason_x - pad)

    strip_h = line_h * len(items) + pad * 2
    strip = np.full((strip_h, strip_w, 3), 35, dtype=np.uint8)  # dark grey

    y = pad + int(28 * s)

    for name, status, reason in items:
        color = _STATUS_COLOR.get(status, _WHITE)
        _put(strip, status, (status_x, y), color, fs, thick)
        _put(strip, name, (name_x, y), color, fs, thick)
        if reason and reason_col_w > 0:
            short = _truncate_to_width(reason, reason_col_w, fs, thick)
            if short:
                _put(strip, short, (reason_x, y), color, fs, thick)
        y += line_h

    # A thin divider line between photo and strip.
    cv2.line(strip, (0, 0), (strip_w, 0), (80, 80, 80), max(1, int(2 * s)))

    # If the strip is wider than the photo, centre-pad the photo so the two
    # bands have equal width and stack cleanly (np.vstack requirement).
    if strip_w > img_w:
        left = (strip_w - img_w) // 2
        right = strip_w - img_w - left
        img = cv2.copyMakeBorder(img, 0, 0, left, right,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return np.vstack([img, strip])


def _normalize_check(value):
    """Accept either a bare status string OR a (status, reason) tuple.

    The runner now passes (status, reason); older callers may pass just the
    status. Returns (status, reason) with reason="" when only a status is given.
    """
    if isinstance(value, (tuple, list)):
        status = value[0] if len(value) > 0 else ""
        reason = value[1] if len(value) > 1 else ""
        return str(status), str(reason or "")
    return str(value), ""


def _truncate_to_width(text, max_w, fs, thick, ellipsis=".."):
    """Trim `text` so it fits within `max_w` px, appending ".." if cut.

    Measures with cv2.getTextSize (the real rendered width), shrinking from the
    end until text+".." fits. Returns "" for empty input; returns the ellipsis
    alone only if even that does not fit (rare).
    """
    if not text:
        return ""
    full_w = cv2.getTextSize(text, _FONT, fs, thick)[0][0]
    if full_w <= max_w:
        return text
    # Binary-ish shrink: drop characters until text+ellipsis fits.
    lo, hi = 0, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = text[:mid].rstrip() + ellipsis
        cw = cv2.getTextSize(cand, _FONT, fs, thick)[0][0]
        if cw <= max_w:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _draw_check_panel(img, checks: dict, w: int, h: int, base_fs: float):
    """Bottom-left panel: one row per check, columns STATUS | name | reason.

    Mirrors the face overlay's _draw_check_panel. `checks` maps
    check_name -> (status, reason) (or bare status; see _normalize_check). The
    reason column is truncated with ".." to the panel's right edge.
    """
    if not checks:
        return

    # Scale relative to the image, like the face panel (which scales by a
    # per-frame `s`). Here base_fs already encodes image size; derive a panel
    # scale from it so rows are readable on a 4032px photo and a small crop.
    s = max(0.5, min(2.5, base_fs / 0.85))
    fs = 0.85 * s
    thick = max(1, int(round(1.25 * s)))
    line_h = int(34 * s)
    pad = int(12 * s)

    names = sorted(checks.keys())
    items = [(nm, *_normalize_check(checks[nm])) for nm in names]

    # Column geometry: STATUS | name | reason. Size the name column to the
    # widest "STATUS  name" so the reason column never overlaps it.
    label_texts = [f"{st:5s} {nm}" for nm, st, _r in items]
    max_label_w = max(
        cv2.getTextSize(t, _FONT, fs, thick)[0][0] for t in label_texts)
    gap = int(30 * s)

    status_col_w = int(95 * s)            # width reserved for the STATUS word
    name_col_w = max_label_w + gap        # reason starts after the widest label

    # Panel width: leave a right margin; cap the reason column to the room left.
    side_margin = int(20 * s)
    x0 = int(12 * s)
    panel_w_max = w - x0 - side_margin
    # Target a generous reason column but never exceed the image.
    reason_col_w = max(int(300 * s), panel_w_max - (pad + name_col_w + pad))
    panel_w = min(pad + name_col_w + reason_col_w + pad, panel_w_max)
    # Recompute the real reason width inside the (possibly clamped) panel.
    reason_col_w = max(0, panel_w - (pad + name_col_w + pad))

    panel_h = line_h * len(items) + pad * 2
    y0 = h - panel_h - int(16 * s)

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), _PANEL, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    status_x = x0 + pad
    name_x = status_x + status_col_w
    reason_x = x0 + pad + name_col_w
    y = y0 + pad + int(26 * s)

    for name, status, reason in items:
        color = _STATUS_COLOR.get(status, _WHITE)
        _put(img, status, (status_x, y), color, fs, thick)
        _put(img, name, (name_x, y), color, fs, thick)
        if reason and reason_col_w > 0:
            short = _truncate_to_width(reason, reason_col_w, fs, thick)
            if short:
                _put(img, short, (reason_x, y), color, fs, thick)
        y += line_h