"""Debug overlay for the WALK (gait) pipeline.

Layout (matches what the reviewer asked for, aligning with face + palm):
  - HEADER drawn ON the video, top-left (like the face overlay): id, filename,
    frame index, timestamp, the detected body box label (WxH), and pose status.
    Nothing in the header covers the walker (it sits in the top corner over
    background).
  - SKELETON (33-point limbs + joints) and the body BOUNDING BOX drawn on the
    frame, colored by the worst frame-check status.
  - CHECKS in a strip BELOW the frame (like the palm below-panel): one row per
    check (STATUS | name | reason). The strip GROWS in width to fit the longest
    reason, so descriptions are never cropped; the video is centre-padded to the
    strip width so the two bands stack cleanly. Minimum strip width is 1000px.

Per-frame video: add_frame is called once per sampled frame and the whole
composed canvas (frame + skeleton + header on top, check strip below) is written
to an .mp4. The output is written at the sample rate so its duration matches the
source (the pipeline passes fps=overlay_fps).

Styling (colors, font, outlined-text) is imported from qc.utils.overlay so the
appearance is single-sourced, exactly as overlay_below.py / palm_overlay.py do.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

from qc.utils.overlay import (
    _put, _STATUS_COLOR,
    _GREEN, _RED, _AMBER, _GREY, _WHITE, _BLACK, _PANEL, _FONT,
)
from qc.utils.report import report_sort_key

logger = logging.getLogger(__name__)

# Below-strip minimum / maximum width (px). The single canvas width is decided
# ONCE (from the video's own width, clamped to this band) and used for EVERY
# frame, so the output has a constant size (a video writer requires it, and a
# per-frame-varying width would stretch frames). min keeps a narrow clip
# readable; max keeps a huge clip from producing an oversized canvas. [DESIGN]
# reviewer decision: min 1000, max 3000.
_MIN_CANVAS_WIDTH = 1000
_MAX_CANVAS_WIDTH = 3000

# Font-scale band. Scale tracks canvas width (s = w / 1280) but is clamped so
# text stays legible at 1000 and does not become billboard-sized at 3000.
# [DESIGN] reviewer decision: clamp(s, 0.6, 1.6).
_MIN_FONT_SCALE = 0.6
_MAX_FONT_SCALE = 1.6

# Upscale a source narrower than this before drawing so the header/skeleton/text
# render at a legible absolute size. bbox and landmark px coords are scaled by
# the same factor to stay aligned.
_MIN_RENDER_WIDTH = 960


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

# Standard MediaPipe 33-point pose skeleton edges (index pairs).
_POSE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)


def _truncate_to_width(text, max_w, fs, thick, ellipsis=".."):
    """Trim text to fit within max_w px (safety net; the strip is sized to fit
    reasons, so this rarely triggers). Mirrors palm_overlay._truncate_to_width."""
    if not text:
        return ""
    if cv2.getTextSize(text, _FONT, fs, thick)[0][0] <= max_w:
        return text
    lo, hi, best = 0, len(text), ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = text[:mid].rstrip() + ellipsis
        if cv2.getTextSize(cand, _FONT, fs, thick)[0][0] <= max_w:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best


class WalkOverlayWriter:
    """Per-frame walk overlay writer: header + skeleton + bbox ON the frame,
    checks in a grow-to-fit strip below. One .mp4 out, one frame per add_frame."""

    def __init__(self, out_path, *, fps, volunteer_id, filename,
                 canvas_width=None):
        self.out_path = out_path
        self.fps = max(float(fps), 1.0)
        self.volunteer_id = volunteer_id
        self.filename = filename
        # The ONE canvas width for every frame. If given (from the video's
        # probed width, clamped 1000..3000 by the caller), it is used verbatim.
        # If None, it is derived from the FIRST frame's width on that frame and
        # then held constant for the rest. Either way it never varies per frame.
        self._canvas_width = (int(_clamp(canvas_width, _MIN_CANVAS_WIDTH,
                                         _MAX_CANVAS_WIDTH))
                              if canvas_width else None)
        self._writer = None
        self._size = None            # (canvas_w, canvas_h)
        self._frames_written = 0

    def _ensure_writer(self, canvas_w, canvas_h):
        if self._writer is not None:
            return
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.out_path, fourcc, self.fps,
                                       (canvas_w, canvas_h))
        self._size = (canvas_w, canvas_h)
        if not self._writer.isOpened():
            logger.warning("WalkOverlayWriter: could not open %s", self.out_path)

    def _to_bgr(self, image, color_space):
        if color_space == "RGB":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def add_frame(self, image, color_space, frame_index, timestamp_sec, *,
                  pose_detected, landmarks_px=None, bbox=None, checks=None):
        checks = checks or {}
        frame = self._to_bgr(image, color_space)
        h, w = frame.shape[:2]

        label_dims = (bbox[2], bbox[3]) if bbox is not None else None

        # ---- decide the ONE canvas width (first frame only) ----
        # If the caller passed canvas_width, it is already set. Otherwise derive
        # it from THIS (first) frame's width, clamped to the band, and hold it.
        if self._canvas_width is None:
            self._canvas_width = int(_clamp(w, _MIN_CANVAS_WIDTH,
                                            _MAX_CANVAS_WIDTH))
        canvas_w = self._canvas_width

        # ---- fit the source frame into canvas_w (uniform scale, no distortion) ----
        # Scale the frame so its WIDTH == canvas_w (up for narrow clips, down for
        # huge ones). Aspect ratio preserved: the height changes proportionally.
        # bbox/landmarks are scaled by the same factor so they stay aligned.
        fit = canvas_w / float(w)
        if fit != 1.0:
            frame = cv2.resize(frame, (canvas_w, int(round(h * fit))),
                               interpolation=cv2.INTER_LINEAR)
            h, w = frame.shape[:2]
            bbox, landmarks_px = self._scale_coords(bbox, landmarks_px, fit)

        # Font scale from the CANVAS width (constant across frames since canvas_w
        # is fixed), clamped so text stays legible without becoming oversized.
        s = _clamp(canvas_w / 1280.0, _MIN_FONT_SCALE, _MAX_FONT_SCALE)

        # ---- draw skeleton + bbox + header ON the frame ----
        vid = frame.copy()
        color = self._frame_color(pose_detected, checks)
        self._draw_pose(vid, landmarks_px, bbox, color, w, h)
        self._draw_header(vid, frame_index, timestamp_sec, pose_detected,
                          label_dims, color, s)

        # ---- check strip below, at the SAME fixed width ----
        canvas = self._append_check_strip(vid, checks, canvas_w, s)

        ch, cw = canvas.shape[:2]
        # The video height can vary slightly if the source aspect changes (it
        # shouldn't within one clip), so lock the FULL canvas size on frame 1 and
        # conform later frames to it. Width is already constant by construction.
        self._ensure_writer(cw, ch)
        if self._writer is None:
            return
        if (cw, ch) != self._size:
            canvas = cv2.resize(canvas, (self._size[0], self._size[1]))

        self._writer.write(canvas)
        self._frames_written += 1

    @staticmethod
    def _scale_coords(bbox, landmarks_px, factor):
        if factor == 1.0:
            return bbox, landmarks_px
        if bbox is not None:
            bx, by, bw, bh = bbox
            bbox = (int(round(bx * factor)), int(round(by * factor)),
                    int(round(bw * factor)), int(round(bh * factor)))
        if landmarks_px:
            landmarks_px = [(int(round(px * factor)), int(round(py * factor)), z)
                            for (px, py, z) in landmarks_px]
        return bbox, landmarks_px

    def _frame_color(self, pose_detected, checks):
        statuses = [s_ for (s_, _r) in checks.values()]
        if not pose_detected:
            return _GREY
        if "FAIL" in statuses:
            return _RED
        return _GREEN

    def _draw_pose(self, vid, landmarks_px, bbox, color, w, h):
        body_dim = max(bbox[2], bbox[3]) if bbox is not None else h // 2
        limb_thick = max(1, min(5, int(round(body_dim * 0.004))))
        joint_r = max(1, min(6, int(round(body_dim * 0.005))))

        if landmarks_px:
            n = len(landmarks_px)
            for a, b in _POSE_EDGES:
                if a < n and b < n:
                    ax, ay = landmarks_px[a][0], landmarks_px[a][1]
                    bx, by = landmarks_px[b][0], landmarks_px[b][1]
                    cv2.line(vid, (ax, ay), (bx, by), color, limb_thick,
                             cv2.LINE_AA)
            for (px, py, _z) in landmarks_px:
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(vid, (px, py), joint_r, _WHITE, -1, cv2.LINE_AA)

        if bbox is not None:
            bx, by, bw, bh = bbox
            box_thick = max(1, min(4, int(round(body_dim * 0.003))))
            cv2.rectangle(vid, (bx, by), (bx + bw, by + bh), color, box_thick)

    def _draw_header(self, vid, frame_index, timestamp_sec, pose_detected,
                     label_dims, color, s):
        fs = max(0.5, 0.9 * s)
        thick = max(1, int(round(2 * s)))
        pad = int(14 * s)
        line_h = int(34 * s)

        label = f"{label_dims[0]}x{label_dims[1]}" if label_dims else "no-body"
        lines = [
            (f"id {self.volunteer_id}   {self.filename}", _WHITE),
            (f"frame {frame_index}   t={timestamp_sec:.2f}s", _WHITE),
            (f"label={label}   {'pose OK' if pose_detected else 'NO POSE'}",
             color),
        ]

        panel_w = min(
            vid.shape[1],
            max(cv2.getTextSize(t, _FONT, fs, thick)[0][0] for t, _ in lines)
            + pad * 2,
        )
        panel_h = line_h * len(lines) + pad
        ov = vid.copy()
        cv2.rectangle(ov, (0, 0), (panel_w, panel_h), _PANEL, -1)
        cv2.addWeighted(ov, 0.55, vid, 0.45, 0, vid)

        y = pad + int(line_h * 0.7)
        for text, col in lines:
            _put(vid, text, (pad, y), col, fs, thick)
            y += line_h

    def _append_check_strip(self, vid, checks, canvas_w, s):
        """Append a grey check strip below the frame, at the FIXED canvas width.

        Unlike the palm version (which grew the strip to the reason and padded
        the photo), the walk strip is built at exactly `canvas_w` for EVERY frame
        so the video size is constant. The reason column takes whatever width is
        left after the status + name columns; a reason too long for that space is
        truncated with '..' (rare at the 3000px cap). The video is already
        canvas_w wide (fitted in add_frame), so no padding is needed here.
        """
        if not checks:
            return vid

        thick = max(1, int(round(1.4 * s)))
        fs = 0.9 * s
        line_h = int(42 * s)
        pad = int(16 * s)
        gap = int(30 * s)

        # Order the strip by the SAME report order the CSV uses (not alphabetical)
        # so the overlay and the report agree row-for-row.
        names = sorted(checks.keys(),
                       key=lambda nm: report_sort_key(nm, "frame", "walk"))
        items = []
        for nm in names:
            v = checks[nm]
            st = str(v[0]) if isinstance(v, (tuple, list)) else str(v)
            rs = str(v[1]) if isinstance(v, (tuple, list)) and len(v) > 1 else ""
            items.append((nm, st, rs))

        label_texts = [f"{st:5s} {nm}" for nm, st, _r in items]
        max_label_w = max(cv2.getTextSize(t, _FONT, fs, thick)[0][0]
                          for t in label_texts)

        status_col_w = int(95 * s)
        name_col_w = max_label_w + gap
        status_x = pad
        name_x = status_x + status_col_w
        reason_x = pad + name_col_w
        reason_col_w = max(0, canvas_w - reason_x - pad)

        strip_h = line_h * len(items) + pad * 2
        strip = np.full((strip_h, canvas_w, 3), 18, dtype=np.uint8)

        y = pad + int(28 * s)
        for name, status, reason in items:
            col = _STATUS_COLOR.get(status, _WHITE)
            _put(strip, f"{status:5s}", (status_x, y), col, fs, thick)
            _put(strip, name, (name_x, y), col, fs, thick)
            if reason and reason_col_w > 0:
                short = _truncate_to_width(reason, reason_col_w, fs, thick)
                if short:
                    _put(strip, short, (reason_x, y), col, fs, thick)
            y += line_h

        cv2.line(strip, (0, 0), (canvas_w, 0), (80, 80, 80), max(1, int(2 * s)))
        return np.vstack([vid, strip])

    def close(self):
        if self._writer is not None:
            self._writer.release()
            logger.info("WalkOverlayWriter: wrote %d frames to %s",
                        self._frames_written, self.out_path)

    @property
    def frames_written(self):
        return self._frames_written