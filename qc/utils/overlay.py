"""Debug overlay video writer.

Draws a "full debug" overlay onto each SAMPLED frame as the face_rgb pipeline
runs, so a reviewer (e.g. พี่ยอ) can SEE why a case broke instead of reading the
CSV. The writer is fed live from inside the per-frame loop, so it draws the
EXACT bbox / landmarks / pose / check results the checks used — no re-detection,
no second pass, no drift between the video and the report.

Design contract:
- This module never runs a model and never re-reads the video. It only draws.
- The pipeline holds an OverlayWriter (or None). When None, every hook is a
  no-op, so behavior with --overlay absent is byte-for-byte unchanged.
- Output contains only the SAMPLED frames (matches the QC timeline, ~sample_fps),
  written back-to-back. Playback is therefore time-compressed vs the source;
  that is intentional for review (you see every judged frame, nothing skipped).

What is drawn (full debug):
  - face bounding box (green = all front-only checks passed this frame,
    red = at least one FAILed, amber = non-frontal/SKIP, grey = no face)
  - all 468 face-mesh landmarks as faint dots (only when a face was detected)
  - pose readout: yaw / pitch / roll and the classified label (front/left/...)
  - per-check status lines: PASS / FAIL / SKIP for every frame-level check,
    each with the measured value pulled from the check's own reason string
  - a header strip: frame index, timestamp, volunteer id, filename
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# BGR colors (OpenCV order).
_GREEN = (0, 200, 0)
# BGR colors (OpenCV order). Status colors are deliberately LIGHT/bright so
# they stand out against both dark and light footage, and they sit on a
# near-black panel at the bottom for maximum contrast.
_GREEN = (90, 255, 90)     # light green
_RED = (90, 90, 255)       # light red
_AMBER = (60, 200, 255)    # light amber
_GREY = (190, 190, 190)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_PANEL = (20, 20, 20)      # near-black panel behind the check list

_STATUS_COLOR = {
    "PASS": _GREEN,
    "FAIL": _RED,
    "SKIP": _AMBER,
    "REVIEW": _AMBER,
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put(img, text, org, color, scale=0.5, thick=2):
    """Draw text with a thick black outline so it stays readable on any frame.

    A heavier outline (thick+3) keeps light-colored text legible even over
    bright or busy footage.
    """
    cv2.putText(img, text, org, _FONT, scale, _BLACK, thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, thick, cv2.LINE_AA)


class OverlayWriter:
    """Collects per-frame draw data and writes an annotated video.

    Lifecycle:
        w = OverlayWriter(out_path, fps=..., volunteer_id=..., filename=...)
        # inside the loop, once per sampled frame:
        w.add_frame(image, color_space, frame_index, timestamp_sec,
                    face_detected=..., landmarks=..., bbox=..., pose=...,
                    label=..., checks={check_name: (status, reason), ...})
        w.close()

    The writer is lazily initialized on the first frame, because the output
    VideoWriter needs the frame width/height and those are only known then.
    """

    def __init__(self, out_path: str, *, fps: float,
                 volunteer_id: str, filename: str):
        self.out_path = out_path
        # Output fps: play sampled frames at the sampling rate so 5fps sampling
        # plays at 5fps. Clamp to a sane floor so a <1fps sample still plays.
        self.fps = max(float(fps), 1.0)
        self.volunteer_id = volunteer_id
        self.filename = filename
        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Optional[tuple[int, int]] = None  # (w, h)
        self._frames_written = 0

    # -- internal ----------------------------------------------------------

    def _ensure_writer(self, w: int, h: int):
        if self._writer is not None:
            return
        # mp4v is broadly available in OpenCV builds and writes .mp4.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (w, h))
        self._size = (w, h)
        if not self._writer.isOpened():
            logger.warning("OverlayWriter: could not open %s for writing",
                           self.out_path)

    def _to_bgr(self, image: Any, color_space: str):
        if color_space == "RGB":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    # -- public hook -------------------------------------------------------

    def add_frame(self, image: Any, color_space: str,
                  frame_index: int, timestamp_sec: float, *,
                  face_detected: bool,
                  landmarks: Optional[list] = None,
                  bbox: Optional[tuple] = None,
                  pose: Optional[dict] = None,
                  label: Optional[str] = None,
                  checks: Optional[dict] = None):
        """Draw one sampled frame and write it. Safe to call every frame."""
        checks = checks or {}
        img = self._to_bgr(image, color_space).copy()
        h, w = img.shape[:2]
        self._ensure_writer(w, h)
        if self._writer is None:
            return

        # If a previous frame set a size and this one differs, resize so the
        # writer stays valid (sampled frames should all match, but be safe).
        if self._size and (w, h) != self._size:
            img = cv2.resize(img, self._size)
            h, w = self._size[1], self._size[0]

        # --- scale everything by frame height so text is readable but compact ---
        # Reference 900p: s=0.8 at 720p, s=1.2 at 1080p, s=0.53 at 480p.
        # Tuning history: h/480 (too small) -> h/480 w/ big multipliers (too big)
        #  -> h/720 (still too big) -> h/900 (this). To resize, change the 900.0:
        #  smaller divisor = bigger text, larger divisor = smaller text.
        s = h / 900.0
        fs_head = 0.95 * s      # header (id / file) — text tracks the FRAME
        fs_sub = 0.85 * s       # frame/time + pose line

        # --- decide bbox color from the worst frame-level status ---
        statuses = [s_ for (s_, _r) in checks.values()]
        if not face_detected:
            box_color = _GREY
        elif "FAIL" in statuses:
            box_color = _RED
        elif label and label != "front":
            box_color = _AMBER
        else:
            box_color = _GREEN

        # Box / dot / box-label scale with the FACE bbox, not the frame, so a
        # large (close) face gets bigger dots and a small (far) face gets
        # smaller ones. (Fixes "dots invisible on large video".) Text above
        # stays frame-scaled because legibility should track the frame.
        if bbox is not None:
            _bw, _bh = bbox[2], bbox[3]
            face_dim = max(_bw, _bh)
        else:
            face_dim = h // 4   # fallback if no face this frame

        lm_radius = max(1, min(3, int(round(face_dim * 0.003))))
        box_thick = max(1, min(9, int(round(face_dim * 0.009))))
        
        fs_box = max(0.5, face_dim * 0.005)   # the WxH label on the box

        # --- landmarks (faint) ---
        if landmarks:
            for (px, py, _z) in landmarks:
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(img, (px, py), lm_radius, (210, 210, 210),
                               -1, cv2.LINE_AA)

        # --- bounding box ---
        if bbox is not None:
            bx, by, bw, bh = bbox
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh),
                          box_color, box_thick)
            # Draw the WxH label INSIDE the box's top-left corner. Placing it
            # above the box (the old behavior) collided with the header/pose
            # lines whenever the face was large and the box reached near the top
            # of the frame. Inside-the-corner is always clear of the header.
            lbl_x = bx + int(8 * s)
            lbl_y = by + int(28 * s)
            _put(img, f"{bw}x{bh}", (lbl_x, lbl_y), box_color, fs_box)

        # --- header panel (same translucency as the bottom check panel) ---
        # The white/colored header text is hard to read over bright skies or
        # busy foliage, so sit it on a translucent dark strip like the bottom
        # panel. Compute its size from the three header lines.
        hx = int(12 * s)
        hline = int(34 * s)
        # build the three strings up-front to measure the widest
        header_lines = [
            f"id {self.volunteer_id}  {self.filename}",
            f"frame {frame_index}   t={timestamp_sec:.2f}s",
        ]
        if pose:
            yaw = pose.get("yaw"); pitch = pose.get("pitch"); roll = pose.get("roll")
            def _f(v):
                return f"{v:+.1f}" if isinstance(v, (int, float)) else "--"
            pose_line = (f"label={label or '?'}  "
                         f"yaw={_f(yaw)} pitch={_f(pitch)} roll={_f(roll)}")
        else:
            pose_line = f"label={label or 'no-face'}"
        header_lines.append(pose_line)

        h_thick = max(2, int(round(2 * s)))
        h_w = max(cv2.getTextSize(t, _FONT, fs_head, h_thick)[0][0]
                  for t in header_lines)
        hpad = int(10 * s)
        hp_x0 = hx - hpad
        hp_y0 = int(8 * s)
        hp_x1 = hx + h_w + hpad
        hp_y1 = hp_y0 + hline * len(header_lines) + hpad
        hp_x1 = min(hp_x1, w - int(8 * s))
        hov = img.copy()
        cv2.rectangle(hov, (hp_x0, hp_y0), (hp_x1, hp_y1), _PANEL, -1)
        cv2.addWeighted(hov, 0.55, img, 0.45, 0, img)

        # --- header strip (scaled), drawn on top of the panel ---
        ly = int(34 * s)
        _put(img, header_lines[0], (hx, ly), _WHITE, fs_head)
        ly += int(34 * s)
        _put(img, header_lines[1], (hx, ly), _WHITE, fs_sub)

        # --- pose readout ---
        ly += int(32 * s)
        _put(img, pose_line, (hx, ly), box_color, fs_sub)

        # --- per-check panel (bottom-left), one line per check ---
        self._draw_check_panel(img, checks, w, h, s)

        self._writer.write(img)
        self._frames_written += 1

    def _draw_check_panel(self, img, checks: dict, w: int, h: int, s: float):
        if not checks:
            return
        fs = 0.85 * s                       # check-line font scale
        thick = max(2, int(round(2 * s)))
        line_h = int(34 * s)                # row height
        pad = int(12 * s)
        names = sorted(checks.keys())

        # Measure the widest "STATUS name" label so the value column starts
        # clear of it (the previous fixed offset overlapped at large sizes).
        label_texts = [f"{st:5s} {nm}" for nm, (st, _r) in
                       ((nm, checks[nm]) for nm in names)]
        max_label_w = max(
            cv2.getTextSize(t, _FONT, fs, thick)[0][0] for t in label_texts)
        gap = int(30 * s)
        col2_rel = max_label_w + gap        # value column, relative to text start

        # Widest value, to size the panel.
        val_texts = [_shorten_reason(checks[nm][1]) for nm in names]
        max_val_w = max([cv2.getTextSize(t, _FONT, fs, thick)[0][0]
                         for t in val_texts] + [0])

        panel_h = line_h * len(names) + pad * 2
        panel_w = pad * 2 + col2_rel + max_val_w + pad
        panel_w = min(panel_w, w - int(20 * s))   # never overflow the frame
        y0 = h - panel_h - int(16 * s)
        x0 = int(12 * s)
        # near-opaque dark panel for strong contrast (your "light text on dark"
        # request): 0.78 panel + 0.22 image.
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      _PANEL, -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        y = y0 + pad + int(26 * s)
        col2 = x0 + pad + col2_rel          # absolute value-column x
        for name in names:
            status, reason = checks[name]
            color = _STATUS_COLOR.get(status, _WHITE)
            short = _shorten_reason(reason)
            _put(img, f"{status:5s} {name}", (x0 + pad, y), color, fs, thick)
            if short:
                _put(img, short, (col2, y), color, fs, thick)
            y += line_h

    def close(self):
        if self._writer is not None:
            self._writer.release()
            logger.info("OverlayWriter: wrote %d frames to %s",
                        self._frames_written, self.out_path)

    @property
    def frames_written(self) -> int:
        return self._frames_written


def _shorten_reason(reason: Optional[str]) -> str:
    """Pull the measured value out of a check reason for compact display.

    Reasons look like 'frame=60 blur variance=49.1 < 90'. The 'frame=NNN'
    prefix is redundant (the header already shows it), so strip it and keep
    the informative tail.
    """
    if not reason:
        return ""
    r = reason
    # drop a leading 'frame=NNN ' token if present
    if r.startswith("frame="):
        sp = r.find(" ")
        if sp != -1:
            r = r[sp + 1:]
    # keep it short enough for the panel
    return r[:46]