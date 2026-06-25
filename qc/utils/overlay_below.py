"""Letterbox-style debug overlay (ALTERNATIVE to qc.utils.overlay).

Same purpose and SAME PUBLIC INTERFACE as qc.utils.overlay.OverlayWriter, but a
different layout: instead of drawing on top of the video frame, it keeps the
video frame CLEAN and writes all debug info into a fixed-height strip BELOW the
frame (a "letterbox" bar). Nothing covers the face — preferred when a reviewer
wants to judge the footage itself while reading the QC verdict underneath.

Why a separate file:
  - The on-video writer (qc.utils.overlay) is left completely untouched, so it
    keeps working exactly as before. This file is purely additive.
  - This class is also named `OverlayWriter` and exposes the identical methods
    (__init__, add_frame, close, frames_written), so the pipeline does not need
    to change at all. Switching styles is a one-line import swap in run_face.py:

        from qc.utils.overlay import OverlayWriter          # draws ON the video
        # from qc.utils.overlay_below import OverlayWriter  # draws BELOW it

    Comment one, uncomment the other. Nothing else changes.

Shared styling (colors, fonts, the outlined-text helper, reason shortener) is
imported from qc.utils.overlay so there is ONE source of truth for appearance.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

# Reuse the single source of truth for styling from the on-video overlay.
from qc.utils.overlay import (
    _put, _shorten_reason, _STATUS_COLOR,
    _GREEN, _RED, _AMBER, _GREY, _WHITE, _BLACK, _PANEL, _FONT,
)

logger = logging.getLogger(__name__)

# The strip is a fixed fraction of the video height. ~7 check lines + header fit
# comfortably for the face pipeline (6 frame checks). If a future modality emits
# more lines than fit, the text auto-shrinks to avoid clipping (see _fit_scale).
_STRIP_FRACTION = 0.42      # strip height = 42% of the video height
_STRIP_BG = (18, 18, 18)    # near-black bar behind everything

# Minimum width (px) the annotated canvas is rendered at. A low-res source
# (e.g. a heavily-compressed shareable clip) makes every scaled dimension
# collapse — text, dots and the WxH label turn into unreadable blobs, because
# the strip scale is s = w / 1280. To keep annotations legible regardless of
# the source resolution, a frame narrower than this is upscaled to this width
# BEFORE anything is drawn (bbox/landmark coords are scaled to match). The
# video content is unchanged in proportion; it is just enlarged for review.
# Raise this if text is still too small on your smallest clips. [DESIGN]
_MIN_RENDER_WIDTH = 960


class OverlayWriter:
    """Letterbox overlay: clean video on top, all debug info in a strip below.

    Drop-in replacement for qc.utils.overlay.OverlayWriter — identical
    constructor and methods, so run_face.py / the pipeline need no changes.
    """

    def __init__(self, out_path: str, *, fps: float,
                 volunteer_id: str, filename: str):
        self.out_path = out_path
        self.fps = max(float(fps), 1.0)
        self.volunteer_id = volunteer_id
        self.filename = filename
        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Optional[tuple[int, int]] = None   # (canvas_w, canvas_h)
        self._vid_h: int = 0                            # original video height
        self._strip_h: int = 0
        self._frames_written = 0

    # -- internal ----------------------------------------------------------

    def _ensure_writer(self, canvas_w: int, canvas_h: int):
        if self._writer is not None:
            return
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.out_path, fourcc, self.fps,
                                       (canvas_w, canvas_h))
        self._size = (canvas_w, canvas_h)
        if not self._writer.isOpened():
            logger.warning("OverlayWriter(below): could not open %s", self.out_path)

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
        """Compose [clean video | debug strip] and write it. Same signature as
        the on-video OverlayWriter (landmarks/bbox accepted but not drawn on the
        frame, by design — the video stays clean)."""
        checks = checks or {}
        frame = self._to_bgr(image, color_space)
        h, w = frame.shape[:2]

        # Preserve the TRUE source face dimensions for the WxH label, so the
        # reviewer sees the real px size (e.g. 553x650), not the upscaled one.
        label_dims = (bbox[2], bbox[3]) if bbox is not None else None

        # --- render-width floor ---------------------------------------------
        # If the source is narrower than _MIN_RENDER_WIDTH, upscale the frame so
        # annotations render at a readable absolute size. bbox/landmarks arrive
        # in SOURCE pixels, so scale them by the same factor or they'd land in
        # the wrong place. A frame already >= the floor is left untouched
        # (up_scale = 1.0), so normal-res output is byte-for-byte as before.
        up_scale = 1.0
        if w < _MIN_RENDER_WIDTH:
            up_scale = _MIN_RENDER_WIDTH / float(w)
            new_w = _MIN_RENDER_WIDTH
            new_h = int(round(h * up_scale))
            # INTER_LINEAR is fine here; this is a review aid, not analysis.
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = new_h, new_w
            bbox, landmarks = self._scale_face_coords(bbox, landmarks, up_scale)

        if self._strip_h == 0:
            self._vid_h = h
            self._strip_h = int(round(h * _STRIP_FRACTION))
        canvas_h = h + self._strip_h
        self._ensure_writer(w, canvas_h)
        if self._writer is None:
            return

        # Build the canvas: clean video on top, dark strip below.
        canvas = np.empty((canvas_h, w, 3), dtype=np.uint8)
        # keep size stable across frames (sampled frames should match)
        if (w, canvas_h) != self._size:
            frame = cv2.resize(frame, (self._size[0], self._vid_h))
            w = self._size[0]
            canvas = np.empty((self._size[1], w, 3), dtype=np.uint8)
        canvas[:self._vid_h] = frame
        canvas[self._vid_h:] = _STRIP_BG

        # Draw bbox + landmarks ON the video region of the canvas (the top
        # self._vid_h rows). This is the SAME visual data the on-video overlay
        # draws; here it sits on the frame while the verdict text stays in the
        # clean strip below. Scaling matches qc.utils.overlay exactly.
        self._draw_face(canvas[:self._vid_h], face_detected, landmarks, bbox,
                        label, checks, self._size[0], self._vid_h,
                        label_dims=label_dims)

        self._draw_strip(canvas, w, frame_index, timestamp_sec,
                         face_detected, pose, label, checks)

        self._writer.write(canvas)
        self._frames_written += 1

    # -- face drawing (bbox + landmarks) -----------------------------------

    @staticmethod
    def _scale_face_coords(bbox, landmarks, factor):
        """Scale bbox + landmark pixel coords by `factor` for an upscaled frame.

        bbox/landmarks come in SOURCE-pixel coordinates. When the frame is
        upscaled to the render-width floor, these must be scaled by the same
        factor so the box and dots still land on the face. Returns new objects;
        inputs are not mutated. factor == 1.0 returns them unchanged.
        """
        if factor == 1.0:
            return bbox, landmarks
        if bbox is not None:
            bx, by, bw, bh = bbox
            bbox = (int(round(bx * factor)), int(round(by * factor)),
                    int(round(bw * factor)), int(round(bh * factor)))
        if landmarks:
            landmarks = [(int(round(px * factor)), int(round(py * factor)), z)
                         for (px, py, z) in landmarks]
        return bbox, landmarks

    def _draw_face(self, vid, face_detected, landmarks, bbox, label, checks,
                   w, h, *, label_dims=None):
        """Draw the bbox + landmarks ON the video region (in place).

        `vid` is a view onto canvas[:vid_h], so drawing here writes straight
        onto the frame portion of the canvas. The scaling logic is copied from
        qc.utils.overlay so both styles look identical: dot radius / box
        thickness / box-label font all track the FACE size, not the frame, so a
        close (large) face gets bigger dots and a far (small) face smaller ones.
        """
        # --- bbox color from the worst frame-level status (same rule as overlay) ---
        statuses = [s_ for (s_, _r) in checks.values()]
        if not face_detected:
            box_color = _GREY
        elif "FAIL" in statuses:
            box_color = _RED
        elif label and label != "front":
            box_color = _AMBER
        else:
            box_color = _GREEN

        # text scale (for the WxH box label) — frame-relative, like overlay.py
        s = h / 900.0

        # Box / dot / box-label scale with the FACE bbox, not the frame.
        if bbox is not None:
            _bw, _bh = bbox[2], bbox[3]
            face_dim = max(_bw, _bh)
        else:
            face_dim = h // 4   # fallback if no face this frame

        lm_radius = max(1, min(3, int(round(face_dim * 0.003))))
        box_thick = max(1, min(3, int(round(face_dim * 0.003))))
        fs_box = max(0.5, face_dim * 0.0016)   # the WxH label on the box

        # --- landmarks (faint) ---
        if landmarks:
            for (px, py, _z) in landmarks:
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(vid, (px, py), lm_radius, (210, 210, 210),
                               -1, cv2.LINE_AA)

        # --- bounding box + WxH label inside its top-left corner ---
        if bbox is not None:
            bx, by, bw, bh = bbox
            cv2.rectangle(vid, (bx, by), (bx + bw, by + bh),
                          box_color, box_thick)
            lbl_x = bx + int(8 * s)
            lbl_y = by + int(28 * s)
            # Show the TRUE source dimensions (label_dims) when the frame was
            # upscaled; fall back to the drawn box size otherwise.
            disp_w, disp_h = label_dims if label_dims is not None else (bw, bh)
            _put(vid, f"{disp_w}x{disp_h}", (lbl_x, lbl_y), box_color, fs_box)

    # -- strip drawing -----------------------------------------------------

    def _draw_strip(self, canvas, w, frame_index, timestamp_sec,
                    face_detected, pose, label, checks):
        top = self._vid_h
        strip_h = self._strip_h
        s = w / 1280.0                       # scale relative to width

        # status color used for the pose/label line
        statuses = [st for (st, _r) in checks.values()]
        if not face_detected:
            line_color = _GREY
        elif "FAIL" in statuses:
            line_color = _RED
        elif label and label != "front":
            line_color = _AMBER
        else:
            line_color = _GREEN

        pad = int(18 * s)
        x = pad
        # --- header lines (left column) ---
        header_lines = [
            (f"id {self.volunteer_id}   {self.filename}", _WHITE),
            (f"frame {frame_index}   t={timestamp_sec:.2f}s", _WHITE),
        ]
        if pose:
            yaw = pose.get("yaw"); pitch = pose.get("pitch"); roll = pose.get("roll")
            def _f(v):
                return f"{v:+.1f}" if isinstance(v, (int, float)) else "--"
            pose_line = (f"label={label or '?'}   "
                         f"yaw={_f(yaw)}  pitch={_f(pitch)}  roll={_f(roll)}")
        else:
            pose_line = f"label={label or 'no-face'}"
        header_lines.append((pose_line, line_color))

        # --- check lines (the verdict list) ---
        names = sorted(checks.keys())
        check_lines = []
        for nm in names:
            st, reason = checks[nm]
            check_lines.append((st, nm, _shorten_reason(reason),
                                _STATUS_COLOR.get(st, _WHITE)))

        # Auto-fit: choose a font scale so header + checks fit the strip height
        # without clipping. Start from a target and shrink if needed.
        n_rows = len(header_lines) + len(check_lines) + 1   # +1 spacer
        fs, line_h = self._fit_scale(s, strip_h, n_rows, pad)
        thick = max(1, int(round(2 * s)))

        y = top + pad + int(line_h * 0.75)
        for text, color in header_lines:
            _put(canvas, text, (x, y), color, fs, thick)
            y += line_h
        y += int(line_h * 0.4)   # small gap between header and checks

        # checks: two columns (STATUS+name | value), measured to avoid overlap
        if check_lines:
            label_w = max(
                cv2.getTextSize(f"{st:5s} {nm}", _FONT, fs, thick)[0][0]
                for (st, nm, _v, _c) in check_lines)
            col2 = x + label_w + int(40 * s)
            for st, nm, val, color in check_lines:
                _put(canvas, f"{st:5s} {nm}", (x, y), color, fs, thick)
                if val:
                    _put(canvas, val, (col2, y), color, fs, thick)
                y += line_h

    def _fit_scale(self, s, strip_h, n_rows, pad):
        """Pick a font scale + line height that fits n_rows into the strip.

        Prevents clipping: if there are more rows than a comfortable size would
        allow, the text shrinks rather than spilling past the strip.
        """
        target_fs = 0.85 * s
        usable = strip_h - 2 * pad
        
        # line height proportional to font scale
        def line_h_for(fs):
            return int(38 * (fs / (0.85 * s)) * s) if s else 38
        fs = target_fs
        lh = line_h_for(fs)
        if n_rows * lh > usable and n_rows > 0:
            # shrink to fit
            lh = max(int(usable / n_rows), int(16 * s))
            fs = max(0.45 * s, (lh / (38.0 * s)) * (0.85 * s)) if s else 0.5
        return fs, lh

    def close(self):
        if self._writer is not None:
            self._writer.release()
            logger.info("OverlayWriter(below): wrote %d frames to %s",
                        self._frames_written, self.out_path)

    @property
    def frames_written(self) -> int:
        return self._frames_written