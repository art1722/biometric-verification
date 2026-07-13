"""Letterbox-style debug overlay for the WALK (gait) pipeline.

Same idea as qc.utils.overlay_below (clean video on top, all debug info in a
strip BELOW the frame), but retargeted from FACE to POSE:

  - the video region shows the 33-point MediaPipe pose as a SKELETON (joints +
    limb lines) plus the body bounding box, instead of face dots,
  - the strip below shows the walk checks (MVP: body_height ratio only),
  - it writes EVERY frame back into an .mp4 so the reviewer watches true
    playback with the skeleton tracking the walker.

Why a separate file (mirrors the palm precedent)
-------------------------------------------------
The palm pipeline got its own palm_overlay.py rather than bending the face
overlay; walk does the same. qc.utils.overlay (on-video) and
qc.utils.overlay_below (face letterbox) are left COMPLETELY untouched, so both
keep working exactly as before. This file is purely additive.

Styling (colors, fonts, the outlined-text helper, reason shortener) is imported
from qc.utils.overlay so there is ONE source of truth for appearance, exactly as
overlay_below.py does.

Public interface
----------------
    writer = WalkOverlayWriter(out_path, fps=fps,
                               volunteer_id="002", filename="002_walk_F.mp4")
    for sf in iter_sampled_frames(path, sample_fps=None):   # every frame
        pose = detect_pose(sf.image, detector=detector,
                           input_color_space=sf.color_space)
        writer.add_frame(sf.image, sf.color_space,
                         sf.frame_index, sf.timestamp_sec,
                         pose_detected=pose.ok,
                         landmarks_px=pose.landmarks_px,
                         bbox=pose.bbox,
                         checks={"check_person_height": (status, reason)})
    writer.close()

`checks` is the same {name: (STATUS, reason)} dict the face overlay uses, so the
strip renderer is shared in spirit. In this MVP the pipeline passes only the one
height row; adding more checks later needs no change here (the strip auto-fits).
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

# Strip height as a fraction of the video height. Walk's MVP emits ONE check
# line (body_height) + a couple header lines, so it needs far less room than the
# face strip's six checks. Kept smaller; the strip auto-shrinks text if a future
# walk build emits more lines (see _fit_scale). [DESIGN]
_STRIP_FRACTION = 0.22      # strip height = 22% of the video height
_STRIP_BG = (18, 18, 18)    # near-black bar behind everything

# Below this source width, upscale before drawing so joints/limbs/text stay
# legible. Same rationale and mechanism as overlay_below._MIN_RENDER_WIDTH: the
# strip scale is s = w / 1280, so a tiny source collapses annotations. bbox and
# landmark pixel coords are scaled by the same factor to stay aligned. [DESIGN]
_MIN_RENDER_WIDTH = 960

# Standard MediaPipe 33-point pose skeleton edges (index pairs), grouped so the
# limbs read as a stick figure. These are the canonical POSE_CONNECTIONS; kept
# literal here so this file does not import mp.solutions just for the constant.
_POSE_EDGES = (
    # face (light — kept minimal so the body reads clearly)
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)


class WalkOverlayWriter:
    """Letterbox walk overlay: clean video on top, pose skeleton + bbox drawn on
    it, walk-check verdicts in a strip below. One .mp4 out, one frame per
    add_frame call."""

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
            logger.warning("WalkOverlayWriter: could not open %s", self.out_path)

    def _to_bgr(self, image: Any, color_space: str):
        if color_space == "RGB":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    # -- public hook -------------------------------------------------------

    def add_frame(self, image: Any, color_space: str,
                  frame_index: int, timestamp_sec: float, *,
                  pose_detected: bool,
                  landmarks_px: Optional[list] = None,
                  bbox: Optional[tuple] = None,
                  checks: Optional[dict] = None):
        """Compose [clean video + skeleton/bbox | debug strip] and write it.

        Args:
            image: the frame (BGR or RGB per color_space).
            color_space: "BGR" or "RGB".
            frame_index, timestamp_sec: from the SampledFrame.
            pose_detected: pose.ok for this frame. When False the skeleton is
                skipped and the strip says so, but the frame is STILL written
                (true playback: a no-pose frame is visible, not dropped).
            landmarks_px: PoseResult.landmarks_px — [(x, y, z), ...] in SOURCE
                pixels (33 items). Used for both the skeleton and the box color.
            bbox: PoseResult.bbox — (x, y, w, h) in SOURCE pixels.
            checks: {check_name: (STATUS, reason)} for the strip. MVP passes the
                single body/person-height row.
        """
        checks = checks or {}
        frame = self._to_bgr(image, color_space)
        h, w = frame.shape[:2]

        # --- render-width floor (identical mechanism to overlay_below) -------
        up_scale = 1.0
        if w < _MIN_RENDER_WIDTH:
            up_scale = _MIN_RENDER_WIDTH / float(w)
            new_w = _MIN_RENDER_WIDTH
            new_h = int(round(h * up_scale))
            frame = cv2.resize(frame, (new_w, new_h),
                               interpolation=cv2.INTER_LINEAR)
            h, w = new_h, new_w
            bbox, landmarks_px = self._scale_coords(bbox, landmarks_px, up_scale)

        if self._strip_h == 0:
            self._vid_h = h
            self._strip_h = int(round(h * _STRIP_FRACTION))
        canvas_h = h + self._strip_h
        self._ensure_writer(w, canvas_h)
        if self._writer is None:
            return

        # Build the canvas: clean video on top, dark strip below.
        canvas = np.empty((canvas_h, w, 3), dtype=np.uint8)
        # keep size stable across frames (source frames are all one resolution,
        # but guard against a stray odd frame the way overlay_below does).
        if (w, canvas_h) != self._size:
            frame = cv2.resize(frame, (self._size[0], self._vid_h))
            w = self._size[0]
            canvas = np.empty((self._size[1], w, 3), dtype=np.uint8)
        canvas[:self._vid_h] = frame
        canvas[self._vid_h:] = _STRIP_BG

        # skeleton + bbox on the video region (top self._vid_h rows)
        self._draw_pose(canvas[:self._vid_h], pose_detected, landmarks_px, bbox,
                        checks, w, self._vid_h)

        self._draw_strip(canvas, w, frame_index, timestamp_sec,
                         pose_detected, checks)

        self._writer.write(canvas)
        self._frames_written += 1

    # -- pose drawing (skeleton + bbox) ------------------------------------

    @staticmethod
    def _scale_coords(bbox, landmarks_px, factor):
        """Scale bbox + landmark pixel coords by `factor` for an upscaled frame.
        Returns new objects; inputs are not mutated. factor == 1.0 is a no-op."""
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

    def _box_color(self, pose_detected, checks):
        """Box/skeleton color from the worst frame-level status (same rule as the
        face overlay): grey if no pose, red on any FAIL, else green."""
        statuses = [s_ for (s_, _r) in checks.values()]
        if not pose_detected:
            return _GREY
        if "FAIL" in statuses:
            return _RED
        return _GREEN

    def _draw_pose(self, vid, pose_detected, landmarks_px, bbox, checks, w, h):
        """Draw the skeleton (limb lines + joint dots) and bbox ON the video
        region (in place). `vid` is a view onto canvas[:vid_h]."""
        color = self._box_color(pose_detected, checks)

        # Line/dot size tracks the BODY bbox, not the frame, so a far (small)
        # walker gets thinner limbs and a near (large) one thicker — same idea
        # the face overlay uses to scale by face size.
        if bbox is not None:
            body_dim = max(bbox[2], bbox[3])
        else:
            body_dim = h // 2
        limb_thick = max(1, min(5, int(round(body_dim * 0.004))))
        joint_r = max(1, min(6, int(round(body_dim * 0.005))))

        # --- skeleton limbs ---
        if landmarks_px:
            n = len(landmarks_px)
            for a, b in _POSE_EDGES:
                if a < n and b < n:
                    ax, ay = landmarks_px[a][0], landmarks_px[a][1]
                    bx, by = landmarks_px[b][0], landmarks_px[b][1]
                    cv2.line(vid, (ax, ay), (bx, by), color, limb_thick,
                             cv2.LINE_AA)
            # joints on top of the limbs
            for (px, py, _z) in landmarks_px:
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(vid, (px, py), joint_r, _WHITE, -1, cv2.LINE_AA)

        # --- bounding box + WxH label ---
        if bbox is not None:
            bx, by, bw, bh = bbox
            box_thick = max(1, min(4, int(round(body_dim * 0.003))))
            cv2.rectangle(vid, (bx, by), (bx + bw, by + bh), color, box_thick)
            s = h / 900.0
            fs_box = max(0.5, body_dim * 0.0012)
            _put(vid, f"{bw}x{bh}", (bx + int(8 * s), by + int(28 * s)),
                 color, fs_box)

    # -- strip drawing -----------------------------------------------------

    def _draw_strip(self, canvas, w, frame_index, timestamp_sec,
                    pose_detected, checks):
        top = self._vid_h
        strip_h = self._strip_h
        s = w / 1280.0

        line_color = self._box_color(pose_detected, checks)
        pad = int(18 * s)
        x = pad

        header_lines = [
            (f"id {self.volunteer_id}   {self.filename}", _WHITE),
            (f"frame {frame_index}   t={timestamp_sec:.2f}s", _WHITE),
            (("pose OK" if pose_detected else "NO POSE"), line_color),
        ]

        names = sorted(checks.keys())
        check_lines = []
        for nm in names:
            st, reason = checks[nm]
            check_lines.append((st, nm, _shorten_reason(reason),
                                _STATUS_COLOR.get(st, _WHITE)))

        n_rows = len(header_lines) + len(check_lines) + 1
        fs, line_h = self._fit_scale(s, strip_h, n_rows, pad)
        thick = max(1, int(round(2 * s)))

        y = top + pad + int(line_h * 0.75)
        for text, color in header_lines:
            _put(canvas, text, (x, y), color, fs, thick)
            y += line_h
        y += int(line_h * 0.4)

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
        """Pick a font scale + line height so n_rows fit the strip without
        clipping (copied from overlay_below so the two strips match)."""
        target_fs = 0.85 * s
        usable = strip_h - 2 * pad

        def line_h_for(fs):
            return int(38 * (fs / (0.85 * s)) * s) if s else 38
        fs = target_fs
        lh = line_h_for(fs)
        if n_rows * lh > usable and n_rows > 0:
            lh = max(int(usable / n_rows), int(16 * s))
            fs = max(0.45 * s, (lh / (38.0 * s)) * (0.85 * s)) if s else 0.5
        return fs, lh

    def close(self):
        if self._writer is not None:
            self._writer.release()
            logger.info("WalkOverlayWriter: wrote %d frames to %s",
                        self._frames_written, self.out_path)

    @property
    def frames_written(self) -> int:
        return self._frames_written