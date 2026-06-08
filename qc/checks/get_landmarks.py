"""Face landmark detection (ported from Face-Verification repo's func/get_landmarks.py).

What this does
--------------
Runs MediaPipe Face Mesh on an image and returns the face landmarks plus a
bounding box around the face. This is the FOUNDATION check: several other
checks (face size, eyes) consume the landmarks/bbox this produces, so it runs
first in the face pipeline.

Changes from the original repo version (all intentional)
--------------------------------------------------------
1. ARRAY-FIRST: accepts either a file path (str) or an already-decoded BGR
   NumPy array (a video frame). The repo version only accepted a path and
   called cv2.imread internally, which forced video frames to be written to
   temp files first. Now a frame from qc/utils/video.py can be passed directly.
2. DETECTOR REUSE: an optional `detector` (a MediaPipe FaceMesh) can be passed
   in. If none is given, the function creates and closes its own (preserving the
   repo's standalone behavior). The face pipeline will create ONE detector and
   pass it to every frame, avoiding the slow per-call model setup at scale.
3. CONFIG-DRIVEN: max_num_faces / min_detection_confidence / refine_landmarks
   come from arguments (wired to config.yml by the caller) instead of being
   hardcoded.
4. RETURN SHAPE FIXED: the repo's error paths returned only 4 values while the
   success path returned 5, which crashes any caller that unpacks 5. Every path
   here returns the SAME 5-tuple, so callers never crash on a no-face frame
   (which is common in turn-sequence videos: profile turns, looking down, etc.).
5. LOGGING: replaced the colored rich.console prints with the standard logging
   module. Per-frame colored prints become noise across thousands of frames;
   results belong in the CSV/JSON report, not the terminal.

Return value (always a 5-tuple)
-------------------------------
    (success, message, landmarks, bbox, norm_box)
    - success    : bool, True if a face was detected
    - message    : str, status or error description
    - landmarks  : list[(x, y, z)] in PIXEL coords (z stays relative), or None
    - bbox       : (x, y, w, h) in pixels, 10% margin added, clamped to image
    - norm_box   : (x, y, w, h) normalized to 0..1, or None
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import cv2
import mediapipe as mp

logger = logging.getLogger(__name__)

# Same margin the repo used: pad the landmark-derived box by 10% each side so
# the bbox comfortably covers the whole face, not just the outermost landmarks.
_BBOX_MARGIN_RATIO = 0.1


def get_lm(
    image: Any,
    *,
    detector: Optional["mp.solutions.face_mesh.FaceMesh"] = None,
    max_num_faces: int = 2,
    min_detection_confidence: float = 0.5,
    refine_landmarks: bool = True,
    input_color_space: Literal["BGR", "RGB"] = "BGR",
    require_single_face: bool = True,
):
    """Detect face landmarks and a bounding box.

    Args:
        image: A file path (str) OR a decoded BGR image array (e.g. a video
            frame from qc.utils.video). BGR is OpenCV's default channel order.
        detector: Optional pre-created MediaPipe FaceMesh to reuse. If None, a
            temporary one is created and closed inside this call.
        max_num_faces: Max faces MediaPipe will look for (config-driven). Default
            2, not 1, so we can DETECT a second face in order to flag it; we only
            need to know "is there more than one", not find all ten.
        min_detection_confidence: Detection confidence threshold (config-driven).
        refine_landmarks: Whether to use the refined landmark model.
        input_color_space: Whether `image` is "BGR" (OpenCV default, from
            cv2.imread or a default video.py frame) or "RGB" (a frame sampled
            with as_rgb=True). Pipelines should pass SampledFrame.color_space
            here so the conversion is never guessed wrong.
        require_single_face: If True, a frame with more than one detected face
            returns success=False with a "Multiple faces" message. The caller
            decides whether that becomes a REVIEW flag or a FAIL — per the
            project's screening-filter philosophy, REVIEW is the safer default
            (the spec does not explicitly forbid extra faces, so a human should
            confirm rather than auto-rejecting).

    Returns:
        (success, message, landmarks, bbox, norm_box) — always 5 values.
    """
    # --- Resolve the input into a BGR array (array-first pattern) ---
    if isinstance(image, str):
        frame = cv2.imread(image)
        if frame is None:
            logger.debug("LANDMARKS | failed to load image: %s", image)
            return (False, "Failed to load image", None, None, None)
    else:
        frame = image

    if frame is None or getattr(frame, "size", 0) == 0:
        logger.debug("LANDMARKS | empty or invalid image array")
        return (False, "Empty or invalid image", None, None, None)

    try:
        height, width = frame.shape[:2]

        # Convert to RGB for MediaPipe, but only if the input is actually BGR.
        # If video.py already handed us RGB (as_rgb=True), converting again
        # would swap the channels and feed MediaPipe wrong colors.
        if input_color_space == "BGR":
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif input_color_space == "RGB":
            image_rgb = frame
        else:
            return (False, f"Unsupported color space: {input_color_space}",
                    None, None, None)

        # --- Detector: reuse the one passed in, or make a temporary one ---
        owns_detector = detector is None
        if owns_detector:
            detector = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=max_num_faces,
                min_detection_confidence=min_detection_confidence,
                refine_landmarks=refine_landmarks,
            )

        try:
            results = detector.process(image_rgb)
        finally:
            # Only close a detector we created ourselves. A shared detector
            # passed in by the pipeline must stay alive for the next frame.
            if owns_detector:
                detector.close()

        faces = results.multi_face_landmarks or []
        if not faces:
            logger.debug("LANDMARKS | no faces detected")
            return (False, "No faces detected", None, None, None)

        # QC concern: a recording is one volunteer. A second face (e.g. someone
        # in the background) should not pass silently. We surface it; the caller
        # maps this to REVIEW or FAIL.
        if require_single_face and len(faces) > 1:
            logger.debug("LANDMARKS | multiple faces detected: %d", len(faces))
            return (False, f"Multiple faces detected: {len(faces)}",
                    None, None, None)

        # Use the first detected face.
        face_landmarks = faces[0]

        # Convert normalized landmark coords -> pixel coords (z stays relative).
        landmarks = []
        for lm in face_landmarks.landmark:
            landmarks.append((int(lm.x * width), int(lm.y * height), lm.z))

        # Bounding box from the landmark extents.
        x_coords = [p[0] for p in landmarks]
        y_coords = [p[1] for p in landmarks]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        w = x_max - x_min
        h = y_max - y_min

        # Add a 10% margin on each side, clamped to the image bounds.
        margin_x = int(w * _BBOX_MARGIN_RATIO)
        margin_y = int(h * _BBOX_MARGIN_RATIO)
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(width, x_max + margin_x)
        y_max = min(height, y_max + margin_y)
        w = x_max - x_min
        h = y_max - y_min

        bbox = (x_min, y_min, w, h)
        norm_box = (x_min / width, y_min / height, w / width, h / height)

        logger.debug("LANDMARKS | face ok, bbox=%s", bbox)
        return (True, "Face detected successfully", landmarks, bbox, norm_box)

    except Exception as e:
        logger.debug("LANDMARKS | error: %s", e)
        return (False, f"Error during face detection: {e}", None, None, None)