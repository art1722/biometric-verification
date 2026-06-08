"""Head pose estimation — RESEARCHER'S algorithm (from Copy_of_face_angle_test.ipynb).

Source hierarchy for this project:
  - The NOTEBOOK is the researcher's own code -> it is the authority for head
    pose. This file ports its `calculate_angles` function.
  - The repo (a master's student's helper) is only used to fill gaps the
    notebook does not cover. Its solvePnP pose method is therefore NOT used
    here; it is kept as check_head_pose_solvepnp_backup.py for reference only.

Method (verbatim from the notebook)
-----------------------------------
Build three direction vectors from face landmarks (eye, mouth, vertical), cross
the eye/mouth vectors with the vertical to estimate the face normal (z-axis),
average them, then read yaw/pitch off that normal and roll off eye+mouth tilt.
The 1.05 / 1.45 / 0.7 factors are the researcher's empirical calibration — kept
exactly, since they were presumably tuned for this capture setup.

IMPORTANT — coordinate space
----------------------------
The notebook reads landmark.x/.y/.z straight from MediaPipe, i.e. NORMALIZED
0..1 coords where x and y share the same scale. This file works on the RAW
MediaPipe landmark object, NOT get_lm's pixel-space tuples (which scale x by
width and y by height and would distort the vectors). The pipeline passes the
raw landmark object here.

Changes from the notebook (wrapper only, math untouched):
- ARRAY-FIRST input + optional DETECTOR REUSE.
- Returns (success, info_dict) with yaw/pitch/roll/direction for the future
  turn-sequence check. The notebook only printed the values.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal, Optional

import cv2
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)

_L_EYE_L, _L_EYE_R = 33, 133
_R_EYE_L, _R_EYE_R = 362, 263
_MOUTH_L, _MOUTH_R = 61, 291
_CHIN, _FOREHEAD = 152, 10


def _to_bgr(image, input_color_space):
    if isinstance(image, str):
        return cv2.imread(image)
    if input_color_space == "RGB":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def calculate_angles(face_landmarks):
    """Researcher's geometric pitch/yaw/roll from RAW (normalized) landmarks.

    Returns (pitch, yaw, roll) in degrees.
    """
    def v(i):
        lm = face_landmarks[i]
        return np.array([lm.x, lm.y, lm.z])

    left_eye = (v(_L_EYE_L) + v(_L_EYE_R)) / 2
    right_eye = (v(_R_EYE_L) + v(_R_EYE_R)) / 2
    left_mouth = v(_MOUTH_L)
    right_mouth = v(_MOUTH_R)
    chin = v(_CHIN)
    forehead = v(_FOREHEAD)

    def normalize(vec):
        return vec / np.linalg.norm(vec)

    vector_eye = left_eye - right_eye
    vector_mouth = left_mouth - right_mouth
    vector_vertical = forehead - chin

    z_axis_eye = normalize(np.cross(vector_eye, vector_vertical))
    z_axis_mouth = normalize(np.cross(vector_mouth, vector_vertical))
    z_axis = (z_axis_eye + z_axis_mouth) / 2

    yaw = math.degrees(math.atan2(z_axis[0], z_axis[2])) * 1.05
    pitch = math.degrees(
        math.atan2(z_axis[1], math.sqrt(z_axis[0] ** 2 + z_axis[2] ** 2))
    ) * 1.45

    eye_roll = math.degrees(
        math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    mouth_roll = math.degrees(
        math.atan2(right_mouth[1] - left_mouth[1], right_mouth[0] - left_mouth[0]))
    roll = -((eye_roll + mouth_roll) / 2) * 0.7

    return pitch, yaw, roll


def estimate_head_pose(
    image,
    *,
    detector=None,
    input_color_space: Literal["BGR", "RGB"] = "BGR",
    left_th=-10.0, right_th=10.0, down_th=-10.0, up_th=15.0,
    til_left_th=-10.0, til_right_th=10.0,
):
    """Estimate head pose using the researcher's notebook algorithm.

    Returns (success, info) where info = {yaw, pitch, roll, direction}.
    """
    img = _to_bgr(image, input_color_space)
    if img is None or getattr(img, "size", 0) == 0:
        return (False, {"reason": "Cannot read image"})

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    owns = detector is None
    if owns:
        detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    try:
        results = detector.process(rgb)
    finally:
        if owns:
            detector.close()

    if not results.multi_face_landmarks:
        return (False, {"reason": "No face detected"})

    face_landmarks = results.multi_face_landmarks[0].landmark
    try:
        pitch, yaw, roll = calculate_angles(face_landmarks)
    except (ZeroDivisionError, FloatingPointError, ValueError) as e:
        return (False, {"reason": f"angle calc failed: {e}"})

    if yaw < left_th:
        direction = "Looking Left"
    elif yaw > right_th:
        direction = "Looking Right"
    elif pitch < down_th:
        direction = "Looking Down"
    elif pitch > up_th:
        direction = "Looking Up"
    elif roll < til_left_th:
        direction = "Tilting Left"
    elif roll > til_right_th:
        direction = "Tilting Right"
    else:
        direction = "Forward"

    return (True, {"yaw": yaw, "pitch": pitch, "roll": roll, "direction": direction})
