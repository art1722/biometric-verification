from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Optional

import mediapipe as mp

from qc.utils.video import probe_video, iter_sampled_frames
from qc.checks.hand_landmarker import create_hand_landmarker, detect_hand
from qc.checks.check_face_size import check_face_min_size
from qc.checks.check_head_fully import check_head_fully
from qc.checks.check_face_blur import check_face_blur
from qc.checks.check_light_pollution import check_lightpol
from qc.checks.check_head_pose import estimate_head_pose
from qc.checks.check_eye import check_eye_status
from qc.checks.check_turn_sequence_seg import (
    check_turn_sequence_seg, apply_gap_split_to_detection_rows,
)
from qc.checks._turn_common import TurnThresholds, classify_frame, FRONT
from qc.checks import check_metadata as md
from qc.schemas import CheckRow

logger = logging.getLogger(__name__)

DATA_TYPE = "palm"