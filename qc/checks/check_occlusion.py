"""Occlusion check — landmark-ROI + YCrCb skin-colour appearance test.

Why this method (and NOT a landmark-confidence cutoff)
------------------------------------------------------
MediaPipe Face Landmarker is a REGRESSION model with a strong facial prior. It
does not expose a per-landmark confidence/visibility score (FaceResult carries
only landmarks, a bbox, and blendshapes — there is no per-point reliability
field, because the model does not produce one). When a region is occluded, the
model does NOT report low confidence — it HALLUCINATES the landmark from its
learned prior. Empirically (verified on occluded samples) a covered mouth/eye
still yields confidently-placed landmarks, so "is the landmark placed?" and
"use confidence as a cutoff" both fail: the model always places the point.

So this check stops asking the landmark MODEL about occlusion and asks the
PIXELS instead. The landmarks tell us WHERE each face region is; we then test
whether the pixels inside that region actually LOOK like skin. A face region
whose pixels are not skin-toned is likely covered by something that is not skin
(a mask, sunglasses, a hat brim, hair, an object).

Method
------
1. Use the pixel-space landmarks (already produced once per frame by the shared
   FaceLandmarker — NO re-detection here) to build a small ROI for each region
   we care about: forehead, left eye, right eye, nose, mouth, left cheek,
   right cheek. Each ROI is the bounding box of a few named landmark indices,
   padded by a margin.
2. Convert the image to YCrCb and test each ROI's pixels against skin-chroma
   bounds (Cr in [cr_min, cr_max], Cb in [cb_min, cb_max]). These are the
   classic face-skin chroma bounds and are robust to luma (brightness) changes,
   which is why YCrCb is preferred over raw RGB for skin detection.
3. A region PASSES if its fraction of skin pixels >= min_skin_ratio. If the
   fraction is below that, the region is flagged as occluded.
4. The check FAILs if any REQUIRED region is occluded. Which regions are
   required, and all thresholds, come from config (face.occlusion.skin.*), so
   the spec owner can tune without code changes.

Known limitation (state this to reviewers)
-------------------------------------------
Colour alone cannot catch a SKIN-COLOURED occluder — e.g. a bare hand over the
mouth has skin chroma and will read as "skin present". This catches masks,
sunglasses, hats, hair, and non-skin objects (the spec's named cases:
no_mask / no_sunglasses / no_hat / no_hair_covering_face_or_eyes). If
skin-toned occluders prove to be a real failure mode in the data, layer a
texture/edge cue on top of this colour test rather than replacing it.

Contract (mirrors the other checks)
-----------------------------------
- Consumes pixel-space landmarks (list[(x, y, z)]) — the SAME shape get_lm /
  detect_face returns and the same shape check_head_fully consumes. No model is
  run in this function.
- Returns (success: bool, message: str). The message embeds per-region skin
  ratios so the report/overlay can show WHY a frame failed, e.g.
  "occluded: mouth(skin=0.12); forehead=0.95 left_eye=0.88 ...".
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Region -> representative MediaPipe Face Mesh landmark indices.
#
# Each region's ROI is the bounding box of these points (then padded). The
# indices are chosen to span the region without straying onto neighbouring
# skin: e.g. the mouth set traces the outer lip ring, the eye sets trace the
# eye opening. Standard MediaPipe FaceMesh (478-pt) indices.
# ---------------------------------------------------------------------------
_REGION_LANDMARKS: Dict[str, List[int]] = {
    # Forehead: a band above the brows. (No landmarks sit on the upper
    # forehead, so this uses brow-top + temple points; the margin pushes the
    # ROI upward into the forehead.)
    "forehead": [10, 67, 69, 104, 108, 151, 337, 299, 333, 297],
    # Left eye opening (subject's left = image right in a non-mirrored frame;
    # naming follows MediaPipe's own Left/Right convention).
    "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173],
    "right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    # Nose: bridge + tip + alae.
    "nose": [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327],
    # Mouth: outer lip ring.
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              409, 270, 269, 267, 0, 37, 39, 40, 185],
    # Cheeks: a patch between eye, nose, and jaw.
    "left_cheek": [50, 101, 118, 117, 123, 187, 205, 36],
    "right_cheek": [280, 330, 347, 346, 352, 411, 425, 266],
}

# The order regions appear in the message (stable, readable).
_REGION_ORDER = [
    "forehead", "left_eye", "right_eye", "nose",
    "mouth", "left_cheek", "right_cheek",
]

# Default regions whose occlusion should FAIL the frame. Eyes, nose, and mouth
# are the spec-critical ones (mask covers nose+mouth; sunglasses cover eyes).
# Forehead/cheeks default to informational unless config promotes them, because
# hair across a cheek is more tolerable than a mask. Overridable via config.
_DEFAULT_REQUIRED = ["left_eye", "right_eye", "nose", "mouth"]

# Default YCrCb skin-chroma bounds. Classic face-skin range; tuned to be
# permissive enough for varied skin tones while still excluding most fabric /
# plastic / hair. All overridable via config.
_DEFAULT_CR = (133, 173)
_DEFAULT_CB = (77, 127)
_DEFAULT_MIN_SKIN_RATIO = 0.40   # a region needs >=40% skin pixels to "pass"
_DEFAULT_ROI_MARGIN = 0.15       # pad each ROI by 15% of its own size


def _roi_from_landmarks(
    landmarks: Sequence,
    idxs: Sequence[int],
    img_w: int,
    img_h: int,
    margin: float,
) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box (x0, y0, x1, y1) of the given landmark indices, padded by
    `margin` (fraction of the box's own w/h) and clamped to the image."""
    xs: List[int] = []
    ys: List[int] = []
    for i in idxs:
        try:
            p = landmarks[i]
        except (IndexError, TypeError):
            continue
        xs.append(int(p[0]))
        ys.append(int(p[1]))
    if not xs or not ys:
        return None

    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    bw, bh = x1 - x0, y1 - y0
    mx = int(round(bw * margin))
    my = int(round(bh * margin))
    # For thin regions (e.g. a closed eye), ensure at least a few px of height
    # so the ROI is not degenerate.
    mx = max(mx, 2)
    my = max(my, 2)

    x0 = max(0, x0 - mx)
    y0 = max(0, y0 - my)
    x1 = min(img_w, x1 + mx)
    y1 = min(img_h, y1 + my)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _skin_ratio_ycrcb(
    ycrcb_roi: np.ndarray,
    cr_bounds: Tuple[int, int],
    cb_bounds: Tuple[int, int],
) -> float:
    """Fraction of pixels in the ROI whose Cr/Cb fall within the skin bounds."""
    if ycrcb_roi.size == 0:
        return 0.0
    cr = ycrcb_roi[:, :, 1]
    cb = ycrcb_roi[:, :, 2]
    cr_lo, cr_hi = cr_bounds
    cb_lo, cb_hi = cb_bounds
    mask = (cr >= cr_lo) & (cr <= cr_hi) & (cb >= cb_lo) & (cb <= cb_hi)
    return float(np.count_nonzero(mask)) / float(mask.size)


def check_occlusion(
    image: Any,
    landmarks: Sequence,
    *,
    input_color_space: Literal["BGR", "RGB"] = "BGR",
    required_regions: Optional[Sequence[str]] = None,
    cr_bounds: Tuple[int, int] = _DEFAULT_CR,
    cb_bounds: Tuple[int, int] = _DEFAULT_CB,
    min_skin_ratio: float = _DEFAULT_MIN_SKIN_RATIO,
    roi_margin: float = _DEFAULT_ROI_MARGIN,
) -> Tuple[bool, str]:
    """Detect whether a required face region is occluded by a non-skin object.

    Args:
        image: file path (str) OR a decoded image array. (Array-first, like the
            other checks — pass the frame the pipeline already decoded.)
        landmarks: pixel-space landmarks list[(x, y, z)] from detect_face /
            get_lm. NOT re-detected here.
        input_color_space: "BGR" (OpenCV default) or "RGB".
        required_regions: which regions FAIL the frame if occluded. Defaults to
            eyes+nose+mouth. Pass face.occlusion.skin.required_regions from cfg.
        cr_bounds, cb_bounds: YCrCb skin-chroma bounds (inclusive).
        min_skin_ratio: a region passes if its skin-pixel fraction >= this.
        roi_margin: padding added to each region ROI (fraction of its size).

    Returns:
        (success, message). success is True when NO required region is occluded.
        The message lists per-region skin ratios so the report shows the reason.
    """
    # --- resolve image to a BGR array ---
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            return (False, "Failed to load image")
    elif input_color_space == "RGB":
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img = image

    if img is None or getattr(img, "size", 0) == 0:
        return (False, "Empty or invalid image")

    if not landmarks:
        return (False, "No landmarks provided")

    img_h, img_w = img.shape[:2]
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    req = list(required_regions) if required_regions else list(_DEFAULT_REQUIRED)

    ratios: Dict[str, Optional[float]] = {}
    for region in _REGION_ORDER:
        idxs = _REGION_LANDMARKS[region]
        roi_box = _roi_from_landmarks(landmarks, idxs, img_w, img_h, roi_margin)
        if roi_box is None:
            ratios[region] = None
            continue
        x0, y0, x1, y1 = roi_box
        roi = ycrcb[y0:y1, x0:x1]
        ratios[region] = _skin_ratio_ycrcb(roi, cr_bounds, cb_bounds)

    # --- decide: any REQUIRED region below the skin-ratio floor is occluded ---
    occluded: List[str] = []
    for region in req:
        r = ratios.get(region)
        # A region we could not build an ROI for (missing landmarks) is treated
        # as occluded for a REQUIRED region — we cannot confirm skin is present.
        if r is None or r < min_skin_ratio:
            occluded.append(region)

    # --- build a readable message with every region's ratio ---
    def _fmt(region: str) -> str:
        r = ratios.get(region)
        return f"{region}={'NA' if r is None else f'{r:.2f}'}"

    all_ratios_str = " ".join(_fmt(rg) for rg in _REGION_ORDER)
    thr = f"min_skin_ratio={min_skin_ratio:.2f}"

    if occluded:
        bad = ", ".join(
            f"{rg}(skin={'NA' if ratios.get(rg) is None else f'{ratios[rg]:.2f}'})"
            for rg in occluded
        )
        return (False, f"occluded: {bad}; {all_ratios_str}; {thr}")

    return (True, f"no occlusion; {all_ratios_str}; {thr}")