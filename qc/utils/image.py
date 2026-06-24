"""Still-image metadata probe — the palm-side mirror of probe_video().

Why this file exists
--------------------
The palm modality is single JPEGs, not video. The existing metadata checks
(`check_container`, `check_resolution` in qc/checks/check_metadata.py) read a
metadata object with fields like `.readable`, `.extension`, `.width`,
`.height`, `.is_color`. `probe_video()` produces that object for an .mp4; this
function produces the SAME-shaped object for a still image, so the palm
pipeline reuses the identical check functions without a re-write.

We deliberately reuse VideoMetadata (from qc.utils.video) rather than invent a
parallel dataclass: the check functions already accept it, and the video-only
fields (fps, duration, frame_count) simply stay None for a still — which is
correct, since a still has no such properties and the palm pipeline never runs
check_fps / check_duration anyway.

`readable` means OpenCV could decode the file into an array. As with video,
width/height/channels may be None for a corrupt file, so the checks already
handle None (they FAIL closed, the project's strict default).
"""

from __future__ import annotations

import os
import cv2
import numpy as np

from qc.utils.video import VideoMetadata


def probe_image(path: str) -> VideoMetadata:
    """Read still-image metadata into a VideoMetadata (shared with video).

    Fields populated for a still:
        readable      : OpenCV decoded the file into an array
        extension     : lowercased file extension (e.g. ".jpg")
        width/height  : pixel dimensions, or None if undecodable
        channel_count : 1 (gray) or 3 (color), or None
        is_color      : True if the 3 channels carry real chroma (not a
                        grayscale image stored as 3 identical channels)
        codec_fourcc  : None (not meaningful for a still)
        fps/duration/frame_count : None (a still has none)
        reason        : failure detail when not readable
    """
    filename = os.path.basename(path)
    extension = os.path.splitext(filename)[1].lower()

    if not os.path.isfile(path):
        return VideoMetadata(
            path=path, filename=filename, extension=extension,
            readable=False, width=None, height=None, fps=None,
            frame_count=None, duration_sec=None, codec_fourcc=None,
            channel_count=None, is_color=None,
            reason="file not found",
        )

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return VideoMetadata(
            path=path, filename=filename, extension=extension,
            readable=False, width=None, height=None, fps=None,
            frame_count=None, duration_sec=None, codec_fourcc=None,
            channel_count=None, is_color=None,
            reason="OpenCV could not decode the image",
        )

    height, width = img.shape[:2]
    channel_count = 1 if img.ndim == 2 else img.shape[2]

    # is_color: a 3-channel image whose channels are identical is grayscale
    # stored as RGB. Same content-based test the video probe applies so the
    # rgb-required container check behaves identically for stills.
    is_color = None
    if channel_count is not None and channel_count >= 3:
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        is_color = not (np.array_equal(b, g) and np.array_equal(g, r))

    return VideoMetadata(
        path=path, filename=filename, extension=extension,
        readable=True, width=int(width), height=int(height),
        fps=None, frame_count=None, duration_sec=None,
        codec_fourcc=None, channel_count=int(channel_count),
        is_color=is_color, reason="",
    )
