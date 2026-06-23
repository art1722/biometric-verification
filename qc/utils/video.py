"""Video utility helpers for biometric QC pipelines.

This module is intentionally small and dependency-light: it only wraps OpenCV's
``VideoCapture`` so checks/pipelines can share ONE way to inspect videos and
sample frames.

Design assumption for this project:
    - The pipeline receives a whole .mp4 file path.
    - OpenCV decodes the file frame-by-frame.
    - Image-based checks then run on sampled frames.
    - Metadata checks use fps/resolution/duration from the same probe.

Typical usage:
    from qc.utils.video import probe_video, sample_video_frames

    meta = probe_video("001_walk_F.mp4")
    sampled = sample_video_frames("001_walk_F.mp4", sample_fps=1, max_frames=600)

For strict gait validation, pass ``sample_fps=None`` to inspect every frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional
import math
import os
import numpy as np


PathLike = str | os.PathLike[str]


class VideoError(RuntimeError):
    """Base error for video probing/extraction problems."""


class VideoOpenError(VideoError):
    """Raised when OpenCV cannot open a video file."""


class VideoReadError(VideoError):
    """Raised when OpenCV opens a file but cannot decode requested frames."""


@dataclass(frozen=True)
class VideoMetadata:
    """Basic video metadata used by QC checks.

    ``readable`` means OpenCV could open the file and decode at least one frame.
    Width/height/fps/frame_count may still be unavailable for unusual/corrupt
    files, so downstream checks should handle ``None``.
    """

    path: str
    filename: str
    extension: str
    readable: bool
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    frame_count: Optional[int]
    duration_sec: Optional[float]
    codec_fourcc: Optional[str]
    channel_count: Optional[int]
    is_color: Optional[bool] = None  # True=real chroma, False=grayscale-as-3ch, None=unknown
    reason: str = ""

    @property
    def resolution(self) -> Optional[tuple[int, int]]:
        """Return ``(width, height)`` when both values are known."""
        if self.width is None or self.height is None:
            return None
        return self.width, self.height

    @property
    def is_full_hd_or_higher(self) -> Optional[bool]:
        """Convenience check for the gait requirement: at least 1920x1080."""
        if self.width is None or self.height is None:
            return None
        return self.width >= 1920 and self.height >= 1080


@dataclass(frozen=True)
class SampledFrame:
    """A decoded frame plus timing/index metadata.

    ``image`` is a NumPy array from OpenCV. By default it is BGR. If
    ``as_rgb=True`` is used, it is RGB.
    """

    frame_index: int
    timestamp_sec: float
    timestamp_ms: int
    image: Any
    color_space: str = "BGR"


@dataclass(frozen=True)
class VideoSample:
    """Combined result: metadata + extracted frames."""

    metadata: VideoMetadata
    frames: list[SampledFrame]


def _require_cv2():
    """Import OpenCV lazily so config/filename tools can run without cv2."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            "OpenCV is required for video utilities. Install with: "
            "pip install opencv-python"
        ) from exc
    return cv2


def _normalise_path(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def _clean_float(value: float | int | None) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v) or v <= 0:
        return None
    return v


def _clean_int(value: float | int | None) -> Optional[int]:
    if value is None:
        return None
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return v


def _fourcc_to_string(raw_fourcc: float | int | None) -> Optional[str]:
    """Convert OpenCV CAP_PROP_FOURCC numeric value to a readable code."""
    if raw_fourcc is None:
        return None
    try:
        code = int(raw_fourcc)
    except (TypeError, ValueError):
        return None
    if code <= 0:
        return None
    chars = [chr((code >> 8 * i) & 0xFF) for i in range(4)]
    text = "".join(chars).strip("\x00").strip()
    return text or None


def _frame_channel_count(frame: Any) -> Optional[int]:
    """Return 1 for grayscale, 3 for BGR/RGB, 4 for BGRA/RGBA, etc."""
    shape = getattr(frame, "shape", None)
    if shape is None:
        return None
    if len(shape) == 2:
        return 1
    if len(shape) >= 3:
        return int(shape[2])
    return None


def _is_truly_RGB(frame, *, sample_stride=4, tol=8, min_color_frac=0.02):
    """True if the frame has real chroma (R/G/B differ), not grayscale-as-3ch."""
    if frame is None or getattr(frame, "ndim", 0) < 3 or frame.shape[2] < 3:
        return False
    f = frame[::sample_stride, ::sample_stride, :3].astype(np.int16)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    # max channel spread per pixel
    spread = np.maximum(np.maximum(abs(r - g), abs(g - b)), abs(r - b))
    color_frac = float((spread > tol).mean())
    return color_frac >= min_color_frac


def probe_video(path: PathLike, *, decode_first_frame: bool = True) -> VideoMetadata:
    """Read metadata from a video file.

    This function never raises for ordinary bad input. Instead, it returns a
    ``VideoMetadata`` with ``readable=False`` and a reason. That makes it easier
    for check files to produce FAIL rows instead of crashing the whole batch.

    Args:
        path: Video path.
        decode_first_frame: If true, verify that at least one frame can be
            decoded and use that frame as a fallback for width/height/channels.

    Returns:
        ``VideoMetadata``.
    """
    cv2 = _require_cv2()
    p = _normalise_path(path)

    if not p.exists():
        return VideoMetadata(
            path=str(p),
            filename=p.name,
            extension=p.suffix.lower(),
            readable=False,
            width=None,
            height=None,
            fps=None,
            frame_count=None,
            duration_sec=None,
            codec_fourcc=None,
            channel_count=None,
            reason="file does not exist",
        )

    if not p.is_file():
        return VideoMetadata(
            path=str(p),
            filename=p.name,
            extension=p.suffix.lower(),
            readable=False,
            width=None,
            height=None,
            fps=None,
            frame_count=None,
            duration_sec=None,
            codec_fourcc=None,
            channel_count=None,
            reason="path is not a file",
        )

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        cap.release()
        return VideoMetadata(
            path=str(p),
            filename=p.name,
            extension=p.suffix.lower(),
            readable=False,
            width=None,
            height=None,
            fps=None,
            frame_count=None,
            duration_sec=None,
            codec_fourcc=None,
            channel_count=None,
            reason="OpenCV could not open video",
        )

    fps = _clean_float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = _clean_int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = _clean_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = _clean_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec_fourcc = _fourcc_to_string(cap.get(cv2.CAP_PROP_FOURCC))
    channel_count: Optional[int] = None
    is_color: Optional[bool] = None
    readable = True
    reason = ""

    if decode_first_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            readable = False
            reason = "OpenCV opened file but could not decode first frame"
        else:
            frame_h, frame_w = frame.shape[:2]
            width = width or int(frame_w)
            height = height or int(frame_h)
            channel_count = _frame_channel_count(frame)
            is_color = _is_truly_RGB(frame)

    cap.release()

    duration_sec: Optional[float]
    if fps is not None and frame_count is not None:
        duration_sec = frame_count / fps
    else:
        duration_sec = None

    return VideoMetadata(
        path=str(p),
        filename=p.name,
        extension=p.suffix.lower(),
        readable=readable,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
        codec_fourcc=codec_fourcc,
        channel_count=channel_count,
        is_color=is_color,
        reason=reason,
    )


def _build_sample_indices(
    *,
    frame_count: Optional[int],
    fps: Optional[float],
    sample_fps: Optional[float],
    include_first_frame: bool,
    include_last_frame: bool,
    max_frames: Optional[int],
    start_sec: float,
    end_sec: Optional[float],
) -> Optional[list[int]]:
    """Return frame indices to read when frame count is known.

    Returns ``None`` when we cannot safely pre-compute indices, in which case
    callers should fall back to sequential reading.
    """
    if frame_count is None or frame_count <= 0:
        return None

    start_index = 0
    end_index = frame_count - 1

    if fps is not None:
        start_index = max(0, int(math.floor(max(0.0, start_sec) * fps)))
        if end_sec is not None:
            end_index = min(end_index, int(math.ceil(max(0.0, end_sec) * fps)))

    if start_index > end_index:
        return []

    if sample_fps is None or sample_fps <= 0 or fps is None:
        step = 1
    else:
        step = max(1, int(round(fps / sample_fps)))

    indices = list(range(start_index, end_index + 1, step))

    if include_first_frame:
        indices.append(start_index)
    if include_last_frame:
        indices.append(end_index)

    indices = sorted(set(i for i in indices if start_index <= i <= end_index))

    if max_frames is not None and max_frames > 0 and len(indices) > max_frames:
        # Even-stride downsample. The previous approach used per-position
        # rounding (round(i*(n-1)/(max_frames-1))), which produced an UNEVEN
        # grid: mostly the base step, but with periodic double-steps where the
        # rounding skipped an index. That showed up in the detail CSV as
        # occasional frames missing from an otherwise regular cadence (e.g. a
        # +0.133s jump among +0.067s steps). Instead, pick a single integer
        # stride over the candidate list so every kept frame is equally spaced.
        # This keeps <= max_frames frames with a uniform gap and no holes.
        if max_frames == 1:
            indices = [indices[0]]
        else:
            stride = math.ceil(len(indices) / max_frames)
            indices = indices[::stride]

    return indices


def _convert_color(frame: Any, *, as_rgb: bool) -> tuple[Any, str]:
    if not as_rgb:
        return frame, "BGR"
    cv2 = _require_cv2()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "RGB"


def _timestamp_for_index(frame_index: int, fps: Optional[float], cap: Any) -> float:
    if fps is not None:
        return frame_index / fps
    # CAP_PROP_POS_MSEC is not always populated, but use it as a fallback.
    try:
        pos_msec = float(cap.get(_require_cv2().CAP_PROP_POS_MSEC))
    except Exception:
        pos_msec = 0.0
    if pos_msec > 0:
        return pos_msec / 1000.0
    return float(frame_index)


def iter_sampled_frames(
    path: PathLike,
    *,
    sample_fps: Optional[float] = 1.0,
    include_first_frame: bool = True,
    include_last_frame: bool = True,
    max_frames: Optional[int] = 600,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    as_rgb: bool = False,
) -> Iterator[SampledFrame]:
    """Yield sampled frames from a video.

    Args:
        path: Video path.
        sample_fps: Desired sampling rate. Use ``None`` or ``<= 0`` to inspect
            every frame. This is useful for strict gait checks.
        include_first_frame: Always include the first frame in the selected
            time range.
        include_last_frame: Always include the last frame in the selected time
            range when frame count is known.
        max_frames: Safety cap. Use ``None`` for no cap.
        start_sec: Start time in seconds.
        end_sec: Optional end time in seconds.
        as_rgb: Convert OpenCV BGR frames to RGB before returning.

    Yields:
        ``SampledFrame`` objects.

    Raises:
        VideoOpenError: if the video cannot be opened/read.
        VideoReadError: if requested frames cannot be decoded.
    """
    cv2 = _require_cv2()
    meta = probe_video(path, decode_first_frame=False)
    if not meta.readable:
        raise VideoOpenError(f"cannot open video: {meta.path} ({meta.reason})")

    cap = cv2.VideoCapture(meta.path)
    if not cap.isOpened():
        cap.release()
        raise VideoOpenError(f"cannot open video: {meta.path}")

    indices = _build_sample_indices(
        frame_count=meta.frame_count,
        fps=meta.fps,
        sample_fps=sample_fps,
        include_first_frame=include_first_frame,
        include_last_frame=include_last_frame,
        max_frames=max_frames,
        start_sec=start_sec,
        end_sec=end_sec,
    )

    try:
        if indices is not None:
            seen_indices = set()
            
            for frame_index in indices:
                if frame_index in seen_indices:
                    continue
                seen_indices.add(frame_index)

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()

                if not ok or frame is None:
                    # Common near the end of MP4 files when metadata says the frame exists
                    # but OpenCV cannot decode it. Skip instead of crashing the whole QC.
                    continue

                image, color_space = _convert_color(frame, as_rgb=as_rgb)
                timestamp_sec = _timestamp_for_index(frame_index, meta.fps, cap)

                yield SampledFrame(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    timestamp_ms=int(round(timestamp_sec * 1000)),
                    image=image,
                    color_space=color_space,
                )

            return

        # Fallback path when frame count is unavailable. Read sequentially.
        frame_index = 0
        yielded = 0
        next_sample_time = max(0.0, start_sec)
        
        ASSUMED_FPS_WHEN_UNKNOWN = 30.0
        every_frame = sample_fps is None or sample_fps <= 0
        effective_fps = meta.fps if meta.fps is not None and meta.fps > 0 else ASSUMED_FPS_WHEN_UNKNOWN
        sample_period = None if every_frame else 1.0 / float(sample_fps)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            timestamp_sec = frame_index / effective_fps
            if timestamp_sec < start_sec:
                frame_index += 1
                continue
            if end_sec is not None and timestamp_sec > end_sec:
                break

            should_yield = every_frame or timestamp_sec + 1e-9 >= next_sample_time
            if should_yield:
                image, color_space = _convert_color(frame, as_rgb=as_rgb)
                yield SampledFrame(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    timestamp_ms=int(round(timestamp_sec * 1000)),
                    image=image,
                    color_space=color_space,
                )
                yielded += 1
                if sample_period is not None:
                    next_sample_time += sample_period
                if max_frames is not None and max_frames > 0 and yielded >= max_frames:
                    break

            frame_index += 1
    finally:
        cap.release()


def sample_video_frames(
    path: PathLike,
    *,
    sample_fps: Optional[float] = 1.0,
    include_first_frame: bool = True,
    include_last_frame: bool = False,
    max_frames: Optional[int] = 600,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    as_rgb: bool = False,
) -> VideoSample:
    """Return metadata and a list of sampled frames.

    This is the main helper most pipelines should call.
    """
    metadata = probe_video(path)
    if not metadata.readable:
        return VideoSample(metadata=metadata, frames=[])

    frames = list(
        iter_sampled_frames(
            path,
            sample_fps=sample_fps,
            include_first_frame=include_first_frame,
            include_last_frame=include_last_frame,
            max_frames=max_frames,
            start_sec=start_sec,
            end_sec=end_sec,
            as_rgb=as_rgb,
        )
    )
    return VideoSample(metadata=metadata, frames=frames)


def iter_all_frames(path: PathLike, *, as_rgb: bool = False) -> Iterator[SampledFrame]:
    """Yield every decodable frame from a video.

    Useful for strict gait checks where the spec says the full body / walking
    posture should be visible every frame.
    """
    yield from iter_sampled_frames(
        path,
        sample_fps=None,
        include_first_frame=True,
        include_last_frame=True,
        max_frames=None,
        as_rgb=as_rgb,
    )


def frame_indices(frames: Iterable[SampledFrame]) -> list[int]:
    """Small convenience helper for debugging/reporting."""
    return [f.frame_index for f in frames]


def metadata_to_dict(meta: VideoMetadata) -> dict[str, Any]:
    """Serialize ``VideoMetadata`` to a JSON/CSV-friendly dictionary."""
    return {
        "path": meta.path,
        "filename": meta.filename,
        "extension": meta.extension,
        "readable": meta.readable,
        "width": meta.width,
        "height": meta.height,
        "fps": meta.fps,
        "frame_count": meta.frame_count,
        "duration_sec": meta.duration_sec,
        "codec_fourcc": meta.codec_fourcc,
        "channel_count": meta.channel_count,
        "is_color": meta.is_color,
        "reason": meta.reason,
    }


def format_metadata(meta: VideoMetadata) -> str:
    """Human-readable one-line metadata summary."""
    if not meta.readable:
        return f"{meta.filename}: unreadable ({meta.reason})"

    fps = "unknown" if meta.fps is None else f"{meta.fps:.2f} fps"
    duration = (
        "unknown"
        if meta.duration_sec is None
        else f"{meta.duration_sec:.2f} sec"
    )
    resolution = (
        "unknown"
        if meta.width is None or meta.height is None
        else f"{meta.width}x{meta.height}"
    )
    frames = "unknown" if meta.frame_count is None else str(meta.frame_count)
    codec = meta.codec_fourcc or "unknown codec"
    return (
        f"{meta.filename}: {resolution}, {fps}, {duration}, "
        f"{frames} frames, {codec}"
    )