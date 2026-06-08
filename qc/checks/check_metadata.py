"""Video metadata checks (NEW — not in the repo).

These operate on the VideoMetadata that qc.utils.video.probe_video already
produced. They run no model and read no pixels, so they are cheap and run once
per video (not per frame).

Each returns (status, message) where status is one of the config status
strings: PASS / FAIL / REVIEW / SKIP. Metadata that could not be read (None)
yields REVIEW, not FAIL — "we couldn't measure it" is different from "it failed",
and a human should confirm rather than auto-rejecting (screening-filter
philosophy).

Spec values (source of truth):
  - container: MPEG-4 / .mp4, and RGB (3-channel) for the rgb stream
  - fps:       >= 5 (face), >= 30 (gait)
  - duration:  >= 40s (face), >= 15s (gait)
  - resolution:>= 180x180 (face), >= 1920x1080 (gait)
"""

from __future__ import annotations

PASS = "PASS"
FAIL = "FAIL"
REVIEW = "REVIEW"
SKIP = "SKIP"


def check_container(meta, *, require_rgb: bool = True, expected_ext: str = ".mp4"):
    """Is the file a readable .mp4, and (for rgb) 3-channel color?"""
    if not meta.readable:
        return (FAIL, f"unreadable: {meta.reason}")
    if meta.extension != expected_ext:
        return (FAIL, f"extension={meta.extension} != {expected_ext}")
    if require_rgb:
        if meta.channel_count is None:
            return (REVIEW, "channel count unknown")
        if meta.channel_count < 3:
            return (FAIL, f"channels={meta.channel_count} (not RGB color)")
    return (PASS, f"container ok ({meta.extension}, "
                  f"{meta.channel_count} channels, codec={meta.codec_fourcc})")


def check_fps(meta, *, min_fps: float = 5.0):
    if meta.fps is None:
        return (REVIEW, "fps unknown")
    if meta.fps >= min_fps:
        return (PASS, f"fps={meta.fps:.2f} >= {min_fps}")
    return (FAIL, f"fps={meta.fps:.2f} < {min_fps}")


def check_duration(meta, *, min_duration_sec: float = 40.0):
    if meta.duration_sec is None:
        return (REVIEW, "duration unknown")
    if meta.duration_sec >= min_duration_sec:
        return (PASS, f"duration={meta.duration_sec:.1f} >= {min_duration_sec}")
    return (FAIL, f"duration={meta.duration_sec:.1f} < {min_duration_sec}")


def check_resolution(meta, *, min_width: int = 180, min_height: int = 180):
    if meta.width is None or meta.height is None:
        return (REVIEW, "resolution unknown")
    if meta.width >= min_width and meta.height >= min_height:
        return (PASS, f"resolution={meta.width}x{meta.height} "
                      f">= {min_width}x{min_height}")
    return (FAIL, f"resolution={meta.width}x{meta.height} "
                  f"< {min_width}x{min_height}")
