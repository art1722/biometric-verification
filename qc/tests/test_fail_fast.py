"""Tests for the three fail-fast structural gates in run_face_rgb.

  GATE 1: a video-level metadata FAIL (here: too-short duration / low fps)
          stops the pipeline before any frame work -> frame_checks SKIP.
  GATE 2: a file that opens but yields 0 frames -> frames_sampled FAIL.
  GATE 3: a frame with MULTIPLE faces -> check_face_detected FAIL + loop break,
          and the per-frame loop does NOT continue past the offending frame.

GATES 1 and 2 use real synthetic videos written with OpenCV and need no face
model. GATE 3 monkeypatches the detector so it needs no .task bundle either.

Also checks fail_fast=False keeps the OLD behaviour (no early stop on metadata).
"""
import os
import subprocess
import tempfile

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from qc.pipelines import face_rgb
from qc.pipelines.face_rgb import run_face_rgb


@pytest.fixture
def stub_face_det(monkeypatch):
    """The real env has mp.solutions.face_detection (used by blur/brightness),
    but this CI MediaPipe wheel is stripped of `solutions`. For tests that
    stub the detector and never reach a real frame's blur/brightness, provide a
    no-op FaceDetection so the shared-detector setup line doesn't crash."""
    import types
    fake = types.SimpleNamespace(
        solutions=types.SimpleNamespace(
            face_detection=types.SimpleNamespace(
                FaceDetection=lambda **k: types.SimpleNamespace(
                    close=lambda: None))))
    monkeypatch.setattr(face_rgb, "mp", fake)


# Minimal config: only the keys run_face_rgb reads. Spec thresholds matter for
# the metadata gate (min_fps=5, min_duration_sec=40); everything else is a
# sensible default. turn_sequence disabled so an aborted run is unambiguous.
def _cfg(**over):
    cfg = {
        "face": {
            "metadata": {"min_fps": 5, "min_duration_sec": 40},
            "size": {"min_head_width_px": 180, "min_head_height_px": 180},
            "checks": {},
            "turn_sequence": {"enabled": False},
        },
        "video": {"max_frames": 6000},
        "models": {
            "mediapipe": {}, "face_landmarker": {}, "face_detection": {},
        },
    }
    cfg.update(over)
    return cfg


def _rows_by_check(rows):
    out = {}
    for r in rows:
        out.setdefault(r.check_name, []).append(r)
    return out


def _write_video(path, n_frames, fps, size=(320, 320)):
    """Write an mp4 of solid grey frames. n_frames may be 0."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = size
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for _ in range(n_frames):
        vw.write(np.full((h, w, 3), 127, dtype=np.uint8))
    vw.release()


# ----------------------------------------------------------------------------
# GATE 1 — video-level metadata FAIL halts before frame work
# ----------------------------------------------------------------------------
def test_gate1_metadata_fail_stops_before_frames():
    # 10 frames @ 5fps = 2s, well under the 40s minimum -> check_duration FAIL.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "001_face_rgb.mp4")
        _write_video(p, n_frames=10, fps=5)
        rows, timeline = run_face_rgb(p, "001", _cfg(), sample_fps=1,
                                      fail_fast=True)

    by = _rows_by_check(rows)
    # duration failed...
    assert by["check_duration"][0].status == "FAIL"
    # ...and the pipeline stopped: a frame_checks SKIP row, no per-frame rows.
    assert "frame_checks" in by
    assert by["frame_checks"][0].status == "SKIP"
    assert "fail-fast" in by["frame_checks"][0].reason
    assert "check_face_detected" not in by   # never entered the frame loop
    assert timeline == []


def test_gate1_disabled_when_fail_fast_false(stub_face_det):
    # Same off-spec file, fail_fast=False: the metadata FAIL is still recorded,
    # but the pipeline does NOT early-stop on it (it proceeds to frame work).
    # We assert only that the early frame_checks SKIP gate did NOT fire.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "001_face_rgb.mp4")
        _write_video(p, n_frames=10, fps=5)
        # Stub the model so frame work doesn't need a real .task bundle.
        face_rgb.create_face_landmarker = lambda **k: type("L", (), {"close": lambda self: None})()
        face_rgb.detect_face = lambda *a, **k: type(
            "R", (), {"ok": False, "message": "No faces", "landmarks_px": None,
                      "bbox": None, "blendshapes": None, "landmarks_norm": None})()
        rows, timeline = run_face_rgb(p, "001", _cfg(), sample_fps=1,
                                      fail_fast=False)

    by = _rows_by_check(rows)
    assert by["check_duration"][0].status == "FAIL"      # still recorded
    # The fail-fast SKIP gate did not fire (its reason text is the tell):
    fc = by.get("frame_checks", [])
    assert not any("fail-fast" in r.reason for r in fc)


# ----------------------------------------------------------------------------
# GATE 2 — zero frames -> FAIL (not a silent PASS)
# ----------------------------------------------------------------------------
def test_gate2_zero_frames_is_fail(stub_face_det):
    # A spec-conforming HEADER (>=40s, >=5fps) but ZERO frames written, so the
    # metadata gate passes and the loop runs zero times. Old code: PASS. New: FAIL.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "001_face_rgb.mp4")
        # Write then truncate to 0 frames: many encoders won't store a header
        # for an empty stream, so instead fake a readable-but-frameless source
        # by stubbing iter_sampled_frames to yield nothing while metadata is OK.
        _write_video(p, n_frames=300, fps=10)  # 30s... still < 40, so also stub meta

        from qc.utils import video as videomod

        class _Meta:
            readable = True
            fps = 10.0
            duration_sec = 45.0
            width = 320
            height = 320
            container = "mp4"
            has_rgb = True

        face_rgb.probe_video = lambda _p: _Meta()
        # metadata checks read the meta object; force them to PASS by giving a
        # conforming meta, then yield zero frames.
        import qc.checks.check_metadata as md
        md_ok = ("PASS", "ok")
        face_rgb.md.check_container = lambda *a, **k: md_ok
        face_rgb.md.check_fps = lambda *a, **k: md_ok
        face_rgb.md.check_duration = lambda *a, **k: md_ok
        face_rgb.md.check_resolution = lambda *a, **k: md_ok
        face_rgb.iter_sampled_frames = lambda *a, **k: iter(())
        face_rgb.create_face_landmarker = lambda **k: type("L", (), {"close": lambda self: None})()

        rows, timeline = run_face_rgb(p, "001", _cfg(), sample_fps=1,
                                      fail_fast=True)

    by = _rows_by_check(rows)
    assert "frames_sampled" in by
    assert by["frames_sampled"][0].status == "FAIL"          # was a silent PASS
    assert "0 frames" in by["frames_sampled"][0].reason


# ----------------------------------------------------------------------------
# GATE 3 — multiple faces -> FAIL + loop break
# ----------------------------------------------------------------------------
def test_gate3_multiple_faces_breaks_loop(monkeypatch, stub_face_det):
    # Conforming header + several frames. The stubbed detector reports MULTIPLE
    # faces on the FIRST frame. Expect: one check_face_detected FAIL, the loop
    # breaks (only ONE check_face_detected row total even though 5 frames exist),
    # and the disabled turn check produces no sequence rows.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "001_face_rgb.mp4")
        _write_video(p, n_frames=5, fps=10)

        class _Meta:
            readable = True; fps = 10.0; duration_sec = 45.0
            width = 320; height = 320; container = "mp4"; has_rgb = True

        monkeypatch.setattr(face_rgb, "probe_video", lambda _p: _Meta())
        md_ok = ("PASS", "ok")
        monkeypatch.setattr(face_rgb.md, "check_container", lambda *a, **k: md_ok)
        monkeypatch.setattr(face_rgb.md, "check_fps", lambda *a, **k: md_ok)
        monkeypatch.setattr(face_rgb.md, "check_duration", lambda *a, **k: md_ok)
        monkeypatch.setattr(face_rgb.md, "check_resolution", lambda *a, **k: md_ok)
        monkeypatch.setattr(face_rgb, "create_face_landmarker", lambda **k: type("L", (), {"close": lambda self: None})())

        # Fake sampled-frame objects.
        class _SF:
            def __init__(self, i):
                self.frame_index = i
                self.timestamp_sec = i / 10.0
                self.image = np.zeros((320, 320, 3), dtype=np.uint8)
                self.color_space = "BGR"

        monkeypatch.setattr(face_rgb, "iter_sampled_frames",
                            lambda *a, **k: iter(_SF(i) for i in range(5)))

        # Detector: ALWAYS reports multiple faces.
        def _multi(*a, **k):
            return type("R", (), {
                "ok": False, "message": "Multiple faces detected (2)",
                "landmarks_px": None, "bbox": None, "blendshapes": None,
                "landmarks_norm": None})()
        monkeypatch.setattr(face_rgb, "detect_face", _multi)

        rows, timeline = run_face_rgb(p, "001", _cfg(), sample_fps=1,
                                      fail_fast=True)

    by = _rows_by_check(rows)
    fd = by.get("check_face_detected", [])
    assert len(fd) == 1, "loop should break after the first multi-face frame"
    assert fd[0].status == "FAIL"
    assert "Multiple faces" in fd[0].reason
    # frames_sampled still recorded (1 frame seen before the break)...
    assert by["frames_sampled"][0].status == "PASS"
    # ...and the aborted run skipped the (disabled) turn sequence entirely.
    assert not any(r.check_name.startswith("check_turn") for r in rows)


def test_gate3_does_not_break_when_fail_fast_false(monkeypatch, stub_face_det):
    # Same multi-face detector, fail_fast=False: the FAIL is recorded on EVERY
    # frame and the loop does NOT break early (full timeline for the dashboard).
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "001_face_rgb.mp4")
        _write_video(p, n_frames=5, fps=10)

        class _Meta:
            readable = True; fps = 10.0; duration_sec = 45.0
            width = 320; height = 320; container = "mp4"; has_rgb = True

        monkeypatch.setattr(face_rgb, "probe_video", lambda _p: _Meta())
        md_ok = ("PASS", "ok")
        for name in ("check_container", "check_fps", "check_duration",
                     "check_resolution"):
            monkeypatch.setattr(face_rgb.md, name, lambda *a, **k: md_ok)
        monkeypatch.setattr(face_rgb, "create_face_landmarker", lambda **k: type("L", (), {"close": lambda self: None})())

        class _SF:
            def __init__(self, i):
                self.frame_index = i; self.timestamp_sec = i / 10.0
                self.image = np.zeros((320, 320, 3), dtype=np.uint8)
                self.color_space = "BGR"

        monkeypatch.setattr(face_rgb, "iter_sampled_frames",
                            lambda *a, **k: iter(_SF(i) for i in range(5)))
        monkeypatch.setattr(face_rgb, "detect_face", lambda *a, **k: type(
            "R", (), {"ok": False, "message": "Multiple faces detected (2)",
                      "landmarks_px": None, "bbox": None, "blendshapes": None,
                      "landmarks_norm": None})())

        rows, _ = run_face_rgb(p, "001", _cfg(), sample_fps=1, fail_fast=False)

    fd = [r for r in rows if r.check_name == "check_face_detected"]
    assert len(fd) == 5, "no break: every frame should be judged"
    assert all(r.status == "FAIL" for r in fd)