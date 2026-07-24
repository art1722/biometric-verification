# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — biometric QC API
#
# Build it:   docker build -t biometric-qc .
# Run it:     docker run -p 8000:8000 biometric-qc
# Then open:  http://localhost:8000/docs
#
# The `# syntax=` line on line 1 is REQUIRED for `RUN --mount=type=cache` below.
# It must be the very first line of the file — comments above it break it.
# ─────────────────────────────────────────────────────────────────────────────

# 1. BASE IMAGE — start from an official, slim Python 3.11 on Linux.
#    "slim" = small (no extra tools you don't need). Matches your pinned 3.11.
FROM python:3.11-slim

# 2. SYSTEM LIBRARIES — OpenCV/MediaPipe need a few Linux .so libraries that are
#    NOT in the slim image. Without these, `import cv2` crashes at runtime with
#    "libGL.so.1: cannot open shared object file". We install them, then delete
#    the apt cache to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. WORKDIR — every command after this runs inside /app in the container.
WORKDIR /app

# 3b. ULTRALYTICS (YOLO) — keep ALL its state INSIDE the project, not in the
#     user home (~/.config/Ultralytics), so the container is self-contained and
#     reproducible on any machine. YOLO_CONFIG_DIR redirects its settings.json /
#     cache here; YOLO_OFFLINE + no telemetry stop it from phoning home or trying
#     to auto-download at runtime (weights are vendored in models/, see below).
ENV YOLO_CONFIG_DIR=/app/.ultralytics \
    YOLO_OFFLINE=1 \
    ULTRALYTICS_ANALYTICS=False
RUN mkdir -p /app/.ultralytics

# 4. INSTALL DEPENDENCIES FIRST (before copying the rest of the code).
#    WHY separately: Docker caches each step. As long as requirements.txt does
#    not change, Docker reuses the cached "pip install" layer and your rebuilds
#    after editing a .py file take seconds, not minutes.
#
#    --mount=type=cache keeps pip's HTTP wheel cache in a Docker-managed volume
#    that SURVIVES layer invalidation. So when you DO edit requirements.txt, pip
#    re-resolves but re-downloads only what actually changed.
#
#    NOTE: do NOT add --no-cache-dir here. It tells pip to write nothing to
#    ~/.cache/pip, which makes the cache mount above completely pointless. The
#    two options cancel each other out.
#
#    NOTE: we deliberately do NOT run `pip install --upgrade pip`. The base
#    image's pip installs these wheels fine, and upgrading re-downloads pip on
#    every cold build for no benefit.
#
#    The CPU-only PyTorch index lives in requirements.txt (not here) so that a
#    plain host-side `pip install -r requirements.txt` resolves identically.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# 5. COPY YOUR PROJECT into the image (qc/, main.py, config.yml, run_*.py, ...).
#    .dockerignore controls what is skipped (data/, reports/, __pycache__, ...).
COPY . .

# 6. MODELS FOLDER — your config points at models/face_landmarker.task,
#    models/hand_landmarker.task, models/pose_landmarker.task, and
#    models/yolov8n.pt (the COCO YOLO weights for the occlusion check). They are
#    large bundles you download once. They are NOT baked into the image (see
#    .dockerignore). At run time you mount them in (see the README block below).
#    We just ensure the folder exists.
RUN mkdir -p models reports data

# 7. NETWORK PORT — document that the app listens on 8000. This does not open
#    anything by itself; you still publish it with `-p 8000:8000` at run time.
EXPOSE 8000

# 8. START COMMAND — launch the FastAPI app with uvicorn.
#    "main:app" means: in main.py, use the variable named `app`.
#    host 0.0.0.0 = listen on all interfaces so it is reachable from outside the
#    container (127.0.0.1 would only be reachable from INSIDE the container).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]