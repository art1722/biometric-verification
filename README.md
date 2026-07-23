# Biometric QC — Multimodal Volunteer Data Validation

Automated quality-control for multimodal biometric captures collected from volunteers.
Given a folder of volunteer submissions, it validates each file against the capture
spec and reports which files pass and which fail, at scale (~1,500 volunteers).

Developed by the Image Processing Unit (IPU), Artificial Intelligence Research Group
(AINRG), National Electronics and Computer Technology Center (NECTEC).

---

## What it checks

Each volunteer submits 17 files across four biometric modalities. The system validates
four of them:

| Modality | Files/volunteer | Format | What is validated |
|---|---|---|---|
| Face (RGB) | 1 | `.mp4` | container, fps, duration, resolution, face detection, framing, size, eyes open, brightness, blur, occlusion, head-turn sequence |
| Face (depth/ir/thermal) | 4 | `.mp4` | filename presence only |
| Palm vein | 10 | `.jpg` | container, resolution, hand detection, handedness, size, brightness, finger spread, framing, roll/pitch angle |
| Gait (walk) | 2 | `.mp4` | container, fps, duration, resolution, body height, person detection, framing, blur, obstruction (YOLO), walk direction |

Every check returns one of three statuses: **PASS**, **FAIL**, or **SKIP** (not
applicable to this frame/image). A file passes only if it has no FAIL on any check.

---

## Requirements

- **Python 3.11.x** — MediaPipe ships wheels for 3.9–3.12 only; 3.11 is the most-tested.
- Model bundles (not included — see below).
- Optional: Docker Desktop, if running via container.

### Model bundles

Four pretrained models must be placed in `models/` before running. They are **not**
committed to the repo (large + downloaded once):

```
models/
    face_landmarker.task      # MediaPipe Face Landmarker
    hand_landmarker.task      # MediaPipe Hand Landmarker
    pose_landmarker.task      # MediaPipe Pose Landmarker
    yolov8n.pt                # YOLOv8-nano (COCO) — gait obstruction check
```

---

## Quick start (local)

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. put your data under data/ and your models under models/
#    (see "Input layout" below)

# 3. run the full batch over data/
python run_folder.py data
```

Results are written under `reports/` (see "Output" below).

## Quick start (Docker)

```bash
docker compose up -d --build      # build + start the API in the background
```

Then open the interactive API docs at **http://localhost:8001/docs**.

Docker mounts three host folders into the container: `models/`, `data/` (input),
and `reports/` (output). See the Docker & API section of the full documentation for
details.

---

## Input layout

Place each volunteer's files in the input folder. Files are matched by name; the
scanner searches up to 5 folder levels deep, so per-volunteer subfolders are fine.

```
data/
├── 001/
│   ├── 001_face_rgb.mp4
│   ├── 001_face_depth.mp4
│   ├── 001_face_ir1.mp4
│   ├── 001_face_ir2.mp4
│   └── 001_face_thermal.mp4
└── 002/
    ├── 002_face_rgb.mp4
    ├── 002_face_{depth,ir1,ir2,thermal}.mp4
    ├── 002_palm_{L,R}_{N,RL,RR,PU,PD}.jpg   # 10 palm images
    ├── 002_walk_F.mp4
    └── 002_walk_S.mp4
```

Filename patterns are defined in `config.yml` under `filenames.required`.

---

## Output

One batch run writes to `reports/`:

```
reports/
├── all_summary.csv / .json      # cross-modal roll-up (the main deliverable)
├── filenames.csv / .json        # filename validation (missing/extra/misnamed)
├── face_summary.csv             # per-check summary, face
├── palm_summary.csv             # per-check summary, palm
├── walk_summary.csv             # per-check summary, walk
└── <volunteer_id>/              # per-volunteer detail + debug overlays
        face_<id>_result.csv, overall.csv, detail.csv, ...
        face_<id>_overlay.mp4
        palm_<id>_*_overlay.jpg
        walk_<id>_{F,S}_overlay.mp4
```

The `_overlay` files are debug visualizations (skeleton/landmarks + per-check
verdicts drawn on the media); they do not affect grading.

---

## Running

The system can be driven three ways, from most to least configurable:

1. **Command line** — most options, can run modalities independently.
2. **API** (FastAPI) — run over a folder or an upload; poll for results.
3. **Streamlit dashboard** (`api_tester.py`) — easiest, fewest options.

### Command line

```bash
python run_folder.py data                    # all modalities
python run_folder.py data --no-walk          # skip gait
python run_folder.py data --limit 10         # first 10 volunteers only
python run_folder.py data --no-overlay --fail-fast   # faster, no debug videos
```

Per-modality runners also exist: `run_face.py`, `run_palm.py`, `run_walk.py`.

Common `run_folder.py` flags:

| Flag | Effect |
|---|---|
| `--no-face` / `--no-palm` / `--no-walk` | skip a modality |
| `--no-filenames` | skip filename validation |
| `--no-overlay` | do not write debug overlay media (faster) |
| `--fail-fast` | stop a file early on a structural defect (needs `--no-overlay`) |
| `--limit N` | process only the first N volunteers |
| `--sample-fps F` | frames sampled per second for video (default 5; 0 = every frame) |
| `--append` | append to an existing `all_summary` instead of overwriting |
| `--out-root DIR` | output folder (default `reports`) |
| `--config PATH` | config file (default `config.yml`) |

### API

```bash
uvicorn main:app --reload        # http://localhost:8000/docs
```

Key endpoints (full list at `/docs`):

- `POST /checks/uploads` — upload a data set (zip or folder), QC it, get a `job_id`
- `GET  /checks/batch/{job_id}` — poll progress
- `GET  /results` — read results; `GET /results/download` — download `all_summary`

---

## Configuration

All thresholds live in `config.yml`, grouped by modality (`face:`, `palm:`, `walk:`)
plus `models:`, `filenames:`, and `report:`. The spec is the source of truth; config
values marked `[SPEC]` come from the requirements, `[DESIGN]`/`[ASSUMPTION]` are
tunable defaults pending validation. Changing a threshold does not require code
changes.

---

## Project layout

```
qc/
├── pipelines/       # per-modality orchestration (face_rgb.py, palm.py, walk.py)
├── checks/          # individual check functions + model wrappers
├── utils/           # video/image I/O, report writers, overlay drawers
├── schemas/         # CheckRow data structure
└── api/             # FastAPI app, routing, job management
run_folder.py        # batch runner (all modalities)
run_face.py / run_palm.py / run_walk.py   # per-modality runners
main.py              # API entrypoint (uvicorn main:app)
config.yml           # all thresholds
api_tester.py        # Streamlit dashboard
view_results.py      # local results dashboard
```

---

## Documentation

Full technical documentation (system flow, per-modality checks, API reference,
configuration, Docker) is maintained separately. See the project documentation for
the detailed per-check explanations.