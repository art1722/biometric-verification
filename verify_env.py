"""Verify the environment is correctly set up BEFORE running the pipeline.

Run:  python verify_env.py

Checks every dependency the pipeline needs, and specifically confirms that
mediapipe's `solutions` API is importable and a detector can actually be
created — the exact thing that silently failed and produced a CSV full of
"module 'mediapipe' has no attribute 'solutions'" rows.

Exit code 0 = good to go. Non-zero = fix before running anything.
"""
import sys

EXPECTED_PYTHON = (3, 11)
problems = []


def ok(msg):
    print(f"  OK    {msg}")


def fail(msg):
    print(f"  FAIL  {msg}")
    problems.append(msg)


print("=== Environment check ===")

# --- Python version ---
v = sys.version_info
if (v.major, v.minor) == EXPECTED_PYTHON:
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
else:
    fail(f"Python {v.major}.{v.minor} (expected 3.11; mediapipe has no wheels "
         f"for 3.13+, and other minors are less tested)")

# --- imports + versions ---
def check_version(modname, attr="__version__", expected=None):
    try:
        mod = __import__(modname)
    except Exception as e:
        fail(f"import {modname}: {e}")
        return None
    ver = getattr(mod, attr, "unknown")
    if expected and not str(ver).startswith(expected):
        fail(f"{modname} {ver} (expected {expected}.x)")
    else:
        ok(f"{modname} {ver}")
    return mod

check_version("cv2")                       # opencv
check_version("numpy", expected="1.26")
check_version("yaml")
mp = check_version("mediapipe", expected="0.10.21")

# --- the critical one: does mp.solutions actually work? ---
if mp is not None:
    where = getattr(mp, "__file__", "?")
    if "site-packages" not in str(where) and ".venv" not in str(where):
        fail(f"mediapipe imported from {where} — looks like a local file is "
             f"shadowing the real package (rename any mediapipe.py / mediapipe/ "
             f"in your project)")
    else:
        ok(f"mediapipe location: {where}")

    if hasattr(mp, "solutions"):
        ok("mp.solutions attribute present")
        try:
            fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                                 max_num_faces=1)
            fm.close()
            ok("created + closed a FaceMesh detector (solutions API works)")
        except Exception as e:
            fail(f"could not create FaceMesh: {e}")
    else:
        fail("mp.solutions MISSING — this is the broken-install symptom. "
             "Try: pip install -r requirements.txt --force-reinstall")

    # --- Tasks API + Face Landmarker model bundle (needed for blendshape eyes) ---
    import os
    MODEL_PATH = os.path.join("models", "face_landmarker.task")
    DL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
    try:
        _ = mp.tasks.vision.FaceLandmarker
        ok("mp.tasks.vision.FaceLandmarker present (Tasks API works)")
    except Exception as e:
        fail(f"mp.tasks.vision.FaceLandmarker missing: {e}")

    if os.path.isfile(MODEL_PATH):
        ok(f"face_landmarker model bundle found: {MODEL_PATH}")
        try:
            BaseOptions = mp.tasks.BaseOptions
            FLO = mp.tasks.vision.FaceLandmarkerOptions
            VRM = mp.tasks.vision.RunningMode
            det = mp.tasks.vision.FaceLandmarker.create_from_options(
                FLO(base_options=BaseOptions(model_asset_path=MODEL_PATH),
                    running_mode=VRM.IMAGE, num_faces=1,
                    output_face_blendshapes=True))
            det.close()
            ok("created + closed a FaceLandmarker with blendshapes enabled")
        except Exception as e:
            fail(f"could not create FaceLandmarker from {MODEL_PATH}: {e}")
    else:
        fail(f"face_landmarker model bundle MISSING: {MODEL_PATH}\n"
             f"        Download it once with:\n"
             f"          mkdir -p models\n"
             f"          curl -L -o {MODEL_PATH} \\\n"
             f"            {DL_URL}")

print()
if problems:
    print(f"{len(problems)} problem(s) found — fix before running the pipeline.")
    sys.exit(1)
print("All good. Environment is ready.")
sys.exit(0)