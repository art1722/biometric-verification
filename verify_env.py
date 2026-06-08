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

print()
if problems:
    print(f"{len(problems)} problem(s) found — fix before running the pipeline.")
    sys.exit(1)
print("All good. Environment is ready.")
sys.exit(0)
