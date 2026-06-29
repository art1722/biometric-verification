"""FastAPI entrypoint. The app and endpoints live in qc/api/.

    uvicorn main:app --host 0.0.0.0 --port 8000   # how the Dockerfile starts it
    uvicorn main:app --reload                      # local development
    python main.py                                 # uses the __main__ block below
"""

from qc.api import app  # noqa: F401  (re-exported for `uvicorn main:app`)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)