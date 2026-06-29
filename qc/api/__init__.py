"""qc.api — FastAPI service: trigger QC batches, read results, check one video.

The app object lives in app.py; main.py at the repo root imports it.
"""

from .app import app

__all__ = ["app"]