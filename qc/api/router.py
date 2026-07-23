from __future__ import annotations

import functools

from qc.validate_filenames import load_required, classify as _classify


# Outcome codes the API layer maps to HTTP status.
MATCH_RUNNABLE = "runnable"          # recognised AND a pipeline exists -> run it
MATCH_NOT_IMPLEMENTED = "not_implemented"  # recognised, no pipeline yet -> 501
UNRECOGNISED = "unrecognised"        # name matches nothing -> 422

# Which data_keys can be graded from a SINGLE uploaded file via /checks/file.
# Extend as you wire a pipeline in live.check_one.
#
# face_rgb : one video -> full result.
# walk_F/S : one video -> full result (run_walk grades a single clip).
# palm_*   : DELIBERATELY EXCLUDED. A palm image's headline check (angle) is
#            graded ABSOLUTELY per image, but the meaningful palm verdict spans
#            all five poses together (run_palm_participant). One image in
#            isolation cannot produce that, so palm is batch-only: a single palm
#            upload here stays 501 rather than returning a misleadingly partial
#            pass. Run palm through /checks/batch or /checks/uploads instead.
RUNNABLE_MODALITIES = {"face_rgb", "walk_F", "walk_S"}


@functools.lru_cache(maxsize=1)
def _required(config_path: str):
    """Compiled (key, regex) patterns from config.yml, cached per path."""
    required, _keys = load_required(config_path)
    return required


def route(filename: str, config_path: str) -> dict:
    """Classify one filename. Returns a dict the API turns into a response.

    {
      "outcome": runnable | not_implemented | unrecognised,
      "data_key": "face_rgb" | ... | None,
      "volunteer_id": "001" | None,
    }
    """
    data_key, volunteer_id = _classify(filename, _required(config_path))

    if data_key is None:
        return {"outcome": UNRECOGNISED, "data_key": None, "volunteer_id": None}

    if data_key in RUNNABLE_MODALITIES:
        outcome = MATCH_RUNNABLE
    else:
        outcome = MATCH_NOT_IMPLEMENTED

    return {"outcome": outcome, "data_key": data_key, "volunteer_id": volunteer_id}