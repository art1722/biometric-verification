"""router.py — decide which pipeline a file belongs to, from its NAME.

This is the dispatch brain for POST /checks. It does NOT run QC itself; it
parses the filename against the project's naming convention and says either
"this is face_rgb, run it" or "this is palm_L_N, not built yet" or "this name
matches nothing, reject it".

Single source of truth
----------------------
The naming convention lives in config.yml (filenames.required) and is parsed by
validate_filenames.classify() — the SAME function the first-scan validator uses.
We reuse it so the API and the validator can never disagree about what a valid
name is.

classify(name, required) returns:
    (data_key, volunteer_id)   e.g. ("face_rgb", "001")   on a match
    (None, None)               on an unrecognised name

data_key is one of the keys in config: face_rgb, face_depth, face_ir1, face_ir2,
face_thermal, palm_L_N ... palm_R_PD, walk_F, walk_S.

What's actually runnable today
------------------------------
Only face_rgb has a pipeline (run_face_rgb). Everything else is recognised by
name but has no QC code yet, so the router marks it NOT_IMPLEMENTED. As palm /
walk pipelines get written, add a branch here — nothing else changes.
"""

from __future__ import annotations

import functools

# Reuse the validator's parsing so the convention lives in one place.
from validate_filenames import load_required, classify as _classify


# Outcome codes the API layer maps to HTTP status.
MATCH_RUNNABLE = "runnable"          # recognised AND a pipeline exists -> run it
MATCH_NOT_IMPLEMENTED = "not_implemented"  # recognised, no pipeline yet -> 501
UNRECOGNISED = "unrecognised"        # name matches nothing -> 422

# Which data_keys have a real pipeline today. Extend as you build them.
RUNNABLE_MODALITIES = {"face_rgb"}


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