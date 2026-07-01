"""uploads.py — turn an HTTP upload into a temp data/ dir a batch can read.

Why this module exists
----------------------
The researcher's mental model is "upload the folder, get all_summary back". But
HTTP has no "folder" type: multipart/form-data only carries FILES. A folder is
always flattened by the CLIENT into a list of files before it reaches us. So an
upload arrives in one of two shapes, and this module accepts BOTH through the
same endpoint:

  1. ONE .zip           — the user zipped the folder and dragged the zip.
  2. MANY loose files   — the user picked a folder (<input webkitdirectory>),
                          and the browser sent every file, each carrying its
                          relative path in the filename field.

Either way, the job of this module is the same: materialise the upload as a
directory tree on disk that run_folder.py can walk. run_folder discovers files
by FILENAME regex under os.walk, so nested subfolders are fine and even a flat
dump works — but we preserve subpaths anyway so the input mirrors what the user
uploaded (useful for debugging and for the wrong-case warnings run_folder logs).

Isolation
---------
Every upload unpacks into its OWN temp root (tempfile.mkdtemp). Nothing is
written into the shared server data/ dir, so one upload can never see another's
files or the standing data set. The caller is responsible for pointing the job
at this dir and (optionally) cleaning it up afterwards.

Safety
------
Two classes of malicious/broken input are rejected here, before any QC runs:
  - Zip-slip: a zip entry whose path escapes the target dir (e.g. "../../etc/x")
    is refused. We resolve each destination and require it stay inside the root.
  - Absolute paths in upload filenames are stripped to their basename-relative
    form for the same reason.
Non-matching junk (a .DS_Store, a stray .txt) is harmless: run_folder simply
never matches it, so we keep it rather than trying to filter here.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from typing import Iterable, Optional


class BadUpload(ValueError):
    """The upload could not be unpacked (empty, unsafe path, corrupt zip)."""


def _safe_join(root: str, member: str) -> str:
    """Join `member` onto `root`, guaranteeing the result stays inside `root`.

    Defends against zip-slip / absolute-path escapes. `member` may use either
    '/' or '\\' separators (zips and webkitRelativePath both show up). Returns
    an absolute path inside root; raises BadUpload if it would escape.
    """
    # Normalise separators, drop any leading slash / drive, collapse '..'.
    member = member.replace("\\", "/").lstrip("/")
    # Reject Windows drive-absolute like "C:/..." defensively.
    if ":" in member.split("/")[0]:
        raise BadUpload(f"unsafe path in upload: {member!r}")

    root_abs = os.path.abspath(root)
    dest = os.path.abspath(os.path.join(root_abs, member))
    # dest must be root itself or a child of root.
    if dest != root_abs and not dest.startswith(root_abs + os.sep):
        raise BadUpload(f"path escapes upload dir: {member!r}")
    return dest


def _looks_like_zip(filename: str, head: bytes) -> bool:
    """A single upload is treated as a zip if its name ends .zip OR its first
    bytes are the zip magic (PK\\x03\\x04). The magic check catches a zip sent
    without the extension; the name check catches an empty-but-named zip."""
    if filename.lower().endswith(".zip"):
        return True
    return head[:4] == b"PK\x03\x04"


def _extract_zip(zip_path: str, dest_root: str) -> int:
    """Extract every member of `zip_path` under `dest_root`, safely. Returns the
    number of FILES written (directories are not counted)."""
    n = 0
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                name = info.filename
                # Skip directory entries; _safe_join validates file entries.
                if name.endswith("/"):
                    continue
                target = _safe_join(dest_root, name)
                os.makedirs(os.path.dirname(target) or dest_root, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                n += 1
    except zipfile.BadZipFile as e:
        raise BadUpload(f"corrupt zip: {e}") from e
    return n


def unpack_uploads(
    files: Iterable[tuple[str, bytes]],
    *,
    dest_root: Optional[str] = None,
) -> tuple[str, int]:
    """Materialise an upload as a directory tree and return (dest_root, n_files).

    Args:
        files: an iterable of (filename, raw_bytes) pairs. `filename` may carry
            a relative path (folder-picker) — it is preserved under dest_root.
            A single .zip pair is detected and extracted in place.
        dest_root: where to unpack. Defaults to a fresh tempfile.mkdtemp() so
            each upload is isolated. If given, it must already exist.

    Returns:
        (dest_root, n_files) — the directory the batch should read, and how many
        files were written (zip members counted individually).

    Raises:
        BadUpload if the upload is empty, a zip is corrupt, or any path is unsafe.
    """
    materialised = dest_root or tempfile.mkdtemp(prefix="qc_upload_")

    pairs = list(files)
    if not pairs:
        raise BadUpload("no files in upload")

    # Single-file upload that is a zip -> extract it and we're done.
    if len(pairs) == 1:
        name, raw = pairs[0]
        if _looks_like_zip(name, raw):
            tmp_zip = os.path.join(materialised, "_upload.zip")
            with open(tmp_zip, "wb") as f:
                f.write(raw)
            try:
                n = _extract_zip(tmp_zip, materialised)
            finally:
                # The zip itself is not input data; remove it so run_folder's
                # os.walk never sees it.
                try:
                    os.remove(tmp_zip)
                except OSError:
                    pass
            if n == 0:
                raise BadUpload("zip contained no files")
            return materialised, n

    # Otherwise: many loose files (or one non-zip file). Write each, preserving
    # any relative path in its filename so folder structure is reconstructed.
    n = 0
    for name, raw in pairs:
        if not name:
            continue
        target = _safe_join(materialised, name)
        os.makedirs(os.path.dirname(target) or materialised, exist_ok=True)
        with open(target, "wb") as f:
            f.write(raw)
        n += 1

    if n == 0:
        raise BadUpload("no usable files in upload")
    return materialised, n