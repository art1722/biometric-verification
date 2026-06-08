"""Organize a delivery folder into per-volunteer subfolders - safely.

Reads a delivery folder (any layout) and COPIES recognised files into a tidy
structure:

    organized/<volunteer_id>/<filename>              # the keeper
    organized/<volunteer_id>_duplicates/<src_path>/<filename>   # extra copies

Volunteer id comes from the FILENAME (e.g. 001_face_rgb.mp4 -> volunteer 001).
The filename patterns are loaded from config.yml (filenames.required), the SAME
source validate_filenames.py uses, so the naming convention lives in ONE place.

Duplicate rule (a "duplicate" = the same filename found in >1 place):
  - The KEEPER is the copy whose containing folder NAME matches its volunteer
    id (e.g. 001_face_rgb.mp4 located inside a folder called "001"). That copy
    goes to organized/<id>/.
  - EVERY other copy goes to organized/<id>_duplicates/, under a subpath that
    mirrors where it came from, so you can see its origin.
  - If NO copy sits in a matching folder, there is no keeper: ALL copies go to
    _duplicates and a human decides. (organized/<id>/ will be missing that file.)

Safety:
  - Originals are never moved or modified - only copied.
  - Dry-run is the DEFAULT. Pass --apply to actually copy.
  - Unrecognised filenames are skipped and reported (exit code 1).

Usage:
    python organize.py <delivery_folder>                 # preview
    python organize.py <delivery_folder> --apply
    python organize.py <delivery_folder> --apply --out organized
    python organize.py <delivery_folder> --config config.yml
"""
import argparse
import os
import re
import shutil
import sys

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "config.yml")


def load_patterns(config_path):
    """Load the recognised filename patterns from config.yml.

    Returns a list of compiled regexes (from filenames.required). Each pattern
    must capture a named group `volunteer_id`. This is the SAME block
    validate_filenames.py reads, so both scripts agree on what is recognised.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        raw = cfg["filenames"]["required"]
    except (KeyError, TypeError):
        sys.exit(f"config '{config_path}' has no filenames.required section")

    patterns = []
    for key, pattern in raw.items():
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            sys.exit(f"bad regex for '{key}' in config: {e}")
        if "volunteer_id" not in (compiled.groupindex or {}):
            sys.exit(f"pattern for '{key}' must capture a (?P<volunteer_id>...) group")
        patterns.append(compiled)

    if not patterns:
        sys.exit("filenames.required is empty")
    return patterns


def volunteer_id(filename, patterns):
    name = os.path.basename(filename)
    for pat in patterns:
        m = pat.match(name)
        if m:
            return m.group("volunteer_id")
    return None


def scan(src_folder, out_abs, patterns):
    """Walk src. Return (groups, unrecognised).

    groups: dict keyed by (volunteer_id, filename) -> list of source paths.
            A key with >1 path is a duplicate (same name in multiple places).
    """
    groups = {}
    unrecognised = []
    for root, _, files in os.walk(src_folder):
        if os.path.abspath(root).startswith(out_abs):
            continue  # never re-ingest our own output
        for name in sorted(files):
            src = os.path.join(root, name)
            vid = volunteer_id(name, patterns)
            if vid is None:
                unrecognised.append(src)
                continue
            groups.setdefault((vid, name), []).append(src)
    return groups, unrecognised


def plan(src_folder, out_folder, patterns):
    """Return (copies, unrecognised).

    copies: list of (source_path, destination_path, role)
            role is "keeper", "duplicate", or "unrecognised".
    """
    out_abs = os.path.abspath(out_folder)
    groups, unrecognised = scan(src_folder, out_abs, patterns)

    copies = []
    for (vid, name), paths in sorted(groups.items()):
        if len(paths) == 1:
            # not duplicated -> straight to its folder
            copies.append((paths[0],
                           os.path.join(out_folder, vid, name),
                           "keeper"))
            continue

        # duplicated: keeper = the copy whose parent folder name == vid
        keeper = None
        for p in paths:
            parent = os.path.basename(os.path.dirname(p))
            if parent == vid:
                keeper = p
                break

        for p in paths:
            if p is keeper:
                copies.append((p, os.path.join(out_folder, vid, name), "keeper"))
            else:
                # mirror the original location under <id>_duplicates/
                rel = os.path.relpath(p, src_folder)
                rel_dir = os.path.dirname(rel)
                dst = os.path.join(out_folder, f"{vid}_duplicates",
                                   rel_dir, name)
                copies.append((p, dst, "duplicate"))

    # unrecognised files -> organized/_unrecognised/<original_path>/<name>
    # (same treatment as duplicates: copied for review, origin preserved)
    for p in unrecognised:
        rel = os.path.relpath(p, src_folder)
        dst = os.path.join(out_folder, "_unrecognised", rel)
        copies.append((p, dst, "unrecognised"))

    return copies, unrecognised


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("delivery_folder")
    ap.add_argument("--out", default="organized")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="path to config.yml (default: alongside this script)")
    ap.add_argument("--apply", action="store_true",
                    help="actually copy (default is a dry-run preview)")
    args = ap.parse_args()

    patterns = load_patterns(args.config)
    copies, unrecognised = plan(args.delivery_folder, args.out, patterns)
    keepers = [c for c in copies if c[2] == "keeper"]
    dups = [c for c in copies if c[2] == "duplicate"]

    mode = "APPLY" if args.apply else "DRY-RUN (nothing will be copied)"
    print(f"=== organize.py [{mode}] ===")
    print(f"source : {args.delivery_folder}")
    print(f"config : {args.config}")
    print(f"output : {args.out}")
    print(f"keepers: {len(keepers)} | duplicates: {len(dups)} | "
          f"unrecognised: {len(unrecognised)}\n")

    copied = skipped = 0
    for src, dst, role in copies:
        if args.apply:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                skipped += 1
                continue
            shutil.copy2(src, dst)
            copied += 1
        else:
            tag = {"duplicate": "DUP ", "unrecognised": "UNRG",
                   "keeper": "KEEP"}[role]
            print(f"  {tag}  {src}  ->  {dst}")

    if args.apply:
        print(f"Copied {copied} files ({skipped} already existed, skipped).")
    else:
        print("\nPreview only. Re-run with --apply to perform the copy.")

    if dups:
        print(f"\n{len(dups)} duplicate copy(ies) routed to *_duplicates/ "
              f"folders for manual review.")

    if unrecognised:
        print(f"\n!!! {len(unrecognised)} unrecognised file(s) - "
              f"copied to {args.out}/_unrecognised/ for manual review:")
        for p in unrecognised:
            print(f"  - {p}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())