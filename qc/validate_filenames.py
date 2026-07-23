"""First-scan validator: naming convention + per-volunteer completeness.

This is a REPORT-ONLY scan. It does not move, rename, or modify anything - it
walks the delivery folder, parses every filename, groups files by volunteer id,
and reports two things:

  1. Unrecognised files - names that match no known spec pattern, even after
     relaxing extension case. Where the name starts with digits + '_', the
     digits are used to ATTRIBUTE the file to a volunteer id so that a
     volunteer whose every file is malformed still shows up in the
     completeness report (instead of silently vanishing).
  2. Per-volunteer completeness - which of the 17 required files each
     volunteer is missing (or has duplicated).

Matching is STRICT on extension case: a file is recognised only if its name
matches a spec pattern exactly, including a lowercase extension (.mp4 / .jpg).
A wrong-case extension such as '.JPG' or '.MP4' is NOT accepted - it is
reported as unrecognised (and attributed to a volunteer via leading digits
where possible). This replaces the earlier 'wrong_extension_case' accept-and-
warn behaviour, which is found confusing.

Patterns are loaded from config.yml (filenames.required) so the naming
convention lives in ONE place. Until Phase 0 confirms the ID-padding rule,
the config patterns accept an id of one or more digits (\\d+).

Volunteer ids are treated as nominal labels: '001' and '0001' are DIFFERENT
volunteers (no zero-stripping), per the project decision.

Output is written in TWO formats:
  - CSV  (long / tidy: one row per issue) - easy to filter in Excel.
  - JSON (nested: missing/duplicates stay as real lists) - for tooling.

Usage:
    python validate_filenames.py <delivery_folder>
    python validate_filenames.py <delivery_folder> --out reports/filenames.csv
    python validate_filenames.py <delivery_folder> --config config.yml
    (a .json file is written next to the .csv automatically)
"""
import argparse
import csv
import json
import os
import re
import sys

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")

# config.yml lives at the PROJECT ROOT, one level above this file (qc/).
# Resolving it from __file__ (rather than the process CWD) means the default
# works no matter which directory the script is invoked from:
#     python qc/validate_filenames.py data          # from the project root
#     python ../qc/validate_filenames.py ../data    # from anywhere else
# --config still overrides it when a different config is needed.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "config.yml")


def load_required(config_path):
    """Load (data_key, compiled_regex) pairs from config.yml.

    Returns (required, keys) where:
      required : list of (key, compiled_pattern), preserving config order
      keys     : list of keys (the required set per volunteer)
    Each pattern MUST contain a named group `volunteer_id`.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        raw = cfg["filenames"]["required"]
    except (KeyError, TypeError):
        sys.exit(f"config '{config_path}' has no filenames.required section")

    required = []
    for key, pattern in raw.items():
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            sys.exit(f"bad regex for '{key}' in config: {e}")
        if "volunteer_id" not in (compiled.groupindex or {}):
            sys.exit(f"pattern for '{key}' must capture a (?P<volunteer_id>...) group")
        required.append((key, compiled))

    if not required:
        sys.exit("filenames.required is empty")
    return required, [k for k, _ in required]


# Loose fallback for UNRECOGNISED names: leading digits + '_' look like a
# volunteer id. Used only to ATTRIBUTE the file in the report, never to
# accept it.
LOOSE_ID = re.compile(r"^(\d+)_")


def classify(filename, required):
    """Return (data_key, volunteer_id).

    Matching is STRICT: the name must match a config pattern exactly, including
    the extension case (.mp4 / .jpg lowercase). A wrong-case extension such as
    '.JPG' or '.MP4' no longer matches and is reported as unrecognised (it is
    still attributed to a volunteer via the leading-digits guess in scan()).
    Unrecognised -> (None, None).
    """
    name = os.path.basename(filename)
    for key, pat in required:
        m = pat.match(name)
        if m:
            return key, m.group("volunteer_id")

    return None, None


def scan(delivery_folder, required):
    """Walk the folder. Return (volunteers, unrecognised).

    volunteers: dict  volunteer_id -> dict  data_key -> [list of paths]
                (a list, so duplicates are visible)
    unrecognised: list of {"volunteer_guess": str ('' if none), "path": str}
                  Names that match no spec pattern, including wrong-case
                  extensions (e.g. '.JPG'). Where the name starts with
                  digits + '_', those digits attribute the file to a volunteer
                  so the volunteer still appears in the completeness report.
    """
    volunteers = {}
    unrecognised = []

    for root, _, files in os.walk(delivery_folder):
        for name in sorted(files):
            path = os.path.join(root, name)
            key, vid = classify(name, required)
            if key is None:
                m = LOOSE_ID.match(name)
                guess = m.group(1) if m else ""
                if guess:
                    # Surface this volunteer in the completeness report even
                    # if every one of their files is malformed.
                    volunteers.setdefault(guess, {})
                unrecognised.append({"volunteer_guess": guess, "path": path})
                continue
            volunteers.setdefault(vid, {}).setdefault(key, []).append(path)
    return volunteers, unrecognised


def build_report(volunteers, unrecognised, required_keys):
    """Turn the raw scan into a structured report (used for both JSON and CSV)."""
    per_volunteer = []
    complete = 0
    incomplete = 0

    for vid in sorted(volunteers):
        found = volunteers[vid]
        missing = [k for k in required_keys if k not in found]
        duplicates = [
            {"item": k, "paths": sorted(paths)}
            for k, paths in found.items() if len(paths) > 1
        ]
        status = "COMPLETE" if (not missing and not duplicates) else "INCOMPLETE"
        if status == "COMPLETE":
            complete += 1
        else:
            incomplete += 1
        per_volunteer.append({
            "volunteer": vid,
            "status": status,
            "missing": missing,
            "duplicates": duplicates,
            "found_count": sum(len(p) for p in found.values()),
        })

    return {
        "summary": {
            "volunteers": len(volunteers),
            "complete": complete,
            "incomplete": incomplete,
            "unrecognised_file_count": len(unrecognised),
        },
        "volunteers": per_volunteer,
        "unrecognised_files": unrecognised,
    }


def write_json(report, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def write_csv_long(report, path):
    """Long / tidy format: one row per issue.

    Columns: volunteer, status, issue_type, item, path
    The volunteer cell is never left empty (pandas would read it as NaN and
    promote the whole column to float): unattributable files get 'UNKNOWN'.
    """
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["volunteer", "status", "issue_type", "item", "path"])

        for v in report["volunteers"]:
            wrote_any = False
            for m in v["missing"]:
                w.writerow([v["volunteer"], v["status"], "missing", m, ""])
                wrote_any = True
            for d in v["duplicates"]:
                for p in d["paths"]:
                    w.writerow([v["volunteer"], v["status"], "duplicate",
                                d["item"], p])
                    wrote_any = True
            if not wrote_any:
                w.writerow([v["volunteer"], "COMPLETE", "", "", ""])

        for u in report["unrecognised_files"]:
            w.writerow([u["volunteer_guess"] or "UNKNOWN", "UNRECOGNISED",
                        "unrecognised", "", u["path"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("delivery_folder")
    ap.add_argument("--out", default="reports/filenames.csv")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="path to config.yml (default: alongside this script)")
    args = ap.parse_args()

    required, required_keys = load_required(args.config)

    volunteers, unrecognised = scan(args.delivery_folder, required)
    report = build_report(volunteers, unrecognised, required_keys)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    csv_path = args.out
    json_path = os.path.splitext(args.out)[0] + ".json"
    write_csv_long(report, csv_path)
    write_json(report, json_path)

    s = report["summary"]
    print("=== validate_filenames.py ===")
    print(f"folder    : {args.delivery_folder}")
    print(f"config    : {args.config}  ({len(required_keys)} required files/volunteer)")
    print(f"volunteers: {s['volunteers']}  "
          f"({s['complete']} complete, {s['incomplete']} incomplete)")
    print(f"unrecognised files: {s['unrecognised_file_count']}")
    print(f"reports   : {csv_path}  +  {json_path}")

    if s["incomplete"]:
        print("\nIncomplete volunteers:")
        for v in report["volunteers"]:
            if v["missing"]:
                print(f"  {v['volunteer']}: missing {len(v['missing'])} "
                      f"-> {', '.join(v['missing'])}")

    if unrecognised:
        print("\nUnrecognised files (check names against the spec; "
              "wrong-case extensions like .JPG land here too):")
        for u in unrecognised:
            who = f"  [volunteer {u['volunteer_guess']}?]" if u["volunteer_guess"] else ""
            print(f"  - {u['path']}{who}")

    clean = (s["complete"] == s["volunteers"] and not unrecognised)
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())