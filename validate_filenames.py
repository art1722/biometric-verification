"""First-scan validator: naming convention + per-volunteer completeness.

This is a REPORT-ONLY scan. It does not move, rename, or modify anything - it
walks the delivery folder, parses every filename, groups files by volunteer id,
and reports two things:

  1. Unrecognised files - names that match no known spec pattern.
  2. Per-volunteer completeness - which of the 17 required files each
     volunteer is missing (or has duplicated).

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

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "config.yml")


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


def classify(filename, required):
    """Return (data_key, volunteer_id) if recognised, else (None, None)."""
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
    unrecognised: list of paths that matched no pattern
    """
    volunteers = {}
    unrecognised = []

    for root, _, files in os.walk(delivery_folder):
        for name in sorted(files):
            path = os.path.join(root, name)
            key, vid = classify(name, required)
            if key is None:
                unrecognised.append(path)
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
    """
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["volunteer", "status", "issue_type", "item", "path"])

        for v in report["volunteers"]:
            if v["status"] == "COMPLETE":
                w.writerow([v["volunteer"], "COMPLETE", "", "", ""])
                continue
            for m in v["missing"]:
                w.writerow([v["volunteer"], "INCOMPLETE", "missing", m, ""])
            for d in v["duplicates"]:
                for p in d["paths"]:
                    w.writerow([v["volunteer"], "INCOMPLETE", "duplicate",
                                d["item"], p])

        for p in report["unrecognised_files"]:
            w.writerow(["", "UNRECOGNISED", "unrecognised", "", p])


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
        print("\nUnrecognised files (check names against the spec):")
        for p in unrecognised:
            print(f"  - {p}")

    return 0 if (s["complete"] == s["volunteers"] and not unrecognised) else 1


if __name__ == "__main__":
    sys.exit(main())