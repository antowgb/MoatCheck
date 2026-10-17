"""Build static Python crisis constants from checksum-verified source artifacts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from pprint import pformat


HEADER = '''"""Generated historical constants. Provenance and canonical CSVs live in data/.

Regenerate with backend/scripts/embed_crisis_snapshots.py. Tests verify parity.
No filesystem or network access occurs in the scenario compute API.
"""

'''


def main() -> None:
    """Generate constants, or fail --check when checked-in constants have drifted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    core = Path(__file__).resolve().parents[1] / "app/core/risk"
    directory = core / "data"
    manifest = json.loads((directory / "manifest.json").read_text())
    panels = {}
    for item in manifest["scenarios"]:
        source = directory / item["file"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["artifact_sha256"][item["file"]]:
            raise SystemExit(f"Checksum mismatch: {item['file']}")
        with source.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        panels[item["id"]] = {
            "dates": tuple(row["date"] for row in rows),
            "returns": tuple(tuple(float(row[asset]) for asset in item["assets"]) for row in rows),
        }
    content = HEADER + "MANIFEST = " + pformat(manifest, sort_dicts=False, width=100)
    content += "\n\nPANELS = " + pformat(panels, sort_dicts=False, width=100) + "\n"
    target = core / "_crisis_data.py"
    if args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != content:
            raise SystemExit("Embedded crisis constants are stale. Run this script without --check.")
    else:
        target.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
