#!/usr/bin/env python3
"""
AgentDisruptBench — Populate Croissant SHA256 Hashes
=====================================================

Updates croissant.json with SHA256 hashes for all task YAML files.
Run this before submitting to NeurIPS.

Usage:
    python3 scripts/populate_croissant_hashes.py
"""

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CROISSANT_PATH = PROJECT_ROOT / "croissant.json"


def sha256_file(path: Path) -> str:
    """Compute SHA256 hex digest for a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main():
    missing_files = []
    with open(CROISSANT_PATH) as f:
        croissant = json.load(f)

    for dist in croissant.get("distribution", []):
        content_url = dist.get("contentUrl", "")
        file_path = PROJECT_ROOT / content_url
        if file_path.exists():
            sha = sha256_file(file_path)
            dist["sha256"] = sha
            print(f"  ✅ {dist['name']}: {sha[:16]}...")
        else:
            print(f"  ⚠️  {dist['name']}: file not found at {file_path}")
            missing_files.append(dist['name'])

    with open(CROISSANT_PATH, "w") as f:
        json.dump(croissant, f, indent=2)
        f.write("\n")

    if missing_files:
        print(f"\n❌ {len(missing_files)} file(s) not found: {', '.join(missing_files)}")
        sys.exit(1)
    else:
        print(f"\n✅ Updated {CROISSANT_PATH}")


if __name__ == "__main__":
    main()
