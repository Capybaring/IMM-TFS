#!/usr/bin/env python3
"""
Install the expanded-MIMIC parser into an IMM-TFS checkout.

Run from the IMM-TFS repository root:

    python apply_mimic_expanded_patch.py

This script only:
  1. copies lib/parse_datasets_mimic_expanded.py into ./lib/
     (when the source file sits next to this installer or in ./lib/);
  2. changes ONE import line in main.py.

It creates main.py.before_mimic_expanded_patch as a backup.
"""

from pathlib import Path
import shutil
import sys

ROOT = Path.cwd()
MAIN = ROOT / "main.py"
LIB = ROOT / "lib"
TARGET = LIB / "parse_datasets_mimic_expanded.py"

OLD = "from lib.parse_datasets import parse_datasets, get_input_and_pred_len"
NEW = (
    "from lib.parse_datasets_mimic_expanded import "
    "parse_datasets, get_input_and_pred_len"
)

if not MAIN.exists() or not LIB.is_dir():
    raise SystemExit(
        "Run this script from the IMM-TFS repository root "
        "(the directory containing main.py and lib/)."
    )

script_dir = Path(__file__).resolve().parent
candidates = [
    script_dir / "parse_datasets_mimic_expanded.py",
    script_dir / "lib" / "parse_datasets_mimic_expanded.py",
    ROOT / "parse_datasets_mimic_expanded.py",
    ROOT / "lib" / "parse_datasets_mimic_expanded.py",
]
source = next((p for p in candidates if p.exists()), None)

if source is None:
    raise SystemExit(
        "Could not find parse_datasets_mimic_expanded.py next to this "
        "installer or under ./lib/."
    )

if source.resolve() != TARGET.resolve():
    shutil.copy2(source, TARGET)
    print(f"Copied: {source} -> {TARGET}")
else:
    print(f"Adapter already in place: {TARGET}")

text = MAIN.read_text(encoding="utf-8")

if NEW in text:
    print("main.py already imports the expanded-MIMIC adapter.")
elif OLD in text:
    backup = ROOT / "main.py.before_mimic_expanded_patch"
    if not backup.exists():
        shutil.copy2(MAIN, backup)
        print(f"Backup: {backup}")
    text = text.replace(OLD, NEW, 1)
    MAIN.write_text(text, encoding="utf-8")
    print("Patched main.py import.")
else:
    raise SystemExit(
        "Expected import line was not found in main.py. "
        "No main.py changes were made."
    )

print("\nDone.")
print("Next: put/symlink your expanded cohort at data/MIMIC/processed/")
print("Also copy sample_statistics.csv to data/MIMIC/sample_statistics.csv")
