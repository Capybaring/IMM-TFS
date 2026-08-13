#!/usr/bin/env python3
"""
Quick integrity check for the expanded MIMIC layout expected by IMM-TFS.

Run from the IMM-TFS repository root:
    python check_expanded_mimic_layout.py
"""

from pathlib import Path
import pandas as pd

root = Path("data/MIMIC")
proc = root / "processed"
stats_path = root / "sample_statistics.csv"

if not proc.is_dir():
    raise SystemExit(f"Missing: {proc}")

records = sorted(
    p for p in proc.iterdir()
    if p.is_dir() and (p / "time_series.csv").exists()
)

print(f"processed records: {len(records):,}")
if not records:
    raise SystemExit("No record folders found.")

first = records[0]
ts = pd.read_csv(first / "time_series.csv")
txt = pd.read_csv(first / "text.csv")

features = [
    c for c in ts.columns
    if c not in ("date_time", "record_id")
]
print(f"first record: {first.name}")
print(f"numeric feature count: {len(features)}")
print(f"time_series rows: {len(ts):,}")
print(f"text rows: {len(txt):,}")

t = pd.to_datetime(ts["date_time"], errors="coerce")
anchor = t.min().normalize()
rel_h = (t - anchor).dt.total_seconds() / 3600.0
print(
    f"numeric relative-hour range: "
    f"[{rel_h.min():.3f}, {rel_h.max():.3f}]"
)

if stats_path.exists():
    stats = pd.read_csv(
        stats_path, usecols=["subject_id", "record_id"]
    ).dropna()
    stats["record_id"] = stats["record_id"].astype(str)
    rec_set = {p.name for p in records}
    sub = stats[stats["record_id"].isin(rec_set)]
    print(f"mapped records: {len(sub):,}/{len(records):,}")
    print(f"unique subjects: {sub['subject_id'].nunique():,}")
    repeated = len(sub) - sub["subject_id"].nunique()
    print(f"additional admissions from repeated subjects: {repeated:,}")
else:
    print(
        "WARNING: data/MIMIC/sample_statistics.csv is missing; "
        "subject-level split cannot be verified."
    )

print("Layout check finished.")
