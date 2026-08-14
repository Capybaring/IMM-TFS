#!/usr/bin/env python3
"""Prepare the fixed/nested expanded-MIMIC scaling protocol (v3 per-N norm).

This script creates ONE persisted subject split and ONE persisted randomized
training-subject order. It intentionally DOES NOT fit normalization.

Formal protocol
---------------
1. Split ALL processed subjects once into fixed 60/20/20 train/val/test.
2. Shuffle the FULL training pool once with a separate seed.
3. Train_N subsets are nested prefixes:
       T200 ⊂ T500 ⊂ T1000 ⊂ ...
4. Validation and test subjects never change with N.
5. At run time, each Train_N fits its own feature-wise z-score using ONLY
   observed numeric values from that Train_N's HISTORY [0,24h).

Output:
  data/MIMIC/mimic_fixed_protocol.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


UNIT_SECONDS = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
    "weeks": 604800.0,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="data/MIMIC")
    p.add_argument("--history", type=float, default=24.0)
    p.add_argument("--pred-window", type=float, default=24.0)
    p.add_argument("--time-unit", default="hours", choices=list(UNIT_SECONDS))
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-order-seed", type=int, default=2026)
    p.add_argument("--protocol-name", default="mimic_fixed_protocol.json")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def episode_anchor(ts: pd.Series) -> pd.Timestamp:
    ts = pd.to_datetime(ts, errors="coerce").dropna()
    if ts.empty:
        raise ValueError("empty/invalid timestamps")
    return ts.min().normalize()


def load_record_subject_mapping(dataset_dir: Path, record_ids: list[str]):
    stats_path = dataset_dir / "sample_statistics.csv"
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"{stats_path} is required for a fixed SUBJECT-level split"
        )

    stats = pd.read_csv(stats_path, usecols=["subject_id", "record_id"]).dropna()
    stats["record_id"] = stats["record_id"].astype(str)
    stats["subject_id"] = stats["subject_id"].astype(str)
    mapping = dict(zip(stats["record_id"], stats["subject_id"]))

    missing = [r for r in record_ids if r not in mapping]
    if missing:
        raise ValueError(
            f"{len(missing)} processed record IDs are missing from {stats_path}; "
            f"examples={missing[:10]}"
        )
    return {r: mapping[r] for r in record_ids}


def flatten_records(subjects: list[str], subject_to_records: dict[str, list[str]]):
    out: list[str] = []
    for s in subjects:
        out.extend(subject_to_records[s])
    return out


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    proc_dir = dataset_dir / "processed"
    protocol_path = dataset_dir / args.protocol_name

    if not proc_dir.is_dir():
        raise FileNotFoundError(proc_dir)

    if protocol_path.exists() and not args.force:
        raise FileExistsError(
            "Protocol output already exists. Use --force only if you intentionally "
            f"want to rebuild the subject split/order.\n  {protocol_path}"
        )

    record_ids = sorted(
        p.name
        for p in proc_dir.iterdir()
        if p.is_dir() and (p / "time_series.csv").is_file()
    )
    if not record_ids:
        raise RuntimeError(f"No processed records under {proc_dir}")

    rec_to_subject = load_record_subject_mapping(dataset_dir, record_ids)
    subject_to_records: dict[str, list[str]] = defaultdict(list)
    for rec in record_ids:
        subject_to_records[rec_to_subject[rec]].append(rec)
    subject_to_records = {s: sorted(rs) for s, rs in subject_to_records.items()}

    subjects = sorted(subject_to_records)
    if len(subjects) < 5:
        raise RuntimeError(f"Only {len(subjects)} subjects found")

    trainval_subj, test_subj = train_test_split(
        subjects,
        train_size=0.8,
        random_state=args.split_seed,
        shuffle=True,
    )
    train_subj, val_subj = train_test_split(
        trainval_subj,
        train_size=0.75,
        random_state=args.split_seed,
        shuffle=True,
    )

    train_subj = sorted(map(str, train_subj))
    val_subj = sorted(map(str, val_subj))
    test_subj = sorted(map(str, test_subj))

    if (
        set(train_subj) & set(val_subj)
        or set(train_subj) & set(test_subj)
        or set(val_subj) & set(test_subj)
    ):
        raise RuntimeError("Subject overlap detected")

    rng = np.random.RandomState(args.train_order_seed)
    train_subject_order = [
        str(x) for x in rng.permutation(train_subj).tolist()
    ]

    train_records_full = flatten_records(train_subject_order, subject_to_records)
    val_records = flatten_records(val_subj, subject_to_records)
    test_records = flatten_records(test_subj, subject_to_records)

    # We scan the episode files only to lock feature schema and padding maxima.
    # No target values are used for normalization here; no scaler is fitted.
    sec_per_unit = UNIT_SECONDS[args.time_unit]
    total_window = args.history + args.pred_window
    feature_cols: list[str] | None = None
    global_max_history_timestamps = 0
    global_max_prediction_timestamps = 0

    print("=" * 72)
    print("Preparing fixed/nested expanded-MIMIC protocol")
    print("=" * 72)
    print(f"dataset              : {dataset_dir}")
    print(f"records              : {len(record_ids):,}")
    print(f"subjects             : {len(subjects):,}")
    print(
        f"fixed subjects       : train={len(train_subj):,}, "
        f"val={len(val_subj):,}, test={len(test_subj):,}"
    )
    print(f"split seed           : {args.split_seed}")
    print(f"train-order seed     : {args.train_order_seed}")
    print("normalization        : NOT fitted here; each Train_N fits HISTORY only")
    print("=" * 72)

    for k, rec in enumerate(record_ids, start=1):
        path = proc_dir / rec / "time_series.csv"
        df = pd.read_csv(path)
        if "date_time" not in df.columns:
            raise ValueError(f"{path}: missing date_time")

        df["_ts_raw"] = pd.to_datetime(df["date_time"], errors="coerce")
        df = df.dropna(subset=["_ts_raw"]).sort_values("_ts_raw").copy()
        if df.empty:
            raise ValueError(f"{path}: no valid timestamps")

        feats = [
            c for c in df.columns
            if c not in ("date_time", "record_id", "_ts_raw")
        ]
        if feature_cols is None:
            feature_cols = feats
        elif feats != feature_cols:
            raise ValueError(
                f"{rec}: feature schema/order differs from first record"
            )

        anchor = episode_anchor(df["_ts_raw"])
        rel = (
            (df["_ts_raw"] - anchor).dt.total_seconds().to_numpy(dtype=np.float64)
            / sec_per_unit
        )
        h = (rel >= 0.0) & (rel < args.history)
        p = (rel >= args.history) & (rel < total_window)
        global_max_history_timestamps = max(
            global_max_history_timestamps, int(h.sum())
        )
        global_max_prediction_timestamps = max(
            global_max_prediction_timestamps, int(p.sum())
        )

        if k % 1000 == 0 or k == len(record_ids):
            print(f"scanned {k:,}/{len(record_ids):,}")

    assert feature_cols is not None

    protocol = {
        "version": "3-per-n-normalization",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "processed_record_count": len(record_ids),
        "subject_count": len(subjects),
        "split_seed": int(args.split_seed),
        "train_order_seed": int(args.train_order_seed),
        "history": float(args.history),
        "pred_window": float(args.pred_window),
        "time_unit": args.time_unit,
        "normalization_source": "each Train_N; history observations only",
        "feature_cols": feature_cols,
        "global_max_history_timestamps": int(global_max_history_timestamps),
        "global_max_prediction_timestamps": int(global_max_prediction_timestamps),
        "train_subject_count": len(train_subj),
        "val_subject_count": len(val_subj),
        "test_subject_count": len(test_subj),
        "train_record_count": len(train_records_full),
        "val_record_count": len(val_records),
        "test_record_count": len(test_records),
        "train_subject_order": train_subject_order,
        "val_subjects": val_subj,
        "test_subjects": test_subj,
        "subject_to_records": subject_to_records,
        "val_records": val_records,
        "test_records": test_records,
    }

    with open(protocol_path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("Fixed protocol written")
    print("=" * 72)
    print(f"protocol             : {protocol_path}")
    print(f"train subjects       : {len(train_subj):,}")
    print(f"val subjects         : {len(val_subj):,}")
    print(f"test subjects        : {len(test_subj):,}")
    print(f"max history rows     : {global_max_history_timestamps:,}")
    print(f"max prediction rows  : {global_max_prediction_timestamps:,}")
    print(f"feature count        : {len(feature_cols)}")
    print("normalization file   : none")
    print("normalization rule   : fit separately from each Train_N HISTORY")
    print("=" * 72)


if __name__ == "__main__":
    main()
