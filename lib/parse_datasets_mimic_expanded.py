"""
Expanded-MIMIC adapter for IMM-TFS.

Why this file exists
--------------------
The original IMM-TFS loader assumes long records that are turned into sliding
(history + prediction) chunks.  The expanded MIMIC preprocessing used here
already produces ONE fixed 0-48 h episode per admission:

    0 <= t < 24 h   : history
    24 <= t < 48 h  : prediction target

Using the original ChunkedTimeSeriesDataset on these files causes several
problems:
  1. it requires st + 48 <= t_max, but the saved data is strictly < 48 h;
  2. it resets t=0 to each record's earliest numeric observation, shifting the
     clinically-defined 0 h / 24 h boundary;
  3. split_method="sample" is invalid when each record has one episode;
  4. it materializes tensors directly on GPU, which does not scale to ~20k
     admissions;
  5. get_input_and_pred_len() scans a whole extra epoch.

This adapter changes ONLY the MIMIC data-loading path.  Other datasets are
delegated to lib.parse_datasets unchanged.

Expected layout
---------------
data/MIMIC/
├── processed/
│   └── <hadm_id>/
│       ├── time_series.csv
│       ├── text.csv
│       └── text_embeddings_model=...pt   # optional
├── sample_statistics.csv                 # strongly recommended
└── tfsimm_dataset_config.json            # optional metadata

The adapter keeps tensors on CPU until collate time, then moves only the
current batch to args.device.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

from lib.parse_datasets import (
    get_input_and_pred_len as _original_get_input_and_pred_len,
    parse_datasets as _original_parse_datasets,
    patch_variable_time_collate_fn,
    variable_time_collate_fn,
    variable_time_collate_fn_CRU,
    variable_time_collate_fn_ODE,
)


UNIT_SECONDS = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
    "weeks": 604800.0,
}


def _is_expanded_mimic(dataset_name: str) -> bool:
    # Allows MIMIC, MIMIC_FULL, MIMIC_EXT, etc.
    return str(dataset_name).upper().startswith("MIMIC")


def _resolve_dataset_path(args: argparse.Namespace) -> str:
    base = (
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", args.data_root)
        )
        if not os.path.isabs(args.data_root)
        else args.data_root
    )
    return os.path.join(base, args.dataset)


def _seconds_per_unit(time_unit: str, unit_scale=None) -> float:
    if time_unit == "custom":
        if unit_scale is None:
            raise ValueError("Must set unit_scale when time_unit='custom'")
        return float(unit_scale)
    try:
        return UNIT_SECONDS[time_unit]
    except KeyError as e:
        raise ValueError(f"Unknown time_unit {time_unit!r}") from e


def _episode_anchor(ts: pd.Series) -> pd.Timestamp:
    """
    Expanded-MIMIC files use a synthetic episode clock rooted at the start of
    the first calendar day (currently 2000-01-01 00:00:00).

    Do NOT use ts.min(): the first actual observation may occur after ICU hour
    0, and using it as the origin would shift the 24 h forecasting boundary.
    """
    ts = pd.to_datetime(ts, errors="coerce").dropna()
    if ts.empty:
        raise ValueError("Cannot determine episode anchor from empty timestamps")
    return ts.min().normalize()


class ExpandedMIMICDataset(Dataset):
    """One fixed 0-(history+pred_window) episode per record/admission."""

    def __init__(
        self,
        root: str,
        history: int,
        pred_window: int,
        device: torch.device,
        time_unit: str = "hours",
        unit_scale=None,
        normalize: bool = True,
        enable_text: bool = False,
        use_text_embeddings: bool = False,
        llm_model_fusion: str | None = None,
        llm_layers_fusion: int | None = None,
        max_length: int = 1024,
        args: argparse.Namespace | None = None,
    ):
        super().__init__()

        self.root = root
        self.proc_dir = os.path.join(root, "processed")
        self.history = float(history)
        self.pred_window = float(pred_window)
        self.total_window = float(history + pred_window)
        self.device = device
        self.normalize = normalize
        self.enable_text = enable_text
        self.use_text_embeddings = use_text_embeddings
        self.llm_model_fusion = llm_model_fusion
        self.llm_layers_fusion = llm_layers_fusion
        self.max_length = int(max_length)
        self.sec_per_unit = _seconds_per_unit(time_unit, unit_scale)

        if not os.path.isdir(self.proc_dir):
            raise FileNotFoundError(
                f"Processed directory not found: {self.proc_dir}"
            )

        rec_ids = sorted(
            d
            for d in os.listdir(self.proc_dir)
            if os.path.isdir(os.path.join(self.proc_dir, d))
            and os.path.isfile(
                os.path.join(self.proc_dir, d, "time_series.csv")
            )
        )

        # Optional explicit record list.
        if (
            isinstance(args, argparse.Namespace)
            and getattr(args, "rec_ids", None) is not None
        ):
            wanted = {str(x) for x in args.rec_ids}
            rec_ids = [r for r in rec_ids if r in wanted]

        # Make the existing -n flag actually useful for smoke tests.
        if isinstance(args, argparse.Namespace):
            n = int(getattr(args, "n", int(1e8)))
            if n < len(rec_ids):
                rec_ids = rec_ids[:n]
                print(
                    f"[ExpandedMIMIC] -n active: using "
                    f"{len(rec_ids):,} records"
                )

        if not rec_ids:
            raise RuntimeError(
                f"No record folders with time_series.csv under {self.proc_dir}"
            )

        self.feature_cols: list[str] | None = None
        self.chunks = []
        self.record_ids = []
        self.max_input_len = 0
        self.max_pred_len = 0
        self.total_numeric_observations = 0
        self.total_notes = 0
        self.skipped_numeric = 0
        self.skipped_text = 0

        for idx, rec in enumerate(rec_ids, start=1):
            item = self._load_record(rec)
            if item is None:
                continue

            chunk_id, tt, vals, mask, texts = item
            self.chunks.append(item)
            self.record_ids.append(rec)

            hist_len = int((tt < self.history).sum().item())
            pred_len = int((tt >= self.history).sum().item())
            self.max_input_len = max(self.max_input_len, hist_len)
            self.max_pred_len = max(self.max_pred_len, pred_len)
            self.total_numeric_observations += int(mask.sum().item())
            self.total_notes += len(texts)

            if idx % 1000 == 0 or idx == len(rec_ids):
                print(
                    f"[ExpandedMIMIC] loaded {idx:,}/{len(rec_ids):,} "
                    f"(kept={len(self.chunks):,}, "
                    f"skip_numeric={self.skipped_numeric:,}, "
                    f"skip_text={self.skipped_text:,})"
                )

        if not self.chunks:
            raise RuntimeError(
                "No MIMIC episodes were created. Check paths, timestamps, "
                "history/pred_window and text-embedding availability."
            )

    def _load_record(self, rec: str):
        rec_dir = os.path.join(self.proc_dir, rec)
        ts_path = os.path.join(rec_dir, "time_series.csv")

        df = pd.read_csv(ts_path)
        if "date_time" not in df.columns:
            raise ValueError(f"{ts_path}: missing date_time")

        df["_ts_raw"] = pd.to_datetime(df["date_time"], errors="coerce")
        df = df.dropna(subset=["_ts_raw"]).sort_values("_ts_raw").copy()
        if df.empty:
            self.skipped_numeric += 1
            return None

        feat_cols = [
            c
            for c in df.columns
            if c not in ("date_time", "record_id", "_ts_raw")
        ]

        if self.feature_cols is None:
            self.feature_cols = feat_cols
            print(
                f"[ExpandedMIMIC] inferred {len(feat_cols)} numeric features"
            )
        elif feat_cols != self.feature_cols:
            raise ValueError(
                f"{rec}: feature schema/order differs from the first record.\n"
                f"Expected: {self.feature_cols}\n"
                f"Got:      {feat_cols}"
            )

        anchor = _episode_anchor(df["_ts_raw"])
        rel = (
            (df["_ts_raw"] - anchor).dt.total_seconds()
            / self.sec_per_unit
        )

        # The preprocessing already defines one episode.  We select that
        # episode directly; no sliding-window condition involving t_max.
        keep = (rel >= 0.0) & (rel < self.total_window)
        df = df.loc[keep].copy()
        rel = rel.loc[keep]

        if df.empty:
            self.skipped_numeric += 1
            return None

        # Preserve the original IMM-TFS normalization semantics: z-score each
        # feature independently within the record before masking/padding.
        # This intentionally does NOT redesign normalization in this patch.
        if self.normalize:
            for c in self.feature_cols:
                col = pd.to_numeric(df[c], errors="coerce")
                mean = col.mean()
                std = col.std()
                if pd.isna(std) or std == 0:
                    df[c] = col - mean
                else:
                    df[c] = (col - mean) / std

        vals_np = df[self.feature_cols].to_numpy(dtype=np.float32)
        mask_np = ~np.isnan(vals_np)

        # IMPORTANT FOR SCALE:
        # keep the full dataset on CPU. Only the current mini-batch is moved
        # to GPU by the collate wrapper below.
        tt = torch.tensor(
            rel.to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        vals = torch.nan_to_num(
            torch.tensor(vals_np, dtype=torch.float32)
        )
        mask = torch.tensor(mask_np, dtype=torch.bool)

        hist = tt < self.history
        pred = (tt >= self.history) & (tt < self.total_window)

        if (
            hist.sum() == 0
            or pred.sum() == 0
            or mask[hist].sum() == 0
            or mask[pred].sum() == 0
        ):
            self.skipped_numeric += 1
            return None

        texts = []
        if self.enable_text:
            texts = self._load_text_for_record(rec_dir, anchor)
            if len(texts) == 0:
                # A multimodal run should not silently include text-empty
                # samples. Numeric-only runs do NOT pay this cost.
                self.skipped_text += 1
                return None

        return (f"{rec}_chunk0", tt, vals, mask, texts)

    def _load_text_for_record(
        self, rec_dir: str, anchor: pd.Timestamp
    ) -> list[tuple[float, object]]:
        text_path = os.path.join(rec_dir, "text.csv")
        if not os.path.isfile(text_path):
            return []

        tdf = pd.read_csv(text_path)
        if "date_time" not in tdf.columns:
            raise ValueError(f"{text_path}: missing date_time")

        text_cols = [
            c
            for c in tdf.columns
            if c not in ("date_time", "record_id")
        ]
        if len(text_cols) != 1:
            raise ValueError(
                f"{text_path}: expected exactly 1 text column, got {text_cols}"
            )
        text_col = text_cols[0]

        tdf["_ts_raw"] = pd.to_datetime(
            tdf["date_time"], errors="coerce"
        )
        valid = tdf["_ts_raw"].notna() & tdf[text_col].notna()
        tdf = tdf.loc[valid].copy()

        rel_times = (
            (tdf["_ts_raw"] - anchor).dt.total_seconds()
            / self.sec_per_unit
        ).to_numpy(dtype=np.float32)

        if self.use_text_embeddings:
            fname = (
                f"text_embeddings_model={self.llm_model_fusion}"
                f"_layers={self.llm_layers_fusion or 'full'}"
                f"_maxlen={self.max_length}.pt"
            )
            emb_path = os.path.join(rec_dir, fname)
            if not os.path.isfile(emb_path):
                raise FileNotFoundError(
                    f"Missing text embeddings file: {emb_path}"
                )

            data = torch.load(emb_path, map_location="cpu")
            emb = data["embeddings"].float().cpu()

            # We intentionally recompute note times from text.csv using the
            # fixed episode anchor.  The old embedding generator's rel_times
            # may use the first numeric observation as t=0, which is wrong for
            # this fixed 0-48h cohort.
            if emb.shape[0] != len(tdf):
                raise ValueError(
                    f"{emb_path}: embeddings rows ({emb.shape[0]}) != "
                    f"text rows ({len(tdf)}). Recompute embeddings."
                )

            pairs = [
                (float(t), emb[i])
                for i, t in enumerate(rel_times)
                if 0.0 <= float(t) < self.history
            ]
        else:
            notes = tdf[text_col].astype(str).tolist()
            pairs = [
                (float(t), notes[i])
                for i, t in enumerate(rel_times)
                if 0.0 <= float(t) < self.history
            ]

        pairs.sort(key=lambda x: x[0])
        return pairs

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]

    def print_summary(self):
        print("=" * 72)
        print("Expanded MIMIC dataset summary")
        print("=" * 72)
        print(f"episodes kept             : {len(self.chunks):,}")
        print(f"numeric features          : {len(self.feature_cols or []):,}")
        print(f"numeric observations      : {self.total_numeric_observations:,}")
        print(f"text notes loaded         : {self.total_notes:,}")
        print(f"max history timestamps    : {self.max_input_len:,}")
        print(f"max prediction timestamps : {self.max_pred_len:,}")
        print(f"skipped numeric-invalid   : {self.skipped_numeric:,}")
        print(f"skipped text-empty        : {self.skipped_text:,}")
        print(
            "episode definition        : "
            f"[0,{self.history:g}) -> [{self.history:g},{self.total_window:g})"
        )
        print("device residency          : CPU dataset / GPU mini-batches")
        print("=" * 72)


def _load_record_to_subject(
    dataset_path: str, record_ids: list[str]
) -> dict[str, str]:
    """
    Prefer subject-level splitting using sample_statistics.csv.  Falls back to
    record-level IDs only if metadata is unavailable.
    """
    stats_path = os.path.join(dataset_path, "sample_statistics.csv")
    if not os.path.isfile(stats_path):
        print(
            "[ExpandedMIMIC][WARN] sample_statistics.csv not found under "
            f"{dataset_path}. Falling back to record/admission-level split. "
            "Copy the generated sample_statistics.csv here to guarantee "
            "subject-level isolation."
        )
        return {r: r for r in record_ids}

    stats = pd.read_csv(
        stats_path, usecols=["subject_id", "record_id"]
    ).dropna()
    stats["record_id"] = stats["record_id"].astype(str)
    stats["subject_id"] = stats["subject_id"].astype(str)

    mapping = dict(
        zip(stats["record_id"].tolist(), stats["subject_id"].tolist())
    )
    missing = [r for r in record_ids if r not in mapping]
    if missing:
        raise ValueError(
            f"{stats_path}: {len(missing)} processed record IDs are missing "
            f"from the subject mapping. Examples: {missing[:10]}"
        )
    return {r: mapping[r] for r in record_ids}


def _subject_level_split(
    chunks,
    dataset_path: str,
    random_state: int = 42,
):
    rec_ids = [
        cid.rsplit("_chunk", 1)[0]
        for cid, *_ in chunks
    ]
    rec_to_subject = _load_record_to_subject(dataset_path, rec_ids)

    subjects = sorted(set(rec_to_subject.values()))
    if len(subjects) < 5:
        raise ValueError(
            f"Need more subjects for a 60/20/20 split; got {len(subjects)}"
        )

    trainval_subj, test_subj = train_test_split(
        subjects,
        train_size=0.8,
        random_state=random_state,
        shuffle=True,
    )
    train_subj, val_subj = train_test_split(
        trainval_subj,
        train_size=0.75,
        random_state=random_state,
        shuffle=True,
    )

    train_subj = set(train_subj)
    val_subj = set(val_subj)
    test_subj = set(test_subj)

    train_idx, val_idx, test_idx = [], [], []
    for i, (cid, *_) in enumerate(chunks):
        rec = cid.rsplit("_chunk", 1)[0]
        subj = rec_to_subject[rec]
        if subj in train_subj:
            train_idx.append(i)
        elif subj in val_subj:
            val_idx.append(i)
        elif subj in test_subj:
            test_idx.append(i)
        else:
            raise RuntimeError(f"Unassigned subject: {subj}")

    # Scientific safety check.
    assert train_subj.isdisjoint(val_subj)
    assert train_subj.isdisjoint(test_subj)
    assert val_subj.isdisjoint(test_subj)

    print(
        "[ExpandedMIMIC] subject-level split 60/20/20: "
        f"subjects train={len(train_subj):,}, "
        f"val={len(val_subj):,}, test={len(test_subj):,}; "
        f"episodes train={len(train_idx):,}, "
        f"val={len(val_idx):,}, test={len(test_idx):,}"
    )

    return train_idx, val_idx, test_idx


def _choose_base_collate(args):
    if args.model == "tPatchGNN":
        args.patch_size = args.patch_size or args.history // 5
        args.npatch = args.npatch or 5
        args.patch_stride = args.patch_stride or args.patch_size
        print(
            "[ExpandedMIMIC] using Patch collate: "
            f"patch_size={args.patch_size}, npatch={args.npatch}, "
            f"patch_stride={args.patch_stride}"
        )
        return patch_variable_time_collate_fn

    if args.model == "CRU":
        print("[ExpandedMIMIC] using CRU collate")
        return variable_time_collate_fn_CRU

    if args.model == "LatentODE":
        print("[ExpandedMIMIC] using LatentODE collate")
        return variable_time_collate_fn_ODE

    print("[ExpandedMIMIC] using standard collate")
    return variable_time_collate_fn


def _make_gpu_batch_collate(base_collate, args, time_max):
    """
    Dataset tensors live on CPU; only this mini-batch is copied to GPU.
    """

    def collate(batch):
        device = args.device

        numeric_batch = []
        for cid, tt, vals, mask, _texts in batch:
            numeric_batch.append(
                (
                    cid,
                    tt.to(device, non_blocking=True),
                    vals.to(device, non_blocking=True),
                    mask.to(
                        device, dtype=torch.float32, non_blocking=True
                    ),
                )
            )

        out = base_collate(numeric_batch, args, time_max)

        raws = [item[4] for item in batch]

        # tau remains in the raw dataset unit (hours for MIMIC), matching the
        # GPINet native fusion path.  GPINet normalizes tau internally.
        time_seqs = [
            torch.tensor(
                [t for (t, _) in seq],
                dtype=torch.float32,
                device=device,
            )
            for seq in raws
        ]
        if time_seqs:
            tau = pad_sequence(
                time_seqs, batch_first=True, padding_value=0.0
            )
        else:
            tau = torch.empty((0, 0), device=device)
        out["tau"] = tau

        if args.enable_text and not args.use_text_embeddings:
            out["notes_text"] = [
                [txt for (_, txt) in seq] for seq in raws
            ]

        if args.enable_text and args.use_text_embeddings:
            d_txt = None
            for seq in raws:
                if seq:
                    d_txt = int(seq[0][1].numel())
                    break

            if d_txt is None:
                out["notes_embeddings"] = torch.zeros(
                    (len(batch), 0, 0), device=device
                )
            else:
                emb_seqs = []
                for seq in raws:
                    if seq:
                        emb_seqs.append(
                            torch.stack(
                                [e for (_, e) in seq], dim=0
                            ).to(device, non_blocking=True)
                        )
                    else:
                        emb_seqs.append(
                            torch.zeros((0, d_txt), device=device)
                        )
                out["notes_embeddings"] = pad_sequence(
                    emb_seqs,
                    batch_first=True,
                    padding_value=0.0,
                )

        return out

    return collate


def parse_datasets(args, show_summary=True):
    """
    Drop-in replacement imported by main.py.

    Expanded MIMIC -> fixed-episode adapter.
    Every other dataset -> original IMM-TFS parser unchanged.
    """
    if not _is_expanded_mimic(args.dataset):
        return _original_parse_datasets(args, show_summary=show_summary)

    dataset_path = _resolve_dataset_path(args)
    print(f"Using expanded MIMIC dataset path: {dataset_path}")

    # For this dataset, one record == one episode.  A within-record "sample"
    # split would produce empty train/val sets.  Always split by subject.
    if getattr(args, "split_method", None) != "instance":
        print(
            f"[ExpandedMIMIC] overriding split_method="
            f"{getattr(args, 'split_method', None)!r} -> 'instance' "
            "because each record contains exactly one fixed episode."
        )
        args.split_method = "instance"

    ds = ExpandedMIMICDataset(
        root=dataset_path,
        history=args.history,
        pred_window=args.pred_window,
        device=args.device,
        time_unit=args.time_unit,
        unit_scale=getattr(args, "unit_scale", None),
        normalize=True,
        enable_text=args.enable_text,
        use_text_embeddings=args.use_text_embeddings,
        llm_model_fusion=args.llm_model_fusion,
        llm_layers_fusion=args.llm_layers_fusion,
        max_length=args.max_length,
        args=args,
    )

    if show_summary:
        ds.print_summary()

    all_chunks = ds.chunks
    _, _, first_vals, _, _ = all_chunks[0]
    input_dim = first_vals.size(-1)

    train_idx, val_idx, test_idx = _subject_level_split(
        all_chunks,
        dataset_path=dataset_path,
        random_state=42,
    )

    base_collate = _choose_base_collate(args)
    time_max = torch.tensor(
        args.history + args.pred_window,
        dtype=torch.float32,
        device=args.device,
    )
    collate_fn = _make_gpu_batch_collate(
        base_collate, args, time_max
    )

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    test_ds = Subset(ds, test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    return {
        "train_dataloader": train_loader,
        "val_dataloader": val_loader,
        "test_dataloader": test_loader,
        "input_dim": input_dim,
        "time_max": time_max,
        "ds": ds,
    }


def get_input_and_pred_len(data_obj):
    """
    Avoid the original full extra dataloader scan for expanded MIMIC.
    """
    ds = data_obj.get("ds")
    if isinstance(ds, ExpandedMIMICDataset):
        return ds.max_input_len, ds.max_pred_len
    return _original_get_input_and_pred_len(data_obj)
