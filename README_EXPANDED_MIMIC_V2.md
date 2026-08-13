# Expanded MIMIC v2 patch for IMM-TFS

This patch updates the expanded-MIMIC adapter after the first successful
numeric smoke test.

## What changed

### 1. Leakage-free normalization

Old behavior:
- load one admission's full 0-48 h episode;
- compute that admission's mean/std from the full 0-48 h;
- then split history/target.

That lets target values influence preprocessing.

v2 behavior:
1. load raw 0-48 h episodes;
2. subject-level 60/20/20 split;
3. fit one global mean/std per variable from **TRAIN subjects, 0-24 h only**;
4. apply those same statistics to train/val/test, history and target.

Missing values remain zero-filled in tensors and are still controlled by masks.

### 2. Text smoke really uses 200 records

`./scripts/run_gpinet.sh --smoke --text` now limits embedding generation to the
same first 200 sorted record folders used by the smoke dataset loader.

### 3. Correct expanded-MIMIC episode anchor for text

MIMIC note relative time uses the synthetic episode-day start, matching the
numeric adapter, rather than the first actual numeric observation.

### 4. Two text time tensors

For expanded MIMIC batches:

- `tau_raw`: raw hours since episode start.
- `tau`: `tau_raw / 48`, matching normalized `tp_to_predict`.

Generic TIME-IMM FusionModel continues to receive `tau`.
GPINet native text fusion receives `tau_raw` via a tiny evaluation.py patch,
so its internal normalization is not applied twice.

### 5. Other datasets remain unchanged

Dataset names not starting with `MIMIC` are delegated to the original
`lib.parse_datasets.py`.

## Install

Copy/unzip this bundle at any location, then from the IMM-TFS repository root:

```bash
python /path/to/imm_tfs_expanded_mimic_v2/apply_expanded_mimic_v2.py
```

The installer creates one-time backups such as:

```text
lib/parse_datasets_mimic_expanded.py.before_expanded_mimic_v2
compute_text_embeddings.py.before_expanded_mimic_v2
scripts/_common.sh.before_expanded_mimic_v2
lib/evaluation.py.before_expanded_mimic_v2
main.py.before_expanded_mimic_v2
```

It does **not** move, delete, create, or symlink dataset directories.

## Test

Numeric:

```bash
./scripts/run_gpinet.sh --smoke
```

Look for:

```text
[ExpandedMIMIC] normalization fitted leakage-free from TRAIN HISTORY only
normalization              : TRAIN HISTORY global z-score
```

Then multimodal:

```bash
./scripts/run_gpinet.sh --smoke --text
```

Look for:

```text
Text embedding records selected: 200 (max_records=200)
```

## MAPE

This patch intentionally does not redefine the repository's MAPE metric.
The current MAPE is computed on normalized values and should not be used as a
clinically meaningful percentage metric. MSE/MAE/RMSE remain the recommended
comparison metrics for the current benchmark.
