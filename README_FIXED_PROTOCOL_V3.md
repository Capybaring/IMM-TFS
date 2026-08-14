# Expanded MIMIC fixed/nested scaling protocol (v3)

This package is for the **formal patient-scaling experiment**. It does not replace the existing smoke-test workflow.

## Formal protocol

1. Use all processed strict-multimodal MIMIC records to make **one persistent subject-level 60/20/20 split**.
2. Keep validation and test subjects fixed for every experiment.
3. Shuffle full-train subjects once with a separate seed and persist the order.
4. `N=200,500,1000,...` are nested prefixes:
   `T200 ⊂ T500 ⊂ T1000 ⊂ ...`.
5. In fixed mode, `-n` means **number of TRAIN subjects**, not total cohort size.
6. Fit one feature-wise z-score from **FULL fixed TRAIN, history [0,24h) only** and reuse it for every N, Uni and Multi.
7. Use a separate fixed DataLoader seed so Uni/Multi see the same minibatch order for the same N.
8. Model seed remains a normal experiment parameter and can later be varied for mean ± std.

With 20,243 unique subjects, the expected split is approximately:

- train: 12,145
- val: 4,049
- test: 4,049

The exact persisted IDs are the source of truth after preparation.

## Install

From the local IMM-TSF repository root:

```bash
python /path/to/imm_tfs_mimic_fixed_protocol_v3/apply_mimic_fixed_protocol_v3.py
```

The installer does **not** touch `data/MIMIC` and does **not** replace `scripts/run_gpinet.sh`.

## Prepare once

```bash
python scripts/prepare_mimic_fixed_protocol.py
```

This creates:

```text
data/MIMIC/mimic_fixed_protocol.json
data/MIMIC/mimic_fixed_normalization.pt
```

Do not rebuild these between N experiments. They define the fixed comparison population.

To intentionally rebuild them with the same/default seeds:

```bash
python scripts/prepare_mimic_fixed_protocol.py --force
```

## Run formal scaling experiments

Numeric-only:

```bash
./scripts/run_gpinet_fixed.sh -n 200
./scripts/run_gpinet_fixed.sh -n 500
./scripts/run_gpinet_fixed.sh -n 1000
./scripts/run_gpinet_fixed.sh -n 2000
./scripts/run_gpinet_fixed.sh -n 5000
./scripts/run_gpinet_fixed.sh -n 10000
```

Multimodal:

```bash
./scripts/run_gpinet_fixed.sh -n 200 --text
./scripts/run_gpinet_fixed.sh -n 500 --text
./scripts/run_gpinet_fixed.sh -n 1000 --text
./scripts/run_gpinet_fixed.sh -n 2000 --text
```

Full train pool (check the preparation output for the exact count, expected ~12,145):

```bash
./scripts/run_gpinet_fixed.sh -n 12145
./scripts/run_gpinet_fixed.sh -n 12145 --text
```

Other options:

```bash
./scripts/run_gpinet_fixed.sh -n 2000 --epochs 50 --patience 10 --batch-size 32 --seed 1
```

## Smoke tests remain separate

Existing debug/smoke commands remain unchanged:

```bash
./scripts/run_gpinet.sh --smoke
./scripts/run_gpinet.sh --smoke --text
```

Those use the earlier subset-and-resplit behavior and are only for debugging, not the formal scaling figure/table.

## Important interpretation

Under this v3 formal protocol, absolute MSE/MAE across different N are evaluated on the same fixed test subjects and the same normalization scale. Therefore the N-scaling curve is directly interpretable.

For each N, Uni and Multi also use the exact same train subjects, validation subjects, test subjects, normalization and minibatch ordering. The textual modality is the intended major difference.
