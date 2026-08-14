# Expanded MIMIC fixed/nested protocol v3 — per-N normalization

## Formal experimental protocol

- Full strict cohort: 20,243 subjects.
- Persist ONE subject-level 60/20/20 split:
  - full train pool: 12,145
  - fixed validation: 4,049
  - fixed test: 4,049
- Persist ONE randomized order of the 12,145 train subjects.
- Training-size subsets are nested prefixes:
  `T200 ⊂ T500 ⊂ T1000 ⊂ T2000 ⊂ T5000 ⊂ T10000 ⊂ T12145`.
- Validation and test sets never change with N.
- **Normalization is fitted separately for each Train_N**, using only observed numeric values from Train_N history `[0,24h)`.
- Uni and Multi at the same N therefore share the same train IDs, validation IDs, test IDs and scaler.
- `-n` means **number of training subjects**, not total cohort size.
- Existing `scripts/run_gpinet.sh --smoke` remains a debugging path and is intentionally untouched.

## Install

From the IMM-TSF repository root:

```bash
python /path/to/imm_tfs_mimic_fixed_protocol_v3_pern/apply_mimic_fixed_protocol_v3.py
```

Then create the fixed subject split/order once:

```bash
python scripts/prepare_mimic_fixed_protocol.py
```

If an older v3 protocol already exists and you intentionally want to replace it:

```bash
python scripts/prepare_mimic_fixed_protocol.py --force
```

The preparer now creates only:

```text
data/MIMIC/mimic_fixed_protocol.json
```

It does **not** create a global normalization file.

## Run

Numeric-only:

```bash
./scripts/run_gpinet_fixed.sh -n 200
./scripts/run_gpinet_fixed.sh -n 500
./scripts/run_gpinet_fixed.sh -n 1000
./scripts/run_gpinet_fixed.sh -n 2000
```

Multimodal:

```bash
./scripts/run_gpinet_fixed.sh -n 200 --text
./scripts/run_gpinet_fixed.sh -n 500 --text
./scripts/run_gpinet_fixed.sh -n 1000 --text
./scripts/run_gpinet_fixed.sh -n 2000 --text
```

At the same N, Uni/Multi should print identical per-feature normalization statistics.

## Important

If you previously created:

```text
data/MIMIC/mimic_fixed_normalization.pt
```

it is no longer used by this version and may be moved/removed after confirming the new protocol works.
