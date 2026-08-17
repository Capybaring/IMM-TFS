# MIMIC-IV chest-radiology cohort

This preprocessing path replaces the TIME-IMM feature/report pairing with a
clinically aligned task:

> irregular numeric history from ICU hour 0-24 + available chest radiology
> reports -> numeric respiratory/cardiopulmonary forecasting for hour 24-48.

The implementation is:

```text
scripts/prepare_mimic_chest_radiology.py
```

## Required source files

Default paths follow the existing preprocessing notebook:

```text
data/mimiciv/3.1/
├── hosp/admissions.csv.gz
├── hosp/patients.csv.gz
├── hosp/labevents.csv.gz
├── hosp/d_labitems.csv.gz
├── icu/icustays.csv.gz
├── icu/chartevents.csv.gz
└── icu/d_items.csv.gz

data/mimic-iv-note/2.2/note/
├── radiology.csv.gz
└── radiology_detail.csv.gz
```

## Numeric variables

The output has 24 continuous numeric channels.

### Forecast targets (16)

```text
spo2
respiratory_rate
heart_rate
sbp
dbp
map
temperature_c
ph
pao2
paco2
bicarbonate_bg
base_excess
lactate
minute_ventilation
tidal_volume_observed
wbc
```

### History-only treatment/device covariates (8)

```text
fio2
oxygen_flow
peep
tidal_volume_set
pressure_support
peak_inspiratory_pressure
plateau_pressure
mean_airway_pressure
```

The eight history covariates are retained before hour 24 but deliberately have
no target observations after hour 24. The existing masked loss therefore
evaluates only the 16 physiologic targets while MTGNN can still learn from the
respiratory treatment/device history.

## Text selection

The script joins `radiology` and `radiology_detail` by `note_id`, then uses
`exam_name`/`exam_code` to retain:

```text
Chest X-ray
CT chest
Chest ultrasound
```

RR and AR are stored as separate timestamped text events. If an AR row lacks
exam metadata, it inherits the metadata of its parent RR when the parent link is
available. Each model input receives a prefix such as:

```text
[EXAM_TYPE=XRAY_CHEST] [EXAM_NAME=CHEST (PORTABLE AP)] [NOTE_TYPE=RR]
<original report text>
```

`text.csv` still has exactly one text column, so it is compatible with the
current embedding script and expanded-MIMIC loader.

## Leakage-safe timing

For a 0-24 h history report, both conditions must hold:

```text
0 <= charttime - ICU_intime < 24 h
storetime < ICU_intime + 24 h
```

`charttime` locates the report token in the irregular event sequence;
`storetime` determines whether the report was available at prediction time.
Reports with missing `storetime` are excluded by default. A sensitivity run can
use `--allow-null-note-storetime` to fall back to `charttime`.

Numeric history uses the same event-time/availability-time distinction. A
numeric row is located by `charttime`; before hour 24 it must have been stored
before the cutoff. Target observations are selected retrospectively by their
24-48 h `charttime`.

## First run in a separate output directory

```bash
python -u scripts/prepare_mimic_chest_radiology.py \
  --mimic-root data/mimiciv/3.1 \
  --note-root data/mimic-iv-note/2.2/note \
  --output-root data/MIMIC_chest
```

Useful smoke run:

```bash
python -u scripts/prepare_mimic_chest_radiology.py \
  --mimic-root data/mimiciv/3.1 \
  --note-root data/mimic-iv-note/2.2/note \
  --output-root data/MIMIC_chest_smoke \
  --max-patients 200
```

The default quality requirements are configurable:

```text
history observations >= 20
target observations >= 20 (16 targets only)
history variables >= 5
target variables >= 5 (16 targets only)
direct respiratory history variables >= 2
direct respiratory target variables >= 2
```

Do not tighten them until inspecting `sample_statistics.csv`, since aggressive
future-density filtering changes the clinical cohort.

## Output

```text
data/MIMIC_chest/
├── processed_full/              # numeric-valid, text may be absent
├── processed/                   # numeric-valid + >=1 chest report
├── sample_statistics.csv
├── cohort_summary.csv
├── feature_dictionary.csv
├── text_event_metadata.csv
└── tfsimm_dataset_config.json
```

For a Uni/Multi ablation on identical patients, use `processed/` for both
modes. Numeric-only training ignores its `text.csv`; multimodal training reads
the same patient episodes and enables text.

## Replace the current `data/MIMIC` build

After inspecting the separate output, rebuild the training location:

```bash
python -u scripts/prepare_mimic_chest_radiology.py \
  --mimic-root data/mimiciv/3.1 \
  --note-root data/mimic-iv-note/2.2/note \
  --output-root data/MIMIC \
  --overwrite
```

The feature schema changes from the TIME-IMM variables to 24 chest-task
variables, so the old fixed protocol and old text embeddings must not be
reused. `--overwrite` removes the generated `processed*` folders and stale
`mimic_fixed_protocol.json` only inside the selected output root. Rebuild it:

```bash
python -u scripts/prepare_mimic_fixed_protocol.py \
  --dataset-dir data/MIMIC \
  --history 24 \
  --pred-window 24 \
  --time-unit hours \
  --split-seed 42 \
  --train-order-seed 2026 \
  --force
```

Then the existing experiment command remains:

```bash
./scripts/run_gpinet_fixed.sh --sweep
```

## Files to inspect before training

1. `feature_dictionary.csv`: resolved feature order, item IDs, units and roles.
2. `sample_statistics.csv`: numeric/text coverage and filtering outcome per stay.
3. `text_event_metadata.csv`: exam type, RR/AR, charttime, storetime and parent link.
4. `tfsimm_dataset_config.json`: full reproducibility configuration and source counts.

