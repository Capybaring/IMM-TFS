#!/usr/bin/env bash
set -euo pipefail

# Formal sample-size runner for expanded MIMIC.
#
# IMPORTANT: in this script -n means NUMBER OF TRAIN SUBJECTS.
# Validation/test subjects are fixed globally and never shrink/change with N.
# Train subsets are nested prefixes of one persisted random train order.

TRAIN_N=1000
ENABLE_TEXT=0
GPU=0
EPOCHS=20
BATCH_SIZE=32
PATIENCE=5
MODEL_SEED=1
LOADER_SEED=314159

DATASET="MIMIC"
LLM_MODEL="BERT"
LLM_LAYERS=6
MAX_LENGTH=512

HISTORY=24
PRED_WINDOW=24
STRIDE=24
TIME_UNIT="hours"

N_LAYER=3
HOP=2
TE_DIM=10
NODE_DIM=10
HID_DIM=64
DROPOUT=0.3

TTF_MODULE="TTF_T2V_XAttn"
MMF_MODULE="MMF_XAttn_Add"

usage() {
cat <<'EOF'
Usage:
  ./scripts/run_gpinet_fixed.sh -n TRAIN_N [options]

Options:
  -n, --num N           Number of TRAIN subjects (default: 1000)
  --text                Enable BERT/radiology multimodal run
  --epochs N            Epochs (default: 20)
  --batch-size N        Batch size (default: 32)
  --patience N          Early stopping patience (default: 5)
  --gpu ID              GPU id for main.py (default: 0)
  --seed N              Model/training seed (default: 1)
  --loader-seed N       Fixed DataLoader shuffle seed (default: 314159)
  -h, --help            Show this help

Examples:
  ./scripts/run_gpinet_fixed.sh -n 200
  ./scripts/run_gpinet_fixed.sh -n 200 --text
  ./scripts/run_gpinet_fixed.sh -n 1000 --epochs 50 --patience 10
  ./scripts/run_gpinet_fixed.sh -n 1000 --text --epochs 50 --patience 10

Protocol file (created once):
  data/MIMIC/mimic_fixed_protocol.json

If missing, this runner creates it automatically using:
  split_seed       = 42
  train_order_seed = 2026

Normalization is NOT persisted globally. For every N, the loader fits:
  scaler_N = Train_N HISTORY [0,24h) only
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num) TRAIN_N="$2"; shift 2 ;;
        --text) ENABLE_TEXT=1; shift ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --patience) PATIENCE="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --seed) MODEL_SEED="$2"; shift 2 ;;
        --loader-seed) LOADER_SEED="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

for pair in "TRAIN_N:$TRAIN_N" "EPOCHS:$EPOCHS" "BATCH_SIZE:$BATCH_SIZE" "PATIENCE:$PATIENCE"; do
    name="${pair%%:*}"; value="${pair#*:}"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -le 0 ]]; then
        echo "Error: $name must be a positive integer, got $value" >&2
        exit 1
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PROTOCOL="data/MIMIC/mimic_fixed_protocol.json"
if [[ ! -f "$PROTOCOL" ]]; then
    echo "### Fixed protocol not found; preparing it once from the full cohort ###"
    python scripts/prepare_mimic_fixed_protocol.py \
        --dataset-dir data/MIMIC \
        --history "$HISTORY" \
        --pred-window "$PRED_WINDOW" \
        --time-unit "$TIME_UNIT" \
        --split-seed 42 \
        --train-order-seed 2026
else
    PROTOCOL_VERSION="$(python - "$PROTOCOL" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(json.load(f).get("version", ""))
PY
)"
    if [[ "$PROTOCOL_VERSION" != "3-per-n-normalization" ]]; then
        echo "### Existing fixed protocol is from the older full-train-normalization v3." >&2
        echo "### Rebuild once with:" >&2
        echo "python scripts/prepare_mimic_fixed_protocol.py --force" >&2
        exit 2
    fi
fi

# Resolve actual train N and display the fixed split.
readarray -t META < <(python - "$PROTOCOL" "$TRAIN_N" <<'PY'
import json, sys
p, req = sys.argv[1], int(sys.argv[2])
with open(p, encoding='utf-8') as f:
    x = json.load(f)
full = len(x['train_subject_order'])
n = min(req, full)
print(n)
print(full)
print(x['val_subject_count'])
print(x['test_subject_count'])
print(x['subject_count'])
PY
)
ACTUAL_N="${META[0]}"
FULL_TRAIN_N="${META[1]}"
VAL_N="${META[2]}"
TEST_N="${META[3]}"
TOTAL_N="${META[4]}"
if [[ "$ACTUAL_N" != "$TRAIN_N" ]]; then
    echo "### Requested train N=$TRAIN_N exceeds full train pool; using $ACTUAL_N ###"
    TRAIN_N="$ACTUAL_N"
fi

MODE="numeric-only"
[[ "$ENABLE_TEXT" -eq 1 ]] && MODE="multimodal"

echo "=================================================================="
echo "GPINet fixed/nested expanded-MIMIC experiment"
echo "=================================================================="
echo "full cohort subjects : $TOTAL_N"
echo "full train pool      : $FULL_TRAIN_N"
echo "TRAIN subjects used  : $TRAIN_N"
echo "VAL subjects fixed   : $VAL_N"
echo "TEST subjects fixed  : $TEST_N"
echo "mode                  : $MODE"
echo "model seed            : $MODEL_SEED"
echo "loader seed           : $LOADER_SEED"
echo "epochs / patience     : $EPOCHS / $PATIENCE"
echo "batch size            : $BATCH_SIZE"
echo "normalization         : current Train_N HISTORY only"
echo "=================================================================="

if [[ "$ENABLE_TEXT" -eq 1 ]]; then
    echo
    echo "### [1/2] Ensuring embeddings for train_N + fixed val + fixed test ###"
    python - "$PROTOCOL" "$TRAIN_N" "$DATASET" "$LLM_MODEL" "$LLM_LAYERS" "$MAX_LENGTH" "$TIME_UNIT" <<'PY'
import json
import sys
import torch
from compute_text_embeddings import compute_text_embeddings

protocol_path = sys.argv[1]
train_n = int(sys.argv[2])
dataset = sys.argv[3]
llm_model = sys.argv[4]
llm_layers = int(sys.argv[5])
max_length = int(sys.argv[6])
time_unit = sys.argv[7]

with open(protocol_path, encoding="utf-8") as f:
    p = json.load(f)

subject_to_records = {str(k): [str(r) for r in v] for k, v in p["subject_to_records"].items()}
selected_subjects = [str(s) for s in p["train_subject_order"][:train_n]]
record_ids = []
for s in selected_subjects:
    record_ids.extend(subject_to_records[s])
record_ids.extend(str(r) for r in p["val_records"])
record_ids.extend(str(r) for r in p["test_records"])
record_ids = sorted(set(record_ids))

print(
    f"Fixed-protocol embedding requirement: {len(record_ids):,} records "
    f"= train({train_n:,} subjects) + fixed val/test"
)

compute_text_embeddings(
    dataset,
    llm_model,
    llm_layers,
    max_length,
    "cuda" if torch.cuda.is_available() else "cpu",
    time_unit=time_unit,
    episode_anchor_from_day_start=True,
    record_ids=record_ids,
)
PY
    echo
    echo "### [2/2] Training GPINet multimodal ###"
else
    echo
    echo "### Training GPINet numeric-only ###"
fi

export MIMIC_FIXED_PROTOCOL=1
export MIMIC_PROTOCOL_PATH="$REPO_ROOT/$PROTOCOL"
export MIMIC_LOADER_SEED="$LOADER_SEED"

CMD=(
    python main.py
    --gpu "$GPU"
    --seed "$MODEL_SEED"
    --dataset "$DATASET"
    --model GPINet
    --history "$HISTORY"
    --pred_window "$PRED_WINDOW"
    --stride "$STRIDE"
    --time_unit "$TIME_UNIT"
    --epoch "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --patience "$PATIENCE"
    --nlayer "$N_LAYER"
    --hop "$HOP"
    --te_dim "$TE_DIM"
    --node_dim "$NODE_DIM"
    --hid_dim "$HID_DIM"
    --dropout "$DROPOUT"
    -n "$TRAIN_N"
)

if [[ "$ENABLE_TEXT" -eq 1 ]]; then
    CMD+=(
        --enable_text
        --use_text_embeddings
        --llm_model_fusion "$LLM_MODEL"
        --llm_layers_fusion "$LLM_LAYERS"
        --max_length "$MAX_LENGTH"
        --TTF_module "$TTF_MODULE"
        --MMF_module "$MMF_MODULE"
    )
fi

"${CMD[@]}"

echo
echo "### Done: fixed protocol / train_N=$TRAIN_N / text=$ENABLE_TEXT / model_seed=$MODEL_SEED ###"
