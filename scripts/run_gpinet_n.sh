#!/usr/bin/env bash
set -euo pipefail

# GPINet runner for expanded MIMIC with a configurable number of records.
#
# Examples:
#   ./scripts/run_gpinet_n.sh -n 2000
#   ./scripts/run_gpinet_n.sh -n 2000 --text
#   ./scripts/run_gpinet_n.sh -n 5000 --epochs 50 --patience 10
#   ./scripts/run_gpinet_n.sh -n 20243 --text --epochs 200
#
# Notes:
# - Numeric and multimodal runs with the same -n and --seed use the same
#   expanded-MIMIC prefix, subject split and train-history normalization.
# - With --text, embedding generation is limited to the same first N sorted
#   record folders. Existing embeddings are skipped automatically.
# - This script assumes the v2 expanded-MIMIC files are already installed.

NUM_RECORDS=2000
ENABLE_TEXT=0

GPU=0
EPOCHS=20
BATCH_SIZE=32
PATIENCE=5
SEED=1

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
  ./scripts/run_gpinet_n.sh [options]

Main options:
  -n, --num N           Number of MIMIC records to use (default: 2000)
  --text                Enable text/BERT multimodal GPINet
  --epochs N            Training epochs (default: 20)
  --batch-size N        Batch size (default: 32)
  --patience N          Early-stop patience (default: 5)
  --gpu ID              GPU id passed to main.py (default: 0)
  --seed N              Random seed passed to main.py (default: 1)

Examples:
  ./scripts/run_gpinet_n.sh -n 200
  ./scripts/run_gpinet_n.sh -n 2000
  ./scripts/run_gpinet_n.sh -n 2000 --text
  ./scripts/run_gpinet_n.sh -n 5000 --epochs 50
  ./scripts/run_gpinet_n.sh -n 20243 --text --epochs 200 --patience 10

Environment overrides are intentionally not used here so each command is
self-contained and reproducible.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num)
            NUM_RECORDS="$2"
            shift 2
            ;;
        --text)
            ENABLE_TEXT=1
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if ! [[ "$NUM_RECORDS" =~ ^[0-9]+$ ]] || [[ "$NUM_RECORDS" -le 0 ]]; then
    echo "Error: -n/--num must be a positive integer." >&2
    exit 1
fi

if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]] || [[ "$EPOCHS" -le 0 ]]; then
    echo "Error: --epochs must be a positive integer." >&2
    exit 1
fi

if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [[ "$BATCH_SIZE" -le 0 ]]; then
    echo "Error: --batch-size must be a positive integer." >&2
    exit 1
fi

if ! [[ "$PATIENCE" =~ ^[0-9]+$ ]] || [[ "$PATIENCE" -le 0 ]]; then
    echo "Error: --patience must be a positive integer." >&2
    exit 1
fi

# Run from repository root regardless of caller's working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f main.py ]]; then
    echo "Error: main.py not found under $REPO_ROOT" >&2
    exit 1
fi

if [[ ! -d data/MIMIC/processed ]]; then
    echo "Error: data/MIMIC/processed not found." >&2
    exit 1
fi

AVAILABLE_RECORDS="$(
    find data/MIMIC/processed -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' '
)"

if [[ "$NUM_RECORDS" -gt "$AVAILABLE_RECORDS" ]]; then
    echo "### Requested $NUM_RECORDS records, but only $AVAILABLE_RECORDS are available."
    echo "### Using all $AVAILABLE_RECORDS records instead."
    NUM_RECORDS="$AVAILABLE_RECORDS"
fi

echo "=================================================================="
echo "GPINet expanded-MIMIC run"
echo "=================================================================="
echo "records      : $NUM_RECORDS / $AVAILABLE_RECORDS"
echo "mode         : $([[ "$ENABLE_TEXT" -eq 1 ]] && echo multimodal || echo numeric-only)"
echo "epochs       : $EPOCHS"
echo "batch size   : $BATCH_SIZE"
echo "patience     : $PATIENCE"
echo "gpu          : $GPU"
echo "seed         : $SEED"
echo "history      : ${HISTORY}h"
echo "prediction   : ${PRED_WINDOW}h"
echo "=================================================================="

if [[ "$ENABLE_TEXT" -eq 1 ]]; then
    echo
    echo "### [1/2] Ensuring BERT embeddings for first $NUM_RECORDS records ###"

    python - \
        "$DATASET" \
        "$LLM_MODEL" \
        "$LLM_LAYERS" \
        "$MAX_LENGTH" \
        "$TIME_UNIT" \
        "$NUM_RECORDS" <<'PY'
import sys
import torch
from compute_text_embeddings import compute_text_embeddings

dataset = sys.argv[1]
llm_model = sys.argv[2]
llm_layers = int(sys.argv[3])
max_length = int(sys.argv[4])
time_unit = sys.argv[5]
num_records = int(sys.argv[6])

device = "cuda" if torch.cuda.is_available() else "cpu"

compute_text_embeddings(
    dataset,
    llm_model,
    llm_layers,
    max_length,
    device,
    time_unit=time_unit,
    episode_anchor_from_day_start=dataset.upper().startswith("MIMIC"),
    max_records=num_records,
)
PY

    echo
    echo "### [2/2] Training GPINet multimodal on $NUM_RECORDS records ###"

    python main.py \
        --gpu "$GPU" \
        --seed "$SEED" \
        --dataset "$DATASET" \
        --model GPINet \
        --history "$HISTORY" \
        --pred_window "$PRED_WINDOW" \
        --stride "$STRIDE" \
        --time_unit "$TIME_UNIT" \
        --enable_text \
        --use_text_embeddings \
        --llm_model_fusion "$LLM_MODEL" \
        --llm_layers_fusion "$LLM_LAYERS" \
        --max_length "$MAX_LENGTH" \
        --TTF_module "$TTF_MODULE" \
        --MMF_module "$MMF_MODULE" \
        --epoch "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --patience "$PATIENCE" \
        --nlayer "$N_LAYER" \
        --hop "$HOP" \
        --te_dim "$TE_DIM" \
        --node_dim "$NODE_DIM" \
        --hid_dim "$HID_DIM" \
        --dropout "$DROPOUT" \
        -n "$NUM_RECORDS"
else
    echo
    echo "### Training GPINet numeric-only on $NUM_RECORDS records ###"

    python main.py \
        --gpu "$GPU" \
        --seed "$SEED" \
        --dataset "$DATASET" \
        --model GPINet \
        --history "$HISTORY" \
        --pred_window "$PRED_WINDOW" \
        --stride "$STRIDE" \
        --time_unit "$TIME_UNIT" \
        --epoch "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --patience "$PATIENCE" \
        --nlayer "$N_LAYER" \
        --hop "$HOP" \
        --te_dim "$TE_DIM" \
        --node_dim "$NODE_DIM" \
        --hid_dim "$HID_DIM" \
        --dropout "$DROPOUT" \
        -n "$NUM_RECORDS"
fi

echo
echo "### Done: GPINet / records=$NUM_RECORDS / text=$ENABLE_TEXT ###"
