#!/usr/bin/env bash
set -euo pipefail

# Command-line interface: only mode, total-subject count, and model seed.
MODE="uni"
TOTAL_N=1000
MODEL_SEED=1

# Edit the remaining experiment settings directly here.
GPU=0
EPOCHS=50
BATCH_SIZE=16
PATIENCE=10
DATA_SEED=42
LOADER_SEED=314159
USE_AMP=0
OUTPUT_DIR="logs/gpinet_runs"

TTF_MODULE_LIST=("TTF_SemTime_Slots" "TTF_T2V_XAttn")
MMF_MODULE_LIST=("MMF_VarTime_SlotGate" "MMF_GR_Add")
INDEX=1
TTF_MODULE=${TTF_MODULE_LIST[$INDEX]}
MMF_MODULE=${MMF_MODULE_LIST[$INDEX]}
TEXT_HEADS=1
TEXT_DIM=128
SEMANTIC_SLOTS=2
RECENCY_SIGMA=0.25
SEMANTIC_TIME_GATE_BIAS=-1.0
ABSOLUTE_RECENCY_FLOOR=0.1
MMF_SLOT_ATTN_DIM=128
MMF_SLOT_GATE_BIAS=0.0
MMF_DELTA_INIT_STD=0.01
FUSION_GATE_WARMUP_EPOCHS=5
FUSION_GATE_WARMUP_VALUE=0.5
FUSION_LR_MULTIPLIER=2.0
KAPPA=0.1

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_gpinet.sh --mode uni|multi --num N --seed N

Options:
  --mode MODE  Experiment mode: uni or multi (default: uni)
  --num N      Total subjects before the 60/20/20 split (default: 1000)
  --seed N     Model seed (default: 1)
  -h, --help   Show this help

All other experiment settings are constants at the top of this file.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --num) TOTAL_N="$2"; shift 2 ;;
        --seed) MODEL_SEED="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$MODE" in
    uni|multi) ;;
    *) echo "--mode must be uni or multi" >&2; exit 2 ;;
esac
[[ "$TOTAL_N" =~ ^[1-9][0-9]*$ ]] || {
    echo "--num must be a positive integer" >&2
    exit 2
}
[[ "$MODEL_SEED" =~ ^[0-9]+$ ]] || {
    echo "--seed must be a non-negative integer" >&2
    exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ensure_embeddings() {
    python - "$TOTAL_N" "$DATA_SEED" <<'PY'
import sys

import torch

from compute_text_embeddings import compute_text_embeddings
from lib.parse_datasets import select_mimic_subject_records

selection = select_mimic_subject_records(
    "data/MIMIC",
    requested_total_n=int(sys.argv[1]),
    data_seed=int(sys.argv[2]),
)
compute_text_embeddings(
    "MIMIC",
    "BERT",
    6,
    512,
    "cuda" if torch.cuda.is_available() else "cpu",
    time_unit="hours",
    episode_anchor_from_day_start=True,
    record_ids=selection["required_records"],
)
PY
}

filter_final_log() {
    awk '
        BEGIN { final = 0; traceback = 0 }
        final || traceback { print; next }
        /^Exp has been early stopped!$/ || /^loss: / { final = 1; print; next }
        /^Traceback \(most recent call last\):/ { traceback = 1; print; next }
        /Error|ERROR|Exception|OOM|out of memory|NaN|Inf|Killed/ { print }
    '
}

export MIMIC_LOADER_SEED="$LOADER_SEED"

cmd=(
    python main.py
    --gpu "$GPU"
    --seed "$MODEL_SEED"
    --data_seed "$DATA_SEED"
    --dataset MIMIC
    --model GPINet
    --history 24
    --pred_window 24
    --stride 24
    --time_unit hours
    --epoch "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --patience "$PATIENCE"
    --nlayer 3
    --hop 2
    --te_dim 10
    --node_dim 10
    --hid_dim 64
    --dropout 0.3
    --num "$TOTAL_N"
)

if [[ "$MODE" == "multi" ]]; then
    ensure_embeddings
    cmd+=(
        --enable_text
        --use_text_embeddings
        --llm_model_fusion BERT
        --llm_layers_fusion 6
        --max_length 512
        --n_heads_fusion "$TEXT_HEADS"
        --d_txt "$TEXT_DIM"
        --TTF_module "$TTF_MODULE"
        --MMF_module "$MMF_MODULE"
        --semantic_slots "$SEMANTIC_SLOTS"
        --recency_sigma "$RECENCY_SIGMA"
        --semantic_time_gate_bias "$SEMANTIC_TIME_GATE_BIAS"
        --absolute_recency_floor "$ABSOLUTE_RECENCY_FLOOR"
        --mmf_slot_attn_dim "$MMF_SLOT_ATTN_DIM"
        --mmf_slot_gate_bias "$MMF_SLOT_GATE_BIAS"
        --mmf_delta_init_std "$MMF_DELTA_INIT_STD"
        --fusion_gate_warmup_epochs "$FUSION_GATE_WARMUP_EPOCHS"
        --fusion_gate_warmup_value "$FUSION_GATE_WARMUP_VALUE"
        --fusion_lr_multiplier "$FUSION_LR_MULTIPLIER"
        --kappa "$KAPPA"
    )
fi
[[ "$USE_AMP" -eq 1 ]] && cmd+=(--use_amp)

mkdir -p "$OUTPUT_DIR"
timestamp="$(date '+%Y%m%d_%H%M%S')"
log_file="$OUTPUT_DIR/gpinet_n${TOTAL_N}_${MODE}_seed${MODEL_SEED}_${timestamp}.log"

echo "### GPINet mode=$MODE num=$TOTAL_N model_seed=$MODEL_SEED data_seed=$DATA_SEED"
set +e
"${cmd[@]}" 2>&1 | tee >(filter_final_log > "$log_file")
status=${PIPESTATUS[0]}
wait || true
set -e

echo "### Results: $log_file"
exit "$status"
