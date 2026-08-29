#!/usr/bin/env bash
set -euo pipefail

# One entry point for independent-size GPINet Uni/Multi experiments.

TRAIN_N=1000
ENABLE_TEXT=0
SWEEP=0
SIZES="1000,2000,4000,6000,8000"
MODES="both"
OUTPUT_DIR=""

GPU=0
EPOCHS=50
BATCH_SIZE=16
PATIENCE=10
MODEL_SEED=1
DATA_SEED=42
LOADER_SEED=314159
USE_AMP=0

TTF_MODULE="TTF_SemTime_Slots"
MMF_MODULE="MMF_VarTime_SlotGate"
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
  ./scripts/run_gpinet.sh -n N [--text] [options]
  ./scripts/run_gpinet.sh --sweep [options]

Core options:
  -n, --num N                 Number of independently sampled TRAIN subjects
  --text                      Enable multimodal training
  --sweep                     Run all configured sizes and modes
  --sizes CSV                 Sweep sizes (default: 1000,2000,4000,6000,8000)
  --modes both|uni|text       Sweep modes
  --epochs N                  Training epochs
  --batch-size N              Batch size
  --patience N                Early-stop patience
  --gpu ID                    GPU id
  --seed N                    Model seed
  --data-seed N               Training-subject sampling seed
  --loader-seed N             DataLoader seed
  --output-dir DIR            Log directory
  --amp                       Enable AMP

Fusion options:
  --TTF_module NAME
  --MMF_module NAME
  --text-heads N
  --text-dim N
  --semantic_slots N
  --recency_sigma X
  --semantic_time_gate_bias X
  --absolute-recency-floor X
  --mmf_slot_attn_dim N
  --mmf_slot_gate_bias X
  --mmf-delta-init-std X
  --fusion-gate-warmup-epochs N
  --fusion-gate-warmup-value X
  --fusion-lr-multiplier X
  --kappa X
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num) TRAIN_N="$2"; shift 2 ;;
        --text) ENABLE_TEXT=1; shift ;;
        --sweep) SWEEP=1; shift ;;
        --sizes) SIZES="$2"; shift 2 ;;
        --modes) MODES="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --patience) PATIENCE="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --seed) MODEL_SEED="$2"; shift 2 ;;
        --data-seed) DATA_SEED="$2"; shift 2 ;;
        --loader-seed) LOADER_SEED="$2"; shift 2 ;;
        --amp) USE_AMP=1; shift ;;
        --TTF_module) TTF_MODULE="$2"; shift 2 ;;
        --MMF_module) MMF_MODULE="$2"; shift 2 ;;
        --text-heads) TEXT_HEADS="$2"; shift 2 ;;
        --text-dim) TEXT_DIM="$2"; shift 2 ;;
        --semantic_slots) SEMANTIC_SLOTS="$2"; shift 2 ;;
        --recency_sigma) RECENCY_SIGMA="$2"; shift 2 ;;
        --semantic_time_gate_bias) SEMANTIC_TIME_GATE_BIAS="$2"; shift 2 ;;
        --absolute-recency-floor) ABSOLUTE_RECENCY_FLOOR="$2"; shift 2 ;;
        --mmf_slot_attn_dim) MMF_SLOT_ATTN_DIM="$2"; shift 2 ;;
        --mmf_slot_gate_bias) MMF_SLOT_GATE_BIAS="$2"; shift 2 ;;
        --mmf-delta-init-std) MMF_DELTA_INIT_STD="$2"; shift 2 ;;
        --fusion-gate-warmup-epochs) FUSION_GATE_WARMUP_EPOCHS="$2"; shift 2 ;;
        --fusion-gate-warmup-value) FUSION_GATE_WARMUP_VALUE="$2"; shift 2 ;;
        --fusion-lr-multiplier) FUSION_LR_MULTIPLIER="$2"; shift 2 ;;
        --kappa) KAPPA="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ensure_embeddings() {
    local train_n="$1"
    python - "$train_n" "$DATA_SEED" <<'PY'
import sys, torch
from compute_text_embeddings import compute_text_embeddings
from lib.parse_datasets import select_mimic_subject_records

selection = select_mimic_subject_records(
    "data/MIMIC",
    requested_train_n=int(sys.argv[1]),
    data_seed=int(sys.argv[2]),
)

compute_text_embeddings(
    "MIMIC", "BERT", 6, 512,
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

run_one() {
    local train_n="$1"
    local text_enabled="$2"
    local log_file="$3"
    local mode="uni"
    mkdir -p "$(dirname "$log_file")"
    : > "$log_file"
    local cmd=(
        python main.py
        --gpu "$GPU"
        --seed "$MODEL_SEED"
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
        --data_seed "$DATA_SEED"
        -n "$train_n"
    )

    if [[ "$text_enabled" -eq 1 ]]; then
        mode="text"
        ensure_embeddings "$train_n" || return $?
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

    echo "### GPINet train_n=$train_n mode=$mode model_seed=$MODEL_SEED data_seed=$DATA_SEED"
    set +e
    "${cmd[@]}" 2>&1 | tee >(filter_final_log > "$log_file")
    local status=${PIPESTATUS[0]}
    wait || true
    set -e
    return "$status"
}

metric_from_log() {
    [[ -f "$1" ]] || return 0
    awk -F': ' -v key="$2" '$1 == key { value=$2 } END { print value }' "$1"
}

export MIMIC_LOADER_SEED="$LOADER_SEED"

if [[ "$SWEEP" -eq 1 ]]; then
    case "$MODES" in
        both) TEXT_FLAGS=(0 1) ;;
        uni) TEXT_FLAGS=(0) ;;
        text) TEXT_FLAGS=(1) ;;
        *) echo "--modes must be both, uni, or text" >&2; exit 2 ;;
    esac
    IFS=',' read -r -a TRAIN_SIZES <<< "$SIZES"
    OUTPUT_DIR="${OUTPUT_DIR:-logs/gpinet_independent_sweep/$(date '+%Y%m%d_%H%M%S')}"
    mkdir -p "$OUTPUT_DIR"
    SUMMARY="$OUTPUT_DIR/final_metrics.csv"
    echo "train_n,mode,model_seed,data_seed,loader_seed,mse,mae,status,run_log" > "$SUMMARY"

    failures=0
    for train_n in "${TRAIN_SIZES[@]}"; do
        for text_enabled in "${TEXT_FLAGS[@]}"; do
            mode="uni"
            [[ "$text_enabled" -eq 1 ]] && mode="text"
            log_file="$OUTPUT_DIR/gpinet_n${train_n}_${mode}_seed${MODEL_SEED}.log"
            if run_one "$train_n" "$text_enabled" "$log_file"; then
                status="ok"
            else
                status="failed"
                failures=$((failures + 1))
            fi
            mse="$(metric_from_log "$log_file" mse)"
            mae="$(metric_from_log "$log_file" mae)"
            echo "$train_n,$mode,$MODEL_SEED,$DATA_SEED,$LOADER_SEED,${mse:-NA},${mae:-NA},$status,\"$log_file\"" >> "$SUMMARY"
        done
    done
    echo "### Results: $SUMMARY"
    [[ "$failures" -eq 0 ]]
    exit
fi

mode="uni"
[[ "$ENABLE_TEXT" -eq 1 ]] && mode="text"
OUTPUT_DIR="${OUTPUT_DIR:-logs/gpinet_independent_runs}"
log_file="$OUTPUT_DIR/gpinet_n${TRAIN_N}_${mode}_seed${MODEL_SEED}_$(date '+%Y%m%d_%H%M%S').log"
run_one "$TRAIN_N" "$ENABLE_TEXT" "$log_file"
echo "### Results: $log_file"
