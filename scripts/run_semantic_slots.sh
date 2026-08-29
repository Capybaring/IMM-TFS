#!/usr/bin/env bash
# BUILD_ID: semantic-slot-effective-training-v6-20260824
set -euo pipefail

# Semantic-slot GPINet runner for expanded MIMIC.
#
# IMPORTANT: in this script -n means NUMBER OF TRAIN SUBJECTS.
# Validation/test subjects are fixed globally and never shrink/change with N.
# Train subsets are nested prefixes of one persisted random train order.

TRAIN_N=1000
ENABLE_TEXT=0
SWEEP=0
SWEEP_SIZES_CSV="1000,2000,4000,6000,8000"
SWEEP_MODES="both"
OUTPUT_DIR=""
RUN_LOG_OVERRIDE=""
GPU=0
EPOCHS=50
BATCH_SIZE=16
PATIENCE=10
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

TEXT_HEADS=1
TEXT_GATE_BIAS=-1.0

TEXT_DIM=128
TTF_MODULE="TTF_SemTime_Slots"
MMF_MODULE="MMF_VarTime_SlotGate"
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
USE_AMP=0
DETECT_ANOMALY=0

usage() {
cat <<'EOF'
Usage:
  ./scripts/run_semantic_slots.sh -n TRAIN_N [options]
  ./scripts/run_semantic_slots.sh --sweep [options]

Options:
  -n, --num N           Number of TRAIN subjects (default: 1000)
  --text                Enable BERT/radiology multimodal run
  --sweep               Run the fixed 1000/2000/4000/6000/8000 Uni+Multi experiment set
  --sizes CSV           Sweep sizes (default: 1000,2000,4000,6000,8000)
  --modes MODE          Sweep modes: both, uni, or text (default: both)
  --output-dir DIR      Sweep log directory (default: timestamped directory)
  --run-log FILE        Full log file for one run (normally set automatically)
  --epochs N            Epochs (default: 50)
  --batch-size N        Batch size (default: 16)
  --patience N          Early stopping patience (default: 5)
  --gpu ID              GPU id for main.py (default: 0)
  --seed N              Model/training seed (default: 1)
  --loader-seed N       Fixed DataLoader shuffle seed (default: 314159)
  --text-heads N        Attention heads used by fusion modules (default: 1)
  --text-gate-bias X    Legacy GPINet internal text-gate bias (default: -1.0)
  --text-dim N          Projected text dimension (default: 128)
  --TTF_module NAME     TTF module passed to main.py (default: TTF_SemTime_Slots)
  --MMF_module NAME     MMF module passed to main.py (default: MMF_VarTime_SlotGate)
  --semantic_slots N    Semantic slots for TTF_SemTime_Slots (default: 2)
  --recency_sigma X      Gaussian recency sigma on normalized time (default: 0.25)
  --semantic_time_gate_bias X
                        Initial adaptive relative-time bias (default: -1.0)
  --absolute-recency-floor X
                        Minimum absolute text strength after age decay (default: 0.1)
  --mmf_slot_attn_dim N MMF slot-attention dimension (default: 128)
  --mmf_slot_gate_bias X
                        Initial MMF residual-gate bias (default: 0.0)
  --mmf-delta-init-std X
                        Residual output initialization std (default: 0.01)
  --fusion-gate-warmup-epochs N
                        Epochs with protected non-zero text gate (default: 5)
  --fusion-gate-warmup-value X
                        Fixed gate during warmup (default: 0.5)
  --fusion-lr-multiplier X
                        Fusion learning-rate multiplier (default: 2.0)
  --kappa X             Maximum text residual scale (default: 0.1)
  --amp                 Enable automatic mixed precision
  --detect-anomaly      Enable expensive autograd anomaly detection
  -h, --help            Show this help

Examples:
  ./scripts/run_semantic_slots.sh -n 1000
  ./scripts/run_semantic_slots.sh -n 1000 --text
  ./scripts/run_semantic_slots.sh --sweep
  ./scripts/run_semantic_slots.sh --sweep --epochs 50 --patience 10
  ./scripts/run_semantic_slots.sh --sweep --modes text --sizes 1000,2000,4000,6000,8000
  ./scripts/run_semantic_slots.sh -n 1000 --epochs 50 --patience 10
  ./scripts/run_semantic_slots.sh -n 1000 --text --epochs 50 --patience 10
  ./scripts/run_semantic_slots.sh -n 1000 --text \
    --TTF_module TTF_SemTime_Slots \
    --text-dim 128 \
    --semantic_slots 2 \
    --recency_sigma 0.25 \
    --semantic_time_gate_bias -1.0 \
    --absolute-recency-floor 0.1 \
    --MMF_module MMF_VarTime_SlotGate \
    --mmf_slot_attn_dim 128 \
    --mmf_slot_gate_bias 0.0 \
    --mmf-delta-init-std 0.01 \
    --fusion-gate-warmup-epochs 5 \
    --fusion-gate-warmup-value 0.5 \
    --fusion-lr-multiplier 2.0 \
    --kappa 0.1

Sweep outputs:
  <output-dir>/final_metrics.log                  readable MSE/MAE summary
  <output-dir>/final_metrics.csv                  machine-readable summary
  <output-dir>/gpinet_n<N>_<uni|text>_seed<S>.log full console log per run

Protocol files (created once):
  data/MIMIC/mimic_fixed_protocol.json
  data/MIMIC/mimic_fixed_normalization.pt

If missing, this runner creates it automatically using:
  split_seed       = 42
  train_order_seed = 2026

Normalization is persisted once and shared by every N, Uni, and Text run:
  scaler = FULL fixed TRAIN HISTORY [0,24h) only
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num) TRAIN_N="$2"; shift 2 ;;
        --text) ENABLE_TEXT=1; shift ;;
        --sweep) SWEEP=1; shift ;;
        --sizes) SWEEP_SIZES_CSV="$2"; shift 2 ;;
        --modes) SWEEP_MODES="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --run-log) RUN_LOG_OVERRIDE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --patience) PATIENCE="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --seed) MODEL_SEED="$2"; shift 2 ;;
        --loader-seed) LOADER_SEED="$2"; shift 2 ;;
        --text-heads) TEXT_HEADS="$2"; shift 2 ;;
        --text-gate-bias) TEXT_GATE_BIAS="$2"; shift 2 ;;
        --text-dim) TEXT_DIM="$2"; shift 2 ;;
        --TTF_module|--ttf-module) TTF_MODULE="$2"; shift 2 ;;
        --MMF_module|--mmf-module) MMF_MODULE="$2"; shift 2 ;;
        --semantic_slots|--semantic-slots) SEMANTIC_SLOTS="$2"; shift 2 ;;
        --recency_sigma|--recency-sigma) RECENCY_SIGMA="$2"; shift 2 ;;
        --semantic_time_gate_bias|--semantic-time-gate-bias)
            SEMANTIC_TIME_GATE_BIAS="$2"; shift 2 ;;
        --absolute_recency_floor|--absolute-recency-floor)
            ABSOLUTE_RECENCY_FLOOR="$2"; shift 2 ;;
        --mmf_slot_attn_dim|--mmf-slot-attn-dim)
            MMF_SLOT_ATTN_DIM="$2"; shift 2 ;;
        --mmf_slot_gate_bias|--mmf-slot-gate-bias)
            MMF_SLOT_GATE_BIAS="$2"; shift 2 ;;
        --mmf_delta_init_std|--mmf-delta-init-std)
            MMF_DELTA_INIT_STD="$2"; shift 2 ;;
        --fusion_gate_warmup_epochs|--fusion-gate-warmup-epochs)
            FUSION_GATE_WARMUP_EPOCHS="$2"; shift 2 ;;
        --fusion_gate_warmup_value|--fusion-gate-warmup-value)
            FUSION_GATE_WARMUP_VALUE="$2"; shift 2 ;;
        --fusion_lr_multiplier|--fusion-lr-multiplier)
            FUSION_LR_MULTIPLIER="$2"; shift 2 ;;
        --kappa) KAPPA="$2"; shift 2 ;;
        --amp) USE_AMP=1; shift ;;
        --detect-anomaly) DETECT_ANOMALY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

for pair in \
    "TRAIN_N:$TRAIN_N" \
    "EPOCHS:$EPOCHS" \
    "BATCH_SIZE:$BATCH_SIZE" \
    "PATIENCE:$PATIENCE" \
    "TEXT_HEADS:$TEXT_HEADS" \
    "TEXT_DIM:$TEXT_DIM" \
    "SEMANTIC_SLOTS:$SEMANTIC_SLOTS" \
    "MMF_SLOT_ATTN_DIM:$MMF_SLOT_ATTN_DIM"; do
    name="${pair%%:*}"; value="${pair#*:}"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -le 0 ]]; then
        echo "Error: $name must be a positive integer, got $value" >&2
        exit 1
    fi
done

if ! [[ "$FUSION_GATE_WARMUP_EPOCHS" =~ ^[0-9]+$ ]]; then
    echo "Error: FUSION_GATE_WARMUP_EPOCHS must be >= 0" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ "$SWEEP" -eq 1 ]]; then
    case "$SWEEP_MODES" in
        both) SWEEP_TEXT_FLAGS=(0 1) ;;
        uni)  SWEEP_TEXT_FLAGS=(0) ;;
        text) SWEEP_TEXT_FLAGS=(1) ;;
        *)
            echo "Error: --modes must be one of: both, uni, text" >&2
            exit 2
            ;;
    esac

    IFS=',' read -r -a SWEEP_SIZES <<< "$SWEEP_SIZES_CSV"
    if [[ "${#SWEEP_SIZES[@]}" -eq 0 ]]; then
        echo "Error: --sizes must contain at least one positive integer" >&2
        exit 2
    fi
    for size in "${SWEEP_SIZES[@]}"; do
        if ! [[ "$size" =~ ^[0-9]+$ ]] || [[ "$size" -le 0 ]]; then
            echo "Error: invalid train size in --sizes: $size" >&2
            exit 2
        fi
    done

    if [[ -z "$OUTPUT_DIR" ]]; then
        OUTPUT_DIR="$REPO_ROOT/logs/gpinet_semantic_slots_sweep/$(date '+%Y%m%d_%H%M%S')"
    elif [[ "$OUTPUT_DIR" != /* ]]; then
        OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
    fi
    mkdir -p "$OUTPUT_DIR"

    RESULTS_LOG="$OUTPUT_DIR/final_metrics.log"
    RESULTS_CSV="$OUTPUT_DIR/final_metrics.csv"
    SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

    {
        echo "GPINet semantic-slot fixed-protocol sweep"
        echo "started_at       : $(date '+%Y-%m-%d %H:%M:%S %z')"
        echo "train_sizes      : $SWEEP_SIZES_CSV"
        echo "modes            : $SWEEP_MODES"
        echo "model_seed       : $MODEL_SEED"
        echo "loader_seed      : $LOADER_SEED"
        echo "TTF module       : $TTF_MODULE"
        echo "MMF module       : $MMF_MODULE"
        echo "text dimension   : $TEXT_DIM"
        echo "semantic slots   : $SEMANTIC_SLOTS"
        echo "recency sigma    : $RECENCY_SIGMA"
        echo "time gate bias   : $SEMANTIC_TIME_GATE_BIAS"
        echo "absolute floor   : $ABSOLUTE_RECENCY_FLOOR"
        echo "MMF attn dim     : $MMF_SLOT_ATTN_DIM"
        echo "MMF gate bias    : $MMF_SLOT_GATE_BIAS"
        echo "delta init std   : $MMF_DELTA_INIT_STD"
        echo "gate warmup      : $FUSION_GATE_WARMUP_EPOCHS epochs @ $FUSION_GATE_WARMUP_VALUE"
        echo "fusion LR factor : $FUSION_LR_MULTIPLIER"
        echo "text scale kappa : $KAPPA"
        echo "epochs/patience  : $EPOCHS/$PATIENCE"
        echo "batch_size       : $BATCH_SIZE"
        echo "output_directory : $OUTPUT_DIR"
        echo
        printf '%-10s %-10s %-18s %-18s %-10s\n' \
            "train_n" "mode" "mse" "mae" "status"
    } > "$RESULTS_LOG"

    echo "train_n,mode,text_enabled,model_seed,loader_seed,mse,mae,status,run_log" \
        > "$RESULTS_CSV"

    echo "### Sweep started: sizes=$SWEEP_SIZES_CSV modes=$SWEEP_MODES seed=$MODEL_SEED"
    echo "### Sweep output: $OUTPUT_DIR"

    FAILED_RUNS=0
    TOTAL_RUNS=0
    for size in "${SWEEP_SIZES[@]}"; do
        for text_flag in "${SWEEP_TEXT_FLAGS[@]}"; do
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
            mode="uni"
            if [[ "$text_flag" -eq 1 ]]; then
                mode="text"
            fi

            RUN_LOG="$OUTPUT_DIR/gpinet_n${size}_${mode}_seed${MODEL_SEED}.log"
            RUN_CMD=(
                "$SCRIPT_PATH"
                -n "$size"
                --epochs "$EPOCHS"
                --batch-size "$BATCH_SIZE"
                --patience "$PATIENCE"
                --gpu "$GPU"
                --seed "$MODEL_SEED"
                --loader-seed "$LOADER_SEED"
                --text-heads "$TEXT_HEADS"
                --text-gate-bias "$TEXT_GATE_BIAS"
                --text-dim "$TEXT_DIM"
                --TTF_module "$TTF_MODULE"
                --MMF_module "$MMF_MODULE"
                --semantic_slots "$SEMANTIC_SLOTS"
                --recency_sigma "$RECENCY_SIGMA"
                --semantic_time_gate_bias "$SEMANTIC_TIME_GATE_BIAS"
                --absolute-recency-floor "$ABSOLUTE_RECENCY_FLOOR"
                --mmf_slot_attn_dim "$MMF_SLOT_ATTN_DIM"
                --mmf_slot_gate_bias "$MMF_SLOT_GATE_BIAS"
                --mmf-delta-init-std "$MMF_DELTA_INIT_STD"
                --fusion-gate-warmup-epochs "$FUSION_GATE_WARMUP_EPOCHS"
                --fusion-gate-warmup-value "$FUSION_GATE_WARMUP_VALUE"
                --fusion-lr-multiplier "$FUSION_LR_MULTIPLIER"
                --kappa "$KAPPA"
                --run-log "$RUN_LOG"
            )
            if [[ "$USE_AMP" -eq 1 ]]; then
                RUN_CMD+=(--amp)
            fi
            if [[ "$DETECT_ANOMALY" -eq 1 ]]; then
                RUN_CMD+=(--detect-anomaly)
            fi
            if [[ "$text_flag" -eq 1 ]]; then
                RUN_CMD+=(--text)
            fi

            set +e
            "${RUN_CMD[@]}"
            RUN_STATUS=$?
            set -e

            MSE="$(
                tr '\r' '\n' < "$RUN_LOG" \
                    | awk -F': ' '$1 == "mse" {value=$2} END {print value}'
            )"
            MAE="$(
                tr '\r' '\n' < "$RUN_LOG" \
                    | awk -F': ' '$1 == "mae" {value=$2} END {print value}'
            )"

            status="ok"
            if [[ "$RUN_STATUS" -ne 0 || -z "$MSE" || -z "$MAE" ]]; then
                status="failed"
                FAILED_RUNS=$((FAILED_RUNS + 1))
                [[ -n "$MSE" ]] || MSE="NA"
                [[ -n "$MAE" ]] || MAE="NA"
            fi

            printf '%-10s %-10s %-18s %-18s %-10s\n' \
                "$size" "$mode" "$MSE" "$MAE" "$status" \
                >> "$RESULTS_LOG"
            printf '%s,%s,%s,%s,%s,%s,%s,%s,"%s"\n' \
                "$size" "$mode" "$text_flag" "$MODEL_SEED" \
                "$LOADER_SEED" "$MSE" "$MAE" "$status" "$RUN_LOG" \
                >> "$RESULTS_CSV"
        done
    done

    {
        echo
        echo "finished_at : $(date '+%Y-%m-%d %H:%M:%S %z')"
        echo "total_runs  : $TOTAL_RUNS"
        echo "failed_runs : $FAILED_RUNS"
    } >> "$RESULTS_LOG"

    echo
    echo "### Sweep complete"
    echo "### Final MSE/MAE log: $RESULTS_LOG"
    echo "### CSV summary      : $RESULTS_CSV"
    if [[ "$FAILED_RUNS" -ne 0 ]]; then
        exit 1
    fi
    exit 0
fi

RUN_MODE="uni"
if [[ "$ENABLE_TEXT" -eq 1 ]]; then
    RUN_MODE="text"
fi
if [[ -n "$RUN_LOG_OVERRIDE" ]]; then
    if [[ "$RUN_LOG_OVERRIDE" = /* ]]; then
        RUN_LOG="$RUN_LOG_OVERRIDE"
    else
        RUN_LOG="$REPO_ROOT/$RUN_LOG_OVERRIDE"
    fi
else
    RUN_DIR="$REPO_ROOT/logs/gpinet_semantic_slots_runs"
    RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
    RUN_LOG="$RUN_DIR/gpinet_n${TRAIN_N}_${RUN_MODE}_seed${MODEL_SEED}_${RUN_STAMP}.log"
fi
mkdir -p "$(dirname "$RUN_LOG")"
: > "$RUN_LOG"

PROTOCOL="data/MIMIC/mimic_fixed_protocol.json"
NORMALIZATION="data/MIMIC/mimic_fixed_normalization.pt"
if [[ ! -f "$PROTOCOL" && ! -f "$NORMALIZATION" ]]; then
    echo "### Fixed protocol not found; preparing it once from the full cohort ###"
    python scripts/prepare_mimic_fixed_protocol.py \
        --dataset-dir data/MIMIC \
        --history "$HISTORY" \
        --pred-window "$PRED_WINDOW" \
        --time-unit "$TIME_UNIT" \
        --split-seed 42 \
        --train-order-seed 2026
elif [[ ! -f "$PROTOCOL" || ! -f "$NORMALIZATION" ]]; then
    echo "### Fixed protocol/scaler is incomplete; rebuild both once with:" >&2
    echo "python scripts/prepare_mimic_fixed_protocol.py --force" >&2
    exit 2
else
    PROTOCOL_VERSION="$(python - "$PROTOCOL" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(json.load(f).get("version", ""))
PY
)"
    if [[ "$PROTOCOL_VERSION" != "4-fixed-full-train-normalization" ]]; then
        echo "### Existing fixed protocol uses an incompatible normalization rule." >&2
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

{
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
    echo "AMP / anomaly detect  : $USE_AMP / $DETECT_ANOMALY"
    echo "normalization         : fixed FULL-TRAIN HISTORY"
    if [[ "$ENABLE_TEXT" -eq 1 ]]; then
        echo "TTF / MMF             : $TTF_MODULE / $MMF_MODULE"
        echo "fusion heads          : $TEXT_HEADS"
        echo "text dimension        : $TEXT_DIM"
        echo "semantic slots        : $SEMANTIC_SLOTS"
        echo "recency sigma         : $RECENCY_SIGMA"
        echo "semantic time bias    : $SEMANTIC_TIME_GATE_BIAS"
        echo "absolute recency floor: $ABSOLUTE_RECENCY_FLOOR"
        echo "MMF attn dim / bias   : $MMF_SLOT_ATTN_DIM / $MMF_SLOT_GATE_BIAS"
        echo "MMF delta init std    : $MMF_DELTA_INIT_STD"
        echo "gate warmup           : $FUSION_GATE_WARMUP_EPOCHS @ $FUSION_GATE_WARMUP_VALUE"
        echo "fusion LR multiplier  : $FUSION_LR_MULTIPLIER"
        echo "text scale kappa      : $KAPPA"
        echo "legacy GP gate bias   : $TEXT_GATE_BIAS"
    fi
    echo "full log              : $RUN_LOG"
    echo "=================================================================="
}

if [[ "$ENABLE_TEXT" -eq 1 ]]; then
    echo
    echo "### [1/2] Ensuring embeddings for train_N + fixed val + fixed test ###"
    set +e
    python - "$PROTOCOL" "$TRAIN_N" "$DATASET" "$LLM_MODEL" "$LLM_LAYERS" "$MAX_LENGTH" "$TIME_UNIT" 2>&1 <<'PY'
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
    EMBED_STATUS=$?
    set -e
    if [[ "$EMBED_STATUS" -ne 0 ]]; then
        echo "### Embedding preparation failed with exit code $EMBED_STATUS ###"
        exit "$EMBED_STATUS"
    fi
    echo
    echo "### [2/2] Training GPINet multimodal ###"
else
    echo
    echo "### Training GPINet numeric-only ###"
fi

export MIMIC_FIXED_PROTOCOL=1
export MIMIC_PROTOCOL_PATH="$REPO_ROOT/$PROTOCOL"
export MIMIC_NORMALIZATION_PATH="$REPO_ROOT/$NORMALIZATION"
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
        --n_heads_fusion "$TEXT_HEADS"
        --d_txt "$TEXT_DIM"
        --gpinet_text_gate_bias "$TEXT_GATE_BIAS"
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

if [[ "$USE_AMP" -eq 1 ]]; then
    CMD+=(--use_amp)
fi
if [[ "$DETECT_ANOMALY" -eq 1 ]]; then
    CMD+=(--detect_anomaly)
fi

echo
printf "### Command:"
printf " %q" "${CMD[@]}"
echo

filter_final_log() {
    tr '\r' '\n' \
        | awk '
            BEGIN {
                final_output = 0
                traceback = 0
            }

            final_output {
                print
                fflush()
                next
            }
            /^Exp has been early stopped!$/ {
                final_output = 1
                print
                fflush()
                next
            }
            /^loss: / {
                final_output = 1
                print
                fflush()
                next
            }

            traceback {
                print
                fflush()
                next
            }
            /^Traceback \(most recent call last\):/ {
                traceback = 1
                print
                fflush()
                next
            }
            /Error|ERROR|Exception|OOM|out of memory|NaN|Inf|Killed|Segmentation fault/ {
                print
                fflush()
            }
        ' > "$RUN_LOG"
}

set +e
"${CMD[@]}" 2>&1 \
    | tee >(filter_final_log)
RUN_STATUS=${PIPESTATUS[0]}
FILTER_PID="$!"
if [[ -n "$FILTER_PID" ]]; then
    wait "$FILTER_PID"
fi
set -e

if [[ "$RUN_STATUS" -eq 0 ]]; then
    echo >> "$RUN_LOG"
    echo "### Done: fixed protocol / train_N=$TRAIN_N / text=$ENABLE_TEXT / model_seed=$MODEL_SEED ###" \
        >> "$RUN_LOG"
    echo "### Done: train_N=$TRAIN_N mode=$RUN_MODE seed=$MODEL_SEED"
    echo "### Full results: $RUN_LOG"
else
    echo >> "$RUN_LOG"
    echo "### Failed with exit code $RUN_STATUS ###" >> "$RUN_LOG"
    echo "### Failed: train_N=$TRAIN_N mode=$RUN_MODE seed=$MODEL_SEED"
    echo "### See full log: $RUN_LOG"
    exit "$RUN_STATUS"
fi
