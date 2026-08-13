#!/usr/bin/env bash
# Shared runner logic for IMM-TFS model scripts.
#
# Expanded-MIMIC v2 additions:
#   * --smoke limits BOTH training data and text-embedding generation to 200
#     sorted record IDs.
#   * expanded MIMIC embeddings use the fixed synthetic episode-day anchor.
#
# We still deliberately do NOT pass --overwrite_args; see the original repo's
# comments for why command-line args would otherwise be clobbered.

GPU="${GPU:-0}"
DATASET="${DATASET:-MIMIC}"
LLM_MODEL="${LLM_MODEL:-BERT}"
MAX_LENGTH="${MAX_LENGTH:-512}"
LLM_LAYERS="${LLM_LAYERS:-6}"
TTF_MODULE="${TTF_MODULE:-TTF_T2V_XAttn}"
MMF_MODULE="${MMF_MODULE:-MMF_XAttn_Add}"
EPOCH="${EPOCH:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PATIENCE="${PATIENCE:-10}"

HISTORY="${HISTORY:-24}"
PRED_WINDOW="${PRED_WINDOW:-24}"
STRIDE="${STRIDE:-24}"
TIME_UNIT="${TIME_UNIT:-hours}"

ENABLE_TEXT=0
SMOKE=0

parse_common_flags() {
    for arg in "$@"; do
        case "$arg" in
            --text) ENABLE_TEXT=1 ;;
            --smoke) SMOKE=1 ;;
            *)
                echo "Unknown argument: $arg (expected --text and/or --smoke)" >&2
                exit 1
                ;;
        esac
    done
}

run_baseline() {
    local model="$1"
    shift
    local extra_args=("$@")

    local n_arg=()
    local embed_max_records="None"

    if [ "$SMOKE" -eq 1 ]; then
        n_arg=(-n 200)
        embed_max_records="200"
        EPOCH=2
        PATIENCE=1
        echo "### Smoke mode: -n 200, text max_records=200, epoch=2, patience=1 ###"
    fi

    if [ "$ENABLE_TEXT" -eq 1 ]; then
        echo "### [1/2] Ensuring text embeddings exist for $DATASET ($LLM_MODEL, layers=$LLM_LAYERS, maxlen=$MAX_LENGTH) ###"

        python - \
            "$DATASET" \
            "$LLM_MODEL" \
            "$MAX_LENGTH" \
            "$LLM_LAYERS" \
            "$TIME_UNIT" \
            "$embed_max_records" <<'PY'
import sys
import torch

from compute_text_embeddings import compute_text_embeddings

(
    data_name,
    llm_model_fusion,
    max_length,
    llm_layers_fusion,
    time_unit,
    max_records_raw,
) = sys.argv[1:7]

max_length = int(max_length)
llm_layers_fusion = int(llm_layers_fusion)
max_records = (
    None if max_records_raw == "None" else int(max_records_raw)
)
device = "cuda" if torch.cuda.is_available() else "cpu"

is_expanded_mimic = data_name.upper().startswith("MIMIC")

compute_text_embeddings(
    data_name,
    llm_model_fusion,
    llm_layers_fusion,
    max_length,
    device,
    time_unit=time_unit,
    # Expanded MIMIC's t=0 is the synthetic episode-day start, not the
    # first actual numeric observation.
    episode_anchor_from_day_start=is_expanded_mimic,
    # Critical for --smoke: do not embed all ~20k records.
    max_records=max_records,
)
PY

        echo "### [2/2] Training $model + FusionModel ($TTF_MODULE / $MMF_MODULE) on $DATASET ###"
        python main.py \
            --gpu "$GPU" \
            --dataset "$DATASET" \
            --model "$model" \
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
            --epoch "$EPOCH" \
            --batch_size "$BATCH_SIZE" \
            --patience "$PATIENCE" \
            "${extra_args[@]}" \
            "${n_arg[@]}"
    else
        echo "### Training $model (numeric-only) on $DATASET ###"
        python main.py \
            --gpu "$GPU" \
            --dataset "$DATASET" \
            --model "$model" \
            --history "$HISTORY" \
            --pred_window "$PRED_WINDOW" \
            --stride "$STRIDE" \
            --time_unit "$TIME_UNIT" \
            --epoch "$EPOCH" \
            --batch_size "$BATCH_SIZE" \
            --patience "$PATIENCE" \
            "${extra_args[@]}" \
            "${n_arg[@]}"
    fi
}
