#!/usr/bin/env bash
# Shared runner logic for the run_<model>.sh scripts in this directory.
# Not meant to be executed directly — each run_<model>.sh does:
#   cd "$(dirname "$0")"
#   source ./_common.sh
#   parse_common_flags "$@"
#   run_baseline <ModelName> [extra --flag value ...]
#
# Design notes (apply to every run_<model>.sh, not just this file):
#
# 1. We deliberately never pass --overwrite_args to main.py. In this
#    codebase that flag does NOT mean "fill in dataset/model defaults" — it
#    means "ignore every --dataset/--model/--enable_text/... flag you just
#    passed on the command line and instead overwrite args with the
#    `fixed_params` dict hardcoded in main.py's `if __name__ == "__main__":`
#    block" (see update_args_from_fixed_params / update_args in main.py).
#    That block is someone's leftover debug config (dataset=ClusterTrace,
#    model=LatentODE, enable_text=False, ...) — passing --overwrite_args
#    here would silently clobber whatever model/dataset/text setting you
#    asked for. Instead, every run_<model>.sh passes each dataset- and
#    model-specific hyperparameter explicitly, mirroring exactly what
#    update_args_for_dataset()/update_args_for_model() would have set for
#    that combination (see main.py) — so from the outside it behaves the
#    same as if --overwrite_args had "worked as advertised", just without
#    the CLI-clobbering side effect.
#
# 2. --text needs text embeddings cached under
#    data/<DATASET>/processed/<record_id>/text_embeddings_model=<LLM_MODEL>_layers=<LLM_LAYERS>_maxlen=<MAX_LENGTH>.pt
#    Those three values (LLM_MODEL/LLM_LAYERS/MAX_LENGTH) MUST be identical
#    between the embedding-generation step and the training step, or the
#    dataset loader raises FileNotFoundError looking for a filename that
#    doesn't exist. run_baseline() below generates with, and trains with,
#    the same LLM_MODEL/LLM_LAYERS/MAX_LENGTH — don't override one without
#    the other.
#
# 3. A handful of update_args_for_model() fields (e.g. CRU's
#    cru_enc_var_activation/cru_dec_var_activation) are set via plain Python
#    attribute assignment in main.py and have NO corresponding argparse
#    flag — there is no way to pass them on the command line at all. Where
#    that happens, the affected run_<model>.sh does not try; it relies on
#    the model's own `getattr(args, "...", <same default>)` fallback inside
#    its __init__, which matches update_args_for_model()'s value anyway (so
#    the end result is identical either way — just flagging it so it's not
#    mysterious if you go looking for the flag and can't find it).

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

# Mirrors update_args_for_dataset() for MIMIC. Override via env vars if you
# point this at a different --dataset (and check main.py's
# update_args_for_dataset() for that dataset's own history/pred_window/
# stride/time_unit before trusting these).
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

# run_baseline <ModelName> [extra main.py flags for that model's hyperparams...]
run_baseline() {
    local model="$1"
    shift
    local extra_args=("$@")

    local n_arg=()
    if [ "$SMOKE" -eq 1 ]; then
        n_arg=(-n 200)
        EPOCH=2
        PATIENCE=1
        echo "### Smoke mode: -n 200, epoch=2, patience=1 ###"
    fi

    if [ "$ENABLE_TEXT" -eq 1 ]; then
        echo "### [1/2] Ensuring text embeddings exist for $DATASET ($LLM_MODEL, layers=$LLM_LAYERS, maxlen=$MAX_LENGTH) ###"
        python - "$DATASET" "$LLM_MODEL" "$MAX_LENGTH" "$LLM_LAYERS" "$TIME_UNIT" <<'PY'
import sys
import torch
from compute_text_embeddings import compute_text_embeddings

data_name, llm_model_fusion, max_length, llm_layers_fusion, time_unit = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
)
device = "cuda" if torch.cuda.is_available() else "cpu"
# time_unit + align_base_to_series fix the text/time_series unit+reference-point
# mismatch bug in compute_text_embeddings.py (see IMM-TSF/results/mimic_comparison.md
# / project notes). Scoped here (not in the function's defaults or the standalone
# __main__ batch script) so only this pipeline's runs (MIMIC by default) are affected;
# other datasets' embedding generation is untouched until reviewed separately.
compute_text_embeddings(
    data_name, llm_model_fusion, llm_layers_fusion, max_length, device,
    time_unit=time_unit, align_base_to_series=True,
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
