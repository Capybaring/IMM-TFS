#!/usr/bin/env bash
# Run Time-LLM inside the IMM-TSF benchmark.
#
# Usage: ./run_timellm.sh [--text] [--smoke]
# Env var overrides: see _common.sh, plus INPUT_TOKEN_LEN/OUTPUT_TOKEN_LEN/
# D_MODEL/D_FF/LLM_MODEL_TIMELLM/LLM_LAYERS_TIMELLM below.
#
# NOTE: this baseline loads its own LLM backbone (GPT2 by default, via
# --llm_model_timellm) separately from the --llm_model_fusion used for text
# FusionModel — the two are independent settings that happen to both default
# to GPT2. Time-LLM is also the heaviest baseline here to run on CPU; if it's
# slow or OOMs, that's expected, not a bug in this script.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
cd "$SCRIPT_DIR/.."

# Mirrors update_args_for_model() for TimeLLM in main.py.
INPUT_TOKEN_LEN="${INPUT_TOKEN_LEN:-16}"
OUTPUT_TOKEN_LEN="${OUTPUT_TOKEN_LEN:-96}"
D_MODEL="${D_MODEL:-32}"
D_FF="${D_FF:-128}"
LLM_MODEL_TIMELLM="${LLM_MODEL_TIMELLM:-GPT2}"
LLM_LAYERS_TIMELLM="${LLM_LAYERS_TIMELLM:-6}"

parse_common_flags "$@"
run_baseline TimeLLM \
    --input_token_len "$INPUT_TOKEN_LEN" \
    --output_token_len "$OUTPUT_TOKEN_LEN" \
    --d_model "$D_MODEL" \
    --d_ff "$D_FF" \
    --llm_model_timellm "$LLM_MODEL_TIMELLM" \
    --llm_layers_timellm "$LLM_LAYERS_TIMELLM"
