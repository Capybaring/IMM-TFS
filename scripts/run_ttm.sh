#!/usr/bin/env bash
# Run TTM inside the IMM-TSF benchmark.
#
# Usage: ./run_ttm.sh [--text] [--smoke]
# Env var overrides: see _common.sh, plus INPUT_TOKEN_LEN/OUTPUT_TOKEN_LEN/
# D_MODEL/AP_LEVELS/E_LAYERS/D_LAYERS/D_D_MODEL/PATCH_SIZE below.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
cd "$SCRIPT_DIR/.."

# Mirrors update_args_for_model() for TTM in main.py, including
# `args.patch_size = args.history // 4` — computed here from HISTORY so it
# stays correct if you override HISTORY for a different dataset.
INPUT_TOKEN_LEN="${INPUT_TOKEN_LEN:-16}"
OUTPUT_TOKEN_LEN="${OUTPUT_TOKEN_LEN:-96}"
D_MODEL="${D_MODEL:-1024}"
AP_LEVELS="${AP_LEVELS:-3}"
E_LAYERS="${E_LAYERS:-3}"
D_LAYERS="${D_LAYERS:-2}"
D_D_MODEL="${D_D_MODEL:-64}"
PATCH_SIZE="${PATCH_SIZE:-$(( HISTORY / 4 ))}"

parse_common_flags "$@"
run_baseline TTM \
    --input_token_len "$INPUT_TOKEN_LEN" \
    --output_token_len "$OUTPUT_TOKEN_LEN" \
    --d_model "$D_MODEL" \
    --AP_levels "$AP_LEVELS" \
    --e_layers "$E_LAYERS" \
    --d_layers "$D_LAYERS" \
    --d_d_model "$D_D_MODEL" \
    --patch_size "$PATCH_SIZE"
