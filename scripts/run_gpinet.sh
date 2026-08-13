#!/usr/bin/env bash
# Run GPINet inside the IMM-TSF benchmark.
#
# Usage (run from anywhere; the script cd's into IMM-TSF/ itself):
#   ./run_gpinet.sh                    # full numeric-only run
#   ./run_gpinet.sh --text             # full run + FusionModel (TTF+MMF) on text
#   ./run_gpinet.sh --smoke            # fast sanity run, tiny subset, numeric-only
#   ./run_gpinet.sh --smoke --text     # fast sanity run, tiny subset, with text
#
# Env var overrides (all optional): GPU, DATASET, LLM_MODEL, MAX_LENGTH,
# LLM_LAYERS, TTF_MODULE, MMF_MODULE, EPOCH, BATCH_SIZE, PATIENCE, HISTORY,
# PRED_WINDOW, STRIDE, TIME_UNIT, NLAYER, HOP, TE_DIM, NODE_DIM, HID_DIM,
# DROPOUT.
# See _common.sh for shared behavior and important caveats (why
# --overwrite_args is never passed, why LLM_LAYERS must match between the
# embedding step and the training step).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
cd "$SCRIPT_DIR/.."

# GPINet's own originally-tuned capacity (see gpinet_mm.py / model.py
# reference implementations), no longer matched to run_tpatchgnn.sh's
# NLAYER/HOP/HID_DIM. That capacity-matching was a deliberate earlier
# choice for a "backbone-only, fair comparison" ablation; it's been
# dropped per 2026-08-09 decision to restore GPINet to its intended
# settings (hid_dim=64 confirmed best by the user).
NLAYER="${NLAYER:-3}"
HOP="${HOP:-2}"
TE_DIM="${TE_DIM:-10}"
NODE_DIM="${NODE_DIM:-10}"
HID_DIM="${HID_DIM:-64}"
# GPINet-specific override of the global --dropout default (0.1 in
# main.py). Original reference implementations use 0.3; only GPINet gets
# this override, other models keep the shared 0.1 default.
DROPOUT="${DROPOUT:-0.3}"

parse_common_flags "$@"
run_baseline GPINet \
    --nlayer "$NLAYER" \
    --hop "$HOP" \
    --te_dim "$TE_DIM" \
    --node_dim "$NODE_DIM" \
    --hid_dim "$HID_DIM" \
    --dropout "$DROPOUT"
