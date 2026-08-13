#!/usr/bin/env bash
# Run tPatchGNN inside the IMM-TSF benchmark — the baseline to compare
# GPINet against (see run_gpinet.sh). Same structure/flags/conventions on
# purpose, so the two are apples-to-apples: same dataset split, same
# history/pred_window/stride, same FusionModel (TTF+MMF) when --text is
# passed. The only thing that should differ is the backbone itself.
#
# Usage (run from anywhere; the script cd's into IMM-TSF/ itself):
#   ./run_tpatchgnn.sh                    # full numeric-only run
#   ./run_tpatchgnn.sh --text             # full run + FusionModel (TTF+MMF) on text
#   ./run_tpatchgnn.sh --smoke            # fast sanity run, tiny subset, numeric-only
#   ./run_tpatchgnn.sh --smoke --text     # fast sanity run, tiny subset, with text
#
# Env var overrides: see _common.sh, plus PATCH_SIZE/N_HEADS/TF_LAYER/
# OUTLAYER/NLAYER/HOP/TE_DIM/NODE_DIM/HID_DIM below.
#
# *** IMPORTANT CAVEAT — read before treating this as "the" tPatchGNN result ***
# PATCH_SIZE defaults to 24, mirroring this repo's own update_args_for_model()
# default for tPatchGNN, which hardcodes patch_size=24 for every dataset
# regardless of that dataset's history length. For MIMIC, history is also
# 24, and args.npatch = ceil((history - patch_size) / stride) + 1 (computed
# in main.py) evaluates to 1 — i.e. with these literal defaults, tPatchGNN's
# whole patch-based temporal modeling collapses to a single patch covering
# the entire history window (the transformer-over-patches layer runs on a
# sequence of length 1). This may be an intentional shared cross-dataset
# default, or a placeholder meant to be swept by a hyperparameter search
# whose results (referenced as
# exp_settings_and_results/.../tPatchGNN/MIMIC.json in main.py's
# commented-out code) aren't included in this copy of the repo. Treat a
# PATCH_SIZE=24 result as a starting point, not a validated baseline number.
# For a real multi-patch config, try e.g.:
#   PATCH_SIZE=4 ./run_tpatchgnn.sh
# which gives npatch = ceil((24-4)/24)+1 = 2 (STRIDE here is shared with the
# dataset's chunk-sliding stride, not a patch-only knob — changing it also
# changes how training chunks are cut).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
cd "$SCRIPT_DIR/.."

# Mirrors update_args_for_model() for tPatchGNN in main.py. NLAYER/HOP/
# TE_DIM/NODE_DIM/HID_DIM kept identical to run_gpinet.sh on purpose.
PATCH_SIZE="${PATCH_SIZE:-24}"
N_HEADS="${N_HEADS:-1}"
TF_LAYER="${TF_LAYER:-1}"
OUTLAYER="${OUTLAYER:-Linear}"
NLAYER="${NLAYER:-1}"
HOP="${HOP:-1}"
TE_DIM="${TE_DIM:-10}"
NODE_DIM="${NODE_DIM:-10}"
HID_DIM="${HID_DIM:-32}"

parse_common_flags "$@"
run_baseline tPatchGNN \
    --patch_size "$PATCH_SIZE" \
    --n_heads "$N_HEADS" \
    --tf_layer "$TF_LAYER" \
    --outlayer "$OUTLAYER" \
    --nlayer "$NLAYER" \
    --hop "$HOP" \
    --te_dim "$TE_DIM" \
    --node_dim "$NODE_DIM" \
    --hid_dim "$HID_DIM"
