#!/usr/bin/env bash
set -euo pipefail

# Conservative defaults for the competitive-slot TTF and identity-initialized
# MMF.  User-supplied arguments are appended last and therefore can override
# any default below.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_RUNNER="$SCRIPT_DIR/run_semantic_slots.sh"

if [[ ! -f "$BASE_RUNNER" ]]; then
    echo "Missing base runner: $BASE_RUNNER" >&2
    exit 2
fi

exec bash "$BASE_RUNNER" \
    --TTF_module TTF_SemTime_Slots \
    --MMF_module MMF_VarTime_SlotGate \
    --semantic_slots 1 \
    --recency_sigma 0.25 \
    --semantic_time_gate_bias -2.0 \
    --mmf_slot_attn_dim 128 \
    --mmf_slot_gate_bias -2.0 \
    --kappa 0.1 \
    --batch-size 16 \
    "$@"
