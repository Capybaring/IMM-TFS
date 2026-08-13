#!/usr/bin/env bash
# Run the full comparison sweep: every model in this framework, numeric-only
# vs +text, using the run_<model>.sh scripts.
#
# NOTE (2026-08-08): compute_text_embeddings.py had a bug where note
# timestamps were always converted assuming time_unit="days" (hardcoded
# /86400.0) and referenced to text.csv's own earliest note instead of the
# record's time_series.csv earliest reading — wrong for MIMIC (time_unit=
# "hours") both in scale (~24x) and in reference point (notes were shifted
# by the offset between the first note and the first vitals reading, often
# a few hours). This has been fixed for the MIMIC pipeline specifically:
# _common.sh now passes time_unit=$TIME_UNIT and align_base_to_series=True
# into compute_text_embeddings(). The function's own defaults (time_unit=
# "days", align_base_to_series=False) are untouched, so other datasets'
# embedding generation (e.g. via the standalone __main__ block in
# compute_text_embeddings.py) is unaffected until reviewed separately.
#
# IMPORTANT: if you previously generated MIMIC text embeddings before this
# fix, delete the stale .pt files first — compute_text_embeddings() skips
# regeneration when the output file already exists:
#   rm -f data/MIMIC/processed/*/text_embeddings_*.pt
#
# Usage:
#   ./run_all_comparison.sh                       # default models, uni + multi
#   ./run_all_comparison.sh GPINet tPatchGNN       # just these models
#   ./run_all_comparison.sh --smoke                # fast sanity sweep first
#   ./run_all_comparison.sh --smoke GPINet         # smoke-test just one
#   ./run_all_comparison.sh CRU LatentODE          # explicitly run the slow ones
#
# Model names are matched case-insensitively to run_<model_lowercased>.sh.
#
# NOTE (2026-08-08): CRU, LatentODE, and NeuralFlow are excluded from the
# default sweep (ALL_MODELS below) — all three are ODE-solver-based and much
# slower to train than everything else here (LatentODE in particular may not
# finish in reasonable time on MIMIC's irregular long sequences). Pass them
# by name explicitly (see usage above) if you want them included.
#
# Output goes to scripts/results/<timestamp>/:
#   <model>_uni.log, <model>_multi.log   — full stdout/stderr of each run
#   summary.csv                          — Model,Modal,MSE,MAE, same shape
#                                           as the paper's appendix table,
#                                           for direct side-by-side diffing.
#
# Env var overrides (GPU, DATASET, EPOCH, ...) are the same ones each
# run_<model>.sh already supports — see _common.sh. They apply to every
# model in the sweep, not per-model.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ALL_MODELS=(GPINet tPatchGNN DLinear PatchTST Informer TimesNet TimeMixer TimeLLM TTM)

SMOKE_FLAG=()
MODELS=()
for a in "$@"; do
    case "$a" in
        --smoke) SMOKE_FLAG=(--smoke) ;;
        *) MODELS+=("$a") ;;
    esac
done
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=("${ALL_MODELS[@]}")
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$SCRIPT_DIR/results/$TS"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.csv"
echo "Model,Modal,MSE,MAE" > "$SUMMARY"
echo "### Logs and summary going to $OUT_DIR ###"

parse_and_record() {
    local model="$1" modal="$2" log="$3"
    local mse mae
    mse="$(grep -E '^mse:' "$log" | tail -1 | awk '{print $2}')"
    mae="$(grep -E '^mae:' "$log" | tail -1 | awk '{print $2}')"
    if [ -z "$mse" ] || [ -z "$mae" ]; then
        echo "[WARN] Could not parse mse/mae for $model/$modal from $log — check the log." >&2
        mse="${mse:-NA}"
        mae="${mae:-NA}"
    fi
    echo "$model,$modal,$mse,$mae" >> "$SUMMARY"
}

for model in "${MODELS[@]}"; do
    script_name="run_$(echo "$model" | tr '[:upper:]' '[:lower:]').sh"
    if [ ! -f "$script_name" ]; then
        echo "[SKIP] no $script_name for model '$model'" >&2
        continue
    fi

    echo "=================================================================="
    echo "### $model — Uni (numeric-only) ###"
    echo "=================================================================="
    uni_log="$OUT_DIR/${model}_uni.log"
    if bash "./$script_name" "${SMOKE_FLAG[@]}" 2>&1 | tee "$uni_log"; then
        parse_and_record "$model" "Uni" "$uni_log"
    else
        echo "[FAIL] $model uni run failed — see $uni_log" >&2
        echo "$model,Uni,FAILED,FAILED" >> "$SUMMARY"
    fi

    echo "=================================================================="
    echo "### $model — Multi (+text) ###"
    echo "=================================================================="
    multi_log="$OUT_DIR/${model}_multi.log"
    if bash "./$script_name" "${SMOKE_FLAG[@]}" --text 2>&1 | tee "$multi_log"; then
        parse_and_record "$model" "Multi" "$multi_log"
    else
        echo "[FAIL] $model multi run failed — see $multi_log" >&2
        echo "$model,Multi,FAILED,FAILED" >> "$SUMMARY"
    fi
done

echo
echo "### Done. Summary: $SUMMARY ###"
column -s, -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
