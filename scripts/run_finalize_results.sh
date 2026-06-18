#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_finalize_results.sh  —  finish all remaining UNETR HD95 + consolidate
#
# Run this AFTER exp07 (UNETR global MTL) training has finished. It executes the
# whole remaining GPU queue sequentially, then the CPU analysis, so the results
# section has every metric. Each step is independent (a failure is logged and the
# queue continues); the consolidation + significance run at the end regardless.
#
# GPU steps (in order):
#   1. exp07 test inference + HD95   (UNETR global MTL, RQ3)
#   2. exp05 test HD95 re-inference  (UNETR per-patch MTL, RQ3)
#   3. abl_unetr ×4 test HD95        (RQ2)
#   4. exp03 RQ1 CV HD95             (UNETR baseline, 5-fold CV)
# CPU steps:
#   5. consolidate_results.py + significance (RQ1/RQ2 + RQ3)
#
# exp02 RQ1 HD95 is already done (from disk, no GPU).
#
# Usage (in tmux):
#   bash scripts/run_finalize_results.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail   # NOT -e: we want the queue to continue past a failed step

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="${REPO_ROOT}/venv"
LOG_DIR="${REPO_ROOT}/experiments/_finalize_logs"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"
[[ -f "${VENV}/bin/activate" ]] && source "${VENV}/bin/activate"
[[ -f "${SCRIPT_DIR}/setup_nnunet_env.sh" ]] && source "${SCRIPT_DIR}/setup_nnunet_env.sh"

run_step() {
    local name="$1"; shift
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ${name}   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "═══════════════════════════════════════════════════════════════"
    if "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"; then
        echo "  ✓ ${name} OK"
    else
        echo "  ✗ ${name} FAILED (exit ${PIPESTATUS[0]}) — continuing"
    fi
}

# ── 1. exp07 test inference + HD95 ───────────────────────────────────────────
# Guard: all 3 folds must be DONE. best_model.pth exists mid-training (running
# best), so checking one fold is not enough; require run_info.json (written by
# run_multitask.sh only after a fold finishes) for all 3 folds.
EXP07_BASE="${REPO_ROOT}/experiments/exp07_unetr_mtl_global"
exp07_done=true
for f in 0 1 2; do
    [[ -f "${EXP07_BASE}/fold_${f}/run_info.json" ]] || exp07_done=false
done
if [[ "$exp07_done" == true ]]; then
    run_step "infer_exp07" python scripts/infer_exp07_unetr_mtl_global.py
else
    echo "SKIP exp07 inference: not all 3 folds finished yet (run_info.json missing)."
    echo "  Re-run this script once exp07 training is complete."
fi

# ── 2. exp05 test HD95 re-inference ──────────────────────────────────────────
run_step "infer_exp05" python scripts/infer_exp05_unetr_mtl.py

# ── 3. abl_unetr ×4 test HD95 re-inference ───────────────────────────────────
for tier in disabled reduced default extended; do
    run_step "infer_abl_unetr_${tier}" python scripts/infer_ablation_unetr.py --tier "$tier"
done

# ── 4. exp03 RQ1 CV HD95 re-inference ────────────────────────────────────────
run_step "infer_exp03_rq1" python scripts/infer_exp03.py

# ── 5. CPU: consolidate + significance ───────────────────────────────────────
run_step "consolidate" python scripts/consolidate_results.py
run_step "significance_mtl" python scripts/analyze_mtl_significance.py
run_step "significance_rq1_rq2" python scripts/analyze_significance_rq1_rq2.py

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  DONE. Master table: experiments/results_master.csv"
echo "  Step logs: ${LOG_DIR}/"
echo "═══════════════════════════════════════════════════════════════"
python scripts/consolidate_results.py --status 2>&1 | tail -20
