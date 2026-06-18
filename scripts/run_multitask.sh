#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_multitask.sh  —  RQ3 Multi-Task Learning (UNETR + nnU-Net)
#
# Runs a SINGLE multi-task experiment (model) across the chosen folds, with full
# tracking (GPU monitor, metadata, fold_info, timing) — same structure as
# run_ablation.sh. Segmentation + auxiliary ET-volume ordinal classification.
#
# Usage:
#   bash scripts/run_multitask.sh --model unetr                 # EXP-05, folds 0,1,2
#   bash scripts/run_multitask.sh --model unetr --fold 1
#   bash scripts/run_multitask.sh --model unetr --folds 0,2
#   bash scripts/run_multitask.sh --model unetr --debug          # 2 epochs, fold 0
#   bash scripts/run_multitask.sh --model nnunet                 # EXP-04 (v1, per-patch)
#   bash scripts/run_multitask.sh --model nnunet --variant v2    # EXP-06 (per-case global)
#   bash scripts/run_multitask.sh --model nnunet --variant v3    # EXP-08 (global + FiLM + learned weighting)
#   bash scripts/run_multitask.sh --model nnunet --variant v3 --debug   # 2-epoch sanity run
#
# Comparison baselines: ABL-07 (abl_unetr_default) / ABL-03 (abl_nnunet_default)
# — same splits, same protocol, seg-only. Only the auxiliary head differs.
#
# Splits: splits_ablation.json (3-fold CV, 1126 train, 125 test held out)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

VENV="${REPO_ROOT}/venv"
NUM_EPOCHS=100
FOLDS=(0 1 2)
DEBUG=false
MODEL=""
SPLITS_FILE="/data/paul_daten/nnunet/nnUNet_preprocessed/Dataset001_BraTS/splits_ablation.json"

# ── Parse CLI arguments ───────────────────────────────────────────────────────
VARIANT="v1"   # nnU-Net only: v1=EXP-04 (per-patch ET), v2=EXP-06 (per-case global ET)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2"; shift 2 ;;
        --variant)    VARIANT="$2"; shift 2 ;;
        --fold)       FOLDS=("$2"); shift 2 ;;
        --folds)      IFS=',' read -ra FOLDS <<< "$2"; shift 2 ;;
        --num_epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --debug)      DEBUG=true; NUM_EPOCHS=2; FOLDS=(0); shift ;;
        *)            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate required args
if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 --model {unetr|nnunet} [--variant {v1|v2|v3}] [--folds 0,1,2] [--num_epochs N] [--debug]"
    echo "       (--variant v3 = EXP-08, nnU-Net only: global target + FiLM + learned weighting)"
    exit 1
fi
if [[ "$MODEL" != "nnunet" && "$MODEL" != "unetr" ]]; then
    echo "ERROR: --model must be 'unetr' or 'nnunet', got '$MODEL'"; exit 1
fi
if [[ "$VARIANT" != "v1" && "$VARIANT" != "v2" && "$VARIANT" != "v3" ]]; then
    echo "ERROR: --variant must be 'v1', 'v2' or 'v3', got '$VARIANT'"; exit 1
fi
if [[ "$VARIANT" == "v3" && "$MODEL" != "nnunet" ]]; then
    echo "ERROR: --variant v3 (EXP-08) is implemented for nnU-Net only"; exit 1
fi
# v1 = per-patch ET target (EXP-04/05); v2 = per-case global ET tertile (EXP-06/07);
# v3 = v2 target + FiLM feature fusion + learned (uncertainty) loss weighting (EXP-08)

# ── nnU-Net branch: not yet implemented (intentional guard) ──────────────────
# ── Activate environment ──────────────────────────────────────────────────────
if [[ -f "${VENV}/bin/activate" ]]; then
    source "${VENV}/bin/activate"
fi
# nnU-Net needs its env vars (nnUNet_raw/preprocessed/results) for training.
if [[ "$MODEL" == "nnunet" && -f "${SCRIPT_DIR}/setup_nnunet_env.sh" ]]; then
    source "${SCRIPT_DIR}/setup_nnunet_env.sh"
fi
cd "$REPO_ROOT"

# nnU-Net multi-task specifics (trainer subclass added in models/nnunet_trainers.py)
DATASET_ID="001"
NNUNET_CONFIG="3d_fullres"
case "$VARIANT" in
    v3) MTL_TRAINER="nnUNetTrainer_MTL_v3" ;;
    v2) MTL_TRAINER="nnUNetTrainer_MTL_RawImages" ;;
    *)  MTL_TRAINER="nnUNetTrainer_MTL" ;;
esac

# v2 (EXP-06/07) and v3 (EXP-08) need per-case ET tertile labels; generate once if missing.
if [[ ( "$VARIANT" == "v2" || "$VARIANT" == "v3" ) && ! -f "${REPO_ROOT}/experiments/et_tertiles_ablation.json" ]]; then
    echo "Pre-computing per-case ET tertiles for ${VARIANT}..."
    python "${SCRIPT_DIR}/precompute_et_tertiles.py"
fi

# ── Directories ───────────────────────────────────────────────────────────────
if [[ "$MODEL" == "nnunet" ]]; then
    case "$VARIANT" in
        v3) EXP_ID="exp08_nnunet_mtl_v3" ;;
        v2) EXP_ID="exp06_nnunet_mtl_global" ;;
        *)  EXP_ID="exp04_nnunet_mtl" ;;
    esac
else
    if [[ "$VARIANT" == "v2" ]]; then
        EXP_ID="exp07_unetr_mtl_global"
    else
        EXP_ID="exp05_unetr_mtl"
    fi
fi
CONFIG="${REPO_ROOT}/configs/exp05_unetr_mtl.yaml"
# Logs + metrics go to repo (lightweight, git-trackable)
EXP_BASE="${REPO_ROOT}/experiments/${EXP_ID}"
# Checkpoints (~1.3 GB per .pth) go to /data
CKPT_BASE="/data/paul_daten/experiments/${EXP_ID}"

mkdir -p "${EXP_BASE}" "${CKPT_BASE}"

# ── Experiment metadata ───────────────────────────────────────────────────────
EXP_START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%S+00:00")
if [[ "$MODEL" == "nnunet" ]]; then
    MODEL_DESC="nnU-Net (3d_fullres, ${MTL_TRAINER}) + aux ET-volume ordinal head (CORAL)"
else
    MODEL_DESC="UNETR (ViT-B/16 + CNN decoder) + aux ET-volume ordinal head (CORAL)"
fi
# v3 (EXP-08) adds two mechanisms on top of the v2 global target; lambda_aux is
# no longer a fixed scalar but learned via homoscedastic uncertainty weighting.
LAMBDA_AUX_JSON="0.3"
AUX_BANNER="ET-volume ordinal classification (CORAL, λ_aux=0.3)"
if [[ "$VARIANT" == "v3" ]]; then
    MODEL_DESC="${MODEL_DESC} + FiLM feature fusion + learned loss weighting (Kendall 2018)"
    LAMBDA_AUX_JSON='"learned (uncertainty weighting)"'
    AUX_BANNER="ET-volume ordinal (CORAL) + FiLM fusion + learned loss weighting"
fi

python3 -c "
import json, torch
debug_mode = '${DEBUG}' == 'true'
meta = {
    'experiment_id': '${EXP_ID}',
    'research_question': 'RQ3 - Multi-Task Learning (ET-volume ordinal classification)',
    'timestamp_start': '${EXP_START_ISO}',
    'config': {
        'model': '${MODEL}',
        'model_description': '${MODEL_DESC}',
        'aux_task': 'ET-volume ordinal classification (3 tertiles, CORAL)',
        'lambda_seg': 1.0,
        'lambda_aux': ${LAMBDA_AUX_JSON},
        'num_epochs': ${NUM_EPOCHS},
        'folds': [$(echo "${FOLDS[*]}" | tr ' ' ',')],
        'splits_file': '${SPLITS_FILE}',
        'baseline_comparison': 'abl_unetr_default (ABL-07)',
        'test_cases': 125,
        'debug': debug_mode
    },
    'hardware': {
        'gpu': '$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)',
        'vram_gb': '$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo unknown)',
        'hostname': '$(hostname)',
        'torch_version': torch.__version__,
    },
    'paths': {
        'experiment_dir': '${EXP_BASE}',
        'checkpoints_dir': '${CKPT_BASE}',
        'splits_file': '${SPLITS_FILE}'
    }
}
with open('${EXP_BASE}/experiment_metadata.json', 'w') as f:
    json.dump(meta, f, indent=4)
print('Metadata saved.')
" 2>&1

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  RQ3 Multi-Task: ${MODEL}  (${EXP_ID})"
echo "  Aux    : ${AUX_BANNER}"
echo "  Folds  : ${FOLDS[*]}"
echo "  Epochs : ${NUM_EPOCHS}"
echo "  Splits : ${SPLITS_FILE}"
echo "  Output : ${EXP_BASE}"
echo "  Ckpts  : ${CKPT_BASE}"
echo "  Debug  : ${DEBUG}"
echo "═══════════════════════════════════════════════════════════════"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
echo ""

# ── Start GPU monitor ─────────────────────────────────────────────────────────
GPU_LOG="${EXP_BASE}/gpu_monitor.csv"
GPU_MONITOR_PID=""
if [[ -f "${SCRIPT_DIR}/monitor_gpu.py" ]]; then
    python "${SCRIPT_DIR}/monitor_gpu.py" --output "${GPU_LOG}" --interval 30 &
    GPU_MONITOR_PID=$!
    echo "GPU monitor started (PID ${GPU_MONITOR_PID})"
fi
stop_gpu_monitor() {
    if [[ -n "${GPU_MONITOR_PID}" ]] && kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
        kill "${GPU_MONITOR_PID}" 2>/dev/null
        wait "${GPU_MONITOR_PID}" 2>/dev/null || true
        echo "GPU monitor stopped."
    fi
}
trap stop_gpu_monitor EXIT INT TERM

# ── Run folds ─────────────────────────────────────────────────────────────────
declare -A FOLD_RESULTS

for FOLD in "${FOLDS[@]}"; do
    FOLD_TAG="fold_${FOLD}"
    OUT_DIR="${EXP_BASE}/${FOLD_TAG}"
    CKPT_DIR="${CKPT_BASE}/${FOLD_TAG}"
    mkdir -p "${OUT_DIR}" "${CKPT_DIR}"

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo "  Starting ${MODEL} multi-task fold ${FOLD}  →  ${OUT_DIR}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "───────────────────────────────────────────────────────────────"

    START_TS=$(date +%s)
    START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%S+00:00")

    set +e
    if [[ "$MODEL" == "nnunet" ]]; then
        # nnU-Net manages its own output/checkpoint dirs under nnUNet_results.
        # The MTL trainer reads splits_ablation.json + 100 epochs (inherited from
        # nnUNetTrainer_Ablation). --debug shortens epochs via the env override.
        MTL_NUM_EPOCHS="${NUM_EPOCHS}" \
        nnUNetv2_train ${DATASET_ID} ${NNUNET_CONFIG} ${FOLD} \
            -tr "${MTL_TRAINER}" \
            2>&1 | tee "${OUT_DIR}/training_stdout.log"
        FOLD_EXIT=${PIPESTATUS[0]}
    else
        AUX_TARGET="patch"; [[ "$VARIANT" == "v2" ]] && AUX_TARGET="global"
        python scripts/train_multitask.py \
            --config "${CONFIG}" \
            --fold "${FOLD}" \
            --num_epochs "${NUM_EPOCHS}" \
            --output_dir "${OUT_DIR}" \
            --checkpoint_dir "${CKPT_DIR}" \
            --splits_file "${SPLITS_FILE}" \
            --aux_target "${AUX_TARGET}" \
            2>&1 | tee "${OUT_DIR}/training_stdout.log"
        FOLD_EXIT=${PIPESTATUS[0]}
    fi
    set -e

    END_TS=$(date +%s)
    END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%S+00:00")
    ELAPSED=$(( END_TS - START_TS ))

    python3 -c "
import json
data = {
    'experiment_id': '${EXP_ID}',
    'fold': ${FOLD},
    'model': '${MODEL}',
    'aux_task': 'ET-volume ordinal (CORAL)',
    'timestamp_start': '${START_ISO}',
    'timestamp_end': '${END_ISO}',
    'duration_seconds': ${ELAPSED},
    'duration_hours': round(${ELAPSED}/3600, 2),
    'exit_code': ${FOLD_EXIT},
    'output_dir': '${OUT_DIR}',
    'checkpoint_dir': '${CKPT_DIR}'
}
with open('${OUT_DIR}/run_info.json', 'w') as f:
    json.dump(data, f, indent=4)
" 2>/dev/null || true

    if [[ "${FOLD_EXIT}" -eq 0 ]]; then
        STATUS="OK ($(( ELAPSED / 3600 ))h$(( (ELAPSED % 3600) / 60 ))m)"
        echo "  ✓ Fold ${FOLD} completed in $(( ELAPSED / 3600 ))h$(( (ELAPSED % 3600) / 60 ))m"
    else
        STATUS="FAILED (exit ${FOLD_EXIT})"
        echo "  ✗ Fold ${FOLD} FAILED after $(( ELAPSED / 60 ))m"
    fi
    FOLD_RESULTS["${FOLD}"]="${STATUS}"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Multi-Task ${MODEL} — Summary"
echo "═══════════════════════════════════════════════════════════════"
for FOLD in "${FOLDS[@]}"; do
    echo "  Fold ${FOLD}: ${FOLD_RESULTS[${FOLD}]}"
done
echo ""
echo "  Outputs     : ${EXP_BASE}"
echo "  Checkpoints : ${CKPT_BASE}"
echo "  GPU log     : ${GPU_LOG}"
echo "═══════════════════════════════════════════════════════════════"
