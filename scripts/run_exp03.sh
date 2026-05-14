#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_exp03.sh  —  EXP-03: UNETR 5-fold cross-validation on BraTS 2021
#
# Usage:
#   bash scripts/run_exp03.sh             # full training (200 epochs)
#   bash scripts/run_exp03.sh --debug     # debug run    (20 epochs, fold 0 only)
#   bash scripts/run_exp03.sh --fold 1    # single fold
#   bash scripts/run_exp03.sh --folds 0,1 # specific folds
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="${REPO_ROOT}/configs/exp03_unetr.yaml"
VENV="${REPO_ROOT}/venv"
# Logs + metrics + CSVs go to repo (lightweight, git-trackable)
EXP_BASE="${REPO_ROOT}/experiments/exp03_unetr"
# Checkpoints (~1.3 GB per .pth × 2 per fold × 5 folds ≈ 13 GB) go to /data
CKPT_BASE="/data/paul_daten/experiments/exp03_unetr"

DEBUG=false
NUM_EPOCHS=""       # empty = use config value (200)
FOLDS=(0 1 2 3 4)   # default: 5-fold (matching paper §4.3)

# ── Parse CLI arguments ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            DEBUG=true
            NUM_EPOCHS=20
            FOLDS=(0)
            shift ;;
        --num_epochs)
            NUM_EPOCHS="$2"; shift 2 ;;
        --fold)
            FOLDS=("$2"); shift 2 ;;
        --folds)
            IFS=',' read -ra FOLDS <<< "$2"; shift 2 ;;
        --config)
            CONFIG="$2"; shift 2 ;;
        --output_base)
            EXP_BASE="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Activate virtual environment ─────────────────────────────────────────────
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
else
    echo "WARNING: venv not found at ${VENV}. Using system Python."
fi

cd "$REPO_ROOT"

# ── Create experiment_metadata.json ──────────────────────────────────────────
mkdir -p "${EXP_BASE}" "${CKPT_BASE}"
EXP_START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%S+00:00")
python3 -c "
import json, torch, subprocess
meta = {
    'experiment_id': 'exp03_unetr',
    'timestamp_start': '${EXP_START_ISO}',
    'config': {
        'model': 'unetr',
        'architecture': 'ViT-B/16 + CNN decoder',
        'paper': 'Hatamizadeh et al., WACV 2022',
        'num_epochs': 200,
        'early_stopping_patience': 30,
        'batch_size': 2,
        'gradient_accumulation': 4,
        'effective_batch': 8,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'AdamW',
        'scheduler': 'cosine_warmup',
        'augmentation': 'medium',
        'patch_size': [128, 128, 128],
        'folds': '${FOLDS[*]}',
        'loss': 'DiceCE (include_background=False)'
    },
    'hardware': {
        'gpu': 'NVIDIA GeForce RTX 4080',
        'vram_gb': '16.0',
        'cuda_version': '12.4',
        'hostname': 'ca-crs-mth-dat24-02',
        'torch_version': torch.__version__,
        'monai_version': __import__('monai').__version__
    },
    'paths': {
        'raw_data': '/data/paul_daten/nnunet/nnUNet_raw/Dataset001_BraTS',
        'splits_file': '/data/paul_daten/nnunet/nnUNet_preprocessed/Dataset001_BraTS/splits_final.json',
        'experiment_dir': '${EXP_BASE}',
        'checkpoints_dir': '${CKPT_BASE}'
    }
}
with open('${EXP_BASE}/experiment_metadata.json', 'w') as f:
    json.dump(meta, f, indent=4)
" 2>/dev/null

# ── GPU info ──────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  EXP-03  UNETR BraTS 2021"
echo "  Config : ${CONFIG}"
echo "  Folds  : ${FOLDS[*]}"
echo "  Epochs : ${NUM_EPOCHS:-"from config (200)"}"
echo "  Debug  : ${DEBUG}"
echo "═══════════════════════════════════════════════════════════════"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
echo ""

# ── Start GPU monitor in background ──────────────────────────────────────────
GPU_LOG="${EXP_BASE}/gpu_monitor.csv"
mkdir -p "${EXP_BASE}"

if command -v python &>/dev/null && [[ -f "${SCRIPT_DIR}/monitor_gpu.py" ]]; then
    python "${SCRIPT_DIR}/monitor_gpu.py" --output "${GPU_LOG}" --interval 30 &
    GPU_MONITOR_PID=$!
    echo "GPU monitor started (PID ${GPU_MONITOR_PID})"
else
    GPU_MONITOR_PID=""
    echo "GPU monitor not started (monitor_gpu.py not found)"
fi

stop_gpu_monitor() {
    if [[ -n "${GPU_MONITOR_PID}" ]] && kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
        kill "${GPU_MONITOR_PID}"
        echo "GPU monitor stopped."
    fi
}
trap stop_gpu_monitor EXIT INT TERM

# ── Run each fold ─────────────────────────────────────────────────────────────
declare -A FOLD_RESULTS

for FOLD in "${FOLDS[@]}"; do
    FOLD_TAG="fold_${FOLD}"
    if [[ "${DEBUG}" == "true" ]]; then
        FOLD_TAG="fold_${FOLD}_debug"
    fi

    OUT_DIR="${EXP_BASE}/${FOLD_TAG}"
    mkdir -p "${OUT_DIR}"

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo "  Starting fold ${FOLD}  →  ${OUT_DIR}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "───────────────────────────────────────────────────────────────"

    # Build CLI arguments
    TRAIN_ARGS=(
        scripts/train.py
        --config "${CONFIG}"
        --fold "${FOLD}"
        --output_dir "${OUT_DIR}"
        --checkpoint_dir "${CKPT_BASE}/${FOLD_TAG}"
    )
    if [[ -n "${NUM_EPOCHS}" ]]; then
        TRAIN_ARGS+=(--num_epochs "${NUM_EPOCHS}")
    fi

    START_TS=$(date +%s)
    START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%S+00:00")
    python "${TRAIN_ARGS[@]}" 2>&1 | tee "${OUT_DIR}/training_stdout.log"
    FOLD_EXIT=${PIPESTATUS[0]}
    END_TS=$(date +%s)
    END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%S+00:00")
    ELAPSED=$(( END_TS - START_TS ))

    # Write fold_info.json (same as nnU-Net exp02 structure)
    python3 -c "
import json
data = {
    'fold': ${FOLD},
    'timestamp_start': '${START_ISO}',
    'timestamp_end': '${END_ISO}',
    'duration_seconds': ${ELAPSED},
    'duration_hours': round(${ELAPSED}/3600, 2),
    'exit_code': ${FOLD_EXIT},
    'output_dir': '${OUT_DIR}',
    'checkpoint_dir': '${CKPT_BASE}/${FOLD_TAG}'
}
with open('${OUT_DIR}/fold_info.json', 'w') as f:
    json.dump(data, f, indent=4)
" 2>/dev/null || true

    if [[ "${FOLD_EXIT}" -eq 0 ]]; then
        STATUS="OK"
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
echo "  EXP-03 Summary"
echo "═══════════════════════════════════════════════════════════════"
for FOLD in "${FOLDS[@]}"; do
    FOLD_TAG="fold_${FOLD}"
    if [[ "${DEBUG}" == "true" ]]; then
        FOLD_TAG="fold_${FOLD}_debug"
    fi
    FOLD_INFO="${EXP_BASE}/${FOLD_TAG}/fold_info.json"

    echo "  Fold ${FOLD}: ${FOLD_RESULTS[${FOLD}]}"
    if [[ -f "${FOLD_INFO}" ]]; then
        python3 -c "
import json
d = json.load(open('${FOLD_INFO}'))
best = d.get('best_val_dice_wt', 'N/A')
if isinstance(best, float): print(f\"    Best WT Dice: {best:.4f}\")
else: print(f\"    Best WT Dice: {best}\")
" 2>/dev/null || true
    fi
done
echo ""
echo "  All outputs : ${EXP_BASE}"
echo "  Checkpoints : ${CKPT_BASE}"
echo "═══════════════════════════════════════════════════════════════"
