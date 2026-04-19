#!/bin/bash
# =============================================================================
# EXP-02: nnU-Net Baseline Training (5-Fold CV)
#
# Runs standard nnUNetTrainer on Dataset001_BraTS with 3d_fullres config.
# Includes GPU/system monitoring and automatic log parsing after each fold.
#
# Usage:
#   bash scripts/run_exp02.sh          # Run all 5 folds (0-4)
#   bash scripts/run_exp02.sh 0 2      # Run folds 0, 1, 2
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Configuration ---
DATASET_ID="001"
CONFIG="3d_fullres"
TRAINER="nnUNetTrainer"  # Standard trainer = Medium augmentation
EXPERIMENT_NAME="exp02_nnunet"
EXPERIMENT_DIR="${PROJECT_DIR}/experiments/${EXPERIMENT_NAME}"
GPU_MONITOR_INTERVAL=30  # seconds

# Folds to run (default: 0-4, override via args)
FOLD_START=${1:-0}
FOLD_END=${2:-4}

# --- Setup ---
source "${SCRIPT_DIR}/setup_nnunet_env.sh"
source "${PROJECT_DIR}/venv/bin/activate"

mkdir -p "${EXPERIMENT_DIR}"

# Results base dir (where nnU-Net writes)
NNUNET_OUT="${nnUNet_results}/Dataset001_BraTS/${TRAINER}__nnUNetPlans__${CONFIG}"

# --- Log helper ---
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${EXPERIMENT_DIR}/experiment.log"
}

# --- Start GPU Monitor ---
GPU_LOG="${EXPERIMENT_DIR}/gpu_monitor.csv"
log "Starting GPU monitor → ${GPU_LOG}"
python "${SCRIPT_DIR}/monitor_gpu.py" \
    --output "${GPU_LOG}" \
    --interval ${GPU_MONITOR_INTERVAL} &
GPU_MONITOR_PID=$!
log "GPU monitor PID: ${GPU_MONITOR_PID}"

# Ensure monitor stops on exit
cleanup() {
    log "Stopping GPU monitor (PID ${GPU_MONITOR_PID})..."
    touch "${GPU_LOG%.csv}.stop"
    wait ${GPU_MONITOR_PID} 2>/dev/null || true
    log "Cleanup complete."
}
trap cleanup EXIT

# --- Record experiment metadata ---
cat > "${EXPERIMENT_DIR}/experiment_metadata.json" << EOF
{
    "experiment_id": "${EXPERIMENT_NAME}",
    "timestamp_start": "$(date -Iseconds)",
    "config": {
        "model": "nnunet",
        "trainer": "${TRAINER}",
        "architecture": "${CONFIG}",
        "dataset_id": "${DATASET_ID}",
        "multi_task": false,
        "augmentation": "medium (nnU-Net default)",
        "num_epochs": 1000,
        "folds": "$(seq -s ',' ${FOLD_START} ${FOLD_END})"
    },
    "hardware": {
        "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')",
        "vram_gb": "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{printf "%.1f", $1/1024}' || echo 'unknown')",
        "cuda_version": "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'unknown')",
        "hostname": "$(hostname)",
        "python_version": "$(python --version 2>&1)",
        "torch_version": "$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'unknown')",
        "nnunet_version": "$(python -c 'import nnunetv2; print(nnunetv2.__version__)' 2>/dev/null || echo 'unknown')"
    },
    "paths": {
        "nnunet_raw": "${nnUNet_raw}",
        "nnunet_preprocessed": "${nnUNet_preprocessed}",
        "nnunet_results": "${nnUNet_results}",
        "experiment_dir": "${EXPERIMENT_DIR}"
    }
}
EOF
log "Metadata saved to ${EXPERIMENT_DIR}/experiment_metadata.json"

# --- Training Loop ---
log "=========================================="
log "EXP-02: nnU-Net ${CONFIG} — Folds ${FOLD_START}–${FOLD_END}"
log "=========================================="

for FOLD in $(seq ${FOLD_START} ${FOLD_END}); do
    FOLD_DIR="${EXPERIMENT_DIR}/fold_${FOLD}"
    mkdir -p "${FOLD_DIR}"

    log "------------------------------------------"
    log "Starting Fold ${FOLD}/$(( FOLD_END )) at $(date)"
    log "------------------------------------------"

    FOLD_START_TIME=$(date +%s)

    # Run nnU-Net training
    nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} 2>&1 | tee "${FOLD_DIR}/training_stdout.log"
    TRAIN_EXIT=$?

    FOLD_END_TIME=$(date +%s)
    FOLD_DURATION=$(( FOLD_END_TIME - FOLD_START_TIME ))
    FOLD_HOURS=$(python3 -c "print(f'{${FOLD_DURATION}/3600:.2f}')")

    if [ ${TRAIN_EXIT} -eq 0 ]; then
        log "Fold ${FOLD} completed successfully in ${FOLD_HOURS}h"
    else
        log "ERROR: Fold ${FOLD} failed with exit code ${TRAIN_EXIT}"
        continue
    fi

    # Parse nnU-Net log into metrics.csv
    NNUNET_LOG=$(ls -t "${NNUNET_OUT}/fold_${FOLD}"/training_log_*.txt 2>/dev/null | head -1)
    if [ -n "${NNUNET_LOG}" ]; then
        log "Parsing training log: ${NNUNET_LOG}"
        python "${SCRIPT_DIR}/parse_nnunet_log.py" \
            --log_file "${NNUNET_LOG}" \
            --output "${FOLD_DIR}/metrics.csv"
    else
        log "WARNING: No training log found for fold ${FOLD}"
    fi

    # Save fold timing info
    cat > "${FOLD_DIR}/fold_info.json" << FEOF
{
    "fold": ${FOLD},
    "timestamp_start": "$(date -d @${FOLD_START_TIME} -Iseconds)",
    "timestamp_end": "$(date -d @${FOLD_END_TIME} -Iseconds)",
    "duration_seconds": ${FOLD_DURATION},
    "duration_hours": ${FOLD_HOURS},
    "exit_code": ${TRAIN_EXIT},
    "nnunet_output_dir": "${NNUNET_OUT}/fold_${FOLD}"
}
FEOF

    log "Fold ${FOLD} results saved to ${FOLD_DIR}/"
done

# --- Final Summary ---
log "=========================================="
log "EXP-02 COMPLETE at $(date)"
log "=========================================="
log "Results directory: ${EXPERIMENT_DIR}"
log "nnU-Net results:   ${NNUNET_OUT}"
log ""
log "Per-fold metrics CSVs:"
for FOLD in $(seq ${FOLD_START} ${FOLD_END}); do
    if [ -f "${EXPERIMENT_DIR}/fold_${FOLD}/metrics.csv" ]; then
        LAST_LINE=$(tail -1 "${EXPERIMENT_DIR}/fold_${FOLD}/metrics.csv")
        log "  Fold ${FOLD}: ${LAST_LINE}"
    fi
done

# Update metadata with end time
python -c "
import json
from datetime import datetime
p = '${EXPERIMENT_DIR}/experiment_metadata.json'
with open(p) as f: d = json.load(f)
d['timestamp_end'] = datetime.now().isoformat()
with open(p, 'w') as f: json.dump(d, f, indent=4)
"

log "Done. GPU monitor log: ${GPU_LOG}"
