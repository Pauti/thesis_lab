#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_ablation.sh  —  RQ2 Augmentation Ablation (UNETR + nnU-Net)
#
# Runs a SINGLE ablation experiment (model + tier) across all 3 folds,
# with full tracking (GPU monitor, metadata, fold_info, timing) — same
# structure as run_exp03.sh.
#
# Usage:
#   bash scripts/run_ablation.sh --model unetr --tier default
#   bash scripts/run_ablation.sh --model nnunet --tier disabled
#   bash scripts/run_ablation.sh --model unetr --tier extended --fold 1
#   bash scripts/run_ablation.sh --model nnunet --tier default --debug
#
# Training order (confirmed 20.05.2026):
#   1. Default  → 2. Disabled → 3. Extended → 4. Reduced
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
TIER=""
SPLITS_FILE="/data/paul_daten/nnunet/nnUNet_preprocessed/Dataset001_BraTS/splits_ablation.json"

# nnU-Net specifics
DATASET_ID="001"
NNUNET_CONFIG="3d_fullres"
declare -A NNUNET_TRAINERS=(
    [disabled]="nnUNetTrainer_NoAug"
    [reduced]="nnUNetTrainer_LightAug"
    [default]="nnUNetTrainer_DefaultAug"
    [extended]="nnUNetTrainer_HeavyAug"
)

# ── Parse CLI arguments ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2"; shift 2 ;;
        --tier)       TIER="$2"; shift 2 ;;
        --fold)       FOLDS=("$2"); shift 2 ;;
        --folds)      IFS=',' read -ra FOLDS <<< "$2"; shift 2 ;;
        --num_epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --debug)      DEBUG=true; NUM_EPOCHS=5; FOLDS=(0); shift ;;
        *)            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate required args
if [[ -z "$MODEL" || -z "$TIER" ]]; then
    echo "Usage: $0 --model {nnunet|unetr} --tier {disabled|reduced|default|extended}"
    exit 1
fi
if [[ "$MODEL" != "nnunet" && "$MODEL" != "unetr" ]]; then
    echo "ERROR: --model must be 'nnunet' or 'unetr', got '$MODEL'"; exit 1
fi
if [[ ! "${NNUNET_TRAINERS[$TIER]+_}" ]]; then
    echo "ERROR: --tier must be one of: disabled reduced default extended"; exit 1
fi

# ── Activate environment ──────────────────────────────────────────────────────
if [[ -f "${VENV}/bin/activate" ]]; then
    source "${VENV}/bin/activate"
fi
source "${SCRIPT_DIR}/setup_nnunet_env.sh"
cd "$REPO_ROOT"

# ── Directories ───────────────────────────────────────────────────────────────
EXP_ID="abl_${MODEL}_${TIER}"
# Logs + metrics go to repo (lightweight, git-trackable)
EXP_BASE="${REPO_ROOT}/experiments/ablation/${EXP_ID}"
# Checkpoints (~1.3 GB per .pth) go to /data
CKPT_BASE="/data/paul_daten/experiments/ablation/${EXP_ID}"

mkdir -p "${EXP_BASE}" "${CKPT_BASE}"

# ── Experiment metadata ───────────────────────────────────────────────────────
EXP_START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%S+00:00")

if [[ "$MODEL" == "nnunet" ]]; then
    TRAINER="${NNUNET_TRAINERS[$TIER]}"
    MODEL_DESC="nnU-Net (3d_fullres, ${TRAINER})"
else
    MODEL_DESC="UNETR (ViT-B/16 + CNN decoder)"
    TRAINER="N/A"
fi

python3 -c "
import json, torch

debug_mode = '${DEBUG}' == 'true'

meta = {
    'experiment_id': '${EXP_ID}',
    'research_question': 'RQ2 - Augmentation Ablation',
    'timestamp_start': '${EXP_START_ISO}',
    'config': {
        'model': '${MODEL}',
        'model_description': '${MODEL_DESC}',
        'augmentation_tier': '${TIER}',
        'nnunet_trainer': '${TRAINER}',
        'num_epochs': ${NUM_EPOCHS},
        'folds': [$(echo "${FOLDS[*]}" | tr ' ' ',')],
        'splits_file': '${SPLITS_FILE}',
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
echo "  RQ2 Ablation: ${MODEL} / ${TIER}"
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
    echo "  Starting ${MODEL}/${TIER} fold ${FOLD}  →  ${OUT_DIR}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "───────────────────────────────────────────────────────────────"

    START_TS=$(date +%s)
    START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%S+00:00")

    if [[ "$MODEL" == "nnunet" ]]; then
        # ── nnU-Net training ──
        nnUNetv2_train ${DATASET_ID} ${NNUNET_CONFIG} ${FOLD} \
            -tr "${TRAINER}" \
            2>&1 | tee "${OUT_DIR}/training_stdout.log"
        FOLD_EXIT=${PIPESTATUS[0]}
    else
        # ── UNETR training ──
        # Generate tier-specific config (once per experiment)
        TIER_CONFIG="${EXP_BASE}/config_${TIER}.yaml"
        if [[ ! -f "$TIER_CONFIG" ]]; then
            sed "s/augmentation: \"default\"/augmentation: \"${TIER}\"/" \
                "${REPO_ROOT}/configs/exp03_unetr.yaml" > "$TIER_CONFIG"
        fi

        python scripts/train.py \
            --config "${TIER_CONFIG}" \
            --fold "${FOLD}" \
            --num_epochs "${NUM_EPOCHS}" \
            --output_dir "${OUT_DIR}" \
            --checkpoint_dir "${CKPT_DIR}" \
            --splits_file "${SPLITS_FILE}" \
            2>&1 | tee "${OUT_DIR}/training_stdout.log"
        FOLD_EXIT=${PIPESTATUS[0]}
    fi

    END_TS=$(date +%s)
    END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%S+00:00")
    ELAPSED=$(( END_TS - START_TS ))

    # Write fold_info.json
    python3 -c "
import json
data = {
    'experiment_id': '${EXP_ID}',
    'fold': ${FOLD},
    'tier': '${TIER}',
    'model': '${MODEL}',
    'timestamp_start': '${START_ISO}',
    'timestamp_end': '${END_ISO}',
    'duration_seconds': ${ELAPSED},
    'duration_hours': round(${ELAPSED}/3600, 2),
    'exit_code': ${FOLD_EXIT},
    'output_dir': '${OUT_DIR}',
    'checkpoint_dir': '${CKPT_DIR}'
}
with open('${OUT_DIR}/fold_info.json', 'w') as f:
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
echo "  Ablation ${MODEL}/${TIER} — Summary"
echo "═══════════════════════════════════════════════════════════════"
for FOLD in "${FOLDS[@]}"; do
    echo "  Fold ${FOLD}: ${FOLD_RESULTS[${FOLD}]}"
done
echo ""
echo "  Outputs     : ${EXP_BASE}"
echo "  Checkpoints : ${CKPT_BASE}"
echo "  GPU log     : ${GPU_LOG}"
echo "═══════════════════════════════════════════════════════════════"
