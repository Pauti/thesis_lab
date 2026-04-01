#!/bin/bash
# Setup nnU-Net v2 environment variables
# Usage: source scripts/setup_nnunet_env.sh

NNUNET_BASE="/data/paul_daten/nnunet"

export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_BASE}/nnUNet_results"

# Create directories if they don't exist
mkdir -p "$nnUNet_raw"
mkdir -p "$nnUNet_preprocessed"
mkdir -p "$nnUNet_results"

echo "nnU-Net environment variables set:"
echo "  nnUNet_raw=$nnUNet_raw"
echo "  nnUNet_preprocessed=$nnUNet_preprocessed"
echo "  nnUNet_results=$nnUNet_results"
