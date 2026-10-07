#!/usr/bin/env bash
# run_gradient_norm_analysis.sh — Reproduce Figure 3 using pretrained checkpoints.
# Per plan.md Section 7 runbook step 8.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CKPT_DIR="${BASE_DIR}/results/checkpoints"
OUT_DIR="${BASE_DIR}/results/figures"
LOG_DIR="${BASE_DIR}/results/logs"
# Data dir for CIFAR-10 (needed to load training images for gradient computation)
DATA_DIR="${BASE_DIR}/data/cifar10/cifar-10-batches-py"
NUM_WORKERS="${NUM_WORKERS:-4}"
N_SAMPLES="${N_SAMPLES:-10000}"

echo "=== Gradient norm analysis (Figure 3 reproduction) ==="
echo "Checkpoint dir: ${CKPT_DIR}"
echo "Output dir:     ${OUT_DIR}"
echo "Data dir:       ${DATA_DIR}"
echo "n_samples:      ${N_SAMPLES}"

mkdir -p "${OUT_DIR}"

python -m src.evaluation.gradient_norm_probe \
    --checkpoint_dir "${CKPT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --dataset cifar10 \
    --out_dir "${OUT_DIR}" \
    --n_samples "${N_SAMPLES}" \
    --base_dir "${BASE_DIR}"

echo "=== Figure 3 analysis complete. See ${OUT_DIR}/reproduced_figure3_grid.png ==="
