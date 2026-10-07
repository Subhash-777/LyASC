#!/usr/bin/env bash
# run_table3_experiments.sh — Table 3 CIFAR-10 contracted skip experiments.
# Per plan.md Section 7 runbook step 5.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${BASE_DIR}/configs/cifar10_table3.yaml"
LOG_DIR="${BASE_DIR}/results/logs"
CKPT_DIR="${BASE_DIR}/results/checkpoints"
DEPTH=110
NUM_WORKERS="${NUM_WORKERS:-4}"

VARIANTS=(
    "baseline"
    "0.5xSkip"
    "LN(0.5x+F)"
    "LN(x+2F)"
    "LN(x+3F)"
)
SEEDS=(0 1 2 3 4)

echo "=== Table 3 experiments: ${#VARIANTS[@]} variants × ${#SEEDS[@]} seeds ==="
echo "Config: ${CONFIG}"

total_runs=$(( ${#VARIANTS[@]} * ${#SEEDS[@]} ))
run_num=0

for VARIANT in "${VARIANTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        run_num=$(( run_num + 1 ))
        echo "[${run_num}/${total_runs}] variant=${VARIANT}  seed=${SEED}"
        python -m src.training.train \
            --config "${CONFIG}" \
            --variant "${VARIANT}" \
            --seed "${SEED}" \
            --depth "${DEPTH}" \
            --log_dir "${LOG_DIR}" \
            --checkpoint_dir "${CKPT_DIR}" \
            --num_workers "${NUM_WORKERS}" \
            --base_dir "${BASE_DIR}"
    done
done

echo "=== Table 3 training complete. ==="
