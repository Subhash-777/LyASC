#!/usr/bin/env bash
# run_table6_experiments.sh — Table 6 CIFAR-100 multi-depth baseline vs 2rSkip+LN.
# Trains 5 depths × 2 variants × 5 seeds = 50 runs total.
# Per plan.md Section 7 runbook step 6.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${BASE_DIR}/configs/cifar100_table6.yaml"
LOG_DIR="${BASE_DIR}/results/logs"
CKPT_DIR="${BASE_DIR}/results/checkpoints"
NUM_WORKERS="${NUM_WORKERS:-4}"

DEPTHS=(20 32 44 56 110)
VARIANTS=("baseline" "2rSkip+LN")
SEEDS=(0 1 2)

echo "=== Table 6 experiments: ${#DEPTHS[@]} depths × ${#VARIANTS[@]} variants × ${#SEEDS[@]} seeds ==="
echo "Config: ${CONFIG}"

total_runs=$(( ${#DEPTHS[@]} * ${#VARIANTS[@]} * ${#SEEDS[@]} ))
run_num=0

for DEPTH in "${DEPTHS[@]}"; do
    for VARIANT in "${VARIANTS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            run_num=$(( run_num + 1 ))
            echo "[${run_num}/${total_runs}] depth=${DEPTH}  variant=${VARIANT}  seed=${SEED}"
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
done

echo "=== Table 6 training complete. ==="
