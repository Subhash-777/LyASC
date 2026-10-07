#!/usr/bin/env bash
# run_table1_experiments.sh — Table 1 CIFAR-10 PreAct-ResNet-110 experiments.
# Loops over every variant × seed defined in configs/cifar10_table1.yaml and
# invokes src/training/train.py for each combination.
# Per plan.md Section 7 runbook step 4.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${BASE_DIR}/configs/cifar10_table1.yaml"
LOG_DIR="${BASE_DIR}/results/logs"
CKPT_DIR="${BASE_DIR}/results/checkpoints"
DEPTH=110
NUM_WORKERS="${NUM_WORKERS:-4}"

# All variants defined in cifar10_table1.yaml
VARIANTS=(
    "1xSkip"
    "2xSkip"
    "3xSkip"
    "1xSkip+LN"
    "2xSkip+LN"
    "3xSkip+LN"
    "1rSkip+LN"
    "2rSkip+LN"
    "3rSkip+LN"
    "wSkip+LN"
)
SEEDS=(0 1 2 3 4)

echo "=== Table 1 experiments: ${#VARIANTS[@]} variants × ${#SEEDS[@]} seeds ==="
echo "Config:      ${CONFIG}"
echo "Log dir:     ${LOG_DIR}"
echo "Checkpoint:  ${CKPT_DIR}"
echo ""

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

echo ""
echo "=== Table 1 training complete. Run reproduce_tables.py to aggregate. ==="
