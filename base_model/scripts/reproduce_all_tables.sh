#!/usr/bin/env bash
# reproduce_all_tables.sh — Run evaluate/reproduce_tables.py for all four
# configs and emit the four CSV files + PASS/FAIL summaries.
# Run this AFTER all training experiments have completed.
# Per plan.md Section 7 runbook step 7.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
TABLES_DIR="${BASE_DIR}/results/tables"

mkdir -p "${TABLES_DIR}"

echo "=== Reproducing Table 1 ==="
python -m src.evaluation.reproduce_tables \
    --config "${BASE_DIR}/configs/cifar10_table1.yaml" \
    --out "${TABLES_DIR}/reproduced_table1.csv" \
    --log_dir "${BASE_DIR}/results/logs" \
    --base_dir "${BASE_DIR}"

echo "=== Reproducing Table 3 ==="
python -m src.evaluation.reproduce_tables \
    --config "${BASE_DIR}/configs/cifar10_table3.yaml" \
    --out "${TABLES_DIR}/reproduced_table3.csv" \
    --log_dir "${BASE_DIR}/results/logs" \
    --base_dir "${BASE_DIR}"

echo "=== Reproducing Table 4 ==="
python -m src.evaluation.reproduce_tables \
    --config "${BASE_DIR}/configs/cifar10_table4.yaml" \
    --out "${TABLES_DIR}/reproduced_table4.csv" \
    --log_dir "${BASE_DIR}/results/logs" \
    --base_dir "${BASE_DIR}"

echo "=== Reproducing Table 6 ==="
python -m src.evaluation.reproduce_tables \
    --config "${BASE_DIR}/configs/cifar100_table6.yaml" \
    --out "${TABLES_DIR}/reproduced_table6.csv" \
    --log_dir "${BASE_DIR}/results/logs" \
    --base_dir "${BASE_DIR}"

echo ""
echo "=== All tables reproduced. CSVs in ${TABLES_DIR}/ ==="
ls "${TABLES_DIR}/"
