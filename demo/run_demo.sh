#!/usr/bin/env bash
# LyASC CPU Demo Inference Launcher
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate conda environment or use python path directly
PYTHON_EXE="/home/subhash/miniconda3/envs/lyasc-experiments/bin/python"
if [ ! -f "${PYTHON_EXE}" ]; then
    echo "Error: Python executable not found at ${PYTHON_EXE}"
    exit 1
fi

export CONDA_DEFAULT_ENV="lyasc-experiments"
cd "${WORKSPACE_ROOT}"

"${PYTHON_EXE}" demo/demo_inference.py "$@"
