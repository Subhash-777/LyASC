#!/usr/bin/env bash
# Phase C sequential training — LyASC variant, 3 seeds (0, 1, 2)
# Must be run from the lyasc_model/ directory.
# Uses the unmodified lyasc_table1_equiv.yaml (k=1, N=20, full 64k-iter schedule).
#
# Safety: uses a lockfile per seed so re-launching when already running
# exits immediately rather than spawning a duplicate training process.

set -euo pipefail

PYTHON=/home/subhash/miniconda3/envs/lyasc-experiments/bin/python
TRAIN_SCRIPT="$(dirname "$0")/../src/training/train.py"
CONFIG=configs/lyasc_table1_equiv.yaml
LOG_DIR=results/logs
CKPT_DIR=results/checkpoints
DATA_DIR="$(cd "$(dirname "$0")/../.." && pwd)/base_model/data/cifar10/cifar-10-batches-py"
LOCK_DIR=/tmp/lyasc_phase_c_locks
mkdir -p "$LOCK_DIR"

echo "========================================================"
echo "Phase C: LyASC training  —  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Config:  $CONFIG"
echo "Seeds:   0 1 2"
echo "========================================================"

for SEED in 0 1 2; do
    LOCKFILE="$LOCK_DIR/seed${SEED}.lock"
    RESULT_JSON="$LOG_DIR/lyasc_table1_equiv__LyASC__depth110__seed${SEED}.json"

    # Skip if result already written (train.py also checks, but fast-fail here too)
    if [ -f "$RESULT_JSON" ]; then
        echo "--- Seed $SEED already complete (result JSON exists), skipping. ---"
        continue
    fi

    # Lockfile guard: if lock exists and the PID inside is still alive, skip
    if [ -f "$LOCKFILE" ]; then
        LOCK_PID=$(cat "$LOCKFILE" 2>/dev/null || echo "")
        if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
            echo "--- Seed $SEED already running (PID $LOCK_PID), exiting to avoid duplicate. ---"
            echo "    To force restart: rm $LOCKFILE"
            exit 1
        else
            echo "--- Stale lockfile for seed $SEED (PID $LOCK_PID gone), removing and continuing. ---"
            rm -f "$LOCKFILE"
        fi
    fi

    echo ""
    echo "--- Seed $SEED  $(date -u '+%Y-%m-%dT%H:%M:%SZ') ---"

    # Write lockfile with our own PID
    echo "$$" > "$LOCKFILE"
    trap "rm -f '$LOCKFILE'" EXIT INT TERM

    $PYTHON "$TRAIN_SCRIPT" \
        --config  "$CONFIG" \
        --variant "LyASC"   \
        --seed    "$SEED"   \
        --log_dir "$LOG_DIR" \
        --checkpoint_dir "$CKPT_DIR" \
        --data_dir "$DATA_DIR" \
        --num_workers 4

    rm -f "$LOCKFILE"
    echo "--- Seed $SEED complete  $(date -u '+%Y-%m-%dT%H:%M:%SZ') ---"
done

echo ""
echo "========================================================"
echo "All seeds complete  —  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "========================================================"
