#!/bin/bash
source /home/subhash/miniconda3/etc/profile.d/conda.sh
conda activate lyasc-base
cd /home/subhash/projects/LyASC/base_model

for VARIANT in "2xSkip+BN" "3xSkip+BN" "1rSkip+BN" "2rSkip+BN" "3rSkip+BN"; do
    for SEED in 0 1 2; do
        # string replace for variant same as in train.py: replace("+", "plus").replace("/", "_").replace(" ", "_")
        SAFE_VARIANT=$(echo "$VARIANT" | sed -e 's/+/plus/g' -e 's/\//_/g' -e 's/ /_/g')
        JSON_FILE="results/logs/cifar10_table4__${SAFE_VARIANT}__depth110__seed${SEED}.json"
        
        if [ -f "$JSON_FILE" ]; then
            echo "Skipping already completed $VARIANT seed $SEED (found $JSON_FILE)"
            continue
        fi

        echo "Starting $VARIANT seed $SEED"
        python -m src.training.train \
            --config configs/cifar10_table4.yaml \
            --variant "$VARIANT" --seed $SEED --depth 110 \
            --log_dir results/logs --checkpoint_dir results/checkpoints \
            --num_workers 4 --base_dir .
    done
done
echo "TABLE4_LOOP_COMPLETE" >> results/logs/session_progress.log
