#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lyasc-base

echo "Starting unattended queue..." > results/logs/session_progress.log

# Table 3 remainder
echo "RUNNING TABLE 3 REMAINDER" >> results/logs/session_progress.log
for VARIANT in "LN(x+2F)" "LN(x+3F)"; do
    for SEED in 0 1 2; do
        python -m src.training.train \
            --config configs/cifar10_table3.yaml \
            --variant "$VARIANT" --seed $SEED --depth 110 \
            --log_dir results/logs --checkpoint_dir results/checkpoints \
            --num_workers 4 --base_dir .
    done
done
echo "TABLE3_LOOP_COMPLETE" >> results/logs/session_progress.log

# Table 4
echo "RUNNING TABLE 4" >> results/logs/session_progress.log
for VARIANT in "1xSkip" "1xSkip+BN" "2xSkip+BN" "3xSkip+BN" "1rSkip+BN" "2rSkip+BN" "3rSkip+BN"; do
    for SEED in 0 1 2; do
        python -m src.training.train \
            --config configs/cifar10_table4.yaml \
            --variant "$VARIANT" --seed $SEED --depth 110 \
            --log_dir results/logs --checkpoint_dir results/checkpoints \
            --num_workers 4 --base_dir .
    done
done
echo "TABLE4_LOOP_COMPLETE" >> results/logs/session_progress.log

# Table 6
echo "RUNNING TABLE 6" >> results/logs/session_progress.log
for DEPTH in 20 32 44 56 110; do
    for VARIANT in "baseline" "2rSkip+LN"; do
        python -m src.training.train \
            --config configs/cifar100_table6.yaml \
            --variant "$VARIANT" --seed 0 --depth $DEPTH \
            --log_dir results/logs --checkpoint_dir results/checkpoints \
            --num_workers 4 --base_dir .
    done
done
echo "TABLE6_LOOP_COMPLETE" >> results/logs/session_progress.log

# Evaluation
echo "EVALUATING" >> results/logs/session_progress.log
python -m src.evaluation.reproduce_tables --config configs/cifar10_table1.yaml --out results/tables/reproduced_table1.csv --log_dir results/logs --base_dir .
python -m src.evaluation.reproduce_tables --config configs/cifar10_table3.yaml --out results/tables/reproduced_table3.csv --log_dir results/logs --base_dir .
python -m src.evaluation.reproduce_tables --config configs/cifar10_table4.yaml --out results/tables/reproduced_table4.csv --log_dir results/logs --base_dir .
python -m src.evaluation.reproduce_tables --config configs/cifar100_table6.yaml --out results/tables/reproduced_table6.csv --log_dir results/logs --base_dir .

# Figures
python -m src.evaluation.gradient_norm_probe --checkpoint_dir results/checkpoints --data_dir data/cifar10/cifar-10-batches-py --dataset cifar10 --out_dir results/figures --n_samples 10000 --base_dir .
python -m src.evaluation.figure4_analysis --log_dir results/logs --out_dir results/figures --config_name cifar10_table1

echo "ALL_COMPLETE" >> results/logs/session_progress.log
