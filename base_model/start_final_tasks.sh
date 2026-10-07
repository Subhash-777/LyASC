#!/bin/bash
source /home/subhash/miniconda3/etc/profile.d/conda.sh
conda activate lyasc-base
cd /home/subhash/projects/LyASC/base_model

echo "Launching Figure 3 reproduction..."
bash ./scripts/run_gradient_norm_analysis.sh > results/logs/figure3.log 2>&1 &
P1=$!

echo "Launching Figure 4 reproduction..."
python -m src.evaluation.figure4_analysis --log_dir results/logs --out_dir results/figures --config_name cifar10_table4 > results/logs/figure4.log 2>&1 &
P2=$!

echo "Launching Table 6..."
bash ./scripts/run_table6_experiments.sh > results/logs/table6_queue.log 2>&1 &
P3=$!

echo "All tasks launched successfully. Waiting for completion..."
wait $P1 $P2 $P3
echo "All final tasks completed!"
