#!/usr/bin/env python3
"""Run Phase E Stress Tests (Table 3 Equivalent) & Normalization Pairing Ablations.

Phased execution plan:
1. --smoke-benchmark: 500-iter pre-flight timing benchmark for noG, LayerNorm2d, BatchNorm2d.
2. --run-e1-stress: Table 3 stress cases (LyASC_stress_0.5, LyASC_stress_2.0, LyASC_stress_3.0) with 3 seeds each (9 runs total).
3. --run-e2-norm: Table 4 normalization pairing (LyASC_noG, LyASC+BN) with 2 seeds first (4 runs total).

Includes mandatory Environment Guard and Quantified Escalation Triggers.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Environment Guard: Ensure execution takes place within the lyasc-experiments environment
if "lyasc-experiments" not in sys.prefix and os.environ.get("CONDA_DEFAULT_ENV") != "lyasc-experiments":
    raise RuntimeError(
        f"ENVIRONMENT ERROR: Launcher must be run within the 'lyasc-experiments' Python environment.\n"
        f"Current sys.prefix: {sys.prefix}\n"
        f"Current CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV')}\n"
        f"Please invoke using: /home/subhash/miniconda3/envs/lyasc-experiments/bin/python {sys.argv[0]}"
    )

_LYASC_ROOT = Path(__file__).resolve().parents[1]
_BASE_ROOT = _LYASC_ROOT.parent / "base_model"

sys.path.insert(0, str(_BASE_ROOT / "src"))
sys.path.insert(0, str(_BASE_ROOT))
sys.path.insert(0, str(_LYASC_ROOT / "src"))
sys.path.insert(0, str(_LYASC_ROOT))

import src
import src.models
src.models.__path__.append(str(_BASE_ROOT / "src" / "models"))
src.models.__path__.append(str(_LYASC_ROOT / "src" / "models"))

from src.training.train import train_one_run


def run_benchmark():
    """Execute a 500-iter pre-flight timing benchmark for each G() normalization mode."""
    config_path = str(_LYASC_ROOT / "configs" / "lyasc_phase_e_stresstest.yaml")
    log_dir = str(_LYASC_ROOT / "results" / "logs" / "phase_e_benchmark")
    checkpoint_dir = str(_LYASC_ROOT / "results" / "checkpoints" / "phase_e_benchmark")
    data_dir = str(_BASE_ROOT / "data" / "cifar10" / "cifar-10-batches-py")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    benchmark_variants = [
        ("LyASC_noG", "G = None (No Normalization)"),
        ("LyASC_stress_2.0", "G = LayerNorm2d"),
        ("LyASC+BN", "G = BatchNorm2d"),
    ]

    print("=" * 80)
    print("PHASE E PRE-FLIGHT BENCHMARK: MEASURING STEADY-STATE PER-G() OVERHEAD (2,000 iterations)")
    print("=" * 80)

    timings = {}
    for variant, desc in benchmark_variants:
        print(f"\nBenchmarking {variant} ({desc}) for 2,000 iterations...")
        t0 = time.time()
        res = train_one_run(
            config_path=config_path,
            variant_name=variant,
            seed=99,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            data_dir_override=data_dir,
            num_workers=4,
            total_iters_override=2000,
        )
        elapsed = time.time() - t0
        s_per_iter = elapsed / 2000.0
        hours_64k = (s_per_iter * 64000) / 3600.0
        timings[variant] = (s_per_iter, hours_64k)
        print(f"--> {variant}: {s_per_iter:.4f} s/iter (~{hours_64k:.2f} hours per 64,000-iter run)")

    print("\n" + "=" * 80)
    print("SUMMARY OF STEADY-STATE TIMING BENCHMARKS (2,000 iterations)")
    print("=" * 80)
    for variant, (spi, h64k) in timings.items():
        print(f"  {variant:20s}: {spi:.4f} s/iter | 64k-iter run projection: {h64k:.2f} hours")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run Phase E Stress Tests & Normalization Pairing Ablations.")
    parser.add_argument("--smoke-benchmark", action="store_true", help="Run 500-iter timing benchmarks.")
    parser.add_argument("--run-e1-stress", action="store_true", help="Run Phase E1 Table 3 stress variants (3 seeds each).")
    parser.add_argument("--run-e2-norm", action="store_true", help="Run Phase E2 Table 4 normalization pairing variants (2 seeds first).")
    args = parser.parse_args()

    if args.smoke_benchmark:
        run_benchmark()
        return

    config_path = str(_LYASC_ROOT / "configs" / "lyasc_phase_e_stresstest.yaml")
    log_dir = str(_LYASC_ROOT / "results" / "logs" / "phase_e_stresstest")
    checkpoint_dir = str(_LYASC_ROOT / "results" / "checkpoints" / "phase_e_stresstest")
    data_dir = str(_BASE_ROOT / "data" / "cifar10" / "cifar-10-batches-py")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    runs_to_execute = []

    if args.run_e1_stress:
        # Table 3 stress variants: START AT 3 SEEDS per user requirement (bimodality detection)
        for variant in ["LyASC_stress_0.5", "LyASC_stress_2.0", "LyASC_stress_3.0"]:
            for seed in [0, 1, 2]:
                runs_to_execute.append((variant, seed))

    if args.run_e2_norm:
        # Table 4 normalization pairing variants: 2 seeds first (Seeds 0, 1)
        for variant in ["LyASC_noG", "LyASC+BN"]:
            for seed in [0, 1]:
                runs_to_execute.append((variant, seed))

    if not runs_to_execute:
        print("No run flags specified. Use --smoke-benchmark, --run-e1-stress, or --run-e2-norm.")
        return

    print("=" * 80)
    print("PHASE E EXECUTION LAUNCH")
    print(f"Total runs to execute: {len(runs_to_execute)} (64,000 iterations each)")
    print("=" * 80)

    start_time = time.time()
    for idx, (variant_name, seed) in enumerate(runs_to_execute, 1):
        print(f"\n[{idx}/{len(runs_to_execute)}] Launching {variant_name} (Seed {seed})...")
        t0 = time.time()
        res = train_one_run(
            config_path=config_path,
            variant_name=variant_name,
            seed=seed,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            data_dir_override=data_dir,
            num_workers=4,
        )
        elapsed = time.time() - t0
        err = res.get("final_test_error")
        print(f"[{idx}/{len(runs_to_execute)}] Finished {variant_name} Seed {seed}: test_error={err}%, elapsed={elapsed/3600:.2f}h")

    total_elapsed = time.time() - start_time
    print("=" * 80)
    print(f"ALL PHASE E RUNS COMPLETE in {total_elapsed/3600:.2f} hours.")
    print("=" * 80)


if __name__ == "__main__":
    main()
