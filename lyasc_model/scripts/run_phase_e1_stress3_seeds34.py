#!/usr/bin/env python3
"""Run LyASC_stress_3.0 Seeds 3 & 4 (5-Seed Sample Expansion).

Expands the sample size for LyASC_stress_3.0 (Table 3 3x residual scaling)
from n=3 to n=5 seeds to distinguish whether the 19.12% / 27.90% / 32.73%
spread represents a bimodal distribution vs continuous wide-variance rescue.

Existing seed results:
  - Seed 0: 27.90%  (final alpha_mean = 0.643, loss cracked at iter ~28k)
  - Seed 1: 32.73%  (final alpha_mean = 0.663, loss cracked at iter ~48k)
  - Seed 2: 19.12%  (final alpha_mean = 0.647, loss cracked at iter ~32k)
"""

import os
import sys
import time
import gc
from pathlib import Path

# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Environment Guard
if "lyasc-experiments" not in sys.prefix and os.environ.get("CONDA_DEFAULT_ENV") != "lyasc-experiments":
    raise RuntimeError(
        f"ENVIRONMENT ERROR: Launcher must be run within the 'lyasc-experiments' environment.\n"
        f"Current sys.prefix: {sys.prefix}\n"
        f"Current CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV')}\n"
        f"Invoke via: /home/subhash/miniconda3/envs/lyasc-experiments/bin/python {sys.argv[0]}"
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


def main():
    config_path = str(_LYASC_ROOT / "configs" / "lyasc_phase_e_stresstest.yaml")
    log_dir = str(_LYASC_ROOT / "results" / "logs" / "phase_e_stresstest")
    checkpoint_dir = str(_LYASC_ROOT / "results" / "checkpoints" / "phase_e_stresstest")
    data_dir = str(_BASE_ROOT / "data" / "cifar10" / "cifar-10-batches-py")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    variant_name = "LyASC_stress_3.0"
    target_seeds = [3, 4]

    print("=" * 80)
    print(f"PHASE E1 STRESS 3.0 SAMPLE EXPANSION: SEEDS 3 & 4 (n=5 Total)")
    print(f"Target variant: {variant_name} (Paper baseline: 49.27% [LN(x+3F)])")
    print("Existing seeds:")
    print("  Seed 0: 27.90%")
    print("  Seed 1: 32.73%")
    print("  Seed 2: 19.12%")
    print("=" * 80)

    batch_start = time.time()
    results = {}

    for idx, seed in enumerate(target_seeds, 1):
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n[{idx}/{len(target_seeds)}] Launching {variant_name} (Seed {seed})...")
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
        skipped = res.get("skipped", False)

        if skipped:
            print(f"[{idx}/{len(target_seeds)}] SKIPPED (already exists): {variant_name} seed{seed}")
        else:
            print(f"[{idx}/{len(target_seeds)}] Finished {variant_name} seed{seed}: "
                  f"test_error={err:.3f}%  elapsed={elapsed/3600:.2f}h")

        results[seed] = err

    total_elapsed = (time.time() - batch_start) / 3600
    print()
    print("=" * 80)
    print("STRESS 3.0 SEEDS 3 & 4 COMPLETE")
    print(f"Total elapsed: {total_elapsed:.2f}h")
    print("Results summary:")
    for seed, err in results.items():
        print(f"  Seed {seed}: {err:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
