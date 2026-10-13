#!/usr/bin/env python3
"""Run Phase E2 Normalization Pairing Ablations.

Evaluates two normalization variants post-gate:
- LyASC_noG  (G = Identity, no post-gate normalization, relying on backbone BN)
- LyASC+BN   (G = BatchNorm2d, post-gate Batch Normalization)

Seed Protocol: 2 seeds each (Seeds 0 and 1, 4 runs total @ 64,000 iterations each).
Comparison targets: base_model Table 4 baselines & Phase D LyASC+LN baseline (6.34% mean).
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
        f"ENVIRONMENT ERROR: Launcher must be run within 'lyasc-experiments' environment.\n"
        f"Current sys.prefix: {sys.prefix}\n"
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

    runs = [
        ("LyASC_noG", 0),
        ("LyASC_noG", 1),
        ("LyASC+BN", 0),
        ("LyASC+BN", 1),
        ("LyASC+BN", 2),
    ]

    print("=" * 80)
    print("PHASE E2 NORMALIZATION PAIRING ABLATION MATRIX (Seeds 0, 1, 2)")
    print("Variants: LyASC_noG (G=Identity), LyASC+BN (G=BatchNorm2d)")
    print("Comparison baseline: LyASC+LN (G=LayerNorm2d) = 6.34% 3-seed mean")
    print("=" * 80)

    batch_start = time.time()
    results = {}

    for idx, (variant_name, seed) in enumerate(runs, 1):
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n[{idx}/{len(runs)}] Launching {variant_name} (Seed {seed})...")
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
            print(f"[{idx}/{len(runs)}] SKIPPED (already exists): {variant_name} seed{seed}")
        else:
            print(f"[{idx}/{len(runs)}] Finished {variant_name} seed{seed}: "
                  f"test_error={err:.3f}%  elapsed={elapsed/3600:.2f}h")

        results[f"{variant_name}_seed{seed}"] = err

    total_elapsed = (time.time() - batch_start) / 3600
    print()
    print("=" * 80)
    print("PHASE E2 RUNS COMPLETE")
    print(f"Total batch elapsed: {total_elapsed:.2f} hours")
    print("Results summary:")
    for run_key, err in results.items():
        print(f"  {run_key:<20}: {err:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
