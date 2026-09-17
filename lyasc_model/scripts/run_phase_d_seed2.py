#!/usr/bin/env python3
"""Run Phase D 3rd Seed (Seed 2) for LyASC+LN_none and LyASC+LN_damped.

Required by pre-registered tolerance protocol due to <= 0.40pp gap
between none (7.29%) and damped (7.51%) with damped std = ±0.42pp.
Warmup (7.95%) is excluded as its 0.66pp gap exceeds the 0.40pp trigger.
"""

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


def main():
    config_path = str(_LYASC_ROOT / "configs" / "lyasc_phase_d_coldstart.yaml")
    log_dir = str(_LYASC_ROOT / "results" / "logs" / "phase_d_coldstart")
    checkpoint_dir = str(_LYASC_ROOT / "results" / "checkpoints" / "phase_d_coldstart")
    data_dir = str(_BASE_ROOT / "data" / "cifar10" / "cifar-10-batches-py")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    seed2_runs = [
        ("LyASC+LN_none", 2),
        ("LyASC+LN_damped", 2),
    ]

    print("=" * 80)
    print("PHASE D COLD-START ABLATION: SEED 2 (3RD SEED) LAUNCH")
    print(f"Executing {len(seed2_runs)} runs (64,000 iterations each)")
    print("=" * 80)

    start_time = time.time()
    for idx, (variant_name, seed) in enumerate(seed2_runs, 1):
        print(f"\n[{idx}/{len(seed2_runs)}] Launching {variant_name} (Seed {seed})...")
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
        print(f"[{idx}/{len(seed2_runs)}] Finished {variant_name} Seed {seed}: test_error={err}%, elapsed={elapsed/3600:.2f}h")

    total_elapsed = time.time() - start_time
    print("=" * 80)
    print(f"ALL SEED 2 RUNS COMPLETE in {total_elapsed/3600:.2f} hours.")
    print("=" * 80)


if __name__ == "__main__":
    main()
