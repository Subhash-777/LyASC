#!/usr/bin/env python3
"""Phase E2 Pre-Flight Timing Benchmark.

Runs 1,000 iterations for Phase E2 normalization pairing variants:
- LyASC_noG (G = Identity)
- LyASC+BN  (G = BatchNorm2d)

Measures exact wall-clock throughput (s/iter) to calculate budget for 64k runs.
"""

import os
import sys
import time
import gc
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

    test_variants = ["LyASC_noG", "LyASC+BN"]
    print("=" * 80)
    print("PHASE E2 PRE-FLIGHT TIMING BENCHMARK (1,000 Iters Each)")
    print("=" * 80)

    for variant_name in test_variants:
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t0 = time.time()
        res = train_one_run(
            config_path=config_path,
            variant_name=variant_name,
            seed=999,  # test benchmark seed
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            data_dir_override=data_dir,
            num_workers=4,
            total_iters_override=1000,
        )
        elapsed = time.time() - t0
        s_per_iter = elapsed / 1000.0
        est_64k_hours = (s_per_iter * 64000.0) / 3600.0

        print(f"✓ {variant_name} Benchmark (1k iters): {elapsed:.2f}s total | {s_per_iter:.4f} s/iter | Est 64k run: {est_64k_hours:.2f} hours")

        # Cleanup benchmark artifacts/checkpoints
        json_path = Path(log_dir) / f"lyasc_phase_e_stresstest__{variant_name}__depth110__seed999.json"
        log_path = Path(log_dir) / f"lyasc_phase_e_stresstest__{variant_name}__seed999.log"
        ckpt_path = Path(checkpoint_dir) / f"lyasc_phase_e_stresstest__{variant_name}__depth110__seed999.pt"
        for p in [json_path, log_path, ckpt_path]:
            if p.exists():
                p.unlink()


if __name__ == "__main__":
    main()
