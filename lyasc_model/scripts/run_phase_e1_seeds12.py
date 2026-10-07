#!/usr/bin/env python3
"""Run Phase E1 Seeds 1 & 2 Batch (Table 3 Stress Cases).

Executes 6 runs sequentially (2 seeds × 3 variants = 6 × 64,000 iterations):
  Seed 1: LyASC_stress_0.5, LyASC_stress_2.0, LyASC_stress_3.0
  Seed 2: LyASC_stress_0.5, LyASC_stress_2.0, LyASC_stress_3.0

Each run uses LayerNorm2d after the Lyapunov-adaptive gate, matching the
base_model Table 3 normalization configuration for apples-to-apples comparison.

Paper baselines (Table 3):
  - LN(0.5x+F):  37.14% ± 9.94%    (catastrophic failure)
  - LN(x+2F):    37.59% ± 22.83%   (bimodal failure)
  - LN(x+3F):    49.27% ± 15.31%   (severe failure)

Seed 0 results already complete:
  - LyASC_stress_0.5 seed0:  7.98%
  - LyASC_stress_2.0 seed0:  8.13%
  - LyASC_stress_3.0 seed0: 27.90%  (partial convergence under 3x scaling)
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


# Seed 0 results for context reporting
SEED0_RESULTS = {
    "LyASC_stress_0.5": 7.98,
    "LyASC_stress_2.0": 8.13,
    "LyASC_stress_3.0": 27.90,
}

PAPER_TARGETS = {
    "LyASC_stress_0.5": (37.14, "LN(0.5x+F)"),
    "LyASC_stress_2.0": (37.59, "LN(x+2F)"),
    "LyASC_stress_3.0": (49.27, "LN(x+3F)"),
}


def main():
    config_path = str(_LYASC_ROOT / "configs" / "lyasc_phase_e_stresstest.yaml")
    log_dir = str(_LYASC_ROOT / "results" / "logs" / "phase_e_stresstest")
    checkpoint_dir = str(_LYASC_ROOT / "results" / "checkpoints" / "phase_e_stresstest")
    data_dir = str(_BASE_ROOT / "data" / "cifar10" / "cifar-10-batches-py")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Run seeds 1 and 2, grouped by seed so wall-clock projections are clean
    runs = [
        ("LyASC_stress_0.5", 1),
        ("LyASC_stress_2.0", 1),
        ("LyASC_stress_3.0", 1),
        ("LyASC_stress_0.5", 2),
        ("LyASC_stress_2.0", 2),
        ("LyASC_stress_3.0", 2),
    ]

    print("=" * 80)
    print("PHASE E1 SEEDS 1+2 LAUNCH: TABLE 3 STRESS CASES")
    print(f"Executing {len(runs)} runs (64,000 iterations each)")
    print()
    print("Seed 0 reference results:")
    for v, e in SEED0_RESULTS.items():
        paper_e, paper_lbl = PAPER_TARGETS[v]
        rescue = paper_e - e
        print(f"  {v}: {e:.2f}%  (paper: {paper_e}% [{paper_lbl}], rescue: +{rescue:.2f}pp)")
    print("=" * 80)

    batch_start = time.time()
    all_results = {}

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
            print(f"[{idx}/{len(runs)}] SKIPPED (result already exists): {variant_name} seed{seed}")
        else:
            print(f"[{idx}/{len(runs)}] Finished {variant_name} seed{seed}: "
                  f"test_error={err:.3f}%  elapsed={elapsed/3600:.2f}h")

        if variant_name not in all_results:
            all_results[variant_name] = {}
        all_results[variant_name][seed] = err

    total_elapsed = (time.time() - batch_start) / 3600

    print()
    print("=" * 80)
    print("PHASE E1 SEEDS 1+2 COMPLETE")
    print(f"Total elapsed: {total_elapsed:.2f}h")
    print()
    print("Cross-seed summary (all available seeds):")
    import statistics
    for v in ["LyASC_stress_0.5", "LyASC_stress_2.0", "LyASC_stress_3.0"]:
        paper_e, paper_lbl = PAPER_TARGETS[v]
        all_errors = [SEED0_RESULTS[v]]
        for seed in [1, 2]:
            e = all_results.get(v, {}).get(seed)
            if e is not None:
                all_errors.append(e)
        n = len(all_errors)
        mean = statistics.mean(all_errors)
        std = statistics.pstdev(all_errors) if n > 1 else 0
        rescue = paper_e - mean
        seed_str = ", ".join(f"s{s}={e:.2f}%" for s, e in
                             [(0, SEED0_RESULTS[v])] +
                             [(s, all_results.get(v, {}).get(s)) for s in [1, 2]
                              if all_results.get(v, {}).get(s) is not None])
        print(f"  {v}: mean={mean:.3f}% ±{std:.3f}pp  rescue=+{rescue:.2f}pp  [{seed_str}]")
        print(f"    vs paper: {paper_e}% ({paper_lbl})")
    print("=" * 80)


if __name__ == "__main__":
    main()
