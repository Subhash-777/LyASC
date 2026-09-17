#!/usr/bin/env python3
"""Run truncated diagnostic training runs (2000 iters) for cold-start strategies
(none, warmup, damped) and log exact training-time telemetry:
- loss
- train_acc
- grad_norm
- grad_norm clipping count (clip_active / steps)
- beta_grad_norm & other_grad_norm
- alpha (mean, std, min, max)
- beta (mean, std)
- lambda (mean, std)
"""

import os
import sys
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

sys.path.insert(0, str(_LYASC_ROOT))
sys.path.insert(0, str(_BASE_ROOT))

import src
for r in (_BASE_ROOT, _LYASC_ROOT):
    sp = str(r / "src")
    if os.path.exists(sp) and sp not in src.__path__:
        src.__path__.append(sp)
for sub in ("models", "training", "utils", "data", "tracking", "evaluation"):
    sub_path_base = str(_BASE_ROOT / "src" / sub)
    sub_path_lyasc = str(_LYASC_ROOT / "src" / sub)
    try:
        submod = __import__(f"src.{sub}", fromlist=[sub])
        for sp in (sub_path_base, sub_path_lyasc):
            if os.path.exists(sp) and sp not in submod.__path__:
                submod.__path__.append(sp)
    except ImportError:
        pass

import yaml
from src.training.train import train_one_run

def main():
    # Create temporary config file for 2000-iter diagnostic check
    tmp_config_path = str(_LYASC_ROOT / "tmp_configs" / "diagnostic_coldstart.yaml")
    os.makedirs(os.path.dirname(tmp_config_path), exist_ok=True)

    config_content = {
        "dataset": "cifar10",
        "data_dir": "../../base_model/data/cifar10/cifar-10-batches-py",
        "model": "preact_resnet110",
        "num_seeds": 1,
        "seeds": [0],
        "optimizer": {
            "type": "sgd",
            "momentum": 0.9,
            "weight_decay": 0.0002,
            "nesterov": False,
        },
        "lr_schedule": {
            "initial_lr": 0.1,
            "milestones_iters": [500, 800],
            "total_iters": 1000,
            "gamma": 0.1,
        },
        "batch_size": 128,
        "augmentation": {
            "pad": 4,
            "crop_size": 32,
            "random_flip": True,
        },
        "lyasc": {
            "k": 1,
            "N": 20,
            "ema_decay": 0.9,
            "lambda0": 0.0,
            "use_layer_norm": True,
            "enable_mlflow": False,
        },
        "variants": [
            {
                "name": "LyASC+LN_none",
                "cold_start_strategy": "none",
                "cold_start_warmup_iters": 1000,
                "use_layer_norm": True,
            },
            {
                "name": "LyASC+LN_warmup",
                "cold_start_strategy": "warmup",
                "cold_start_warmup_iters": 1000,
                "use_layer_norm": True,
            },
            {
                "name": "LyASC+LN_damped",
                "cold_start_strategy": "damped",
                "cold_start_warmup_iters": 1000,
                "use_layer_norm": True,
            },
        ],
        "acceptance_tolerance_abs_points": 0.4,
    }

    with open(tmp_config_path, "w") as f:
        yaml.dump(config_content, f)

    log_dir = str(_LYASC_ROOT / "results" / "logs" / "diagnostic_coldstart")
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 80)
    print("RUNNING TRUNCATED COLD-START DIAGNOSTIC RUNS (2000 ITERS PER STRATEGY)")
    print("=" * 80)

    for variant_name in ["LyASC+LN_none", "LyASC+LN_warmup", "LyASC+LN_damped"]:
        print(f"\n---> Launching diagnostic run for: {variant_name}")
        result = train_one_run(
            config_path=tmp_config_path,
            variant_name=variant_name,
            seed=0,
            log_dir=log_dir,
            checkpoint_dir=str(_LYASC_ROOT / "results" / "checkpoints" / "diagnostic_coldstart"),
            data_dir_override=str(_BASE_ROOT / "data" / "cifar10" / "cifar-10-batches-py"),
            num_workers=4,
        )
        time_val = result.get('wall_clock_seconds')
        time_str = f"{time_val:.1f}s" if time_val is not None else "N/A"
        print(f"Finished {variant_name}: final_test_error={result.get('final_test_error')}%, time={time_str}")
        print(f"lyasc_config: {result.get('lyasc_config')}")

    print("\nALL DIAGNOSTIC RUNS COMPLETE!")

if __name__ == "__main__":
    main()
