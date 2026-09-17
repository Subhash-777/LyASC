"""Extended config for lyasc_model — adds LyASC-specific hyperparameters on top
of base_model's ExperimentConfig schema.

Reuses base_model/src/training/config.py's load_config() and ExperimentConfig
without modification; this module adds only the LyASC-specific fields that the
training wrapper needs to read from the YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class LyASCConfig:
    """LyASC-specific hyperparameters, read from the 'lyasc:' block in YAML configs."""
    k: int = 2                        # power-iteration steps per refresh
    N: int = 10                       # recompute interval (steps)
    ema_decay: float = 0.9            # EMA decay for lambda smoothing
    lambda0: float = 0.0              # stability threshold for the gate
    cold_start_strategy: str = "none" # "none", "warmup", "damped"
    cold_start_warmup_iters: int = 1000  # for strategy "warmup"
    beta_lr_multiplier: float = 1.0   # LR multiplier for beta_l param group
    freeze_beta: bool = False         # Step 3: freeze beta_l at 0 (requires_grad=False)
    use_layer_norm: bool = False      # Section 7 Ablation 1: G() = LayerNorm2d
    enable_mlflow: bool = True        # whether to log to MLflow


def load_lyasc_config(yaml_path: str) -> LyASCConfig:
    """Read the 'lyasc:' block from a YAML config file.
    Returns defaults if the block is absent."""
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    lyasc_raw = raw.get("lyasc", {})
    return LyASCConfig(
        k=lyasc_raw.get("k", 2),
        N=lyasc_raw.get("N", 10),
        ema_decay=lyasc_raw.get("ema_decay", 0.9),
        lambda0=lyasc_raw.get("lambda0", 0.0),
        cold_start_strategy=lyasc_raw.get("cold_start_strategy", "none"),
        cold_start_warmup_iters=lyasc_raw.get("cold_start_warmup_iters", 1000),
        beta_lr_multiplier=lyasc_raw.get("beta_lr_multiplier", 1.0),
        freeze_beta=lyasc_raw.get("freeze_beta", False),
        use_layer_norm=lyasc_raw.get("use_layer_norm", False),
        enable_mlflow=lyasc_raw.get("enable_mlflow", True),
    )


def load_lyasc_variant_raw_dicts(yaml_path: str) -> dict[str, dict]:
    """Read raw variant dictionaries directly from YAML 'variants:' list,
    preserving all custom fields (e.g. cold_start_strategy, use_layer_norm)
    without modifying base_model/src/training/config.py.
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    variant_dicts = {}
    for v in raw.get("variants", []):
        if isinstance(v, dict) and "name" in v:
            variant_dicts[v["name"]] = dict(v)
    return variant_dicts

