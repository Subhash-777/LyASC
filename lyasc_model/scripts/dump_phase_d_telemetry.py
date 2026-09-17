#!/usr/bin/env python3
"""Dump per-step beta_l statistics and loss trajectories for warmup vs damped vs none."""

import os
import sys
import importlib.util
from pathlib import Path

_LYASC_ROOT = Path(__file__).resolve().parents[1]
_BASE_ROOT = _LYASC_ROOT.parent / "base_model"

# Explicitly load lyasc_model modules
def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

sys.path.insert(0, str(_BASE_ROOT / "src"))
sys.path.insert(0, str(_LYASC_ROOT / "src"))
sys.path.insert(0, str(_BASE_ROOT))
sys.path.insert(0, str(_LYASC_ROOT))

import src
import src.models
src.models.__path__.append(str(_BASE_ROOT / "src" / "models"))
src.models.__path__.append(str(_LYASC_ROOT / "src" / "models"))

lyasc_config = _load_mod("src.training.lyasc_config", _LYASC_ROOT / "src" / "training" / "lyasc_config.py")
lyasc_skip = _load_mod("src.models.lyasc_skip", _LYASC_ROOT / "src" / "models" / "lyasc_skip.py")
lyasc_train = _load_mod("src.training.train", _LYASC_ROOT / "src" / "training" / "train.py")

import torch
import torch.nn as nn
from functools import partial
from src.training.train import (
    set_seed,
    _lyasc_build_skip_variant,
    _collect_lyasc_blocks,
    _attach_estimator_hooks,
    _build_lyasc_optimizer,
)
from src.models.preact_resnet import preact_resnet
from src.training.lyasc_config import LyASCConfig
from src.training.config import load_config
from src.data.cifar_datamodule import build_dataloaders


def run_telemetry(strategy, total_steps=1000):
    config_path = str(_LYASC_ROOT / "configs" / "lyasc_phase_d_coldstart.yaml")
    data_dir = str(_BASE_ROOT / "data" / "cifar10" / "cifar-10-batches-py")
    cfg = load_config(config_path, base_dir=str(_BASE_ROOT))
    lyasc_cfg = LyASCConfig(k=1, N=20, ema_decay=0.9, lambda0=0.0, use_layer_norm=True)

    train_loader, test_loader = build_dataloaders(
        dataset="cifar10",
        data_dir=data_dir,
        batch_size=128,
        augmentation_cfg={"pad": 4, "crop_size": 32, "random_flip": True},
        seed=0,
        num_workers=4,
    )

    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    var_cfg = {
        "name": f"LyASC+LN_{strategy}",
        "cold_start_strategy": strategy,
        "cold_start_warmup_iters": 1000,
        "use_layer_norm": True,
    }

    import src.models.preact_resnet as base_preact
    original_bsv = base_preact.build_skip_variant
    base_preact.build_skip_variant = partial(_lyasc_build_skip_variant, lyasc_cfg=lyasc_cfg)
    model = preact_resnet(depth=110, num_classes=10, variant_cfg=var_cfg).to(device)
    base_preact.build_skip_variant = original_bsv

    hook_state, handles = _attach_estimator_hooks(model)
    blocks = _collect_lyasc_blocks(model)
    optimizer = _build_lyasc_optimizer(model, cfg, lyasc_cfg)
    criterion = nn.CrossEntropyLoss()

    train_iter = iter(train_loader)
    print(f"=== STRATEGY: {strategy} (Blocks: {len(blocks)}) ===", flush=True)

    for step in range(1, total_steps + 1):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        images, labels = images.to(device), labels.to(device)
        hook_state.global_step = step
        optimizer.zero_grad(set_to_none=True)
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step in (1, 10, 100, 500, 999, 1000, 2000, 4000):
            betas = [b.beta_l.item() for _, _, b in blocks]
            b_mean = sum(betas) / len(betas)
            b_max = max(abs(b) for b in betas)
            b_std = (sum((b - b_mean) ** 2 for b in betas) / len(betas)) ** 0.5
            print(
                f"Step {step:4d} | Loss: {loss.item():.4f} | beta_mean: {b_mean:.8f} | beta_std: {b_std:.8f} | max(|beta|): {b_max:.8f}",
                flush=True,
            )


def main():
    for strat in ["warmup", "damped", "none"]:
        run_telemetry(strat, total_steps=1000)
        print("-" * 70, flush=True)


if __name__ == "__main__":
    main()
