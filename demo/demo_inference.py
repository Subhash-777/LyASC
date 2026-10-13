#!/usr/bin/env python3
"""LyASC Interactive CPU Demo Inference.

Loads a saved LyASC checkpoint on CPU, classifies 8 random CIFAR-10 test images,
prints true vs predicted labels with class names, and outputs a per-block alpha summary.
"""

import sys
import os
import random
import math
from pathlib import Path
from functools import partial

# Environment check
if "lyasc-experiments" not in sys.prefix and os.environ.get("CONDA_DEFAULT_ENV") != "lyasc-experiments":
    raise RuntimeError("Must run in lyasc-experiments conda environment")

_WORKSPACE = Path(__file__).resolve().parents[1]
_LYASC_ROOT = _WORKSPACE / "lyasc_model"
_BASE_ROOT = _WORKSPACE / "base_model"

if str(_BASE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASE_ROOT))
if str(_BASE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_BASE_ROOT / "src"))
if str(_LYASC_ROOT) not in sys.path:
    sys.path.insert(0, str(_LYASC_ROOT))
if str(_LYASC_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_LYASC_ROOT / "src"))

import src
for root_path in (_BASE_ROOT, _LYASC_ROOT):
    sp = str(root_path / "src")
    if sp not in src.__path__:
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

import torch
from src.data.cifar_datamodule import build_dataloaders
from src.models.preact_resnet import preact_resnet
from src.models import preact_resnet as _base_preact_mod
from src.training.train import _lyasc_build_skip_variant
from src.training.lyasc_config import LyASCConfig

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def main():
    device = torch.device("cpu")
    print("=" * 75)
    print("LyASC CPU Demo Inference (PreAct-ResNet-110, CIFAR-10)")
    print("=" * 75)

    default_ckpt = _LYASC_ROOT / "results" / "checkpoints" / "phase_d_coldstart" / "lyasc_phase_d_coldstart__LyASCplusLN_none__depth110__seed0.pt"
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_ckpt

    if not ckpt_path.exists():
        print(f"Error: Checkpoint file not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state_dict"]
    recorded_err = ckpt.get("final_test_error", 0.0)

    # Build model architecture
    original_bsv = _base_preact_mod.build_skip_variant
    lyasc_cfg = LyASCConfig(k=1, N=20, ema_decay=0.9, lambda0=0.0, cold_start_strategy="none")
    _base_preact_mod.build_skip_variant = partial(_lyasc_build_skip_variant, lyasc_cfg=lyasc_cfg)

    # Determine norm from checkpoint name/keys
    is_bn = "LyASCplusBN" in ckpt_path.name or "stages.0.0.skip_variant.g_norm.running_mean" in sd
    use_ln = not is_bn and "stages.0.0.skip_variant.g_norm.weight" in sd
    skip_scale = 0.5 if "stress_0.5" in ckpt_path.name else 1.0
    res_scale = 2.0 if "stress_2.0" in ckpt_path.name else (3.0 if "stress_3.0" in ckpt_path.name else 1.0)

    variant_cfg = {
        "name": ckpt.get("variant_name", "LyASC"),
        "use_layer_norm": use_ln,
        "use_batch_norm": is_bn,
        "skip_scale": skip_scale,
        "residual_scale": res_scale,
        "cold_start_strategy": "none",
    }

    model = preact_resnet(depth=110, num_classes=10, variant_cfg=variant_cfg)
    _base_preact_mod.build_skip_variant = original_bsv

    model.load_state_dict(sd)
    model.eval()
    model.to(device)

    # Extract per-block alphas
    alphas = []
    for s in range(3):
        for b in range(18):
            beta_key = f"stages.{s}.{b}.skip_variant.beta_l"
            lam_key = f"stages.{s}.{b}.skip_variant.estimator._lambda_ema"
            if beta_key in sd and lam_key in sd:
                beta = sd[beta_key].item()
                lam = sd[lam_key].item()
                alpha = 1.0 / (1.0 + math.exp(- beta * lam))
                alphas.append((s, b, alpha))

    # Load CIFAR-10 test set
    data_dir = str(_BASE_ROOT / "data" / "cifar10" / "cifar-10-batches-py")
    _, test_loader = build_dataloaders(
        dataset="cifar10",
        data_dir=data_dir,
        batch_size=256,
        augmentation_cfg={"pad": 4, "crop_size": 32, "random_flip": False},
        seed=42,
        num_workers=0,
    )

    all_images, all_labels = [], []
    for imgs, lbls in test_loader:
        all_images.append(imgs)
        all_labels.append(lbls)
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    random.seed(42)
    sample_indices = random.sample(range(len(all_images)), 8)
    sample_imgs = all_images[sample_indices]
    sample_lbls = all_labels[sample_indices]

    with torch.no_grad():
        outputs = model(sample_imgs)
        preds = outputs.argmax(dim=1)

    print("\n--- Model Checkpoint Overview ---")
    print(f"Variant: {variant_cfg['name']} | Recorded Test Error: {recorded_err:.2f}%")
    print(f"Total Blocks: {len(alphas)} | G Norm: {'BatchNorm2d' if is_bn else ('LayerNorm2d' if use_ln else 'Identity')}")
    print(f"Mean Gate α: {sum(a[2] for a in alphas)/len(alphas):.4f} | Blocks with α > 0.5: {sum(1 for a in alphas if a[2] > 0.5)}/{len(alphas)}")

    print("\n--- 8 Sample Inference Predictions ---")
    print(f"{'Sample':<8} | {'True Class':<14} | {'Predicted Class':<16} | {'Match?':<6}")
    print("-" * 52)
    correct_count = 0
    for idx, (true_idx, pred_idx) in enumerate(zip(sample_lbls.tolist(), preds.tolist()), 1):
        true_name = CIFAR10_CLASSES[true_idx]
        pred_name = CIFAR10_CLASSES[pred_idx]
        matched = (true_idx == pred_idx)
        if matched:
            correct_count += 1
        status = "✓ YES" if matched else "✗ NO"
        print(f"Sample {idx:<2} | {true_name:<14} | {pred_name:<16} | {status}")
    print("-" * 52)
    print(f"Batch Accuracy: {correct_count}/8 ({100.0 * correct_count / 8:.1f}%)")

    print("\n--- Per-Stage Gate Value Summary ---")
    for s_idx in range(3):
        stage_alphas = [a[2] for a in alphas if a[0] == s_idx]
        print(f"Stage {s_idx+1} ({[16, 32, 64][s_idx]} channels, 18 blocks): "
              f"Mean α = {sum(stage_alphas)/len(stage_alphas):.4f} "
              f"(Min: {min(stage_alphas):.4f}, Max: {max(stage_alphas):.4f}, "
              f"α > 0.5: {sum(1 for a in stage_alphas if a > 0.5)}/18)")
    print("=" * 75)

if __name__ == "__main__":
    main()
