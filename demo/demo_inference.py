#!/usr/bin/env python3
"""LyASC Interactive CPU Demo Inference.

Loads saved model checkpoints on CPU, runs inference on CIFAR-10 test samples,
prints true vs predicted labels with class names, and outputs per-block gate summaries
for LyASC models or architectural scaling summaries for baseline models.
"""

import sys
import os
import random
import math
import argparse
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
from src.models.skip_variants import build_skip_variant as _base_build_skip_variant
from src.training.train import _lyasc_build_skip_variant
from src.training.lyasc_config import LyASCConfig

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

PRESETS = {
    "lyasc_ln": {
        "title": "LyASC+LN (Phase D Baseline)",
        "phase": "Phase D (Default Model)",
        "path": "lyasc_model/results/checkpoints/phase_d_coldstart/lyasc_phase_d_coldstart__LyASCplusLN_none__depth110__seed0.pt",
        "is_lyasc": True,
        "variant_cfg": {"name": "LyASC+LN_none", "use_layer_norm": True, "use_batch_norm": False, "skip_scale": 1.0, "residual_scale": 1.0, "cold_start_strategy": "none"},
        "desc": "Primary LyASC model with post-gate LayerNorm2d (7.33% test error, seed 0).",
    },
    "lyasc_bn": {
        "title": "LyASC+BN (Phase E2 Normalization Pairing)",
        "phase": "Phase E2 (Normalization Ablation)",
        "path": "lyasc_model/results/checkpoints/phase_e_stresstest/lyasc_phase_e_stresstest__LyASCplusBN__depth110__seed0.pt",
        "is_lyasc": True,
        "variant_cfg": {"name": "LyASC+BN", "use_layer_norm": False, "use_batch_norm": True, "skip_scale": 1.0, "residual_scale": 1.0, "cold_start_strategy": "none"},
        "desc": "LyASC with post-gate BatchNorm2d (6.35% test error, seed 0).",
    },
    "lyasc_nog": {
        "title": "LyASC_noG (Phase C Un-normalized Ablation)",
        "phase": "Phase C (Architecture Ablation)",
        "path": "lyasc_model/results/checkpoints/lyasc_table1_equiv__LyASC__depth110__seed0.pt",
        "is_lyasc": True,
        "variant_cfg": {"name": "LyASC", "use_layer_norm": False, "use_batch_norm": False, "skip_scale": 1.0, "residual_scale": 1.0, "cold_start_strategy": "none"},
        "desc": "Un-normalized gating G=Identity (10.21% test error, seed 0).",
    },
    "stress_0.5": {
        "title": "LyASC Stress 0.5 (Contracted Skip Scaling Rescue)",
        "phase": "Phase E1 (Stress Testing)",
        "path": "lyasc_model/results/checkpoints/phase_e_stresstest/lyasc_phase_e_stresstest__LyASC_stress_0.5__depth110__seed0.pt",
        "is_lyasc": True,
        "variant_cfg": {"name": "LyASC_stress_0.5", "use_layer_norm": True, "use_batch_norm": False, "skip_scale": 0.5, "residual_scale": 1.0, "cold_start_strategy": "none"},
        "desc": "Recovers to 7.98% error under 0.5x skip scaling where fixed baseline fails at 37.14%.",
    },
    "stress_2.0": {
        "title": "LyASC Stress 2.0 (Expanded Residual Scaling Rescue)",
        "phase": "Phase E1 (Stress Testing)",
        "path": "lyasc_model/results/checkpoints/phase_e_stresstest/lyasc_phase_e_stresstest__LyASC_stress_2.0__depth110__seed0.pt",
        "is_lyasc": True,
        "variant_cfg": {"name": "LyASC_stress_2.0", "use_layer_norm": True, "use_batch_norm": False, "skip_scale": 1.0, "residual_scale": 2.0, "cold_start_strategy": "none"},
        "desc": "Recovers to 8.13% error under 2.0x residual scaling where fixed baseline fails at 37.59%.",
    },
    "stress_3.0_rescue": {
        "title": "LyASC Stress 3.0 Full Rescue (Seed 4)",
        "phase": "Phase E1 (Stress Testing)",
        "path": "lyasc_model/results/checkpoints/phase_e_stresstest/lyasc_phase_e_stresstest__LyASC_stress_3.0__depth110__seed4.pt",
        "is_lyasc": True,
        "variant_cfg": {"name": "LyASC_stress_3.0", "use_layer_norm": True, "use_batch_norm": False, "skip_scale": 1.0, "residual_scale": 3.0, "cold_start_strategy": "none"},
        "desc": "Full rescue seed reaching 8.75% error under 3.0x residual scaling (baseline fails at 49.27%).",
    },
    "stress_3.0_partial": {
        "title": "LyASC Stress 3.0 Partial Rescue (Seed 1)",
        "phase": "Phase E1 (Stress Testing)",
        "path": "lyasc_model/results/checkpoints/phase_e_stresstest/lyasc_phase_e_stresstest__LyASC_stress_3.0__depth110__seed1.pt",
        "is_lyasc": True,
        "variant_cfg": {"name": "LyASC_stress_3.0", "use_layer_norm": True, "use_batch_norm": False, "skip_scale": 1.0, "residual_scale": 3.0, "cold_start_strategy": "none"},
        "desc": "Partial rescue cluster seed finishing at 32.73% error (plateaued near alpha=0.65).",
    },
    "1xskip": {
        "title": "Base Model: 1xSkip (Standard PreAct-ResNet-110)",
        "phase": "Base Paper Table 1",
        "path": "base_model/results/checkpoints/cifar10_table1__1xSkip__depth110__seed0.pt",
        "is_lyasc": False,
        "variant_cfg": {"name": "1xSkip", "equation": "x + F(x)"},
        "desc": "Reproduced standard identity shortcut baseline (6.31% test error, seed 0).",
    },
    "2rskip": {
        "title": "Base Model: 2rSkip+LN (Best Hand-Tuned Fixed Schedule)",
        "phase": "Base Paper Table 1",
        "path": "base_model/results/checkpoints/cifar10_table1__2rSkipplusLN__depth110__seed0.pt",
        "is_lyasc": False,
        "variant_cfg": {"name": "2rSkip+LN", "equation": "LN(x + LN(x + F(x, W)))", "G": "layer_norm", "lambda": 2, "recursive": True},
        "desc": "Liu et al. best fixed recursive scaling schedule (6.09% test error, seed 0).",
    },
}

# Alias shortcuts for CLI convenience
ALIASES = {
    "ln": "lyasc_ln",
    "default": "lyasc_ln",
    "bn": "lyasc_bn",
    "nog": "lyasc_nog",
    "stress0.5": "stress_0.5",
    "stress2.0": "stress_2.0",
    "stress3.0_s4": "stress_3.0_rescue",
    "stress3.0_full": "stress_3.0_rescue",
    "stress3.0_s1": "stress_3.0_partial",
    "base_1xskip": "1xskip",
    "base_2rskip": "2rskip",
}

def print_presets():
    print("=" * 80)
    print(f"{'Preset Key':<20} | {'Phase':<24} | {'Model Variant':<30}")
    print("-" * 80)
    for key, p in PRESETS.items():
        print(f"{key:<20} | {p['phase']:<24} | {p['title']}")
        print(f"  -> Path: {p['path']}")
        print(f"  -> Info: {p['desc']}")
        print()
    print("=" * 80)
    print("Usage examples:")
    print("  bash demo/run_demo.sh --preset ln")
    print("  bash demo/run_demo.sh --preset bn")
    print("  bash demo/run_demo.sh --preset stress_0.5")
    print("  bash demo/run_demo.sh --preset 2rskip")
    print("  bash demo/run_demo.sh /path/to/any_checkpoint.pt")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="LyASC CPU Demo Inference for Models & Experiments")
    parser.add_argument("checkpoint", nargs="?", default=None, help="Path to checkpoint .pt file")
    parser.add_argument("-p", "--preset", default=None, help="Preset model key (use --list to see all)")
    parser.add_argument("-l", "--list", action="store_true", help="List all available presets and experiments")
    parser.add_argument("-n", "--samples", type=int, default=8, help="Number of random test samples (default: 8)")
    parser.add_argument("-s", "--seed", type=int, default=42, help="RNG seed for sample selection (default: 42)")

    args = parser.parse_args()

    if args.list:
        print_presets()
        return

    device = torch.device("cpu")
    print("=" * 75)
    print("LyASC & Baseline CPU Demo Inference (PreAct-ResNet-110, CIFAR-10)")
    print("=" * 75)

    is_lyasc = True
    variant_cfg = None
    preset_meta = None

    if args.preset:
        preset_key = ALIASES.get(args.preset.lower(), args.preset)
        if preset_key not in PRESETS:
            print(f"Error: Unknown preset '{args.preset}'. Available presets:")
            for k in PRESETS:
                print(f"  - {k}")
            sys.exit(1)
        preset_meta = PRESETS[preset_key]
        ckpt_path = _WORKSPACE / preset_meta["path"]
        is_lyasc = preset_meta["is_lyasc"]
        variant_cfg = preset_meta["variant_cfg"]
    elif args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = _WORKSPACE / ckpt_path
    else:
        # Default to Phase D LyASC+LN
        preset_meta = PRESETS["lyasc_ln"]
        ckpt_path = _WORKSPACE / preset_meta["path"]
        is_lyasc = preset_meta["is_lyasc"]
        variant_cfg = preset_meta["variant_cfg"]

    if not ckpt_path.exists():
        print(f"Error: Checkpoint file not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading Checkpoint : {ckpt_path.name}")
    if preset_meta:
        print(f"Experiment / Phase : {preset_meta['phase']} ({preset_meta['title']})")

    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state_dict"]
    recorded_err = ckpt.get("final_test_error", 0.0)
    var_name = ckpt.get("variant_name", "PreAct-ResNet-110")

    # Detect LyASC vs Base Model if not from preset
    if variant_cfg is None:
        has_estimator = any("estimator._lambda_ema" in k for k in sd.keys())
        is_lyasc = has_estimator
        if is_lyasc:
            is_bn = "LyASCplusBN" in ckpt_path.name or any("g_norm.running_mean" in k for k in sd)
            use_ln = not is_bn and any("g_norm.weight" in k for k in sd)
            skip_scale = 0.5 if "stress_0.5" in ckpt_path.name else 1.0
            res_scale = 2.0 if "stress_2.0" in ckpt_path.name else (3.0 if "stress_3.0" in ckpt_path.name else 1.0)
            variant_cfg = {
                "name": var_name,
                "use_layer_norm": use_ln,
                "use_batch_norm": is_bn,
                "skip_scale": skip_scale,
                "residual_scale": res_scale,
                "cold_start_strategy": "none",
            }
        else:
            if "2rSkip" in var_name:
                variant_cfg = {"name": "2rSkip+LN", "equation": "LN(x + LN(x + F(x, W)))", "G": "layer_norm", "lambda": 2, "recursive": True}
            elif "wSkip" in var_name:
                variant_cfg = {"name": "wSkip+LN", "equation": "LN(w * x + F(x))", "G": "layer_norm", "scaling_type": "learned"}
            else:
                variant_cfg = {"name": var_name, "equation": "x + F(x)"}

    original_bsv = _base_preact_mod.build_skip_variant
    if is_lyasc:
        lyasc_cfg = LyASCConfig(k=1, N=20, ema_decay=0.9, lambda0=0.0, cold_start_strategy="none")
        _base_preact_mod.build_skip_variant = partial(_lyasc_build_skip_variant, lyasc_cfg=lyasc_cfg)
    else:
        _base_preact_mod.build_skip_variant = _base_build_skip_variant

    model = preact_resnet(depth=110, num_classes=10, variant_cfg=variant_cfg)
    _base_preact_mod.build_skip_variant = original_bsv

    model.load_state_dict(sd)
    model.eval()
    model.to(device)

    # Extract per-block alphas if LyASC
    alphas = []
    if is_lyasc:
        for s in range(3):
            for b in range(18):
                beta_key = f"stages.{s}.{b}.skip_variant.beta_l"
                lam_key = f"stages.{s}.{b}.skip_variant.estimator._lambda_ema"
                if beta_key in sd and lam_key in sd:
                    beta = sd[beta_key].item()
                    lam = sd[lam_key].item()
                    alpha = 1.0 / (1.0 + math.exp(- beta * lam))
                    alphas.append((s, b, alpha, beta, lam))

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

    random.seed(args.seed)
    num_eval = min(args.samples, len(all_images))
    sample_indices = random.sample(range(len(all_images)), num_eval)
    sample_imgs = all_images[sample_indices]
    sample_lbls = all_labels[sample_indices]

    with torch.no_grad():
        outputs = model(sample_imgs)
        preds = outputs.argmax(dim=1)

    print("\n--- Model Checkpoint Overview ---")
    print(f"Variant Name     : {variant_cfg['name']}")
    print(f"Architecture Type: {'Stability-Driven LyASC' if is_lyasc else 'Fixed/Learned Baseline'}")
    if recorded_err > 0.0:
        print(f"Recorded Test Err: {recorded_err:.2f}%")

    if is_lyasc:
        g_name = "BatchNorm2d" if variant_cfg.get("use_batch_norm") else ("LayerNorm2d" if variant_cfg.get("use_layer_norm") else "Identity")
        mean_alpha = sum(a[2] for a in alphas) / len(alphas) if alphas else 0.5
        skip_dominant = sum(1 for a in alphas if a[2] > 0.5)
        print(f"Post-Gate Norm G : {g_name}")
        print(f"Mean Gate α      : {mean_alpha:.4f} (Identity Weight)")
        print(f"Skip-Favoring    : {skip_dominant}/{len(alphas)} blocks ({100.0*skip_dominant/len(alphas):.1f}% with α > 0.5)")
    else:
        eq = variant_cfg.get("equation", "N/A")
        print(f"Shortcut Equation: {eq}")
        print(f"Skip Weighting   : Fixed design parameter (non-adaptive)")

    print(f"\n--- {num_eval} Sample Inference Predictions ---")
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
    print(f"Batch Accuracy: {correct_count}/{num_eval} ({100.0 * correct_count / num_eval:.1f}%)")

    if is_lyasc and alphas:
        print("\n--- LyASC Per-Stage Gate Distribution Summary ---")
        for s_idx in range(3):
            stage_alphas = [a[2] for a in alphas if a[0] == s_idx]
            ch = [16, 32, 64][s_idx]
            s_mean = sum(stage_alphas) / len(stage_alphas)
            s_min = min(stage_alphas)
            s_max = max(stage_alphas)
            s_dom = sum(1 for a in stage_alphas if a > 0.5)
            print(f"Stage {s_idx+1} ({ch} channels, 18 blocks): "
                  f"Mean α = {s_mean:.4f} | Range: [{s_min:.4f}, {s_max:.4f}] | "
                  f"α > 0.5: {s_dom}/18 blocks")
    print("=" * 75)

if __name__ == "__main__":
    main()
