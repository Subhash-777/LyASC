#!/usr/bin/env python3
"""Re-evaluate saved checkpoints on the 10,000 CIFAR-10 test set using CPU only.
Performs exact comparison to each checkpoint's recorded test error.
Outputs results to FINAL/checkpoint_reeval.csv.
"""

import sys
import os
import csv
from pathlib import Path
from functools import partial

# Environment guard
if "lyasc-experiments" not in sys.prefix and os.environ.get("CONDA_DEFAULT_ENV") != "lyasc-experiments":
    raise RuntimeError("Must run in lyasc-experiments conda environment")

_WORKSPACE = Path(__file__).resolve().parents[2]
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
from src.training.train import _lyasc_build_skip_variant, _evaluate
from src.training.lyasc_config import LyASCConfig

def main():
    device = torch.device("cpu")
    print("Using device:", device)

    data_dir = str(_BASE_ROOT / "data" / "cifar10" / "cifar-10-batches-py")
    _, test_loader = build_dataloaders(
        dataset="cifar10",
        data_dir=data_dir,
        batch_size=256,
        augmentation_cfg={"pad": 4, "crop_size": 32, "random_flip": True},
        seed=0,
        num_workers=2,
    )

    checkpoints_to_eval = [
        ("Phase C LyASC seed 0", "lyasc_model/results/checkpoints/lyasc_table1_equiv__LyASC__depth110__seed0.pt", {
            "is_lyasc": True, "variant_cfg": {"name": "LyASC", "use_layer_norm": False, "use_batch_norm": False, "skip_scale": 1.0, "residual_scale": 1.0, "cold_start_strategy": "none"}
        }),
        ("LyASC+LN (none) seed 0", "lyasc_model/results/checkpoints/phase_d_coldstart/lyasc_phase_d_coldstart__LyASCplusLN_none__depth110__seed0.pt", {
            "is_lyasc": True, "variant_cfg": {"name": "LyASC+LN_none", "use_layer_norm": True, "use_batch_norm": False, "skip_scale": 1.0, "residual_scale": 1.0, "cold_start_strategy": "none"}
        }),
        ("stress_0.5 seed 0", "lyasc_model/results/checkpoints/phase_e_stresstest/lyasc_phase_e_stresstest__LyASC_stress_0.5__depth110__seed0.pt", {
            "is_lyasc": True, "variant_cfg": {"name": "LyASC_stress_0.5", "use_layer_norm": True, "use_batch_norm": False, "skip_scale": 0.5, "residual_scale": 1.0, "cold_start_strategy": "none"}
        }),
        ("stress_2.0 seed 0", "lyasc_model/results/checkpoints/phase_e_stresstest/lyasc_phase_e_stresstest__LyASC_stress_2.0__depth110__seed0.pt", {
            "is_lyasc": True, "variant_cfg": {"name": "LyASC_stress_2.0", "use_layer_norm": True, "use_batch_norm": False, "skip_scale": 1.0, "residual_scale": 2.0, "cold_start_strategy": "none"}
        }),
        ("stress_3.0 seed 1", "lyasc_model/results/checkpoints/phase_e_stresstest/lyasc_phase_e_stresstest__LyASC_stress_3.0__depth110__seed1.pt", {
            "is_lyasc": True, "variant_cfg": {"name": "LyASC_stress_3.0", "use_layer_norm": True, "use_batch_norm": False, "skip_scale": 1.0, "residual_scale": 3.0, "cold_start_strategy": "none"}
        }),
        ("stress_3.0 seed 4", "lyasc_model/results/checkpoints/phase_e_stresstest/lyasc_phase_e_stresstest__LyASC_stress_3.0__depth110__seed4.pt", {
            "is_lyasc": True, "variant_cfg": {"name": "LyASC_stress_3.0", "use_layer_norm": True, "use_batch_norm": False, "skip_scale": 1.0, "residual_scale": 3.0, "cold_start_strategy": "none"}
        }),
        ("LyASC+BN seed 0", "lyasc_model/results/checkpoints/phase_e_stresstest/lyasc_phase_e_stresstest__LyASCplusBN__depth110__seed0.pt", {
            "is_lyasc": True, "variant_cfg": {"name": "LyASC+BN", "use_layer_norm": False, "use_batch_norm": True, "skip_scale": 1.0, "residual_scale": 1.0, "cold_start_strategy": "none"}
        }),
        ("base_model 1xSkip seed 0", "base_model/results/checkpoints/cifar10_table1__1xSkip__depth110__seed0.pt", {
            "is_lyasc": False, "variant_cfg": {"name": "1xSkip", "equation": "x + F(x)"}
        }),
        ("base_model 2rSkip+LN seed 0", "base_model/results/checkpoints/cifar10_table1__2rSkipplusLN__depth110__seed0.pt", {
            "is_lyasc": False, "variant_cfg": {"name": "2rSkip+LN", "equation": "LN(x + LN(x + F(x, W)))", "G": "layer_norm", "lambda": 2, "recursive": True}
        }),
    ]

    out_records = []
    original_bsv = _base_preact_mod.build_skip_variant

    print("=" * 80)
    print("RE-EVALUATING 9 SAVED CHECKPOINTS ON CPU (10,000 CIFAR-10 TEST IMAGES)")
    print("=" * 80)

    for label, rel_path, setup in checkpoints_to_eval:
        ckpt_path = str(_WORKSPACE / rel_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        recorded_err = ckpt.get("final_test_error")
        depth = ckpt.get("depth", 110)

        if setup["is_lyasc"]:
            lyasc_cfg = LyASCConfig(k=1, N=20, ema_decay=0.9, lambda0=0.0, cold_start_strategy="none")
            _base_preact_mod.build_skip_variant = partial(_lyasc_build_skip_variant, lyasc_cfg=lyasc_cfg)
        else:
            _base_preact_mod.build_skip_variant = _base_build_skip_variant

        model = preact_resnet(depth=depth, num_classes=10, variant_cfg=setup["variant_cfg"])
        _base_preact_mod.build_skip_variant = original_bsv

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        # Re-evaluate
        re_eval_err = _evaluate(model, test_loader, device)
        diff = re_eval_err - recorded_err

        print(f"{label:<30}: Recorded = {recorded_err:6.2f}%, Re-evaluated = {re_eval_err:6.2f}%, Diff = {diff:+6.3f}pp", flush=True)
        if abs(diff) > 0.05:
            print(f"ERROR: Checkpoint re-evaluation mismatch > 0.05pp on {label}!", flush=True)
            sys.exit(1)

        out_records.append({
            "checkpoint": rel_path,
            "label": label,
            "recorded": round(recorded_err, 4),
            "re_evaluated": round(re_eval_err, 4),
            "difference": round(diff, 4),
        })

    csv_path = _WORKSPACE / "FINAL" / "checkpoint_reeval.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "label", "recorded", "re_evaluated", "difference"])
        writer.writeheader()
        writer.writerows(out_records)

    print("=" * 80)
    print(f"All 9 checkpoints re-evaluated successfully within ±0.05pp! Output written to {csv_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
