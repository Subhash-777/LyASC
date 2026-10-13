#!/usr/bin/env python3
"""Generate per-block gate maps from saved checkpoints (CPU only).
Extracts alpha, beta, and smoothed lambda for:
- stress_3.0 seeds 0 to 4
- stress_0.5 seed 0
- stress_2.0 seed 0

Generates:
1. Alpha vs block index plot (PNG + PDF) saved to report_materials/03_figures/
2. Summary table of mean alpha and fraction of blocks with alpha > 0.5.
No causal claims in captions.
"""

import os
import math
import csv
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_WORKSPACE = Path(__file__).resolve().parents[2]
_FIGURES_PNG = _WORKSPACE / "report_materials" / "03_figures" / "png"
_FIGURES_PDF = _WORKSPACE / "report_materials" / "03_figures" / "pdf"
_FIGURES_TABLES = _WORKSPACE / "report_materials" / "03_figures" / "tables"
_FINAL_DIR = _WORKSPACE / "FINAL"

os.makedirs(_FIGURES_PNG, exist_ok=True)
os.makedirs(_FIGURES_PDF, exist_ok=True)
os.makedirs(_FIGURES_TABLES, exist_ok=True)
os.makedirs(_FINAL_DIR, exist_ok=True)

targets = [
    ("stress_3.0 Seed 0", _WORKSPACE / "lyasc_model/results/checkpoints/phase_e_stresstest/lyasc_phase_e_stresstest__LyASC_stress_3.0__depth110__seed0.pt", "#d95f02", ":"),
    ("stress_3.0 Seed 1", _WORKSPACE / "lyasc_model/results/checkpoints/phase_e_stresstest/lyasc_phase_e_stresstest__LyASC_stress_3.0__depth110__seed1.pt", "#7570b3", ":"),
    ("stress_3.0 Seed 2", _WORKSPACE / "lyasc_model/results/checkpoints/phase_e_stresstest/lyasc_phase_e_stresstest__LyASC_stress_3.0__depth110__seed2.pt", "#e7298a", "--"),
    ("stress_3.0 Seed 3", _WORKSPACE / "lyasc_model/results/checkpoints/phase_e_stresstest/lyasc_phase_e_stresstest__LyASC_stress_3.0__depth110__seed3.pt", "#66a61e", "--"),
    ("stress_3.0 Seed 4 (Breakthrough)", _WORKSPACE / "lyasc_model/results/checkpoints/phase_e_stresstest/lyasc_phase_e_stresstest__LyASC_stress_3.0__depth110__seed4.pt", "#1b9e77", "-"),
    ("stress_0.5 Seed 0", _WORKSPACE / "lyasc_model/results/checkpoints/phase_e_stresstest/lyasc_phase_e_stresstest__LyASC_stress_0.5__depth110__seed0.pt", "#1f78b4", "-."),
    ("stress_2.0 Seed 0", _WORKSPACE / "lyasc_model/results/checkpoints/phase_e_stresstest/lyasc_phase_e_stresstest__LyASC_stress_2.0__depth110__seed0.pt", "#ff7f00", "-."),
]

def extract_gate_data(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]
    alphas, betas, lambdas = [], [], []
    for s in range(3):
        for b in range(18):
            beta_key = f"stages.{s}.{b}.skip_variant.beta_l"
            lam_key = f"stages.{s}.{b}.skip_variant.estimator._lambda_ema"
            if beta_key in sd and lam_key in sd:
                beta = sd[beta_key].item()
                lam = sd[lam_key].item()
                alpha = 1.0 / (1.0 + math.exp(- beta * lam))
                alphas.append(alpha)
                betas.append(beta)
                lambdas.append(lam)
    return alphas, betas, lambdas

def main():
    table_rows = []
    plot_data = []

    print("=" * 80)
    print("EXTRACTING PER-BLOCK GATE MAPS FROM SAVED CHECKPOINTS")
    print("=" * 80)

    for label, ckpt_path, color, ls in targets:
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint not found: {ckpt_path}")
            continue
        alphas, betas, lambdas = extract_gate_data(ckpt_path)
        n_blocks = len(alphas)
        mean_alpha = sum(alphas) / n_blocks
        frac_gt_half = sum(1 for a in alphas if a > 0.5) / n_blocks
        mean_beta = sum(betas) / n_blocks
        mean_lambda = sum(lambdas) / n_blocks

        print(f"{label:<32}: mean_alpha={mean_alpha:.4f}, frac>0.5={frac_gt_half:.3f} ({sum(1 for a in alphas if a > 0.5)}/{n_blocks}), mean_beta={mean_beta:+.3f}, mean_lambda={mean_lambda:+.3f}")

        table_rows.append({
            "variant_seed": label,
            "mean_alpha": round(mean_alpha, 4),
            "frac_blocks_alpha_gt_0.5": round(frac_gt_half, 4),
            "num_blocks_gt_0.5": sum(1 for a in alphas if a > 0.5),
            "total_blocks": n_blocks,
            "mean_beta": round(mean_beta, 4),
            "mean_lambda": round(mean_lambda, 4),
        })

        plot_data.append((label, alphas, color, ls))

    # Save CSV
    csv_paths = [
        _FIGURES_TABLES / "checkpoint_gate_maps.csv",
        _FINAL_DIR / "gate_maps_summary.csv"
    ]
    for cp in csv_paths:
        with open(cp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "variant_seed", "mean_alpha", "frac_blocks_alpha_gt_0.5",
                "num_blocks_gt_0.5", "total_blocks", "mean_beta", "mean_lambda"
            ])
            writer.writeheader()
            writer.writerows(table_rows)
        print(f"Saved table to {cp}")

    # Plot
    plt.figure(figsize=(12, 6), dpi=300)
    for label, alphas, color, ls in plot_data:
        lw = 2.2 if "Seed 4" in label else 1.5
        plt.plot(range(len(alphas)), alphas, label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.9)

    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, label="Neutral Gate (α=0.5)")
    plt.axvline(17.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    plt.axvline(35.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

    plt.text(8.5, 0.05, "Stage 1 (16 ch)", ha="center", fontsize=10, color="#555555")
    plt.text(26.5, 0.05, "Stage 2 (32 ch)", ha="center", fontsize=10, color="#555555")
    plt.text(44.5, 0.05, "Stage 3 (64 ch)", ha="center", fontsize=10, color="#555555")

    plt.title("Per-Block Gate Value (α) Across Architecture Depths (54 Residual Blocks)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Residual Block Index (0 to 53)", fontsize=11)
    plt.ylabel("Gate Value α", fontsize=11)
    plt.ylim(-0.02, 1.05)
    plt.xlim(0, 53)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=9)
    plt.tight_layout()

    out_png = _FIGURES_PNG / "F13_checkpoint_gate_maps.png"
    out_pdf = _FIGURES_PDF / "F13_checkpoint_gate_maps.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {out_png} and {out_pdf}")

if __name__ == "__main__":
    main()
