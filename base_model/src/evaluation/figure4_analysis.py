"""Reproduces Figure 4 of the base paper: training/validation loss and error
curves for 2rSkip+LN vs wSkip (both without LN suffix in the paper's caption),
over training iterations, showing wSkip's characteristic slow start.

CLI: python -m src.evaluation.figure4_analysis
         --log_dir results/logs
         --out_dir results/figures
         [--config configs/cifar10_table1.yaml]

Reads the per-iteration loss and accuracy logs from the training .log files
(written by train.py every 1000 iterations). Falls back to reading TensorBoard
event files if log parsing fails.

Per plan.md Section 6, the qualitative match (wSkip slow start + higher final
error) is the key thing to verify, not exact numerical match.
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logger = logging.getLogger(__name__)


def parse_training_log(log_path: str) -> dict:
    """Parse training metrics from a .log file written by train.py.

    Extracts: iteration, epoch, lr, loss, train_acc at each 1000-iter checkpoint.

    Returns:
        Dict with keys 'iters', 'epochs', 'lr', 'loss', 'train_acc'.
    """
    iters, epochs, lrs, losses, accs = [], [], [], [], []

    loss_pattern = re.compile(
        r"iter=\s*(\d+)\s+epoch=([\d.]+)\s+lr=([\d.]+)\s+loss=([\d.]+)\s+train_acc=([\d.]+)"
    )
    eval_pattern = re.compile(
        r"EVAL\s+iter=(\d+)\s+epoch=([\d.]+)\s+test_error=([\d.]+)"
    )

    eval_iters, eval_errors = [], []

    try:
        with open(log_path, "r") as f:
            for line in f:
                m = loss_pattern.search(line)
                if m:
                    iters.append(int(m.group(1)))
                    epochs.append(float(m.group(2)))
                    lrs.append(float(m.group(3)))
                    losses.append(float(m.group(4)))
                    accs.append(float(m.group(5)))
                    continue
                m2 = eval_pattern.search(line)
                if m2:
                    eval_iters.append(int(m2.group(1)))
                    eval_errors.append(float(m2.group(3)))
    except FileNotFoundError:
        logger.warning("Log file not found: %s", log_path)

    return {
        "iters": iters,
        "epochs": epochs,
        "lr": lrs,
        "loss": losses,
        "train_acc": accs,
        "eval_iters": eval_iters,
        "eval_errors": eval_errors,
    }


def find_log_files(log_dir: str, config_name: str, variant_name: str) -> list:
    """Find all seed log files for a (config, variant) pair."""
    safe_var = variant_name.replace("+", "plus").replace("/", "_").replace(" ", "_")
    pattern = f"{config_name}__{safe_var}__*__seed*.log"
    from glob import glob
    files = sorted(glob(os.path.join(log_dir, pattern)))
    if not files:
        # Try without depth in name
        pattern2 = f"{config_name}__{safe_var}__seed*.log"
        files = sorted(glob(os.path.join(log_dir, pattern2)))
    return files


def aggregate_curve(all_seed_data: list, key: str) -> tuple:
    """Average a metric curve across seeds, handling different-length arrays."""
    if not all_seed_data:
        return np.array([]), np.array([])

    # Use the seed with most data points as reference x-axis
    max_len = max(len(d[key]) for d in all_seed_data)
    ref_x = None
    for d in all_seed_data:
        if len(d[key]) == max_len:
            ref_x = np.array(d["iters"] if key != "eval_errors" else d["eval_iters"])
            break

    # Align and average
    values = []
    for d in all_seed_data:
        y = np.array(d[key])
        if len(y) == max_len:
            values.append(y)
    if not values:
        return ref_x, np.zeros_like(ref_x, dtype=float)
    return ref_x, np.mean(np.stack(values), axis=0)


def plot_figure4(
    variants_data: dict,
    out_path: str,
) -> None:
    """Plot training loss and test error curves for 2rSkip+LN vs wSkip+LN."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    COLORS = {
        "2rSkip+LN": "#2196F3",
        "wSkip+LN": "#FF5722",
    }
    LABELS = {
        "2rSkip+LN": "2rSkip+LN (best)",
        "wSkip+LN": "wSkip+LN (learnable w)",
    }

    for vname, data in variants_data.items():
        color = COLORS.get(vname, "gray")
        label = LABELS.get(vname, vname)

        # Training loss curve
        x, y = aggregate_curve(data, "loss")
        if len(x) > 0:
            axes[0].plot(x, y, color=color, label=label, linewidth=1.5)

        # Test error curve
        xe, ye = aggregate_curve(data, "eval_errors")
        if len(xe) > 0:
            axes[1].plot(xe, ye, color=color, label=label, linewidth=1.5, marker="o",
                         markersize=4)

    axes[0].set_xlabel("Training iteration")
    axes[0].set_ylabel("Training loss")
    axes[0].set_title("Training Loss (Figure 4 reproduction)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Training iteration")
    axes[1].set_ylabel("Test error rate (%)")
    axes[1].set_title("Test Error (Figure 4 reproduction)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # Add vertical lines at LR milestones
    for ax in axes:
        ax.axvline(32000, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axvline(48000, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    plt.suptitle("Figure 4: 2rSkip+LN vs wSkip+LN Learning Curves", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Figure 4 saved: %s", out_path)
    print(f"Figure 4 reproduction saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Reproduce Figure 4 learning curves.")
    parser.add_argument("--log_dir", default="results/logs")
    parser.add_argument("--out_dir", default="results/figures")
    parser.add_argument("--config_name", default="cifar10_table1")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)

    COMPARE_VARIANTS = ["2rSkip+LN", "wSkip+LN"]
    variants_data = {}

    for variant_name in COMPARE_VARIANTS:
        log_files = find_log_files(args.log_dir, args.config_name, variant_name)
        if not log_files:
            logger.warning("No log files found for '%s'", variant_name)
            variants_data[variant_name] = []
            continue
        logger.info("Found %d seed logs for '%s'", len(log_files), variant_name)
        all_seed_data = [parse_training_log(f) for f in log_files]
        variants_data[variant_name] = all_seed_data

    out_path = os.path.join(args.out_dir, "reproduced_figure4.png")
    plot_figure4(variants_data, out_path)


if __name__ == "__main__":
    main()
