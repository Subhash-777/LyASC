"""Aggregates all logged runs into reproduced_table*.csv files, computes
mean/std per row, and diffs against paper reference values with PASS/FAIL.

CLI: python -m src.evaluation.reproduce_tables --config <path>
                                                --out results/tables/<name>.csv
     [--log_dir results/logs]
     [--base_dir <workspace_root>]

Produces one CSV per config, then prints a PASS/FAIL summary table to stdout.
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.evaluation.evaluate import evaluate_config
from src.training.config import load_config

logger = logging.getLogger(__name__)

CSV_FIELDS = [
    "variant_name", "depth", "n_seeds_found",
    "mean_error", "std_error", "paper_error", "abs_diff", "pass_fail",
]


def write_csv(records: list, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rec in records:
            row = {k: rec.get(k, "") for k in CSV_FIELDS}
            # Format floats
            for key in ("mean_error", "std_error", "paper_error", "abs_diff"):
                if row[key] not in (None, ""):
                    row[key] = f"{float(row[key]):.2f}"
            writer.writerow(row)
    logger.info("CSV written: %s", out_path)


def print_summary(records: list, config_name: str) -> tuple:
    """Print a human-readable PASS/FAIL summary. Returns (n_pass, n_total)."""
    n_pass = sum(1 for r in records if r["pass_fail"] == "PASS")
    n_fail = sum(1 for r in records if r["pass_fail"] == "FAIL")
    n_missing = sum(1 for r in records if r["pass_fail"] == "MISSING")
    n_total = len(records)

    print(f"\n{'='*80}")
    print(f"REPRODUCE TABLE: {config_name}")
    print(f"{'='*80}")
    print(f"{'Variant':<25} {'Depth':>6} {'Repro':>7} {'±std':>6} {'Paper':>7} {'Δ':>6} {'Result':<8}")
    print("-" * 80)

    for r in records:
        me = f"{r['mean_error']:.2f}" if r["mean_error"] is not None else "---"
        se = f"{r['std_error']:.2f}" if r["std_error"] is not None else "---"
        pe = f"{r['paper_error']:.2f}" if r["paper_error"] is not None else "---"
        ad = f"{r['abs_diff']:.2f}" if r["abs_diff"] is not None else "---"
        d = r["depth"] if r["depth"] else ""
        pf = r["pass_fail"]
        flag = "✓" if pf == "PASS" else ("✗" if pf == "FAIL" else "?")
        print(f"{r['variant_name']:<25} {d!s:>6} {me:>7} {se:>6} {pe:>7} {ad:>6} {flag} {pf:<8}")

    print("-" * 80)
    print(f"PASS: {n_pass}/{n_total}  FAIL: {n_fail}/{n_total}  MISSING: {n_missing}/{n_total}")
    print(f"{'='*80}\n")

    return n_pass, n_total


def reproduce_tables(
    config_path: str,
    out_path: str,
    log_dir: str = "results/logs",
    base_dir: str = None,
) -> list:
    """Main reproduction pipeline for one config file."""
    cfg = load_config(config_path, base_dir=base_dir)
    records = evaluate_config(cfg, log_dir=log_dir)
    write_csv(records, out_path)
    print_summary(records, cfg.config_name)
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results and emit reproduced_table*.csv with PASS/FAIL."
    )
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--log_dir", default="results/logs")
    parser.add_argument("--base_dir", default=None,
                        help="Workspace root for resolving relative data_dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    reproduce_tables(
        config_path=args.config,
        out_path=args.out,
        log_dir=args.log_dir,
        base_dir=args.base_dir,
    )


if __name__ == "__main__":
    main()
