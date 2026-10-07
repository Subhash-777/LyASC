"""Per-run evaluation + 5-seed aggregation, per plan.md Section 5 (\"Seeds\")
and Section 6 (experiment matrix).

Loads logged result JSON files for a given (config, variant) across all seeds,
computes mean and std error rate, and returns a structured record with PASS/FAIL
against the acceptance tolerance.
"""

import glob
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _safe_variant_name(variant_name: str) -> str:
    """Convert variant name to the safe filename stem used by train.py."""
    return variant_name.replace("+", "plus").replace("/", "_").replace(" ", "_")


def load_seed_results(
    config_name: str,
    variant_name: str,
    seeds: list,
    log_dir: str = "results/logs",
    depth: Optional[int] = None,
) -> list:
    """Load JSON result files for all seeds of a (config, variant) run.

    Args:
        config_name:  Config name stem (e.g. "cifar10_table1").
        variant_name: Variant name (e.g. "2rSkip+LN").
        seeds:        List of integer seed values.
        log_dir:      Directory where JSON result files are stored.
        depth:        Depth override (for Table 6 multi-depth; None → any depth matched).

    Returns:
        List of result dicts, one per successfully found seed.
    """
    safe_var = _safe_variant_name(variant_name)
    results = []
    missing = []

    for seed in seeds:
        if depth is not None:
            fname = f"{config_name}__{safe_var}__depth{depth}__seed{seed}.json"
        else:
            # Try depth-agnostic glob
            pattern = os.path.join(log_dir, f"{config_name}__{safe_var}__depth*__seed{seed}.json")
            matches = glob.glob(pattern)
            if matches:
                fname = os.path.basename(matches[0])
            else:
                # Fallback: old-style without depth in name
                fname = f"{config_name}__{safe_var}__seed{seed}.json"

        fpath = os.path.join(log_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                results.append(json.load(f))
        else:
            missing.append(seed)
            logger.warning("Result file not found: %s", fpath)

    if missing:
        logger.warning(
            "Missing seeds %s for config=%s variant=%s (depth=%s)",
            missing, config_name, variant_name, depth
        )

    return results


def aggregate_results(
    seed_results: list,
    paper_error: Optional[float],
    acceptance_tolerance: float,
    variant_name: str,
    depth: Optional[int] = None,
) -> dict:
    """Aggregate seed results and compute PASS/FAIL vs paper reference.

    Args:
        seed_results:        List of result dicts from load_seed_results().
        paper_error:         Paper's reported error rate for this row (%).
        acceptance_tolerance: Max allowable |reproduced - paper| in % points.
        variant_name:        Variant label (for output record).
        depth:               Optional depth (for Table 6).

    Returns:
        Dict: {variant_name, depth, n_seeds_found, mean_error, std_error,
               paper_error, abs_diff, pass_fail}
    """
    if not seed_results:
        return {
            "variant_name": variant_name,
            "depth": depth,
            "n_seeds_found": 0,
            "mean_error": None,
            "std_error": None,
            "paper_error": paper_error,
            "abs_diff": None,
            "pass_fail": "MISSING",
        }

    errors = [r["final_test_error"] for r in seed_results]
    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0

    if paper_error is not None:
        abs_diff = abs(mean_err - paper_error)
        pass_fail = "PASS" if abs_diff <= acceptance_tolerance else "FAIL"
    else:
        abs_diff = None
        pass_fail = "NO_REF"

    return {
        "variant_name": variant_name,
        "depth": depth,
        "n_seeds_found": len(errors),
        "mean_error": mean_err,
        "std_error": std_err,
        "paper_error": paper_error,
        "abs_diff": abs_diff,
        "pass_fail": pass_fail,
    }


def evaluate_config(
    config,          # ExperimentConfig from src/training/config.py
    log_dir: str = "results/logs",
) -> list:
    """Evaluate all variants for a given config and return aggregated records.

    Handles both the flat-variant layout (Tables 1/3/4) and the
    depth×variant matrix layout (Table 6).

    Args:
        config:   Parsed ExperimentConfig.
        log_dir:  Directory containing JSON result files.

    Returns:
        List of aggregated result dicts (one per variant or per depth×variant).
    """
    records = []

    if config.depths_and_targets is not None:
        # Table 6: depth × variant matrix
        for depth_target in config.depths_and_targets:
            depth = depth_target.depth
            for variant in config.variants:
                # Look up the per-row paper error from the depth_target
                if variant.name == "baseline":
                    paper_error = depth_target.baseline_paper_error
                else:
                    # Proposal variant (2rSkip+LN)
                    paper_error = depth_target.proposal_paper_error

                seed_results = load_seed_results(
                    config_name=config.config_name,
                    variant_name=variant.name,
                    seeds=config.seeds,
                    log_dir=log_dir,
                    depth=depth,
                )
                record = aggregate_results(
                    seed_results=seed_results,
                    paper_error=paper_error,
                    acceptance_tolerance=config.acceptance_tolerance,
                    variant_name=variant.name,
                    depth=depth,
                )
                records.append(record)
    else:
        # Tables 1/3/4: flat variant list, single depth
        depth = config.get_depth()
        for variant in config.variants:
            seed_results = load_seed_results(
                config_name=config.config_name,
                variant_name=variant.name,
                seeds=config.seeds,
                log_dir=log_dir,
                depth=depth,
            )
            record = aggregate_results(
                seed_results=seed_results,
                paper_error=variant.paper_error,
                acceptance_tolerance=config.acceptance_tolerance,
                variant_name=variant.name,
                depth=depth,
            )
            records.append(record)

    return records
