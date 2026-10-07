"""Structured run logging, per plan.md Section 5/7/8.

Every run logs: iteration count, epoch number, seed, weight-decay-scope
decision, and downsampling-shortcut policy decision for later auditability.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def get_run_logger(
    config_name: str,
    variant_name: str,
    seed: int,
    log_dir: str = "results/logs",
) -> logging.Logger:
    """Return a logger writing to results/logs/<config>__<variant>__seed<N>.log.

    Also mirrors all output to stdout via a StreamHandler so training progress
    is visible in the terminal.

    Args:
        config_name:  Short name of the config (e.g. "cifar10_table1").
        variant_name: Variant label (e.g. "2rSkip+LN").
        seed:         Integer seed for this run.
        log_dir:      Directory for log files (created if absent).

    Returns:
        logging.Logger instance.
    """
    # Build a safe filename from config/variant/seed
    safe_variant = variant_name.replace("+", "plus").replace("/", "_").replace(" ", "_")
    run_id = f"{config_name}__{safe_variant}__seed{seed}"
    logger_name = f"lyasc.{run_id}"

    logger = logging.getLogger(logger_name)
    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # File handler
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_id}.log")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # Don't propagate to root logger (avoids double printing)
    logger.propagate = False

    logger.info("Run logger initialised: config=%s  variant=%s  seed=%d",
                config_name, variant_name, seed)
    logger.info("Log file: %s", log_path)

    return logger


def log_env_report(path: str = "results/logs/env_report.txt") -> None:
    """Write `python -m torch.utils.collect_env` output to disk.

    Per plan.md Section 1 — needed to rule out hardware/driver differences
    if reproduced numbers don't match the paper.

    Args:
        path: Destination file path (created with parent dirs if needed).
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "torch.utils.collect_env"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        report = result.stdout + result.stderr
    except Exception as e:
        report = f"Could not collect env report: {e}\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    logging.getLogger(__name__).info("Environment report written to %s", path)


def log_decision(
    key: str,
    value: str,
    rationale: str,
    log_dir: str = "results/logs",
) -> None:
    """Write an explicit, named implementation decision to decisions.log.

    Per plan.md Section 8 / strict rules: every ambiguity resolved during
    implementation must be recorded here, not buried silently in code.

    Args:
        key:       Short name for the decision (e.g. "shortcut_policy").
        value:     The chosen value/option (e.g. "option_A_zero_pad").
        rationale: One-paragraph justification.
        log_dir:   Directory to write decisions.log.
    """
    os.makedirs(log_dir, exist_ok=True)
    decision_path = os.path.join(log_dir, "decisions.log")
    entry = (
        f"DECISION: {key}\n"
        f"VALUE:     {value}\n"
        f"RATIONALE: {rationale}\n"
        f"{'-'*72}\n"
    )
    with open(decision_path, "a", encoding="utf-8") as f:
        f.write(entry)

    logging.getLogger(__name__).info("Decision logged: %s = %s", key, value)
