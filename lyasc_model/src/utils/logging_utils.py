"""Minimal logging_utils for lyasc_model, matching base_model's decisions.log
format exactly (key/value/rationale, '-'*74 separator) so the same log_decision
discipline carries over. Not uploaded with the original base_model files, so
reconstructed here from its observed output format and call signature in
train.py; if this diverges from the real base_model/src/utils/logging_utils.py,
that file should be copied over verbatim instead of using this shim.
"""

import logging
import os
from pathlib import Path


def log_decision(key: str, value: str, rationale: str, log_dir: str) -> None:
    """Append a DECISION/VALUE/RATIONALE block to <log_dir>/decisions.log.

    Idempotent-friendly (append mode) — callers may log the same decision
    multiple times across runs; duplicates are harmless and preserve the
    full audit trail per plan.md's "log every decision" discipline.
    """
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "decisions.log")
    with open(path, "a") as f:
        f.write(f"DECISION: {key}\n")
        f.write(f"VALUE:     {value}\n")
        f.write(f"RATIONALE: {rationale}\n")
        f.write("-" * 74 + "\n")


def get_run_logger(config_name: str, variant_name: str, seed: int, log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    safe_variant = variant_name.replace("+", "plus").replace("/", "_").replace(" ", "_")
    logger_name = f"{config_name}.{safe_variant}.seed{seed}"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(
            os.path.join(log_dir, f"{config_name}__{safe_variant}__seed{seed}.log")
        )
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    return logger


def log_env_report(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    try:
        import torch
        report = torch.utils.collect_env.get_pretty_env_info()
    except Exception as e:  # pragma: no cover
        report = f"collect_env failed: {e}"
    with open(os.path.join(log_dir, "env_report.txt"), "w") as f:
        f.write(report)
