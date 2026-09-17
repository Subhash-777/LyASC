"""MLflow experiment-tracking integration for the LyASC project.

Design goal (see LyASC_Architecture_Research_Plan.docx, Section 6): wrap the
EXISTING result-JSON + decisions.log discipline, don't replace it. Every
number MLflow ever shows must still trace back to a result JSON on disk --
MLflow here is a queryable index and dashboard over artifacts that are
independently trustworthy, not a new source of truth.

Three integration points into base_model/src/training/train.py's
train_one_run() (or lyasc_model's thin wrapper around it):

    1. At run start           -> start_run() + log_run_params()
    2. Every N training steps -> log_step_metrics()  (same cadence as the
                                  existing run_logger.info(...) grad_norm logging)
    3. At json.dump(result,f) -> log_final_result() + log_artifacts()

Nothing here requires mlflow to be installed to run training -- every
public function is a no-op (with a one-time warning) if mlflow is missing
or if tracking is disabled via config, so this is strictly additive.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

_MLFLOW_AVAILABLE = True
try:
    import mlflow
except ImportError:  # pragma: no cover
    _MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore

_WARNED_ONCE = False


def _warn_unavailable_once() -> None:
    global _WARNED_ONCE
    if not _WARNED_ONCE:
        logger.warning(
            "mlflow is not installed; experiment tracking calls are no-ops. "
            "Install with `pip install mlflow --break-system-packages` to enable. "
            "Training itself is unaffected."
        )
        _WARNED_ONCE = True


# ---------------------------------------------------------------------------
# Naming conventions -- mirror train.py's existing result_json_path scheme
# exactly, so an MLflow run and its backing JSON are trivially cross-referenced.
# ---------------------------------------------------------------------------

def experiment_name_for(config_name: str) -> str:
    """One MLflow Experiment per config family, e.g. 'lyasc_table1_equiv',
    'lyasc_coldstart_ablation', 'lyasc_multiscale_ablation'."""
    return config_name


def run_name_for(variant_name: str, depth: int, seed: int) -> str:
    """One MLflow Run per (variant, depth, seed) -- mirrors the safe_variant
    substitution already used for result_json_path in train.py, so run names
    and JSON filenames match at a glance."""
    safe_variant = variant_name.replace("+", "plus").replace("/", "_").replace(" ", "_")
    return f"{safe_variant}__depth{depth}__seed{seed}"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

from pathlib import Path

_LYASC_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB_PATH = _LYASC_ROOT / "results" / "mlflow.db"
_DEFAULT_ARTIFACT_ROOT = (_LYASC_ROOT / "mlruns").resolve()


def set_experiment_anchored(config_name: str) -> str:
    """Set active experiment by config_name, ensuring default_artifact_root is
    explicitly anchored to an absolute _LYASC_ROOT / 'mlruns' path.
    """
    if not _MLFLOW_AVAILABLE:
        return config_name
    exp_name = experiment_name_for(config_name)
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        art_loc = str(_DEFAULT_ARTIFACT_ROOT)
        mlflow.create_experiment(exp_name, artifact_location=art_loc)
    mlflow.set_experiment(exp_name)
    return exp_name


def init_tracking(tracking_uri: Optional[str] = None) -> bool:
    """Call once at process start (e.g. top of train.py's main()).

    tracking_uri: e.g. 'sqlite:///lyasc_model/results/mlflow.db' for a local
    single-file database backend. Always resolved to an absolute URI to prevent
    accidental path doubling when scripts are invoked from subdirectories.

    Returns True if tracking is actually active.
    """
    if not _MLFLOW_AVAILABLE:
        _warn_unavailable_once()
        return False
    if tracking_uri is None:
        uri = f"sqlite:///{_DEFAULT_DB_PATH}"
    else:
        uri = tracking_uri
        if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
            rel_path = uri[len("sqlite///"):]
            abs_db = (_LYASC_ROOT / rel_path).resolve()
            uri = f"sqlite:///{abs_db}"
    mlflow.set_tracking_uri(uri)
    return True


@contextmanager
def tracked_run(
    config_name: str,
    variant_name: str,
    depth: int,
    seed: int,
    extra_tags: Optional[Dict[str, str]] = None,
) -> Iterator[Optional[Any]]:
    """Context manager wrapping one training run."""
    if not _MLFLOW_AVAILABLE:
        _warn_unavailable_once()
        yield None
        return

    set_experiment_anchored(config_name)
    tags = {
        "variant_name": variant_name,
        "depth": str(depth),
        "seed": str(seed),
    }
    if extra_tags:
        tags.update(extra_tags)

    with mlflow.start_run(run_name=run_name_for(variant_name, depth, seed), tags=tags) as run:
        yield run


# ---------------------------------------------------------------------------
# Logging helpers -- every one is a safe no-op if run is None / mlflow missing
# ---------------------------------------------------------------------------

def log_run_params(run: Optional[Any], params: Dict[str, Any]) -> None:
    """Log resolved hyperparameters and decisions at run start.

    Expected keys (superset across base LyASC + Section 4 extensions;
    pass only what's relevant to a given run -- unset keys are simply
    omitted, not logged as None):
        k, N, ema_decay, lambda0, beta_init_strategy, cold_start_strategy,
        grad_clip_max_norm, optimizer_type, momentum, weight_decay,
        initial_lr, batch_size,
        # Section 4 extension flags (default False/None when not in use):
        multiscale_enabled, adaptive_N_enabled, regularizer_weight,
        conditional_compute_threshold, bayesian_gate_enabled,
        curvature_aware_enabled,
    """
    if run is None or not _MLFLOW_AVAILABLE:
        return
    # MLflow params must be strings; flatten non-trivial values defensively.
    flat = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in params.items() if v is not None}
    mlflow.log_params(flat)


def log_step_metrics(run: Optional[Any], step: int, metrics: Dict[str, float]) -> None:
    """Log per-step training metrics, at the SAME cadence as the existing
    run_logger.info(...) grad-norm logging in train.py (no extra overhead
    beyond what's already computed there).

    Expected keys (all optional per call):
        train_loss, grad_norm, beta_grad_norm, other_grad_norm,
        alpha_mean, alpha_std, lambda_mean, lambda_std,
        # any per-block breakdowns should be prefixed, e.g. "alpha_block12"
    """
    if run is None or not _MLFLOW_AVAILABLE:
        return
    clean = {k: float(v) for k, v in metrics.items() if v is not None}
    if clean:
        mlflow.log_metrics(clean, step=step)


def log_final_result(run: Optional[Any], result: Dict[str, Any]) -> None:
    """Log the exact same dict that train.py's json.dump(result, f) writes,
    as MLflow metrics/params -- this is the point where an MLflow run and
    its backing result JSON become byte-for-byte reconcilable.
    """
    if run is None or not _MLFLOW_AVAILABLE:
        return
    numeric_keys = {"test_error", "final_train_loss", "measured_overhead_pct", "wall_clock_seconds"}
    metrics = {k: float(v) for k, v in result.items() if k in numeric_keys and v is not None}
    if metrics:
        mlflow.log_metrics(metrics)
    other = {k: v for k, v in result.items() if k not in numeric_keys}
    log_run_params(run, other)


def log_artifacts(run: Optional[Any], paths: list) -> None:
    """Register existing files (result JSON, decisions.log, config YAML,
    stability-map plots, optionally a checkpoint) as MLflow artifacts.
    Files that don't exist are skipped with a warning, never fabricated.
    """
    if run is None or not _MLFLOW_AVAILABLE:
        return
    for path in paths:
        if path and os.path.exists(path):
            mlflow.log_artifact(path)
        elif path:
            logger.warning("log_artifacts: path does not exist, skipping: %s", path)


def log_aggregate_comparison(
    config_name: str,
    comparison_table: "list[dict]",
    csv_path: Optional[str] = None,
) -> None:
    """Log a whole comparison table (the LyASC-equivalent of
    reproduce_tables.py's output) as one aggregate MLflow run per config
    family, for dashboard-level viewing without opening every individual run.

    comparison_table: list of dicts, one per row, matching the existing
    reproduced_table*_FINAL.csv schema (variant_name, depth, n_seeds_found,
    mean_error, std_error, paper_or_baseline_error, abs_diff, pass_fail).
    """
    if not _MLFLOW_AVAILABLE:
        _warn_unavailable_once()
        return
    set_experiment_anchored(config_name)
    with mlflow.start_run(run_name=f"{config_name}__AGGREGATE_COMPARISON"):
        for row in comparison_table:
            variant = str(row.get("variant_name", "unknown")).replace("+", "plus").replace(" ", "_")
            for key in ("mean_error", "std_error", "abs_diff"):
                if key in row and row[key] is not None:
                    mlflow.log_metric(f"{variant}__{key}", float(row[key]))
            mlflow.set_tag(f"{variant}__pass_fail", row.get("pass_fail", "UNKNOWN"))
        if csv_path and os.path.exists(csv_path):
            mlflow.log_artifact(csv_path)