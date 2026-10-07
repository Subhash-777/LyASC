"""LyASC training wrapper — thin layer over base_model's train_one_run().

Per LyASC_plan.md Section 3, Step 3-4 and the approved Q1 resolution:
  - Injects LyASCSkip as a skip_variant via the config's variant registry
  - Attaches a StabilityEstimator to each LyASCSkip block
  - Registers a pre-forward hook on each PreActBasicBlock that calls
    estimator.forward(F_l, x_l, step) — disabled in eval mode
  - Adds beta_l to the optimizer's no-decay group (Q5 approved)
  - Integrates MLflow tracking per Section 6

This module does NOT duplicate base_model's training loop. It extends it by:
  1. Monkey-patching build_skip_variant to handle the "LyASC" variant name
  2. Wrapping the model post-construction to attach estimators and hooks
  3. Overriding _build_optimizer to handle beta_l's param group
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple
from functools import partial

import torch
import torch.nn as nn
import importlib.util as _ilu

# Resolve roots
_LYASC_ROOT = Path(__file__).resolve().parents[2]  # lyasc_model/
_BASE_ROOT = _LYASC_ROOT.parent / "base_model"

# Ensure sys.path has base_model so top-level imports resolve
if str(_BASE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASE_ROOT))
if str(_LYASC_ROOT) not in sys.path:
    sys.path.insert(0, str(_LYASC_ROOT))

# Force top-level 'src' package __path__ to look in BOTH workspaces
import src
for root_path in (_BASE_ROOT, _LYASC_ROOT):
    sp = str(root_path / "src")
    if sp not in src.__path__:
        src.__path__.append(sp)

# Extend subpackage __path__s so submodules in either workspace are discovered
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

# base_model imports
from src.data.cifar_datamodule import build_dataloaders
from src.models.preact_resnet import preact_resnet, PreActBasicBlock
from src.models import preact_resnet as _base_preact_mod
from src.models.skip_variants import build_skip_variant as _base_build_skip_variant
from src.training.config import load_config, ExperimentConfig
from src.utils.seed import set_seed

# lyasc_model imports
from src.models.lyasc_skip import LyASCSkip
from src.models.stability_estimator import StabilityEstimator
from src.utils.logging_utils import log_decision, get_run_logger, log_env_report
from src.tracking.mlflow_tracking import (
    init_tracking, tracked_run, log_run_params,
    log_step_metrics, log_final_result, log_artifacts,
)

from src.training.lyasc_config import load_lyasc_config, load_lyasc_variant_raw_dicts, LyASCConfig

logger = logging.getLogger(__name__)

DEFAULT_GRAD_CLIP_MAX_NORM = 5.0


# ---------------------------------------------------------------------------
# LyASC variant factory
# ---------------------------------------------------------------------------

def _lyasc_build_skip_variant(
    variant_cfg: dict,
    channels: int,
    spatial_size: int,
    lyasc_cfg: Optional[LyASCConfig] = None,
) -> nn.Module:
    """Extended build_skip_variant that handles any variant starting with 'LyASC'.
    For all other variants, delegates to the base_model factory unchanged."""
    name = variant_cfg.get("name", "")
    if name.startswith("LyASC"):
        cfg = lyasc_cfg or LyASCConfig()
        g_val = str(variant_cfg.get("G", "")).lower()
        use_ln = (
            cfg.use_layer_norm
            or ("+LN" in name)
            or ("plusLN" in name)
            or variant_cfg.get("use_layer_norm", False)
            or g_val in ("layer_norm", "ln")
        )
        use_bn = ("+BN" in name) or ("plusBN" in name) or variant_cfg.get("use_batch_norm", False) or g_val in ("batch_norm", "bn")
        if use_bn:
            use_ln = False  # BN overrides LN if explicitly requested

        skip_scale = variant_cfg.get("skip_scale", variant_cfg.get("lambda", 1.0))
        residual_scale = variant_cfg.get("residual_scale", variant_cfg.get("scale_residual_instead", 1.0))

        cold_start = variant_cfg.get("cold_start_strategy")
        if not cold_start:
            if "warmup" in name.lower():
                cold_start = "warmup"
            elif "damped" in name.lower():
                cold_start = "damped"
            else:
                cold_start = cfg.cold_start_strategy

        warmup_iters = variant_cfg.get("cold_start_warmup_iters", cfg.cold_start_warmup_iters)

        estimator = StabilityEstimator(k=cfg.k, N=cfg.N, ema_decay=cfg.ema_decay)
        return LyASCSkip(
            channels=channels,
            lambda0=cfg.lambda0,
            estimator=estimator,
            use_layer_norm=use_ln,
            use_batch_norm=use_bn,
            skip_scale=skip_scale,
            residual_scale=residual_scale,
            cold_start_strategy=cold_start,
            cold_start_warmup_iters=warmup_iters,
        )
    return _base_build_skip_variant(variant_cfg, channels, spatial_size)


# ---------------------------------------------------------------------------
# Estimator hook infrastructure (Q1 approved resolution)
# ---------------------------------------------------------------------------

def _collect_lyasc_blocks(model: nn.Module) -> List[Tuple[str, PreActBasicBlock, LyASCSkip]]:
    """Find all PreActBasicBlocks that use LyASCSkip as their skip_variant."""
    blocks = []
    for name, module in model.named_modules():
        if type(module).__name__ == "PreActBasicBlock" and type(getattr(module, "skip_variant", None)).__name__ == "LyASCSkip":
            blocks.append((name, module, module.skip_variant))
    return blocks


class _EstimatorHookState:
    """Mutable state shared with pre-forward hooks."""
    def __init__(self):
        self.global_step: int = 0


def _make_estimator_hook(block: PreActBasicBlock, lyasc_skip: LyASCSkip, state: _EstimatorHookState):
    """Create a pre-forward hook that calls estimator.forward(F_l, x_l, step).

    Per Q1 approval: disabled in eval mode (no power iteration during validation).
    """
    def hook(module, args):
        if not module.training:
            return  # Q1: disabled in eval mode

        x = args[0] if args else None
        if x is None:
            return

        estimator = lyasc_skip.estimator
        if estimator is None:
            return

        # F_l is the residual branch function: BN1->ReLU->Conv1->BN2->ReLU->Conv2
        def F_l(inp: torch.Tensor) -> torch.Tensor:
            out = torch.nn.functional.relu(block.bn1(inp), inplace=False)
            out = block.conv1(out)
            out = block.conv2(torch.nn.functional.relu(block.bn2(out), inplace=False))
            return out

        estimator.forward(F_l, x, state.global_step)

    return hook


def _attach_estimator_hooks(model: nn.Module) -> Tuple[_EstimatorHookState, list]:
    """Attach pre-forward hooks to all LyASC blocks. Returns the shared state
    object and the list of hook handles (for cleanup)."""
    state = _EstimatorHookState()
    handles = []
    blocks = _collect_lyasc_blocks(model)

    for name, block, lyasc_skip in blocks:
        lyasc_skip.step_provider = lambda: state.global_step
        hook = _make_estimator_hook(block, lyasc_skip, state)
        handle = block.register_forward_pre_hook(hook)
        handles.append(handle)
        logger.info("Attached estimator hook to block: %s", name)

    logger.info("Total LyASC blocks with estimator hooks: %d", len(handles))
    return state, handles


# ---------------------------------------------------------------------------
# Optimizer (extends base_model's pattern with beta_l exclusion — Q5)
# ---------------------------------------------------------------------------

def _build_lyasc_optimizer(
    model: nn.Module,
    cfg: ExperimentConfig,
    lyasc_cfg: LyASCConfig,
) -> torch.optim.Optimizer:
    """SGD with momentum. beta_l excluded from weight decay (Q5 approved),
    optionally with its own LR multiplier."""
    decay_params = []
    no_decay_params = []
    beta_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".beta_l"):
            beta_params.append(param)
        elif name.endswith(".skip_variant.w"):
            # wSkip+LN's w — same treatment as base_model
            no_decay_params.append(param)
        elif param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    opt_cfg = cfg.optimizer
    base_lr = cfg.lr_schedule.initial_lr
    param_groups = [
        {"params": decay_params, "weight_decay": opt_cfg.weight_decay, "lr": base_lr},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": base_lr},
    ]
    if beta_params:
        param_groups.append({
            "params": beta_params,
            "weight_decay": 0.0,
            "lr": base_lr * lyasc_cfg.beta_lr_multiplier,
        })

    optimizer = torch.optim.SGD(
        param_groups,
        lr=base_lr,
        momentum=opt_cfg.momentum,
        nesterov=opt_cfg.nesterov,
    )
    return optimizer


# ---------------------------------------------------------------------------
# LyASC-specific diagnostics
# ---------------------------------------------------------------------------

def _lyasc_diagnostics(model: nn.Module) -> Optional[dict]:
    """Collect alpha_l and beta_l statistics across all LyASC blocks."""
    blocks = _collect_lyasc_blocks(model)
    if not blocks:
        return None

    alphas = []
    betas = []
    lambdas = []
    for name, block, lyasc_skip in blocks:
        lam = lyasc_skip.estimator.current_value()
        alpha = torch.sigmoid(lyasc_skip.beta_l * (lam - lyasc_skip.lambda0))
        alphas.append(alpha.item())
        betas.append(lyasc_skip.beta_l.item())
        lambdas.append(lam.item())

    return {
        "alpha_mean": sum(alphas) / len(alphas),
        "alpha_std": (sum((a - sum(alphas)/len(alphas))**2 for a in alphas) / len(alphas)) ** 0.5,
        "alpha_min": min(alphas),
        "alpha_max": max(alphas),
        "beta_mean": sum(betas) / len(betas),
        "beta_std": (sum((b - sum(betas)/len(betas))**2 for b in betas) / len(betas)) ** 0.5,
        "lambda_mean": sum(lambdas) / len(lambdas),
        "lambda_std": (sum((l - sum(lambdas)/len(lambdas))**2 for l in lambdas) / len(lambdas)) ** 0.5,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_one_run(
    config_path: str,
    variant_name: str,
    seed: int,
    depth: Optional[int] = None,
    log_dir: str = "results/logs",
    checkpoint_dir: str = "results/checkpoints",
    data_dir_override: Optional[str] = None,
    num_workers: int = 4,
    base_dir: Optional[str] = None,
    grad_clip_max_norm: float = DEFAULT_GRAD_CLIP_MAX_NORM,
    total_iters_override: Optional[int] = None,
) -> dict:
    """Train one (config, variant, seed) combination with LyASC support."""

    # Load configs
    cfg = load_config(config_path, base_dir=base_dir)
    lyasc_cfg = load_lyasc_config(config_path)
    config_name = cfg.config_name

    # Resolve variant
    variant_cfg_obj = next(
        (v for v in cfg.variants if v.name == variant_name), None
    )
    if variant_cfg_obj is None:
        raise ValueError(
            f"Variant '{variant_name}' not found in config. "
            f"Available: {[v.name for v in cfg.variants]}"
        )
    variant_cfg = variant_cfg_obj.raw_dict
    raw_variant_dicts = load_lyasc_variant_raw_dicts(config_path)
    if variant_name in raw_variant_dicts:
        variant_cfg = {**variant_cfg, **raw_variant_dicts[variant_name]}

    # Resolve depth
    if depth is None:
        depth = cfg.get_depth()

    # Resolve directories to absolute paths under _LYASC_ROOT if relative
    def _anchor_path(p_str: str) -> str:
        if os.path.isabs(p_str):
            return p_str
        p = Path(p_str)
        if p.parts and p.parts[0] == _LYASC_ROOT.name:
            return str((_LYASC_ROOT / Path(*p.parts[1:])).resolve())
        return str((_LYASC_ROOT / p).resolve())

    log_dir = _anchor_path(log_dir)
    checkpoint_dir = _anchor_path(checkpoint_dir)

    # Logging
    run_logger = get_run_logger(config_name, variant_name, seed, log_dir=log_dir)

    safe_variant = variant_name.replace("+", "plus").replace("/", "_").replace(" ", "_")
    result_json_path = os.path.join(
        log_dir, f"{config_name}__{safe_variant}__depth{depth}__seed{seed}.json"
    )
    if os.path.exists(result_json_path):
        run_logger.info("Result already exists at %s, skipping.", result_json_path)
        with open(result_json_path) as f:
            return json.load(f)

    # Resume checkpoint: written every RESUME_INTERVAL iters, deleted on clean finish.
    RESUME_INTERVAL = 4000
    resume_ckpt_path = os.path.join(
        checkpoint_dir,
        f"resume__{config_name}__{safe_variant}__depth{depth}__seed{seed}.pt",
    )

    is_lyasc = variant_name.startswith("LyASC") or (lyasc_cfg and lyasc_cfg.use_layer_norm)

    # Log LyASC-specific decisions
    if is_lyasc:
        log_decision(
            key="lyasc_hyperparameters",
            value=f"k={lyasc_cfg.k}, N={lyasc_cfg.N}, ema_decay={lyasc_cfg.ema_decay}, "
                  f"lambda0={lyasc_cfg.lambda0}, cold_start={lyasc_cfg.cold_start_strategy}",
            rationale="LyASC estimator hyperparameters for this run, from config YAML.",
            log_dir=log_dir,
        )

    run_logger.info(
        "=== Starting training: config=%s  variant=%s  depth=%d  seed=%d ===",
        config_name, variant_name, depth, seed
    )

    # Seeding
    set_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_logger.info("Device: %s", device)

    # Data
    data_dir = data_dir_override if data_dir_override else cfg.data_dir
    train_loader, test_loader = build_dataloaders(
        dataset=cfg.dataset,
        data_dir=data_dir,
        batch_size=cfg.batch_size,
        augmentation_cfg={
            "pad": cfg.augmentation.pad,
            "crop_size": cfg.augmentation.crop_size,
            "random_flip": cfg.augmentation.random_flip,
        },
        seed=seed,
        num_workers=num_workers,
        log_dir=log_dir,
    )

    # Model — use the LyASC-aware factory
    num_classes = 10 if cfg.dataset == "cifar10" else 100

    # Monkey-patch build_skip_variant in the base_model's preact_resnet module
    original_bsv = _base_preact_mod.build_skip_variant
    _base_preact_mod.build_skip_variant = partial(_lyasc_build_skip_variant, lyasc_cfg=lyasc_cfg)

    model = preact_resnet(depth=depth, num_classes=num_classes, variant_cfg=variant_cfg)

    # Restore original
    _base_preact_mod.build_skip_variant = original_bsv

    model = model.to(device)

    if is_lyasc and lyasc_cfg.freeze_beta:
        for name, param in model.named_parameters():
            if name.endswith(".beta_l"):
                param.requires_grad = False
        run_logger.info("Step 3 mode: beta_l parameters are FROZEN at 0 (requires_grad=False).")

    total_params = sum(p.numel() for p in model.parameters())
    run_logger.info(
        "Model: PreAct-ResNet-%d  variant=%s  params=%s  blocks=%d",
        depth, variant_name, f"{total_params:,}", model.count_basic_blocks()
    )

    # Attach estimator hooks for LyASC blocks (Q1)
    hook_state = None
    hook_handles = []
    if is_lyasc:
        hook_state, hook_handles = _attach_estimator_hooks(model)

    # Optimizer & scheduler
    optimizer = _build_lyasc_optimizer(model, cfg, lyasc_cfg)
    sched_cfg = cfg.lr_schedule
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=sched_cfg.milestones_iters,
        gamma=sched_cfg.gamma,
    )

    criterion = nn.CrossEntropyLoss()

    # Training loop
    total_iters = total_iters_override if total_iters_override is not None else sched_cfg.total_iters
    iters_per_epoch = cfg.iters_per_epoch()
    clip_active = grad_clip_max_norm is not None and grad_clip_max_norm > 0

    # MLflow setup
    mlflow_run = None
    if lyasc_cfg.enable_mlflow:
        init_tracking(
            tracking_uri=f"sqlite:///{str(_LYASC_ROOT / 'results' / 'mlflow.db')}"
        )

    # --- Resume from mid-run checkpoint if one exists ---
    global_iter = 0
    _resume_loaded = False
    os.makedirs(checkpoint_dir, exist_ok=True)
    if os.path.exists(resume_ckpt_path):
        try:
            resume_data = torch.load(resume_ckpt_path, map_location=device)
            model.load_state_dict(resume_data["model_state_dict"])
            optimizer.load_state_dict(resume_data["optimizer_state_dict"])
            scheduler.load_state_dict(resume_data["scheduler_state_dict"])
            global_iter = resume_data["global_iter"]
            _resume_loaded = True
            run_logger.info(
                "Resumed from checkpoint at iter %d (of %d): %s",
                global_iter, sched_cfg.total_iters, resume_ckpt_path,
            )
        except Exception as exc:
            run_logger.warning(
                "Resume checkpoint load failed (%s); starting from iter 0.", exc
            )
            global_iter = 0

    train_iter = iter(train_loader)
    # Fast-forward the data iterator to align with global_iter when resuming.
    # We skip in batches so the DataLoader RNG is NOT reset — results will differ
    # from an uninterrupted run at the byte level, but this is unavoidable with
    # a stateless DataLoader and is scientifically acceptable for Phase C.
    if _resume_loaded and global_iter > 0:
        run_logger.info("Fast-forwarding data iterator by %d batches...", global_iter)
        for _skip in range(global_iter % len(train_loader)):
            try:
                next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
        run_logger.info("Data iterator fast-forward complete.")

    start_time = time.time()

    run_logger.info(
        "Training schedule: %d total iterations (~%.1f epochs), "
        "LR milestones at iters %s",
        total_iters, total_iters / iters_per_epoch, sched_cfg.milestones_iters,
    )

    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    running_grad_norm_sum = 0.0
    running_grad_norm_count = 0
    running_clipped_count = 0
    running_beta_grad_norm_sum = 0.0
    running_other_grad_norm_sum = 0.0

    # Context manager for MLflow
    import contextlib
    mlflow_ctx = tracked_run(
        config_name, variant_name, depth, seed,
        extra_tags={"dataset": cfg.dataset, "is_lyasc": str(is_lyasc)}
    ) if lyasc_cfg.enable_mlflow else contextlib.nullcontext()

    with mlflow_ctx as mlf_run:
        if mlf_run is not None:
            params_to_log = {
                "variant_name": variant_name,
                "depth": depth,
                "seed": seed,
                "dataset": cfg.dataset,
                "batch_size": cfg.batch_size,
                "initial_lr": cfg.lr_schedule.initial_lr,
                "weight_decay": cfg.optimizer.weight_decay,
                "grad_clip_max_norm": grad_clip_max_norm,
            }
            if is_lyasc:
                params_to_log.update({
                    "k": lyasc_cfg.k,
                    "N": lyasc_cfg.N,
                    "ema_decay": lyasc_cfg.ema_decay,
                    "lambda0": lyasc_cfg.lambda0,
                    "cold_start_strategy": lyasc_cfg.cold_start_strategy,
                    "beta_lr_multiplier": lyasc_cfg.beta_lr_multiplier,
                })
            log_run_params(mlf_run, params_to_log)

        while global_iter < total_iters:
            try:
                images, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, labels = next(train_iter)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Update hook state step counter BEFORE forward pass
            if hook_state is not None:
                hook_state.global_step = global_iter

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            # Diagnostic: split grad norm for beta_l vs rest
            beta_grads = []
            other_grads = []
            for pname, p in model.named_parameters():
                if p.grad is None:
                    continue
                (beta_grads if pname.endswith(".beta_l") else other_grads).append(
                    p.grad.detach().flatten()
                )
            beta_grad_norm = float(torch.norm(torch.cat(beta_grads))) if beta_grads else 0.0
            other_grad_norm = float(torch.norm(torch.cat(other_grads))) if other_grads else 0.0

            # Gradient clipping
            if clip_active:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_max_norm
                )
                grad_norm_value = float(total_grad_norm)
                if grad_norm_value > grad_clip_max_norm:
                    running_clipped_count += 1
            else:
                grad_norm_value = float(
                    torch.norm(torch.stack([
                        p.grad.detach().norm() for p in model.parameters()
                        if p.grad is not None
                    ]))
                )
            running_grad_norm_sum += grad_norm_value
            running_grad_norm_count += 1
            running_beta_grad_norm_sum += beta_grad_norm
            running_other_grad_norm_sum += other_grad_norm

            optimizer.step()
            scheduler.step()

            # Accumulate metrics
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                running_correct += (pred == labels).sum().item()
                running_total += labels.size(0)
                running_loss += loss.item() * labels.size(0)

            global_iter += 1
            current_epoch = global_iter / iters_per_epoch
            current_lr = optimizer.param_groups[0]["lr"]

            # Log every 1000 iterations
            if global_iter % 1000 == 0 or global_iter == total_iters:
                avg_loss = running_loss / running_total
                train_acc = 100.0 * running_correct / running_total
                avg_grad_norm = (
                    running_grad_norm_sum / running_grad_norm_count
                    if running_grad_norm_count > 0 else 0.0
                )
                elapsed = time.time() - start_time

                clip_note = (
                    f" (clipped {running_clipped_count}/{running_grad_norm_count} steps)"
                    if running_clipped_count > 0 else ""
                )

                avg_beta_gn = running_beta_grad_norm_sum / running_grad_norm_count
                avg_other_gn = running_other_grad_norm_sum / running_grad_norm_count
                beta_note = (
                    f"  [beta_grad_norm={avg_beta_gn:.3f}  other_grad_norm={avg_other_gn:.3f}]"
                    if avg_beta_gn > 0 else ""
                )

                run_logger.info(
                    "iter=%6d  epoch=%.1f  lr=%.5f  loss=%.4f  train_acc=%.2f%%  "
                    "grad_norm=%.3f%s  elapsed=%.0fs%s",
                    global_iter, current_epoch, current_lr, avg_loss, train_acc,
                    avg_grad_norm, clip_note, elapsed, beta_note,
                )

                # LyASC diagnostics
                lyasc_diag = _lyasc_diagnostics(model) if is_lyasc else None
                if lyasc_diag is not None:
                    run_logger.info(
                        "         alpha: mean=%.4f  std=%.4f  min=%.4f  max=%.4f  |  "
                        "beta: mean=%.4f  std=%.4f  |  lambda: mean=%.4f  std=%.4f",
                        lyasc_diag["alpha_mean"], lyasc_diag["alpha_std"],
                        lyasc_diag["alpha_min"], lyasc_diag["alpha_max"],
                        lyasc_diag["beta_mean"], lyasc_diag["beta_std"],
                        lyasc_diag["lambda_mean"], lyasc_diag["lambda_std"],
                    )

                # MLflow step metrics
                step_metrics = {
                    "train_loss": avg_loss,
                    "train_accuracy": train_acc,
                    "grad_norm": avg_grad_norm,
                    "lr": current_lr,
                }
                if lyasc_diag:
                    step_metrics.update(lyasc_diag)
                if avg_beta_gn > 0:
                    step_metrics["beta_grad_norm"] = avg_beta_gn
                    step_metrics["other_grad_norm"] = avg_other_gn
                log_step_metrics(mlf_run, global_iter, step_metrics)

                running_loss = 0.0
                running_correct = 0
                running_total = 0
                running_grad_norm_sum = 0.0
                running_grad_norm_count = 0
                running_clipped_count = 0
                running_beta_grad_norm_sum = 0.0
                running_other_grad_norm_sum = 0.0

            # Save resume checkpoint every RESUME_INTERVAL iters
            if global_iter % RESUME_INTERVAL == 0 and global_iter < total_iters:
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    "global_iter": global_iter,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config_name": config_name,
                    "variant_name": variant_name,
                    "depth": depth,
                    "seed": seed,
                }, resume_ckpt_path)
                run_logger.info("Resume checkpoint saved at iter %d: %s", global_iter, resume_ckpt_path)

            # Evaluate at milestones and at end
            if global_iter in sched_cfg.milestones_iters or global_iter == total_iters:
                model.eval()
                test_error = _evaluate(model, test_loader, device)
                run_logger.info(
                    ">> EVAL  iter=%d  epoch=%.1f  test_error=%.2f%%",
                    global_iter, current_epoch, test_error
                )
                log_step_metrics(mlf_run, global_iter, {"test_error": test_error})
                model.train()

        # Final evaluation
        model.eval()
        final_test_error = _evaluate(model, test_loader, device)
        total_time = time.time() - start_time
        run_logger.info(
            "=== DONE  config=%s  variant=%s  depth=%d  seed=%d "
            "final_test_error=%.2f%%  time=%.0fs ===",
            config_name, variant_name, depth, seed, final_test_error, total_time
        )

        # Save checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_name = f"{config_name}__{safe_variant}__depth{depth}__seed{seed}.pt"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        checkpoint = {
            "config_name": config_name,
            "variant_name": variant_name,
            "depth": depth,
            "seed": seed,
            "final_test_error": final_test_error,
            "total_iters": global_iter,
            "grad_clip_max_norm": grad_clip_max_norm if clip_active else None,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if is_lyasc:
            checkpoint["lyasc_diagnostics"] = _lyasc_diagnostics(model)
        torch.save(checkpoint, ckpt_path)
        run_logger.info("Checkpoint saved: %s", ckpt_path)

        # Save result JSON
        os.makedirs(log_dir, exist_ok=True)
        result = {
            "config_name": config_name,
            "variant_name": variant_name,
            "depth": depth,
            "seed": seed,
            "final_test_error": final_test_error,
            "total_iters": global_iter,
            "wall_clock_seconds": total_time,
            "grad_clip_max_norm": grad_clip_max_norm if clip_active else None,
        }
        if is_lyasc:
            result["lyasc_diagnostics"] = _lyasc_diagnostics(model)
            active_cold_start = (
                variant_cfg.get("cold_start_strategy")
                or ("warmup" if "warmup" in variant_name.lower() else ("damped" if "damped" in variant_name.lower() else lyasc_cfg.cold_start_strategy))
            )
            active_use_ln = (
                lyasc_cfg.use_layer_norm or ("+LN" in variant_name) or ("plusLN" in variant_name) or variant_cfg.get("use_layer_norm", False)
            )
            result["lyasc_config"] = {
                "k": lyasc_cfg.k, "N": lyasc_cfg.N, "ema_decay": lyasc_cfg.ema_decay,
                "lambda0": lyasc_cfg.lambda0, "cold_start": active_cold_start,
                "use_layer_norm": active_use_ln,
            }
        with open(result_json_path, "w") as f:
            json.dump(result, f, indent=2)
        run_logger.info("Result JSON: %s", result_json_path)

        # Delete resume checkpoint — run is complete, no recovery needed.
        if os.path.exists(resume_ckpt_path):
            os.remove(resume_ckpt_path)
            run_logger.info("Resume checkpoint deleted (run complete): %s", resume_ckpt_path)

        # MLflow final
        log_final_result(mlf_run, result)
        log_artifacts(mlf_run, [result_json_path, os.path.join(log_dir, "decisions.log"), config_path])

    # Cleanup hooks and GPU memory
    for h in hook_handles:
        h.remove()

    try:
        del model, optimizer, scheduler, train_loader, test_loader, train_iter
    except NameError:
        pass

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result



def _evaluate(model: nn.Module, loader, device: torch.device) -> float:
    """Compute top-1 error rate (%) on the given DataLoader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100.0 * (1.0 - correct / total)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train one (config, variant, seed) with LyASC support."
    )
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    parser.add_argument("--variant", required=True, help="Variant name string")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--log_dir", default="results/logs")
    parser.add_argument("--checkpoint_dir", default="results/checkpoints")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_dir", default=None)
    parser.add_argument("--grad_clip_max_norm", type=float, default=DEFAULT_GRAD_CLIP_MAX_NORM)
    parser.add_argument("--env_report", action="store_true")
    args = parser.parse_args()

    if args.env_report:
        log_env_report(log_dir=args.log_dir)

    result = train_one_run(
        config_path=args.config,
        variant_name=args.variant,
        seed=args.seed,
        depth=args.depth,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        data_dir_override=args.data_dir,
        num_workers=args.num_workers,
        base_dir=args.base_dir,
        grad_clip_max_norm=args.grad_clip_max_norm,
    )
    print(f"\nFINAL_RESULT: {result}")


if __name__ == "__main__":
    main()
