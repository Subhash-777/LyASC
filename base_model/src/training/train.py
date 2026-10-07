"""Single-run training entrypoint, per plan.md Section 5 and Section 7 (runbook
steps 4-6).

CLI: python -m src.training.train --config <path> --variant <name> --seed <int>
      [--depth <int>]  [--log_dir <path>]  [--checkpoint_dir <path>]
      [--data_dir <override>]  [--num_workers <int>]  [--grad_clip_max_norm <float>]

The --depth argument is needed for Table 6 which runs multiple depths; for
Tables 1/3/4 the depth is implicit in the config model field.

DECISION (plan.md Section 8 "Weight decay scope" risk):
    BN/LN affine parameters and all biases are EXCLUDED from weight decay.
    This is common practice (He et al. do not explicitly state scope, but
    excluding BN params is the standard PyTorch convention). Logged in
    decisions.log at startup.

DECISION (added post-hoc, not in original plan.md Section 8 list):
    Gradient clipping (max_norm=5.0 by default) is applied every step,
    between loss.backward() and optimizer.step(). The base paper does not
    specify clipping. This was added after observing hard, permanent
    collapse to loss=ln(num_classes) on LN-augmented / lambda-scaled
    variants (2xSkip+LN, 2rSkip+LN) during the unclipped lr=0.1 phase,
    before the first scheduled LR drop at iter 32000 — a classic unclipped-
    gradient-spike signature (no NaNs observed, clean training right up to
    a sudden permanent flatline). 1xSkip trained correctly without clipping,
    consistent with it having a smaller/simpler gradient path than the LN-
    and lambda-augmented branches. Logged in decisions.log at startup.

DECISION (added post-hoc, resolved after further debugging):
    weight_decay_scope_correction. The weight-decay-exclusion check below
    (`param.ndim <= 1 or name.endswith(".bias")`) is intended to exclude all
    BN/LN affine parameters from weight decay (per the "weight_decay_scope"
    decision above). This worked correctly for BatchNorm and for the *new*
    channel-wise LayerNorm2d (src/models/skip_variants.py), whose weight/
    bias are both 1-D (shape (C,)). It did NOT work correctly for the
    *previous* nn.LayerNorm([C, H, W]) implementation, whose affine "weight"
    parameter was 3-D (shape (C, H, W)) and is not named ".bias" — so it was
    silently falling into the weight-decayed group, contrary to the intended
    policy. This was discovered while debugging the 2xSkip+LN mid-training
    instability (13.31% vs. paper's 6.29%) and is a second, independent
    contributor alongside the LayerNorm-axis bug fixed in skip_variants.py.
    No code change was needed here once LayerNorm2d replaced the old LN
    (its 1-D affine params are now correctly excluded by the existing
    ndim<=1 check) — logged here for auditability per plan.md's "log every
    assumption/decision, don't bury it silently" discipline.

DECISION (added post-hoc, resolved after the weight_decay_scope_w_parameter
fix FAILED to resolve wSkip+LN's instability):
    w_gate_learning_rate. Excluding LearnableVectorSkip.w from weight decay
    did not fix anything: grad_norm at iter 1000 was 1132 before that fix and
    1105 after — functionally unchanged. This ruled out weight decay (a ~2e-4
    per-step penalty) as the cause of a gradient already in the thousands.

    Root-cause reasoning: LN's own affine `weight` gradient is a sum of
    (upstream_grad * normalized_value) across all spatial positions and batch
    samples for a channel. The normalized_value is, by construction, zero-mean
    across channels at each position, so positive/negative contributions to
    that sum largely cancel. w's gradient is a sum of (upstream_grad * raw_x)
    over the same large number of terms (N*H*W per channel per block), but
    raw_x has no such zero-mean guarantee — if a channel's raw activations are
    systematically biased away from zero (plausible after a ReLU-derived
    feature map), the sum does not cancel and can accumulate into a
    structurally oversized gradient. This is a property of the composition
    (a per-channel gate applied directly to raw, unnormalized skip values,
    with 54 independent instances of it in series), not a coding bug.

    Mitigation: w receives its own optimizer parameter group at a reduced
    effective learning rate (w_lr_multiplier, default 0.1x base LR) — the
    standard treatment for gate/mixing parameters prone to this pattern
    (comparable to how highway-network transform gates are often trained
    conservatively relative to the main weights). A diagnostic split of
    grad_norm into w-only vs. everything-else is also logged every 1000
    iterations so this hypothesis is verifiable directly from the training
    log rather than inferred indirectly.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Allow running as `python -m src.training.train` from the workspace root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.cifar_datamodule import build_dataloaders
from src.models.preact_resnet import preact_resnet
from src.training.config import load_config, ExperimentConfig, VariantConfig
from src.utils.seed import set_seed
from src.utils.logging_utils import get_run_logger, log_env_report, log_decision

logger = logging.getLogger(__name__)


# Default gradient-clipping threshold. See "DECISION (added post-hoc...)" above.
DEFAULT_GRAD_CLIP_MAX_NORM = 5.0

# Default LR multiplier for LearnableVectorSkip.w (wSkip+LN's gate parameter).
# See "w_gate_learning_rate" decision in the module docstring.
DEFAULT_W_LR_MULTIPLIER = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_optimizer(
    model: nn.Module,
    cfg: ExperimentConfig,
    w_lr_multiplier: float = 0.1,
) -> torch.optim.Optimizer:
    """SGD with momentum, weight decay applied only to weight tensors.

    BN/LN affine params and biases are excluded from weight decay per the
    decision logged at startup (plan.md Section 8). This correctly covers
    BatchNorm2d and the channel-wise LayerNorm2d (src/models/skip_variants.py)
    since both have 1-D affine parameters — see the
    "weight_decay_scope_correction" decision above for the history here.

    LearnableVectorSkip.w (wSkip+LN's per-channel skip gate, shape (C,1,1))
    gets its OWN parameter group: no weight decay (per
    "weight_decay_scope_w_parameter") AND a reduced learning rate
    (w_lr_multiplier x base LR, default 0.1x) — per the "w_gate_learning_rate"
    decision (see module docstring) added after excluding w from weight decay
    alone failed to resolve wSkip+LN's grad_norm~1100 instability. For every
    other variant, w_gate_params is simply empty and this group is a no-op.
    """
    decay_params = []
    no_decay_params = []
    w_gate_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".skip_variant.w"):
            w_gate_params.append(param)
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
    if w_gate_params:
        param_groups.append({
            "params": w_gate_params,
            "weight_decay": 0.0,
            "lr": base_lr * w_lr_multiplier,
        })
    optimizer = torch.optim.SGD(
        param_groups,
        lr=base_lr,
        momentum=opt_cfg.momentum,
        nesterov=opt_cfg.nesterov,
    )
    return optimizer


def _w_value_stats(model: nn.Module) -> Optional[dict]:
    """Snapshot statistics of every LearnableVectorSkip.w parameter's CURRENT
    VALUE (not gradient) across the whole model, at the moment this is called.

    ADDED — see decisions.log "w_value_trajectory" entry. The w-gate LR
    multiplier and w-only gradient-norm diagnostic both ruled out w's
    gradient magnitude as the cause of wSkip+LN's persistent instability
    (w_grad_norm=69 vs. other_grad_norm=1103 at iter 1000 — w itself is not
    the dominant gradient source). A control run of 1xSkip+LN (no w
    parameter at all) showed the SAME iter-1000 grad_norm spike but
    recovered cleanly by iter 2000 and converged normally — proving the
    initial spike is a universal, self-healing LayerNorm cold-start
    transient, not the terminal cause of wSkip+LN's failure. The open
    question is therefore what happens to w's VALUES (not gradients) during
    or after that chaotic window that 1xSkip+LN structurally cannot
    experience (it has no per-channel, per-block learnable asymmetry for
    that chaos to get baked into). This function answers that by returning
    global min/max/mean/std across every w parameter's current value, so it
    can be tracked every 1000 iterations and the exact point of divergence
    (if any) identified directly from the training log.

    Returns None if the model has no w parameters (every variant except
    wSkip+LN) — callers should skip logging entirely in that case.
    """
    all_w = []
    for name, param in model.named_parameters():
        if name.endswith(".skip_variant.w"):
            all_w.append(param.detach().flatten())
    if not all_w:
        return None
    cat = torch.cat(all_w)
    return {
        "mean": float(cat.mean()),
        "std": float(cat.std()),
        "min": float(cat.min()),
        "max": float(cat.max()),
        "mean_abs": float(cat.abs().mean()),
    }


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
) -> torch.optim.lr_scheduler.MultiStepLR:
    """MultiStepLR with iteration-level milestones (plan.md Section 5)."""
    sched_cfg = cfg.lr_schedule
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=sched_cfg.milestones_iters,
        gamma=sched_cfg.gamma,
    )


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
    error_rate = 100.0 * (1.0 - correct / total)
    return error_rate


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
    w_lr_multiplier: float = DEFAULT_W_LR_MULTIPLIER,
) -> dict:
    """Train one (config, variant, seed) combination.

    Args:
        config_path:      Path to the configs/*.yaml file.
        variant_name:     Variant name string (must exist in config variants list).
        seed:             Integer seed.
        depth:            Override the depth (needed for Table 6 multi-depth configs).
        log_dir:          Directory for training logs.
        checkpoint_dir:   Directory to save final checkpoint.
        data_dir_override: Override data_dir from config (for CI/testing).
        num_workers:      DataLoader workers.
        base_dir:         Workspace root for resolving relative data_dir.
        grad_clip_max_norm: Max L2 norm for gradient clipping, applied every
                            step between loss.backward() and optimizer.step().
                            Set to 0 or a negative value to disable clipping
                            entirely (kept configurable rather than hardcoded
                            so this can be ablated/tuned if 5.0 turns out to
                            be too loose or too tight for a given variant).
        w_lr_multiplier:    Learning-rate multiplier applied to
                            LearnableVectorSkip.w's optimizer param group
                            (wSkip+LN only; no-op for every other variant).
                            Default 0.1 — see "w_gate_learning_rate" decision
                            in the module docstring.

    Returns:
        Dict with keys: variant_name, seed, final_test_error, depth, config_name.
    """
    cfg = load_config(config_path, base_dir=base_dir)
    config_name = cfg.config_name

    # Resolve variant config
    variant_cfg_obj = next(
        (v for v in cfg.variants if v.name == variant_name), None
    )
    if variant_cfg_obj is None:
        raise ValueError(
            f"Variant '{variant_name}' not found in config. "
            f"Available: {[v.name for v in cfg.variants]}"
        )
    variant_cfg = variant_cfg_obj.raw_dict

    # Resolve depth
    if depth is None:
        depth = cfg.get_depth()

    # Set up logging first
    run_logger = get_run_logger(config_name, variant_name, seed, log_dir=log_dir)

    safe_variant = variant_name.replace("+", "plus").replace("/", "_").replace(" ", "_")
    result_json_path = os.path.join(log_dir, f"{config_name}__{safe_variant}__depth{depth}__seed{seed}.json")
    if os.path.exists(result_json_path):
        run_logger.info("Result already exists at %s, skipping.", result_json_path)
        with open(result_json_path) as f:
            return json.load(f)

    # Log decisions (once per run — idempotent entries are fine in the append-mode decisions.log)
    log_decision(
        key="train_skip_if_exists",
        value="skip_check_enabled",
        rationale=(
            "Added directly in train_one_run. Previously, train.py lacked any native "
            "check for whether a result JSON already existed, meaning any relaunch "
            "blindly overwrote completed results. This skip-logic safely bypasses "
            "the run if the JSON artifact is already present."
        ),
        log_dir=log_dir,
    )
    log_decision(
        key="shortcut_policy",
        value="option_A_zero_pad",
        rationale=(
            "He et al. (2016b) CIFAR experiments use Option A (identity with "
            "zero-padding for extra channels, avg-pool for stride). Option B "
            "(1x1 projection) adds ~3.3K params per transition block. Option A "
            "keeps the network parameter-light and is the published reference. "
            "The base paper does not override this."
        ),
        log_dir=log_dir,
    )
    log_decision(
        key="lambda_at_downsampling_blocks",
        value="applied_uniformly",
        rationale=(
            "Lambda-scaling and G normalisation apply to all blocks including "
            "stage-transition (downsampling) blocks. The base paper describes "
            "the skip variant as a per-block rule without any exception for "
            "downsampling blocks. Treating them differently would require an "
            "undocumented special case."
        ),
        log_dir=log_dir,
    )
    log_decision(
        key="weight_decay_scope",
        value="exclude_BN_LN_affine_and_bias",
        rationale=(
            "BN/LN affine parameters (weight, bias) and all bias terms are "
            "excluded from weight decay. This is standard PyTorch practice; "
            "neither the base paper nor He et al. (2016b) state the scope "
            "explicitly. Excluding is the most common published convention."
        ),
        log_dir=log_dir,
    )
    log_decision(
        key="weight_decay_scope_w_parameter",
        value="excluded",
        rationale=(
            "LearnableVectorSkip.w (wSkip+LN's per-channel skip gate) has shape "
            "(C,1,1) — 3-D — so it was NOT caught by the ndim<=1 exclusion check "
            "and was being weight-decayed like a regular conv weight. Discovered "
            "after wSkip+LN reproduced at 66-73% test error across all 3 seeds "
            "(vs. paper's 8.66%), with erratic non-uniform per-channel w values "
            "and grad_norm spiking to >1000 early in training — inconsistent "
            "with the paper's reported uniform small-norm collapse (~0.24 avg). "
            "w is conceptually a gain parameter (like LN's weight), not a "
            "transformation matrix, so it is now excluded from weight decay "
            "by name match on '.w'."
        ),
        log_dir=log_dir,
    )
    log_decision(
        key="w_gate_learning_rate",
        value=f"multiplier={w_lr_multiplier}",
        rationale=(
            "weight_decay_scope_w_parameter (excluding w from weight decay) did "
            "NOT resolve wSkip+LN's grad_norm~1100 spike (1132 before -> 1105 "
            "after, essentially unchanged), ruling out weight decay as the cause. "
            "w's gradient sums (upstream_grad * raw_x) over all spatial positions "
            "and batch samples per channel; unlike LN's own affine gradient "
            "(summed against a zero-mean normalized value, which cancels), raw x "
            "has no such guarantee, plausibly explaining the structurally "
            "oversized gradient. A reduced LR for w's optimizer param group is "
            "applied as the standard mitigation for gate/mixing parameters with "
            "this property, analogous to conservative gate-LR treatment in "
            "highway networks. A w-only vs. rest-of-model grad_norm diagnostic "
            "split is logged every 1000 iterations to verify this directly."
        ),
        log_dir=log_dir,
    )
    log_decision(
        key="w_value_trajectory",
        value="logged_every_1000_iters",
        rationale=(
            "Both prior wSkip+LN fixes (weight_decay_scope_w_parameter, "
            "w_gate_learning_rate) are ruled out: a diagnostic split showed "
            "w_grad_norm=69 vs. other_grad_norm=1103 at iter 1000 (w's own "
            "gradient is not the dominant source), and a control run of "
            "1xSkip+LN (structurally identical at init, w=1 uniformly, but "
            "with NO learnable w at all) showed the SAME iter-1000 grad_norm "
            "spike yet recovered cleanly by iter 2000 and converged normally. "
            "This proves the initial spike is a universal, self-healing "
            "LayerNorm cold-start transient, not wSkip+LN's terminal problem. "
            "The remaining open question is what happens to w's VALUES "
            "(not gradients) during/after that window that 1xSkip+LN cannot "
            "experience, having no per-channel/per-block learnable asymmetry. "
            "w's global mean/std/min/max/mean_abs (see _w_value_stats) is now "
            "logged every 1000 iterations to identify the exact point and "
            "pattern of any divergence directly from the training log."
        ),
        log_dir=log_dir,
    )
    log_decision(
        key="layernorm_axis",
        value="channel_wise_per_position",
        rationale=(
            "LN is implemented as LayerNorm2d: normalises across the channel "
            "dimension independently at each (H, W) spatial position, matching "
            "the Transformer / Ba et al. (2016) convention the base paper cites "
            "(each spatial position is a 'token', channels are its 'features'). "
            "A prior nn.LayerNorm([C, H, W]) implementation pooled the entire "
            "spatial extent into one statistic per sample instead — a much more "
            "aggressive, different normalisation than the paper describes, and "
            "a plausible cause of mid-training instability observed on "
            "2xSkip+LN (13.31% reproduced vs. paper's 6.29%, with an accuracy "
            "dip across iters ~8000-32000 under sustained high initial LR)."
        ),
        log_dir=log_dir,
    )
    log_decision(
        key="gradient_clipping",
        value=(
            f"max_norm={grad_clip_max_norm}" if grad_clip_max_norm and grad_clip_max_norm > 0
            else "disabled"
        ),
        rationale=(
            "Added post-hoc, not specified by the base paper. LN-augmented and "
            "lambda-scaled skip variants (e.g. 2xSkip+LN, 2rSkip+LN) were observed "
            "to train cleanly then collapse permanently to loss=ln(num_classes) "
            "partway through the unclipped lr=0.1 phase, before the first scheduled "
            "LR drop at iter 32000, with no NaNs present — the standard signature of "
            "an unclipped gradient spike. Global L2-norm clipping via "
            "torch.nn.utils.clip_grad_norm_ is applied every step between "
            "loss.backward() and optimizer.step() to guard against this. "
            "1xSkip (no LN, no lambda-scaling) trained correctly even without "
            "clipping, consistent with the affected variants having a larger/more "
            "sensitive gradient path, not with generic training instability."
        ),
        log_dir=log_dir,
    )

    run_logger.info(
        "=== Starting training: config=%s  variant=%s  depth=%d  seed=%d ===",
        config_name, variant_name, depth, seed
    )
    if grad_clip_max_norm and grad_clip_max_norm > 0:
        run_logger.info(
            "Gradient clipping enabled: max_norm=%.2f (added post-hoc — base paper does not "
            "specify clipping; added after observing hard collapse to ln(num_classes) on "
            "LN-augmented variants under the unclipped lr=0.1 phase).",
            grad_clip_max_norm,
        )
    else:
        run_logger.info("Gradient clipping DISABLED (grad_clip_max_norm <= 0).")
    if variant_cfg.get("learnable_w"):
        run_logger.info(
            "w-gate LR multiplier: %.3f (LearnableVectorSkip.w trains at %.3fx "
            "the base LR — added post-hoc after weight-decay exclusion alone "
            "failed to resolve wSkip+LN's grad_norm~1100 instability).",
            w_lr_multiplier, w_lr_multiplier,
        )

    # Deterministic seeding
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

    # Model
    num_classes = 10 if cfg.dataset == "cifar10" else 100
    model = preact_resnet(depth=depth, num_classes=num_classes, variant_cfg=variant_cfg)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    run_logger.info(
        "Model: PreAct-ResNet-%d  variant=%s  params=%s  blocks=%d",
        depth, variant_name, f"{total_params:,}", model.count_basic_blocks()
    )

    # Optimizer & scheduler
    optimizer = _build_optimizer(model, cfg, w_lr_multiplier=w_lr_multiplier)
    scheduler = _build_lr_scheduler(optimizer, cfg)

    criterion = nn.CrossEntropyLoss()

    # TensorBoard writer
    tb_dir = os.path.join(log_dir, "tb", f"{config_name}__{variant_name}__seed{seed}")
    writer = SummaryWriter(log_dir=tb_dir)

    # ---- Training loop (iteration-level, not epoch-level) ----
    sched_cfg = cfg.lr_schedule
    total_iters = sched_cfg.total_iters
    iters_per_epoch = cfg.iters_per_epoch()

    global_iter = 0
    train_iter = iter(train_loader)
    start_time = time.time()

    run_logger.info(
        "Training schedule: %d total iterations (~%.1f epochs), "
        "LR milestones at iters %s (~epochs %s)",
        total_iters,
        total_iters / iters_per_epoch,
        sched_cfg.milestones_iters,
        [f"{m/iters_per_epoch:.1f}" for m in sched_cfg.milestones_iters],
    )

    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    running_grad_norm_sum = 0.0
    running_grad_norm_count = 0
    running_clipped_count = 0  # FIXED: tracks steps where clipping actually fired
    running_w_grad_norm_sum = 0.0        # NEW: w-only gradient norm diagnostic
    running_other_grad_norm_sum = 0.0    # NEW: everything-else gradient norm
    clip_active = grad_clip_max_norm is not None and grad_clip_max_norm > 0

    while global_iter < total_iters:
        # Refresh iterator at epoch boundary
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        # --- DIAGNOSTIC: split gradient norm into w-only vs. everything-else ---
        # (ADDED — see decisions.log "w_gate_learning_rate" entry). Must run
        # BEFORE clipping, since clip_grad_norm_ rescales gradients in place.
        # For every variant other than wSkip+LN this is cheap (empty w_grads
        # list, w_grad_norm_value stays 0.0) and changes nothing else.
        w_grads = []
        other_grads = []
        for pname, p in model.named_parameters():
            if p.grad is None:
                continue
            (w_grads if pname.endswith(".skip_variant.w") else other_grads).append(
                p.grad.detach().flatten()
            )
        w_grad_norm_value = float(torch.norm(torch.cat(w_grads))) if w_grads else 0.0
        other_grad_norm_value = (
            float(torch.norm(torch.cat(other_grads))) if other_grads else 0.0
        )

        # --- Gradient clipping (ADDED — see decisions.log "gradient_clipping" entry) ---
        # Must run after backward() and before step(), so the clipped gradients
        # are what the optimizer actually applies. clip_grad_norm_ returns the
        # PRE-clipping total norm, which we track for diagnostics below so any
        # future collapse can be diagnosed by looking at whether clipping was
        # firing constantly (norm >> max_norm) right before it happened.
        if clip_active:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip_max_norm
            )
            grad_norm_value = float(total_grad_norm)
            # FIXED (was previously labeling every step "(clipped)" whenever
            # clipping was merely *enabled*, regardless of whether the
            # pre-clip norm actually exceeded max_norm). Now only counted as
            # an actual clip event if the pre-clip norm exceeds the threshold.
            if grad_norm_value > grad_clip_max_norm:
                running_clipped_count += 1
        else:
            grad_norm_value = float(
                torch.norm(
                    torch.stack([
                        p.grad.detach().norm() for p in model.parameters()
                        if p.grad is not None
                    ])
                )
            )
        running_grad_norm_sum += grad_norm_value
        running_grad_norm_count += 1
        running_w_grad_norm_sum += w_grad_norm_value
        running_other_grad_norm_sum += other_grad_norm_value

        optimizer.step()
        scheduler.step()   # step every iteration (MultiStepLR counts calls)

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
            # FIXED: label now reflects how many of the steps in this interval
            # actually had their gradients clipped (pre-clip norm > max_norm),
            # not just whether clipping was enabled. Previously this printed
            # "(clipped)" unconditionally whenever clip_active was True, even
            # for steps whose grad_norm (e.g. 0.9-2.7) was well under the
            # max_norm=5.0 threshold and were never actually rescaled.
            if running_clipped_count > 0:
                clip_note = f" (clipped {running_clipped_count}/{running_grad_norm_count} steps)"
            else:
                clip_note = ""
            # NEW: w-gate diagnostic split, only printed when w actually exists
            # in this model (i.e. wSkip+LN) — avoids cluttering every other
            # variant's log with a permanently-zero field.
            avg_w_grad_norm = running_w_grad_norm_sum / running_grad_norm_count
            avg_other_grad_norm = running_other_grad_norm_sum / running_grad_norm_count
            w_note = (
                f"  [w_grad_norm=%.3f  other_grad_norm=%.3f]" % (
                    avg_w_grad_norm, avg_other_grad_norm
                )
                if avg_w_grad_norm > 0 else ""
            )
            run_logger.info(
                "iter=%6d  epoch=%.1f  lr=%.5f  loss=%.4f  train_acc=%.2f%%  "
                "grad_norm=%.3f%s  elapsed=%.0fs%s",
                global_iter, current_epoch, current_lr, avg_loss, train_acc,
                avg_grad_norm,
                clip_note,
                elapsed,
                w_note,
            )
            writer.add_scalar("train/loss", avg_loss, global_iter)
            writer.add_scalar("train/accuracy", train_acc, global_iter)
            writer.add_scalar("train/lr", current_lr, global_iter)
            writer.add_scalar("train/grad_norm", avg_grad_norm, global_iter)
            writer.add_scalar("train/clipped_fraction",
                               running_clipped_count / max(running_grad_norm_count, 1),
                               global_iter)
            if avg_w_grad_norm > 0:
                writer.add_scalar("train/w_grad_norm", avg_w_grad_norm, global_iter)
                writer.add_scalar("train/other_grad_norm", avg_other_grad_norm, global_iter)

            # NEW: w-value trajectory (snapshot of current values, not gradients).
            # See decisions.log "w_value_trajectory" entry and the _w_value_stats
            # docstring for why this is the key remaining diagnostic.
            w_stats = _w_value_stats(model)
            if w_stats is not None:
                run_logger.info(
                    "         w_values: mean=%.4f  std=%.4f  min=%.4f  max=%.4f  mean_abs=%.4f",
                    w_stats["mean"], w_stats["std"], w_stats["min"],
                    w_stats["max"], w_stats["mean_abs"],
                )
                writer.add_scalar("train/w_mean", w_stats["mean"], global_iter)
                writer.add_scalar("train/w_std", w_stats["std"], global_iter)
                writer.add_scalar("train/w_min", w_stats["min"], global_iter)
                writer.add_scalar("train/w_max", w_stats["max"], global_iter)
                writer.add_scalar("train/w_mean_abs", w_stats["mean_abs"], global_iter)

            running_loss = 0.0
            running_correct = 0
            running_total = 0
            running_grad_norm_sum = 0.0
            running_grad_norm_count = 0
            running_clipped_count = 0
            running_w_grad_norm_sum = 0.0
            running_other_grad_norm_sum = 0.0

        # Evaluate at milestones and at end
        if global_iter in sched_cfg.milestones_iters or global_iter == total_iters:
            test_error = _evaluate(model, test_loader, device)
            run_logger.info(
                ">> EVAL  iter=%d  epoch=%.1f  test_error=%.2f%%",
                global_iter, current_epoch, test_error
            )
            writer.add_scalar("test/error_rate", test_error, global_iter)
            model.train()

    # ---- Final evaluation ----
    final_test_error = _evaluate(model, test_loader, device)
    total_time = time.time() - start_time
    run_logger.info(
        "=== DONE  config=%s  variant=%s  depth=%d  seed=%d "
        "final_test_error=%.2f%%  time=%.0fs ===",
        config_name, variant_name, depth, seed, final_test_error, total_time
    )

    # ---- Save checkpoint ----
    os.makedirs(checkpoint_dir, exist_ok=True)
    safe_variant = variant_name.replace("+", "plus").replace("/", "_").replace(" ", "_")
    ckpt_name = f"{config_name}__{safe_variant}__depth{depth}__seed{seed}.pt"
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

    # For wSkip+LN, log the final w-vector norm (plan.md Section 4.2 note)
    # The LearnableVectorSkip.w parameter is named '*.skip_variant.w' in the model
    w_norm_info = {}
    for name, param in model.named_parameters():
        # LearnableVectorSkip registers w as a (C,1,1) parameter named '...skip_variant.w'
        if name.endswith(".skip_variant.w") and param.ndim == 3 and param.shape[2] == 1:
            w_vals = param.detach().abs().squeeze().cpu().tolist()
            if isinstance(w_vals, float):
                w_vals = [w_vals]
            w_norm_info[name] = {
                "mean_abs": float(sum(w_vals) / len(w_vals)),
                "l2_norm": float(param.detach().norm().item()),
                "per_channel": w_vals,
            }
    if w_norm_info:
        run_logger.info(
            "wSkip+LN weight norms: %d blocks, mean_abs values: %s",
            len(w_norm_info),
            {k: f"{v['mean_abs']:.4f}" for k, v in list(w_norm_info.items())[:5]}
        )

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
        "w_norm_info": w_norm_info,
    }
    torch.save(checkpoint, ckpt_path)
    run_logger.info("Checkpoint saved: %s", ckpt_path)

    writer.close()

    # ---- Write a compact JSON result file for later aggregation ----
    result = {
        "config_name": config_name,
        "variant_name": variant_name,
        "depth": depth,
        "seed": seed,
        "final_test_error": final_test_error,
        "total_iters": global_iter,
        "grad_clip_max_norm": grad_clip_max_norm if clip_active else None,
        "w_norm_info": w_norm_info,
    }
    safe_variant = variant_name.replace("+", "plus").replace("/", "_").replace(" ", "_")
    result_json_path = os.path.join(
        log_dir, f"{config_name}__{safe_variant}__depth{depth}__seed{seed}.json"
    )
    with open(result_json_path, "w") as f:
        json.dump(result, f, indent=2)
    run_logger.info("Result JSON: %s", result_json_path)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train one (config, variant, seed) per plan.md Section 5."
    )
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    parser.add_argument("--variant", required=True, help="Variant name string")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--depth", type=int, default=None,
                        help="Model depth override (for Table 6 multi-depth configs)")
    parser.add_argument("--log_dir", default="results/logs")
    parser.add_argument("--checkpoint_dir", default="results/checkpoints")
    parser.add_argument("--data_dir", default=None, help="Override data_dir from config")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_dir", default=None,
                        help="Workspace root for resolving relative paths in config")
    parser.add_argument("--env_report", action="store_true",
                        help="Write environment report to results/logs/env_report.txt")
    parser.add_argument(
        "--grad_clip_max_norm", type=float, default=DEFAULT_GRAD_CLIP_MAX_NORM,
        help=(
            "Max L2 norm for gradient clipping (applied every step). "
            f"Default: {DEFAULT_GRAD_CLIP_MAX_NORM}. Set to 0 or negative to disable."
        ),
    )
    parser.add_argument(
        "--w_lr_multiplier", type=float, default=DEFAULT_W_LR_MULTIPLIER,
        help=(
            "LR multiplier for LearnableVectorSkip.w's optimizer param group "
            f"(wSkip+LN only; no-op for other variants). Default: {DEFAULT_W_LR_MULTIPLIER}."
        ),
    )
    args = parser.parse_args()

    if args.env_report:
        log_env_report(path=os.path.join(args.log_dir, "env_report.txt"))

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
        w_lr_multiplier=args.w_lr_multiplier,
    )
    print(f"\nFINAL_RESULT: {result}")


if __name__ == "__main__":
    main()