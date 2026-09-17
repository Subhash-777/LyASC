"""Stability-gated conditional computation -- proposed extension, Section 4.3
of LyASC_Architecture_Research_Plan.docx. NOT part of the core LyASC design;
a Phase-1/2 ablation, off by default.

Idea: when a block's gate alpha_l is very close to 1 (the block is behaving
almost like a pure identity -- consistently near-stable), skip computing
F_l(x_l) entirely for that forward pass and just pass x_l through, saving
real wall-clock compute. This is distinct from prior stochastic-depth /
early-exit work because the skip decision is driven by a measured
dynamical-stability signal rather than a learned or random gate.

Status: SKELETON, not wired into preact_resnet.py. Wiring this in changes
PreActBasicBlock.forward()'s control flow (a real architectural change,
unlike LyASCSkip which was designed to require zero changes there) -- so
per this project's discipline, that wiring is itself a decision to be
made deliberately and logged, not defaulted into place silently.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConditionalComputeWrapper(nn.Module):
    """Wraps a PreActBasicBlock so that, at a configurable alpha threshold,
    the residual branch F_l is skipped entirely and the block short-circuits
    to (approximately) an identity pass-through.

    threshold: alpha_l value above which F_l is skipped (e.g. 0.98).
    min_train_steps_before_skipping: guard against skipping during early
        training, before alpha_l estimates are trustworthy (ties into the
        same cold-start caution already required for the core gate,
        Section 6 of LyASC_ARCHITECTURE.md).
    """

    def __init__(self, block: nn.Module, threshold: float = 0.98, min_train_steps_before_skipping: int = 4000):
        super().__init__()
        self.block = block
        self.threshold = threshold
        self.min_train_steps_before_skipping = min_train_steps_before_skipping

    def forward(self, x: torch.Tensor, step: int = 0) -> torch.Tensor:
        skip_variant = getattr(self.block, "skip_variant", None)
        can_check_alpha = skip_variant is not None and hasattr(skip_variant, "estimator") and skip_variant.estimator is not None

        if can_check_alpha and step >= self.min_train_steps_before_skipping:
            with torch.no_grad():
                lam = skip_variant.estimator.current_value()
                alpha_est = torch.sigmoid(skip_variant.beta_l * (lam - skip_variant.lambda0))
            if torch.all(alpha_est > self.threshold):
                # Near-pure-identity block: skip F_l's compute this step.
                # NOTE: still routes through skip_variant's shortcut path
                # (e.g. IdentityZeroPadShortcut at downsampling blocks) so
                # channel/stride bookkeeping stays correct -- only the
                # residual branch's conv/BN/ReLU compute is skipped.
                identity = self.block.shortcut(
                    torch.nn.functional.relu(self.block.bn1(x), inplace=False)
                ) if self.block.shortcut is not None else x
                return identity

        return self.block(x)