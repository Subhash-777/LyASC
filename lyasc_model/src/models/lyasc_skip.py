"""LyASCSkip -- y = G(alpha_l * x + (1 - alpha_l) * F_x),
   alpha_l = sigmoid(w(step) * beta_l * (lambda_l - lambda0))
   G() is either identity (default) or LayerNorm2d (Section 7 Ablation 1).
   Supports cold-start strategies: "none", "warmup", "damped" (Section 6).

Per LyASC_ARCHITECTURE.md Section 3.2, Section 6, & Section 7. Implements the same
    forward(self, x: torch.Tensor, F_x: torch.Tensor) -> torch.Tensor
contract as every class in base_model/src/models/skip_variants.py.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from src.models.skip_variants import LayerNorm2d
from src.models.stability_estimator import StabilityEstimator


class LyASCSkip(nn.Module):
    """y = G(alpha_l * (skip_scale * x) + (1 - alpha_l) * (residual_scale * F_x)),
    with alpha_l = sigmoid(w(step) * beta_l * (lambda_l - lambda0))
    and G() = LayerNorm2d if use_layer_norm=True, BatchNorm2d if use_batch_norm=True, else Identity().
    """

    def __init__(
        self,
        channels: int,
        lambda0: float = 0.0,
        estimator: Optional[StabilityEstimator] = None,
        use_layer_norm: bool = False,
        use_batch_norm: bool = False,
        skip_scale: float = 1.0,
        residual_scale: float = 1.0,
        cold_start_strategy: str = "none",
        cold_start_warmup_iters: int = 1000,
    ):
        super().__init__()
        self.channels = channels  # kept for signature parity with other skip_variants factories
        self.beta_l = nn.Parameter(torch.zeros(1))  # init 0 -> alpha starts at sigmoid(0)=0.5
        self.lambda0 = lambda0
        self.estimator = estimator
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm
        self.skip_scale = float(skip_scale)
        self.residual_scale = float(residual_scale)
        self.cold_start_strategy = cold_start_strategy
        self.cold_start_warmup_iters = cold_start_warmup_iters
        self.step_provider: Optional[Callable[[], int]] = None

        if self.use_layer_norm:
            self.g_norm: nn.Module = LayerNorm2d(channels)
        elif self.use_batch_norm:
            self.g_norm = nn.BatchNorm2d(channels)
        else:
            self.g_norm = nn.Identity()

    def forward(self, x: torch.Tensor, F_x: torch.Tensor) -> torch.Tensor:
        if self.estimator is None:
            raise RuntimeError(
                "LyASCSkip.forward() called with no StabilityEstimator attached. "
                "An estimator must be constructed and assigned before training."
            )
        lam = self.estimator.current_value()  # cheap read, no recompute here
        step = self.step_provider() if self.step_provider is not None else 0

        if self.cold_start_strategy == "warmup" and step < self.cold_start_warmup_iters:
            alpha = torch.tensor(0.5, device=x.device, dtype=x.dtype)
        elif self.cold_start_strategy == "damped" and step < self.cold_start_warmup_iters:
            w = float(step) / float(max(1, self.cold_start_warmup_iters))
            alpha = torch.sigmoid(w * self.beta_l * (lam - self.lambda0))
        else:
            alpha = torch.sigmoid(self.beta_l * (lam - self.lambda0))

        scaled_x = x if self.skip_scale == 1.0 else self.skip_scale * x
        scaled_F = F_x if self.residual_scale == 1.0 else self.residual_scale * F_x
        blend = alpha * scaled_x + (1 - alpha) * scaled_F
        return self.g_norm(blend)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, lambda0={self.lambda0}, "
            f"skip_scale={self.skip_scale}, residual_scale={self.residual_scale}, "
            f"use_layer_norm={self.use_layer_norm}, use_batch_norm={self.use_batch_norm}, "
            f"cold_start={self.cold_start_strategy}"
        )
