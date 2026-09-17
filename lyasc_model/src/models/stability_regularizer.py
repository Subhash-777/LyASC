"""Lyapunov spectrum regularizer -- proposed extension, Section 4.4 of
LyASC_Architecture_Research_Plan.docx. NOT part of the core LyASC design
(LyASC_ARCHITECTURE.md); this is a Phase-1 ABLATION, off by default, kept
in its own module precisely so it can be added/removed from a training run
without touching LyASCSkip or StabilityEstimator.

Idea: rather than only reacting to instability via the gate, add a soft
penalty that discourages any block's smoothed lambda_l from exceeding a
target ceiling, so the network is trained to avoid ever entering the
high-lambda regime associated with the ~1000-iteration cold-start
gradient spike documented in decisions.log, rather than only surviving it
once it happens.

Status: SKELETON. The exact form of the penalty (hinge vs. squared,
whether it targets a band [lambda_min, lambda_max] or a ceiling only) is
itself an open question that should go through the same single-seed
validate-before-scaling discipline as everything else in this project --
not assumed correct from this docstring.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class LyapunovSpectrumRegularizer(nn.Module):
    """Computes a soft penalty over a collection of blocks' current
    (already-estimated, no extra forward passes) lambda_l values.

    penalty = weight * mean( relu(lambda_l - lambda_ceiling) ** 2 )

    Deliberately reuses StabilityEstimator.current_value() (a cheap read,
    Section 3.1/3.2) rather than triggering any additional power-iteration
    computation of its own -- this term is designed to be nearly free to
    evaluate every step, unlike the estimator refresh itself.
    """

    def __init__(self, lambda_ceiling: float = 1.0, weight: float = 0.01):
        super().__init__()
        self.lambda_ceiling = lambda_ceiling
        self.weight = weight

    def forward(self, estimators: Iterable) -> torch.Tensor:
        """estimators: iterable of StabilityEstimator instances (one per
        LyASC block in the network). Returns a scalar tensor to be added
        directly to the training loss (e.g. loss = ce_loss + regularizer(...)).
        """
        values = [est.current_value() for est in estimators]
        if not values:
            return torch.zeros(1)
        lam = torch.cat(values)
        excess = torch.relu(lam - self.lambda_ceiling)
        return self.weight * (excess ** 2).mean()