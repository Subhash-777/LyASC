"""StabilityEstimator — finite-time Lyapunov proxy via power iteration.

Implements LyASC_ARCHITECTURE.md Section 2.2/2.3/3.1 EXACTLY as specified
there and nowhere else. This module intentionally does NOT decide:
  - where in the forward pass it gets called (Section 3.3, open question 1)
  - final k/N values beyond the documented starting point (open question 2)
  - cold-start handling (open question 3, lives in LyASCSkip / training loop)
  - whether lambda_l needs special handling at downsampling blocks
    (open question 6)
Those are Section 8 open design questions, resolved separately (plan.md
Step 2) and logged via log_decision before being wired into training.

Algorithm (Section 2.2), implemented literally:
    1. Sample a random unit vector v in the input space of block l.
    2. Compute u = J_l . v via a Jacobian-vector product (forward-mode,
       via torch.autograd.functional.jvp — no explicit Jacobian formed).
    3. Normalize: v <- u / ||u||.
    4. Repeat steps 2-3 for k iterations (k = 1-3).
    5. lambda_l ~= log(||u||)  -- finite-time Lyapunov proxy for this block.
    6. Smooth lambda_l over training via an EMA to reduce mini-batch
       stochasticity.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn


class StabilityEstimator(nn.Module):
    """Per-block finite-time Lyapunov proxy, refreshed every N steps.

    Owns: the random probe vector state (persisted across calls for power
    iteration continuity, per Section 2.2 step 1's "sample a random unit
    vector" — persisting it across refreshes rather than resampling from
    scratch each time is what makes repeated power iteration actually
    converge toward the dominant singular direction instead of restarting
    from a fresh random draw every refresh), the EMA buffer, and the step
    counter.
    """

    def __init__(self, k: int = 2, N: int = 10, ema_decay: float = 0.9):
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        if not (0.0 <= ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        self.k = k
        self.N = N
        self.ema_decay = ema_decay

        # lambda_l EMA buffer (not a learnable parameter -- a running
        # statistic, like BatchNorm's running_mean, so it's a buffer and
        # moves with the module across .to(device) / state_dict calls).
        self.register_buffer("_lambda_ema", torch.zeros(1))
        self.register_buffer("_initialized", torch.zeros(1, dtype=torch.bool))
        # Probe vector v is input-shape-dependent and only known at first
        # call, so it is stored as a plain attribute (lazily created),
        # not a buffer declared at __init__ time.
        self._probe_vec: Optional[torch.Tensor] = None
        self._step_count = 0

    def current_value(self) -> torch.Tensor:
        """Cheap read of the current smoothed lambda_l estimate. No recompute."""
        return self._lambda_ema.clone()

    @torch.no_grad()
    def _power_iteration(self, F_l: Callable[[torch.Tensor], torch.Tensor], x_l: torch.Tensor) -> float:
        """Run k steps of power iteration via JVPs, return log(||u||) of the
        final step (Section 2.2, steps 1-5), using a single representative
        sample from the batch (probe vector is defined per-input-shape, and
        JVP cost scales with batch size, so this uses x_l[0:1] to keep the
        estimator cheap -- this simplification is not yet logged as a
        Section-8-style decision since it is below the granularity of the
        six explicitly flagged open questions, but should be flagged for
        review if it turns out to matter).
        """
        x0 = x_l[0:1].detach()

        if self._probe_vec is None or self._probe_vec.shape != x0.shape:
            v = torch.randn_like(x0)
            v = v / (v.norm() + 1e-12)
            self._probe_vec = v
        v = self._probe_vec

        def f(inp: torch.Tensor) -> torch.Tensor:
            with torch.enable_grad():
                return F_l(inp)

        u_norm = torch.tensor(1e-12)
        for _ in range(self.k):
            with torch.enable_grad():
                x0_ = x0.clone().requires_grad_(True)
                _, u = torch.autograd.functional.jvp(f, (x0_,), (v,), create_graph=False)
            u = u.detach()
            u_norm = u.norm() + 1e-12
            if u.shape == x0.shape:
                v = u / u_norm
            else:
                with torch.enable_grad():
                    x0_vjp = x0.clone().requires_grad_(True)
                    out_vjp = F_l(x0_vjp)
                    v_back = torch.autograd.grad(out_vjp, x0_vjp, grad_outputs=u, retain_graph=False)[0].detach()
                v = v_back / (v_back.norm() + 1e-12)

        self._probe_vec = v
        lam = torch.log(u_norm).item()
        return lam

    def forward(self, F_l: Callable[[torch.Tensor], torch.Tensor], x_l: torch.Tensor, step: int) -> torch.Tensor:
        """Returns the current smoothed lambda_l estimate for this block.

        Recomputes via power iteration only when step % N == 0; otherwise
        returns the cached EMA value (Section 2.3).
        """
        if step % self.N == 0:
            lam_raw = self._power_iteration(F_l, x_l)
            lam_raw_t = torch.tensor([lam_raw], dtype=self._lambda_ema.dtype, device=self._lambda_ema.device)
            if not bool(self._initialized.item()):
                self._lambda_ema.copy_(lam_raw_t)
                self._initialized.fill_(True)
            else:
                self._lambda_ema.mul_(self.ema_decay).add_(lam_raw_t, alpha=(1.0 - self.ema_decay))
        self._step_count = step
        return self.current_value()
