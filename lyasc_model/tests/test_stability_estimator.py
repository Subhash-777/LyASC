"""Unit tests for StabilityEstimator, per LyASC_plan.md Section 3, Step 1.

Covers (as specified in plan.md):
  - the power-iteration loop converges toward the correct dominant singular
    value on a small, hand-constructed linear layer with a KNOWN Jacobian
    (a solvable, checkable test case)
  - the recompute-every-N-steps logic actually skips recomputation on
    non-multiple steps and returns the cached EMA value

Test-case construction note: the algorithm in LyASC_ARCHITECTURE.md Section
2.2 is v <- (J.v)/||J.v|| repeated k times, i.e. power iteration directly on
J (not on J^T J). This converges to the dominant EIGENVALUE magnitude of J,
which only equals J's dominant SINGULAR value when J is symmetric. So the
hand-constructed linear layer here uses a symmetric weight matrix, making
the "known Jacobian, known answer" test case solvable and unambiguous.
If StabilityEstimator's algorithm is later changed to alternate JVP/VJP
(the standard method for computing singular values of a general, possibly
non-symmetric J), this test's construction should be revisited.
"""

import math

import torch
import torch.nn as nn

from src.models.stability_estimator import StabilityEstimator


def _symmetric_linear_fn(weight: torch.Tensor):
    """Returns F(x) = x @ weight (weight symmetric), matching a shape
    F_l would plausibly produce elsewhere: a flat (N, C) -> (N, C) map,
    with a known, exact Jacobian equal to `weight` itself everywhere
    (linear map, so the Jacobian is constant and does not depend on x)."""
    def f(x: torch.Tensor) -> torch.Tensor:
        return x @ weight
    return f


def test_power_iteration_converges_to_known_dominant_eigenvalue():
    torch.manual_seed(0)
    # Known symmetric weight matrix with a clear dominant eigenvalue.
    # diag(3, 1, 0.5) has eigenvalues {3, 1, 0.5} -> dominant |eigenvalue| = 3.
    weight = torch.diag(torch.tensor([3.0, 1.0, 0.5]))
    f = _symmetric_linear_fn(weight)
    x_l = torch.randn(1, 3)  # batch of 1, matches StabilityEstimator's x_l[0:1] usage

    expected_lambda = math.log(3.0)

    # Run with increasing k and check the estimate gets closer to log(3),
    # i.e. more power-iteration steps should not make the estimate worse.
    errs = []
    for k in (1, 3, 8):
        est = StabilityEstimator(k=k, N=1, ema_decay=0.0)  # ema_decay=0 -> EMA = latest raw value
        # Seed the same probe vector across k values for a fair comparison
        torch.manual_seed(42)
        lam = est(f, x_l, step=0)
        errs.append(abs(lam.item() - expected_lambda))

    assert errs[-1] < errs[0], (
        f"More power-iteration steps should converge closer to log(3)={expected_lambda:.4f}, "
        f"got errors {errs} for k=(1,3,8)"
    )
    assert errs[-1] < 0.05, f"k=8 power iteration should be close to log(3)={expected_lambda:.4f}, got err={errs[-1]:.4f}"


def test_recompute_schedule_skips_on_non_multiple_steps():
    weight = torch.diag(torch.tensor([3.0, 1.0, 0.5]))
    f_strong = _symmetric_linear_fn(weight)
    # A very different function (huge dominant eigenvalue) to make an
    # accidental recompute obvious if the schedule logic is broken.
    f_different = _symmetric_linear_fn(torch.diag(torch.tensor([1000.0, 1.0, 0.5])))

    x_l = torch.randn(1, 3)
    est = StabilityEstimator(k=3, N=5, ema_decay=0.9)

    lam_step0 = est(f_strong, x_l, step=0)  # 0 % 5 == 0 -> recompute
    cached_value = est.current_value().clone()

    for step in (1, 2, 3, 4):
        lam = est(f_different, x_l, step=step)  # step % 5 != 0 -> must NOT recompute
        assert torch.allclose(lam, cached_value), (
            f"step={step} should return the cached EMA value unchanged, "
            f"got {lam.item()} vs cached {cached_value.item()}"
        )

    # step=5 is a multiple of N=5 -> should recompute, and since f_different
    # has a wildly larger dominant eigenvalue, the EMA must move.
    lam_step5 = est(f_different, x_l, step=5)
    assert not torch.allclose(lam_step5, cached_value), (
        "step=5 (a multiple of N) must trigger a recompute and change the EMA value"
    )


def test_current_value_is_a_cheap_read_with_no_grad_side_effects():
    """current_value() must not require grad / must not itself trigger any
    computation -- it is documented as a cheap read in Section 3.1/3.2."""
    weight = torch.diag(torch.tensor([2.0, 1.0]))
    f = _symmetric_linear_fn(weight)
    x_l = torch.randn(1, 2)
    est = StabilityEstimator(k=2, N=10, ema_decay=0.9)
    est(f, x_l, step=0)
    val = est.current_value()
    assert not val.requires_grad
    assert val.shape == (1,)


def test_invalid_hyperparameters_raise():
    import pytest
    with pytest.raises(ValueError):
        StabilityEstimator(k=0)
    with pytest.raises(ValueError):
        StabilityEstimator(N=0)
    with pytest.raises(ValueError):
        StabilityEstimator(ema_decay=1.0)
