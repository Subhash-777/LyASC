"""Unit tests for LyASCSkip, per LyASC_plan.md Section 3, Step 1.

Covers (as specified in plan.md):
  - forward() produces alpha_l * x + (1-alpha_l) * F_x EXACTLY given a
    fixed, mocked estimator.current_value()
  - beta_l initializes to 0 (giving alpha_l=0.5 at a fixed lambda_l=lambda0)
  - gradients flow to both beta_l and through x/F_x
"""

import math

import pytest
import torch
import torch.nn as nn

from src.models.lyasc_skip import LyASCSkip


class _MockEstimator:
    """Stand-in for StabilityEstimator exposing only current_value(), per
    plan.md's "mocked estimator.current_value()" instruction -- this
    deliberately does NOT subclass StabilityEstimator or implement
    forward(), keeping the LyASCSkip unit test decoupled from the
    estimator's own (separately tested) power-iteration correctness.
    """

    def __init__(self, value: float):
        self._value = torch.tensor([value])

    def current_value(self) -> torch.Tensor:
        return self._value


def test_forward_matches_exact_formula():
    torch.manual_seed(0)
    channels = 4
    lam_value = 1.3
    lambda0 = 0.5

    skip = LyASCSkip(channels=channels, lambda0=lambda0, estimator=_MockEstimator(lam_value))
    with torch.no_grad():
        skip.beta_l.fill_(0.7)  # non-zero so the formula is actually exercised

    x = torch.randn(2, channels, 8, 8)
    F_x = torch.randn(2, channels, 8, 8)

    out = skip(x, F_x)

    expected_alpha = torch.sigmoid(torch.tensor(0.7 * (lam_value - lambda0)))
    expected = expected_alpha * x + (1 - expected_alpha) * F_x

    assert torch.allclose(out, expected, atol=1e-6), "LyASCSkip.forward() must equal alpha*x + (1-alpha)*F_x exactly"


def test_beta_l_initializes_to_zero_giving_alpha_half_at_lambda0():
    channels = 4
    lambda0 = 2.0
    # lambda_l == lambda0 exactly -> (lambda_l - lambda0) == 0 regardless of beta_l,
    # but the real point of this test is beta_l's init value itself.
    skip = LyASCSkip(channels=channels, lambda0=lambda0, estimator=_MockEstimator(lambda0))

    assert torch.allclose(skip.beta_l, torch.zeros(1)), "beta_l must initialize to 0"

    x = torch.randn(2, channels, 4, 4)
    F_x = torch.randn(2, channels, 4, 4)
    out = skip(x, F_x)
    expected = 0.5 * x + 0.5 * F_x
    assert torch.allclose(out, expected, atol=1e-6), (
        "With beta_l=0 (its init value), alpha_l = sigmoid(0) = 0.5 regardless of "
        "lambda_l, so forward() must reduce to an even 50/50 blend"
    )


def test_beta_l_initializes_to_zero_at_arbitrary_lambda():
    """Same as above but with lambda_l far from lambda0 -- beta_l=0 must still
    force alpha_l=0.5 since beta_l multiplies (lambda_l - lambda0)."""
    channels = 4
    skip = LyASCSkip(channels=channels, lambda0=0.0, estimator=_MockEstimator(50.0))
    x = torch.randn(2, channels, 4, 4)
    F_x = torch.randn(2, channels, 4, 4)
    out = skip(x, F_x)
    expected = 0.5 * x + 0.5 * F_x
    assert torch.allclose(out, expected, atol=1e-6)


def test_gradients_flow_to_beta_l_and_x_and_F_x():
    channels = 4
    skip = LyASCSkip(channels=channels, lambda0=0.3, estimator=_MockEstimator(1.7))
    with torch.no_grad():
        skip.beta_l.fill_(0.4)  # non-zero: at beta_l=0, d(alpha)/d(beta_l) is nonzero
                                  # via sigmoid'(0)*(lambda-lambda0), so 0-init would
                                  # still pass this test, but a non-zero value makes the
                                  # x/F_x split of alpha itself non-trivial too.

    x = torch.randn(2, channels, 4, 4, requires_grad=True)
    F_x = torch.randn(2, channels, 4, 4, requires_grad=True)

    out = skip(x, F_x)
    loss = out.sum()
    loss.backward()

    assert skip.beta_l.grad is not None and torch.any(skip.beta_l.grad != 0), \
        "gradient must flow to beta_l"
    assert x.grad is not None and torch.any(x.grad != 0), "gradient must flow through x"
    assert F_x.grad is not None and torch.any(F_x.grad != 0), "gradient must flow through F_x"


def test_raises_without_estimator():
    skip = LyASCSkip(channels=4, lambda0=0.0, estimator=None)
    x = torch.randn(1, 4, 2, 2)
    F_x = torch.randn(1, 4, 2, 2)
    with pytest.raises(RuntimeError):
        skip(x, F_x)


def test_interface_matches_skip_variants_contract():
    """forward(self, x, F_x) -> Tensor -- same contract as every class in
    base_model/src/models/skip_variants.py, so PreActBasicBlock requires
    zero changes to accept LyASCSkip as a skip_variant."""
    import inspect
    sig = inspect.signature(LyASCSkip.forward)
    params = list(sig.parameters.keys())
    assert params == ["self", "x", "F_x"], f"forward() signature must be (self, x, F_x), got {params}"


# ---------------------------------------------------------------------------
# Tests for Section 7 Ablation 1: G() = LayerNorm2d normalization branch
# ---------------------------------------------------------------------------

def test_layer_norm_branch_initialization():
    from src.models.skip_variants import LayerNorm2d
    channels = 8
    skip = LyASCSkip(channels=channels, lambda0=0.0, estimator=_MockEstimator(0.0), use_layer_norm=True)
    assert skip.use_layer_norm is True
    assert isinstance(skip.g_norm, LayerNorm2d)
    assert skip.g_norm.weight.shape == (channels,)
    assert skip.g_norm.bias.shape == (channels,)
    assert torch.allclose(skip.g_norm.weight, torch.ones(channels))
    assert torch.allclose(skip.g_norm.bias, torch.zeros(channels))


def test_layer_norm_branch_forward_exact_formula():
    from src.models.skip_variants import LayerNorm2d
    torch.manual_seed(42)
    channels = 4
    lam_value = 1.5
    lambda0 = 0.5

    skip = LyASCSkip(
        channels=channels,
        lambda0=lambda0,
        estimator=_MockEstimator(lam_value),
        use_layer_norm=True,
    )
    with torch.no_grad():
        skip.beta_l.fill_(0.8)

    x = torch.randn(2, channels, 4, 4)
    F_x = torch.randn(2, channels, 4, 4)

    out = skip(x, F_x)

    # Compute expected: LayerNorm2d(alpha*x + (1-alpha)*F_x)
    alpha = torch.sigmoid(torch.tensor(0.8 * (lam_value - lambda0)))
    blend = alpha * x + (1 - alpha) * F_x
    ln = LayerNorm2d(channels)
    with torch.no_grad():
        ln.weight.copy_(skip.g_norm.weight)
        ln.bias.copy_(skip.g_norm.bias)
    expected = ln(blend)

    assert torch.allclose(out, expected, atol=1e-6), (
        "LyASCSkip(use_layer_norm=True) must compute LayerNorm2d(alpha*x + (1-alpha)*F_x) exactly"
    )


def test_layer_norm_branch_gradients_flow():
    channels = 4
    skip = LyASCSkip(
        channels=channels,
        lambda0=0.2,
        estimator=_MockEstimator(1.0),
        use_layer_norm=True,
    )
    with torch.no_grad():
        skip.beta_l.fill_(0.5)

    x = torch.randn(2, channels, 4, 4, requires_grad=True)
    F_x = torch.randn(2, channels, 4, 4, requires_grad=True)

    out = skip(x, F_x)
    loss = out.sum()
    loss.backward()

    assert skip.beta_l.grad is not None and torch.any(skip.beta_l.grad != 0), "gradient must flow to beta_l"
    assert skip.g_norm.weight.grad is not None and torch.any(skip.g_norm.weight.grad != 0), "gradient must flow to LayerNorm2d.weight"
    assert skip.g_norm.bias.grad is not None and torch.any(skip.g_norm.bias.grad != 0), "gradient must flow to LayerNorm2d.bias"
    assert x.grad is not None and torch.any(x.grad != 0), "gradient must flow through x"
    assert F_x.grad is not None and torch.any(F_x.grad != 0), "gradient must flow through F_x"


def test_layer_norm_extra_repr():
    skip = LyASCSkip(channels=16, lambda0=0.0, estimator=_MockEstimator(0.0), use_layer_norm=True)
    repr_str = skip.extra_repr()
    assert "use_layer_norm=True" in repr_str


def test_cold_start_warmup_strategy():
    channels = 4
    skip = LyASCSkip(
        channels=channels,
        lambda0=0.0,
        estimator=_MockEstimator(5.0),
        use_layer_norm=False,
        cold_start_strategy="warmup",
        cold_start_warmup_iters=1000,
    )
    with torch.no_grad():
        skip.beta_l.fill_(2.0)  # non-zero beta_l

    current_step = 500
    skip.step_provider = lambda: current_step

    x = torch.randn(2, channels, 4, 4)
    F_x = torch.randn(2, channels, 4, 4)

    # During warmup (<1000), alpha should be fixed at 0.5 regardless of beta_l / lambda
    out_warmup = skip(x, F_x)
    expected_warmup = 0.5 * x + 0.5 * F_x
    assert torch.allclose(out_warmup, expected_warmup, atol=1e-6)

    # After warmup (>=1000), normal gating applies
    current_step = 1000
    out_post = skip(x, F_x)
    alpha_post = torch.sigmoid(torch.tensor(2.0 * (5.0 - 0.0)))
    expected_post = alpha_post * x + (1 - alpha_post) * F_x
    assert torch.allclose(out_post, expected_post, atol=1e-6)


def test_cold_start_damped_strategy():
    channels = 4
    skip = LyASCSkip(
        channels=channels,
        lambda0=0.0,
        estimator=_MockEstimator(2.0),
        use_layer_norm=False,
        cold_start_strategy="damped",
        cold_start_warmup_iters=1000,
    )
    with torch.no_grad():
        skip.beta_l.fill_(1.0)

    x = torch.randn(2, channels, 4, 4)
    F_x = torch.randn(2, channels, 4, 4)

    # At step 0, w=0 -> alpha = sigmoid(0) = 0.5
    current_step = 0
    skip.step_provider = lambda: current_step
    out_0 = skip(x, F_x)
    expected_0 = 0.5 * x + 0.5 * F_x
    assert torch.allclose(out_0, expected_0, atol=1e-6)

    # At step 500, w=0.5 -> alpha = sigmoid(0.5 * 1.0 * (2.0 - 0.0))
    current_step = 500
    out_500 = skip(x, F_x)
    alpha_500 = torch.sigmoid(torch.tensor(0.5 * 1.0 * 2.0))
    expected_500 = alpha_500 * x + (1 - alpha_500) * F_x
    assert torch.allclose(out_500, expected_500, atol=1e-6)

    # At step 1000+, w=1.0 -> full influence
    current_step = 1000
    out_1000 = skip(x, F_x)
    alpha_1000 = torch.sigmoid(torch.tensor(1.0 * 1.0 * 2.0))
    expected_1000 = alpha_1000 * x + (1 - alpha_1000) * F_x
    assert torch.allclose(out_1000, expected_1000, atol=1e-6)


def test_forward_exact_values_path_scaling_hand_computed():
    """Verify exact hand-calculated tensor values for path scaling formula:
    y = alpha * (skip_scale * x) + (1 - alpha) * (residual_scale * F_x)
    """
    channels = 4
    lam_val = 1.0
    lambda0 = 0.0
    skip_scale = 0.5
    residual_scale = 2.0

    skip = LyASCSkip(
        channels=channels,
        lambda0=lambda0,
        estimator=_MockEstimator(lam_val),
        use_layer_norm=False,
        use_batch_norm=False,
        skip_scale=skip_scale,
        residual_scale=residual_scale,
    )

    with torch.no_grad():
        skip.beta_l.fill_(0.7)  # alpha = sigmoid(0.7 * (1.0 - 0.0)) = sigmoid(0.7)

    x = torch.ones(2, channels, 4, 4)       # x = 1.0 everywhere
    F_x = torch.full((2, channels, 4, 4), 2.0)  # F(x) = 2.0 everywhere

    out = skip(x, F_x)

    alpha_val = float(torch.sigmoid(torch.tensor(0.7)))  # ~0.66818777
    # Hand calculation:
    # scaled_x = 0.5 * 1.0 = 0.5
    # scaled_F = 2.0 * 2.0 = 4.0
    # expected = alpha * 0.5 + (1 - alpha) * 4.0 = 4.0 - 3.5 * alpha
    expected_val = alpha_val * 0.5 + (1.0 - alpha_val) * 4.0  # ~1.6613428
    expected_tensor = torch.full((2, channels, 4, 4), expected_val)

    assert torch.allclose(out, expected_tensor, atol=1e-6), (
        f"LyASCSkip output {out[0,0,0,0].item()} must match hand-computed value {expected_val}"
    )


def test_forward_with_path_scaling_and_batch_norm():
    """Verify path scaling works seamlessly when G() = BatchNorm2d."""
    channels = 4
    skip = LyASCSkip(
        channels=channels,
        lambda0=0.0,
        estimator=_MockEstimator(1.0),
        use_layer_norm=False,
        use_batch_norm=True,
        skip_scale=0.5,
        residual_scale=2.0,
    )
    with torch.no_grad():
        skip.beta_l.fill_(0.0)  # alpha = 0.5

    skip.eval()  # running stats mode
    x = torch.randn(2, channels, 4, 4)
    F_x = torch.randn(2, channels, 4, 4)

    out = skip(x, F_x)
    alpha = 0.5
    raw_blend = alpha * (0.5 * x) + (1.0 - alpha) * (2.0 * F_x)
    expected_bn = skip.g_norm(raw_blend)
    assert torch.allclose(out, expected_bn, atol=1e-5)


