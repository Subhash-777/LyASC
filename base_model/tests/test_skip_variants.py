"""Unit tests for src/models/skip_variants.py, per plan.md Section 7 step 3.

For each variant, constructs a small fixed random tensor x and F_x,
hand-computes the expected output using the exact equation from plan.md
Section 4.2, and asserts the module's forward() output matches.

Note on RecursiveSkipLN: tests verify the literal recurrence
  y1 = LN(x + F_x);  y2 = LN(x + y1)
NOT the closed-form algebraic expansion (Appendix A of the paper is for
interpretability only; the module must implement the recurrence).

FIXED (post LayerNorm-axis correction, see src/models/skip_variants.py
module-level DECISION notes and results/logs/decisions.log "layernorm_axis"
entry): every reference LayerNorm constructed in this file to hand-verify
an equation now uses LayerNorm2d(channels) instead of the stale
nn.LayerNorm([channels, spatial_size, spatial_size]). The old reference
shape (C, H, W) belonged to the joint-(C,H,W) LayerNorm implementation that
was replaced in production after it caused real (non-NaN) mid-training
instability (2xSkip+LN reproduced at 13.31% vs. paper's 6.29%). Once
production switched to the channel-wise LayerNorm2d, these three tests
(TestExpandedSkipLN, TestResidualScaledSkip, TestLearnableVectorSkip)
started failing on `ln_ref.load_state_dict(module.ln.state_dict())` with a
shape mismatch ((C,) vs (C,H,W)) — that failure was stale-test breakage,
not a production regression (confirmed separately via a live training run:
2xSkip+LN seed0 reproduced at 5.97% test error, PASS). Fixed by aligning
each reference construction with the actual LayerNorm2d used in production.
"""

import pytest
import torch
import torch.nn as nn

from src.models.skip_variants import (
    PlainSkip,
    ExpandedSkip,
    ExpandedSkipLN,
    ExpandedSkipBN,
    RecursiveSkipLN,
    RecursiveSkipBN,
    ResidualScaledSkip,
    LearnableVectorSkip,
    LayerNorm2d,
    build_skip_variant,
)


# Fixed small tensor dimensions for all tests
B, C, H, W = 2, 16, 4, 4
torch.manual_seed(42)
_x = torch.randn(B, C, H, W)
_F = torch.randn(B, C, H, W)


class TestPlainSkip:
    """y = x + F(x, W) exactly."""

    def test_equation(self):
        module = PlainSkip()
        expected = _x + _F
        actual = module(_x, _F)
        assert torch.allclose(actual, expected, atol=0.0), \
            "PlainSkip: expected x + F exactly"

    def test_no_parameters(self):
        module = PlainSkip()
        assert sum(p.numel() for p in module.parameters()) == 0, \
            "PlainSkip should have no learnable parameters"


class TestExpandedSkip:
    """y = lambda * x + F(x, W)."""

    @pytest.mark.parametrize("lam", [0.5, 2.0, 3.0])
    def test_equation(self, lam):
        module = ExpandedSkip(lam=lam)
        expected = lam * _x + _F
        actual = module(_x, _F)
        assert torch.allclose(actual, expected, atol=1e-6), \
            f"ExpandedSkip(lam={lam}): equation mismatch"

    def test_no_parameters(self):
        for lam in [0.5, 2.0, 3.0]:
            module = ExpandedSkip(lam=lam)
            assert sum(p.numel() for p in module.parameters()) == 0


class TestExpandedSkipLN:
    """y = LN(lambda * x + F(x, W))."""

    @pytest.mark.parametrize("lam", [0.5, 1.0, 2.0, 3.0])
    def test_equation(self, lam):
        module = ExpandedSkipLN(lam=lam, channels=C, spatial_size=H)
        # FIXED: reference LN must match production's LayerNorm2d (channel-wise,
        # shape (C,) affine params), not the old joint-(C,H,W) nn.LayerNorm.
        ln_ref = LayerNorm2d(C)
        ln_ref.load_state_dict(module.ln.state_dict())
        expected = ln_ref(lam * _x + _F)
        actual = module(_x, _F)
        assert torch.allclose(actual, expected, atol=1e-5), \
            f"ExpandedSkipLN(lam={lam}): equation mismatch"

    def test_has_ln_parameters(self):
        module = ExpandedSkipLN(lam=1.0, channels=C, spatial_size=H)
        ln_params = sum(p.numel() for p in module.ln.parameters())
        assert ln_params > 0

    def test_ln_params_are_1d(self):
        """LayerNorm2d affine params must be shape (C,), not (C,H,W) — this is
        what makes the weight-decay-exclusion check (param.ndim <= 1) in
        train.py's _build_optimizer() correctly exclude them. See the
        "weight_decay_scope_correction" decision in train.py.
        """
        module = ExpandedSkipLN(lam=1.0, channels=C, spatial_size=H)
        assert module.ln.weight.shape == (C,), \
            f"Expected LN weight shape ({C},), got {module.ln.weight.shape}"
        assert module.ln.bias.shape == (C,), \
            f"Expected LN bias shape ({C},), got {module.ln.bias.shape}"

    def test_normalizes_across_channels_not_spatial(self):
        """Confirm LN statistics are computed per (n,h,w) across C only —
        i.e. changing values at one spatial position must not change the
        normalized output at a different spatial position (which WOULD
        happen under the old joint-(C,H,W) normalization)."""
        module = ExpandedSkipLN(lam=1.0, channels=C, spatial_size=H)
        module.eval()
        x1 = torch.randn(1, C, H, W)
        F1 = torch.randn(1, C, H, W)
        out1 = module(x1, F1)

        x2 = x1.clone()
        F2 = F1.clone()
        # Perturb only position (0,0) — under channel-wise LN this must not
        # affect the output at position (0,1).
        x2[0, :, 0, 0] += 5.0
        out2 = module(x2, F2)

        assert torch.allclose(out1[0, :, 0, 1], out2[0, :, 0, 1], atol=1e-5), \
            "Perturbing one spatial position changed another position's output " \
            "— LN appears to be normalizing jointly over (C,H,W), not per-position"


class TestExpandedSkipBN:
    """y = BN(lambda * x + F(x, W))."""

    @pytest.mark.parametrize("lam", [1.0, 2.0, 3.0])
    def test_equation(self, lam):
        module = ExpandedSkipBN(lam=lam, channels=C)
        module.eval()  # disable running-stats update for exact comparison
        bn_ref = nn.BatchNorm2d(C)
        bn_ref.load_state_dict(module.bn.state_dict())
        bn_ref.eval()
        expected = bn_ref(lam * _x + _F)
        actual = module(_x, _F)
        assert torch.allclose(actual, expected, atol=1e-5), \
            f"ExpandedSkipBN(lam={lam}): equation mismatch"


class TestRecursiveSkipLN:
    """Literal recurrence: y_k = LN(x + y_{k-1}), y_1 = LN(x + F)."""

    def test_lam1_equals_expanded_ln(self):
        """rSkip+LN with lam=1 should equal 1xSkip+LN."""
        module_r = RecursiveSkipLN(lam=1, channels=C, spatial_size=H)
        # Copy the single LN weights
        module_x = ExpandedSkipLN(lam=1.0, channels=C, spatial_size=H)
        module_x.ln.load_state_dict(module_r.lns[0].state_dict())
        expected = module_x(_x, _F)
        actual = module_r(_x, _F)
        assert torch.allclose(actual, expected, atol=1e-5), \
            "RecursiveSkipLN(lam=1) should equal 1xSkip+LN"

    def test_lam2_literal_recurrence(self):
        """Verify y = LN_outer(x + LN_inner(x + F)) — literal recurrence."""
        module = RecursiveSkipLN(lam=2, channels=C, spatial_size=H)
        module.eval()
        # Innermost LN is lns[-1], outermost is lns[0]
        ln_inner = module.lns[-1]   # lns[1]
        ln_outer = module.lns[0]    # lns[0]
        y1 = ln_inner(_x + _F)
        y2 = ln_outer(_x + y1)
        actual = module(_x, _F)
        assert torch.allclose(actual, y2, atol=1e-5), \
            "RecursiveSkipLN(lam=2): literal recurrence mismatch"

    def test_lam3_literal_recurrence(self):
        """Verify lam=3 three-step recurrence."""
        module = RecursiveSkipLN(lam=3, channels=C, spatial_size=H)
        module.eval()
        ln1 = module.lns[-1]   # innermost
        ln2 = module.lns[-2]
        ln3 = module.lns[0]    # outermost
        y1 = ln1(_x + _F)
        y2 = ln2(_x + y1)
        y3 = ln3(_x + y2)
        actual = module(_x, _F)
        assert torch.allclose(actual, y3, atol=1e-5), \
            "RecursiveSkipLN(lam=3): literal recurrence mismatch"

    def test_independent_ln_params(self):
        """Each recursion step must have independent LN parameters."""
        module = RecursiveSkipLN(lam=2, channels=C, spatial_size=H)
        # Force different params on each LN
        nn.init.ones_(module.lns[0].weight)
        nn.init.zeros_(module.lns[0].bias)
        nn.init.constant_(module.lns[1].weight, 2.0)
        nn.init.constant_(module.lns[1].bias, 0.5)
        # Verify weights are actually different
        assert not torch.allclose(module.lns[0].weight, module.lns[1].weight), \
            "LN modules must have independent parameters"

    @pytest.mark.parametrize("lam", [1, 2, 3])
    def test_gradient_flow(self, lam):
        """Gradients must flow through all LN parameters."""
        module = RecursiveSkipLN(lam=lam, channels=C, spatial_size=H)
        x = _x.clone().requires_grad_(True)
        F_x = _F.clone().requires_grad_(True)
        out = module(x, F_x)
        out.sum().backward()
        assert x.grad is not None, "No gradient to x"
        assert F_x.grad is not None, "No gradient to F_x"
        for i, ln in enumerate(module.lns):
            assert ln.weight.grad is not None, f"No gradient to lns[{i}].weight"


class TestRecursiveSkipBN:
    """BN version of recursive skip — same recurrence structure."""

    def test_lam2_literal_recurrence(self):
        module = RecursiveSkipBN(lam=2, channels=C)
        module.eval()
        bn_inner = module.bns[-1]
        bn_outer = module.bns[0]
        y1 = bn_inner(_x + _F)
        y2 = bn_outer(_x + y1)
        actual = module(_x, _F)
        assert torch.allclose(actual, y2, atol=1e-5), \
            "RecursiveSkipBN(lam=2): literal recurrence mismatch"


class TestResidualScaledSkip:
    """y = LN(x + k * F(x, W)) — scales F, not x."""

    @pytest.mark.parametrize("k", [2.0, 3.0])
    def test_equation(self, k):
        module = ResidualScaledSkip(k=k, channels=C, spatial_size=H)
        module.eval()
        # FIXED: use LayerNorm2d as the reference, matching production (see
        # module docstring FIXED note at top of file).
        ln_ref = LayerNorm2d(C)
        ln_ref.load_state_dict(module.ln.state_dict())
        expected = ln_ref(_x + k * _F)
        actual = module(_x, _F)
        assert torch.allclose(actual, expected, atol=1e-5), \
            f"ResidualScaledSkip(k={k}): equation mismatch"


class TestLearnableVectorSkip:
    """wSkip+LN: y = LN(w ⊙ x + F), w ∈ R^C, init w=1."""

    def test_w_is_parameter(self):
        module = LearnableVectorSkip(channels=C, spatial_size=H)
        assert isinstance(module.w, nn.Parameter), "w must be nn.Parameter"

    def test_w_init_ones(self):
        module = LearnableVectorSkip(channels=C, spatial_size=H)
        assert torch.all(module.w == 1.0), "w must be initialised to 1.0"

    def test_w_shape(self):
        module = LearnableVectorSkip(channels=C, spatial_size=H)
        assert module.w.shape == (C, 1, 1), \
            f"w shape should be ({C}, 1, 1), got {module.w.shape}"

    def test_equation(self):
        module = LearnableVectorSkip(channels=C, spatial_size=H)
        module.eval()
        # FIXED: use LayerNorm2d as the reference, matching production (see
        # module docstring FIXED note at top of file).
        ln_ref = LayerNorm2d(C, eps=1e-5)
        ln_ref.load_state_dict(module.ln.state_dict())
        expected = ln_ref(module.w.detach() * _x + _F)
        actual = module(_x, _F)
        assert torch.allclose(actual, expected, atol=1e-5), \
            "LearnableVectorSkip: equation mismatch"

    def test_gradient_flows_to_w(self):
        module = LearnableVectorSkip(channels=C, spatial_size=H)
        x = _x.clone().requires_grad_(True)
        F_x = _F.clone().requires_grad_(True)
        out = module(x, F_x)
        out.sum().backward()
        assert module.w.grad is not None, "No gradient to w"
        assert x.grad is not None, "No gradient to x"
        assert F_x.grad is not None, "No gradient to F_x"

    def test_per_channel_different_after_update(self):
        """After one SGD step on random loss, different channels should have different w."""
        module = LearnableVectorSkip(channels=C, spatial_size=H)
        opt = torch.optim.SGD(module.parameters(), lr=0.1)
        x = torch.randn(B, C, H, W)
        F_x = torch.randn(B, C, H, W)
        out = module(x, F_x)
        loss = (out * torch.randn_like(out)).sum()   # random loss → different grads per channel
        loss.backward()
        opt.step()
        # w should no longer be all-ones
        assert not torch.all(module.w == 1.0), "w is still all-ones after training step"


class TestBuildSkipVariant:
    """Factory function must route to correct variant class from YAML-style dict."""

    @pytest.mark.parametrize("variant_cfg,expected_type", [
        ({"name": "1xSkip", "G": None, "lambda": 1}, PlainSkip),
        ({"name": "2xSkip", "G": None, "lambda": 2}, ExpandedSkip),
        ({"name": "0.5xSkip", "G": None, "lambda": 0.5}, ExpandedSkip),
        ({"name": "2xSkip+LN", "G": "layer_norm", "lambda": 2}, ExpandedSkipLN),
        ({"name": "1xSkip+BN", "G": "batch_norm", "lambda": 1}, ExpandedSkipBN),
        ({"name": "2rSkip+LN", "G": "layer_norm", "lambda": 2, "recursive": True}, RecursiveSkipLN),
        ({"name": "2rSkip+BN", "G": "batch_norm", "lambda": 2, "recursive": True}, RecursiveSkipBN),
        ({"name": "LN(x+2F)", "G": "layer_norm", "lambda": 1, "scale_residual_instead": 2}, ResidualScaledSkip),
        ({"name": "wSkip+LN", "G": "layer_norm", "learnable_w": True}, LearnableVectorSkip),
    ])
    def test_correct_type(self, variant_cfg, expected_type):
        module = build_skip_variant(variant_cfg, channels=C, spatial_size=H)
        assert isinstance(module, expected_type), \
            f"Expected {expected_type.__name__} for config {variant_cfg['name']}, " \
            f"got {type(module).__name__}"

    def test_plain_skip_no_params(self):
        module = build_skip_variant({"name": "1xSkip", "G": None, "lambda": 1}, C, H)
        assert sum(p.numel() for p in module.parameters()) == 0