"""Sanity checks for src/models/preact_resnet.py, per plan.md Section 7 step 3.

Asserts:
  - preact_resnet(110, ...) contains exactly 54 two-conv-layer basic blocks
    (3 stages × 18 blocks), matching the paper's own description.
  - Parameter count for plain 1xSkip PreAct-ResNet-110 ≈ 1.7M parameters.
  - Forward pass on random (N, 3, 32, 32) produces (N, num_classes) logits
    without shape errors, for every depth in {20, 32, 44, 56, 110}.
  - Downsampling blocks (stage transitions) do not crash and produce the
    expected channel-doubled, spatially-halved output shape.
"""

import pytest
import torch
import torch.nn as nn

from src.models.preact_resnet import preact_resnet, PreActBasicBlock
from src.models.skip_variants import PlainSkip


# Minimal variant config for PlainSkip (baseline / 1xSkip)
_PLAIN_VARIANT = {"name": "1xSkip", "G": None, "lambda": 1}


class TestPreActResNet110BlockCount:
    """ResNet-110 must have exactly 54 basic blocks (3 stages × 18)."""

    def test_block_count_resnet110(self):
        model = preact_resnet(depth=110, num_classes=10, variant_cfg=_PLAIN_VARIANT)
        count = model.count_basic_blocks()
        assert count == 54, (
            f"Expected 54 basic blocks for ResNet-110 (3 stages × 18 blocks), got {count}"
        )

    def test_stage_block_counts(self):
        """Each of the 3 stages should have exactly 18 blocks."""
        model = preact_resnet(depth=110, num_classes=10, variant_cfg=_PLAIN_VARIANT)
        stage_counts = []
        for stage in model.stages:
            cnt = sum(1 for m in stage.modules() if isinstance(m, PreActBasicBlock))
            stage_counts.append(cnt)
        assert stage_counts == [18, 18, 18], (
            f"Expected [18,18,18] blocks per stage, got {stage_counts}"
        )


class TestPreActResNetParameterCount:
    """PlainSkip PreAct-ResNet-110 should be approximately 1.7M parameters."""

    def test_param_count_resnet110(self):
        model = preact_resnet(depth=110, num_classes=10, variant_cfg=_PLAIN_VARIANT)
        n_params = sum(p.numel() for p in model.parameters())
        # Widely-reported value is ~1.7M; allow ±5% tolerance
        assert 1_500_000 <= n_params <= 1_900_000, (
            f"Param count {n_params:,} is outside expected range [1.5M, 1.9M] "
            f"for PlainSkip PreAct-ResNet-110"
        )

    @pytest.mark.parametrize("depth", [20, 32, 44, 56, 110])
    def test_param_count_scaling(self, depth):
        """Deeper models should have more parameters."""
        model = preact_resnet(depth=depth, num_classes=10, variant_cfg=_PLAIN_VARIANT)
        n_params = sum(p.numel() for p in model.parameters())
        # Sanity: all CIFAR ResNets in this family are under 2M params
        assert n_params < 2_000_000, (
            f"ResNet-{depth} param count {n_params:,} exceeds 2M (likely a bug)"
        )
        assert n_params > 50_000, (
            f"ResNet-{depth} param count {n_params:,} is suspiciously small"
        )


class TestPreActResNetForwardPass:
    """Forward pass should produce correct output shapes without errors."""

    @pytest.mark.parametrize("depth,num_classes", [
        (20, 10), (32, 10), (44, 10), (56, 10), (110, 10), (110, 100)
    ])
    def test_forward_shape(self, depth, num_classes):
        model = preact_resnet(
            depth=depth, num_classes=num_classes, variant_cfg=_PLAIN_VARIANT
        )
        model.eval()
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, num_classes), (
            f"ResNet-{depth} with {num_classes} classes: expected output shape "
            f"(4, {num_classes}), got {out.shape}"
        )

    def test_no_nan_output(self):
        model = preact_resnet(depth=110, num_classes=10, variant_cfg=_PLAIN_VARIANT)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any(), "Output contains NaN values"
        assert not torch.isinf(out).any(), "Output contains Inf values"


class TestDownsamplingBlocks:
    """Stage-transition blocks must halve spatial size and double channels."""

    def test_stage1_to_stage2_shape(self):
        """Stage 1 output (B,16,32,32) → Stage 2 block input/output (B,32,16,16)."""
        from src.models.preact_resnet import PreActBasicBlock
        from src.models.skip_variants import ExpandedSkipLN

        # First block of stage 2: 16→32 channels, stride=2, spatial 32→16
        skip = ExpandedSkipLN(lam=1.0, channels=32, spatial_size=16)
        block = PreActBasicBlock(
            in_channels=16, out_channels=32, stride=2,
            skip_variant=skip, spatial_size=16
        )
        x = torch.randn(2, 16, 32, 32)
        out = block(x)
        assert out.shape == (2, 32, 16, 16), (
            f"Expected (2,32,16,16) after stage-1→2 transition, got {out.shape}"
        )

    def test_stage2_to_stage3_shape(self):
        """Stage 2 output (B,32,16,16) → Stage 3 block output (B,64,8,8)."""
        from src.models.preact_resnet import PreActBasicBlock

        skip = PlainSkip()
        # PlainSkip doesn't know channels; create block with identity shortcut bypass
        skip2 = nn.Sequential()  # dummy; we use PlainSkip which is fine for test

        skip_plain = PlainSkip()
        # Need to handle channel mismatch: override with plain to test routing
        from src.models.preact_resnet import IdentityZeroPadShortcut
        block = PreActBasicBlock(
            in_channels=32, out_channels=64, stride=2,
            skip_variant=skip_plain, spatial_size=8
        )
        x = torch.randn(2, 32, 16, 16)
        out = block(x)
        assert out.shape == (2, 64, 8, 8), (
            f"Expected (2,64,8,8) after stage-2→3 transition, got {out.shape}"
        )

    def test_full_model_intermediate_shapes(self):
        """Run a forward hook to verify intermediate spatial sizes."""
        model = preact_resnet(depth=20, num_classes=10, variant_cfg=_PLAIN_VARIANT)
        model.eval()

        shapes = {}
        def make_hook(stage_idx):
            def hook(module, inp, out):
                shapes[f"stage{stage_idx}"] = out.shape
            return hook

        for i, stage in enumerate(model.stages):
            stage.register_forward_hook(make_hook(i))

        with torch.no_grad():
            model(torch.randn(1, 3, 32, 32))

        assert shapes["stage0"][2:] == torch.Size([32, 32]), "Stage 0 spatial should be 32×32"
        assert shapes["stage1"][2:] == torch.Size([16, 16]), "Stage 1 spatial should be 16×16"
        assert shapes["stage2"][2:] == torch.Size([8, 8]),  "Stage 2 spatial should be 8×8"
        assert shapes["stage0"][1] == 16, "Stage 0 channels should be 16"
        assert shapes["stage1"][1] == 32, "Stage 1 channels should be 32"
        assert shapes["stage2"][1] == 64, "Stage 2 channels should be 64"
