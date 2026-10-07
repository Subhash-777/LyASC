"""PreAct-ResNet-{20,32,44,56,110} for CIFAR, per He et al. (2016b) as used by
the base paper (Liu et al., arXiv:2105.07205). See plan.md Section 4.1.

Architecture decisions logged per plan.md Section 4.1 / Section 8:

DECISION 1 — Downsampling shortcut policy (logged in results/logs/decisions.log):
    The base paper does not state which shortcut option is used at the two
    stage-transition blocks (where channels double and stride=2). He et al. (2016b)
    CIFAR experiments use Option A: identity shortcut with zero-padding for the
    extra channels, keeping the network parameter-light. This implementation
    uses Option A for parity with He et al. (2016b), which the paper explicitly
    builds on. Verification: parameter count for PreAct-ResNet-110 with PlainSkip
    must be ~1.7M (as widely reported); Option A produces ~1.7M, Option B would
    add ~3.3K extra projection params (negligible but distinct).

DECISION 2 — Lambda-scaling at downsampling blocks:
    The base paper is silent on whether lambda applies to the (zero-padded) identity
    at downsampling blocks or only to plain-identity blocks. Choice: lambda-scaling
    (and all normalisation G) applies uniformly to ALL blocks including downsampling
    ones, using the same skip_variant instance at every block in the stage.
    Rationale: the paper describes the variant as a per-block rule; treating
    downsampling blocks differently would require an undocumented special case.
    Logged in results/logs/decisions.log.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.skip_variants import build_skip_variant


# ---------------------------------------------------------------------------
# Downsampling shortcut (Option A: identity + zero-pad channels)
# ---------------------------------------------------------------------------

class IdentityZeroPadShortcut(nn.Module):
    """Option A shortcut for stage-transition blocks.

    Spatially halves with stride-2 average-pooling, then zero-pads channels to
    match the doubled width. No learnable parameters — keeps the network
    parameter-light as in He et al. (2016b) CIFAR experiments.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:
            # Average-pool to match spatial size (matches He et al.'s subsample)
            x = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        # Zero-pad the channel dimension
        pad = self.out_channels - self.in_channels
        if pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad))  # pad last dim group = channels
        return x


# ---------------------------------------------------------------------------
# PreAct Basic Block
# ---------------------------------------------------------------------------

class PreActBasicBlock(nn.Module):
    """Two-conv PreAct residual block: BN→ReLU→Conv3×3→BN→ReLU→Conv3×3.

    The skip-connection equation (the  y = G(λx + F(x,W)) part) is injected
    via the skip_variant argument, following the separation of concerns in
    plan.md Section 4.1 / 4.2.

    For downsampling blocks (stride=2, in_channels != out_channels), the
    identity shortcut uses zero-padding (Option A, see module docstring).
    The skip_variant still applies to the (potentially zero-padded and
    spatially-subsampled) shortcut, per Decision 2 above.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        skip_variant: nn.Module,
        spatial_size: int,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.skip_variant = skip_variant

        # Downsampling shortcut only when dimensions change
        self.shortcut: Optional[nn.Module] = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = IdentityZeroPadShortcut(in_channels, out_channels, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-activation: BN → ReLU before conv1
        out = F.relu(self.bn1(x), inplace=True)

        # Apply shortcut to the pre-activation output (He et al. 2016b convention)
        identity = self.shortcut(out) if self.shortcut is not None else x

        # Residual F(x, W)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))

        # Skip-connection equation: y = G(λ·identity + F(x,W))
        return self.skip_variant(identity, out)


# ---------------------------------------------------------------------------
# PreAct-ResNet
# ---------------------------------------------------------------------------

# Spatial sizes for the three stages at CIFAR input (32×32):
#   Stage 1: 32×32 (no downsampling)
#   Stage 2: 16×16 (stride-2 at first block)
#   Stage 3:  8× 8 (stride-2 at first block)
_CIFAR_STAGE_SPATIAL = [32, 16, 8]
_CIFAR_STAGE_CHANNELS = [16, 32, 64]


class PreActResNet(nn.Module):
    """PreAct-ResNet for CIFAR.

    depth = 6n+2, n blocks per stage, 3 stages, channels {16,32,64}.
    stem: Conv3×3(16)
    head: BN → ReLU → GlobalAvgPool → FC(num_classes)
    """

    def __init__(self, n: int, num_classes: int, variant_cfgs: "list[dict]"):
        """
        Args:
            n:            Number of basic blocks per stage.
            num_classes:  10 for CIFAR-10, 100 for CIFAR-100.
            variant_cfgs: List of per-stage skip-variant config dicts, one per
                          stage. In practice all stages share the same config
                          (from configs/*.yaml), but the spatial size differs per
                          stage, so separate instances are needed.
        """
        super().__init__()
        self.n = n

        # Stem: one 3×3 conv, 16 channels
        self.stem = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        # Build 3 stages
        self.stages = nn.ModuleList()
        in_channels = 16
        for stage_idx, (out_channels, spatial) in enumerate(
            zip(_CIFAR_STAGE_CHANNELS, _CIFAR_STAGE_SPATIAL)
        ):
            blocks = []
            for block_idx in range(n):
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                skip = build_skip_variant(
                    variant_cfg=variant_cfgs[stage_idx],
                    channels=out_channels,
                    spatial_size=spatial,
                )
                blocks.append(
                    PreActBasicBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        skip_variant=skip,
                        spatial_size=spatial,
                    )
                )
                in_channels = out_channels
            self.stages.append(nn.Sequential(*blocks))

        # Head: BN → ReLU → GlobalAvgPool → FC
        self.head_bn = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, num_classes)

        # Kaiming He initialisation for conv layers
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        for stage in self.stages:
            out = stage(out)
        out = F.relu(self.head_bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)

    def count_basic_blocks(self) -> int:
        """Return total number of PreActBasicBlock instances (sanity check)."""
        count = 0
        for stage in self.stages:
            for m in stage.modules():
                if isinstance(m, PreActBasicBlock):
                    count += 1
        return count


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

_DEPTH_TO_N = {20: 3, 32: 5, 44: 7, 56: 9, 110: 18}


def preact_resnet(
    depth: int,
    num_classes: int,
    variant_cfg: dict,
) -> PreActResNet:
    """Build a PreAct-ResNet-`depth` for CIFAR with the given skip-connection variant.

    Args:
        depth:       One of {20, 32, 44, 56, 110}.
        num_classes: 10 or 100.
        variant_cfg: A single variant dict from configs/*.yaml `variants:` list.
                     The same logical variant is applied to all three stages;
                     separate instances are constructed per stage so that each
                     stage's LN/BN has the correct channel count and spatial size.

    Returns:
        PreActResNet instance ready for training.
    """
    if depth not in _DEPTH_TO_N:
        raise ValueError(f"Unsupported depth {depth}; must be one of {sorted(_DEPTH_TO_N)}")
    n = _DEPTH_TO_N[depth]
    # One variant config dict per stage (same logical variant, different spatial sizes)
    variant_cfgs = [variant_cfg, variant_cfg, variant_cfg]
    return PreActResNet(n=n, num_classes=num_classes, variant_cfgs=variant_cfgs)
