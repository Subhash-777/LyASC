"""All skip-connection block-combination equations from the base paper, per
plan.md Section 4.2. Each variant implements the general form
    y = G(lambda * x + F(x, W))
or, for the recursive family, the literal recurrence
    y_k = LN(x + y_{k-1}),  y_1 = LN(x + F(x, W)).

Implements one class per row of the table in plan.md Section 4.2:
    - PlainSkip (1xSkip / baseline): y = x + F(x, W)
    - ExpandedSkip (2xSkip, 3xSkip, 0.5xSkip): y = lambda * x + F(x, W), no G
    - ExpandedSkipLN (xSkip+LN): y = LN(lambda * x + F(x, W))
    - ExpandedSkipBN (xSkip+BN): y = BN(lambda * x + F(x, W))
    - RecursiveSkipLN (rSkip+LN): literal recurrence, G=LayerNorm, for lambda in {1,2,3}
    - RecursiveSkipBN (rSkip+BN): literal recurrence, G=BatchNorm
    - ResidualScaledSkip (LN(x+2F), LN(x+3F) from Table 3): y = LN(x + k * F(x, W))
    - LearnableVectorSkip (wSkip+LN): y = LN(w . x + F(x)), w in R^C, nn.Parameter,
      initialized to 1.0 — THIS IS THE CRITICAL ABLATION FOR LyASC.

Every class exposes:
    def forward(self, x: torch.Tensor, F_x: torch.Tensor) -> torch.Tensor: ...

so that preact_resnet.py's basic block can call
    out = skip_variant(x, F(x))
without knowing which equation is behind it.

DECISION (plan.md Section 8, resolved after debugging — see decisions.log
"layernorm_axis" entry, logged from src/training/train.py at startup):
    LayerNorm is implemented as channel-wise, per-spatial-position
    normalization (LayerNorm2d below), NOT joint normalization over the full
    (C, H, W) volume. The base paper grounds its LN choice explicitly in the
    Transformer / Ba et al. (2016) convention, where LayerNorm normalizes
    across the feature dimension independently for each token. For a conv
    feature map the natural analogue is: each spatial position (h, w) is a
    "token", channels are its "features" — so normalization is computed
    per-(n, h, w) across the C dimension only.
    An earlier implementation used nn.LayerNorm([C, H, W]), which pools the
    *entire* spatial extent into one mean/std per sample — a much more
    aggressive, different normalization than the paper describes. That
    version produced real (non-NaN) mid-training instability on
    lambda-scaled + LN variants (e.g. 2xSkip+LN reproduced at 13.31% test
    error vs. the paper's 6.29%, with a pronounced accuracy dip across
    iters ~8000-32000 under the sustained high initial LR) even after
    gradient clipping had already ruled out raw gradient-spike collapse as
    the cause. LayerNorm2d replaces it.

DECISION (added post-hoc, performance-only, no numerical change):
    LayerNorm2d is implemented via a permute + torch.nn.functional.layer_norm
    (fused kernel) rather than manual mean/var/broadcast ops. The manual
    version is mathematically identical but roughly doubled per-run training
    time (observed: 1xSkip unpatched 11304s vs. 2xSkip+LN with manual
    LayerNorm2d 22017s for the same 64000-iteration schedule). This swap does
    NOT require re-validating any already-obtained result (e.g. the 2xSkip+LN
    seed0 PASS at 5.97% test error) — it produces bit-for-bit-equivalent
    values up to floating point op ordering, only faster.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for conv feature maps: normalizes across the
    channel dimension independently at each (H, W) spatial position, matching
    the Transformer / Ba et al. (2016) convention the base paper cites for its
    combination of skip connection + layer normalization (each spatial
    location is a "token", channels are its "features").

    This replaces a prior nn.LayerNorm([C, H, W]) implementation that jointly
    normalized over the full spatial extent — see the module-level DECISION
    note above for why that was wrong and what it caused.

    Implemented via a permute + F.layer_norm (fused kernel) rather than manual
    mean/var/broadcast ops, purely for speed — see the second module-level
    DECISION note above. Numerically equivalent to the manual formulation.

    weight/bias are per-channel learnable affine parameters, shape (C,),
    matching standard LayerNorm semantics and — importantly — matching the
    the (ndim <= 1) weight-decay-exclusion check in
    src/training/train.py's _build_optimizer(), so these are correctly
    excluded from weight decay automatically (the previous (C, H, W)-shaped
    affine params were 3-D and were silently NOT being excluded — see the
    "weight_decay_scope_correction" decision logged at training startup).
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W). Normalize across C independently at each (n, h, w)
        # via PyTorch's fused F.layer_norm, which normalizes over the last
        # dimension(s) it's given — so permute channels to the end, normalize,
        # then permute back.
        x = x.permute(0, 2, 3, 1)  # N,C,H,W -> N,H,W,C
        x = F.layer_norm(x, (self.channels,), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)  # N,H,W,C -> N,C,H,W

    def extra_repr(self) -> str:
        return f"channels={self.channels}, eps={self.eps}"


def _make_ln(
    channels: int, spatial_size: Optional[int] = None, eps: float = 1e-5
) -> nn.Module:
    """Factory for the LN operator used by every G="layer_norm" variant below.

    `spatial_size` is kept in the signature for call-site compatibility with
    every class in this module (all of them pass it through unchanged) but is
    no longer used now that LN is channel-wise rather than joint-(C,H,W) —
    see the module-level DECISION note.

    `eps` defaults to 1e-5 for every variant EXCEPT LearnableVectorSkip
    (wSkip+LN), which passes a larger value — see the "wskip_ln_eps" DECISION
    note on LearnableVectorSkip below. Every other already-validated variant
    (1xSkip, 2xSkip+LN, 2rSkip+LN, etc.) is unaffected by this parameter.
    """
    return LayerNorm2d(channels, eps=eps)


class PlainSkip(nn.Module):
    """Baseline / 1xSkip: y = x + F(x, W).

    Plan.md Section 4.2, Equation 1 (standard residual connection, lambda=1, G=None).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, F_x: torch.Tensor) -> torch.Tensor:
        return x + F_x


class ExpandedSkip(nn.Module):
    """Scaled skip without normalisation: y = lambda * x + F(x, W).

    Covers 0.5xSkip, 2xSkip, 3xSkip (Table 3 contracted skip, Table 1
    expanded skip). Plan.md Section 4.2, lambda in {0.5, 2, 3}.
    """

    def __init__(self, lam: float):
        super().__init__()
        self.lam = lam

    def forward(self, x: torch.Tensor, F_x: torch.Tensor) -> torch.Tensor:
        return self.lam * x + F_x


class ExpandedSkipLN(nn.Module):
    """xSkip+LN: y = LN(lambda * x + F(x, W)).

    Plan.md Section 4.2.  lambda in {0.5, 1, 2, 3}, G=LayerNorm.
    The LN is applied after the combination, channel-wise per spatial
    position (see LayerNorm2d).
    """

    def __init__(self, lam: float, channels: int, spatial_size: Optional[int] = None):
        super().__init__()
        self.lam = lam
        self.ln = _make_ln(channels, spatial_size)

    def forward(self, x: torch.Tensor, F_x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.lam * x + F_x)


class ExpandedSkipBN(nn.Module):
    """xSkip+BN: y = BN(lambda * x + F(x, W)).

    Plan.md Section 4.2.  lambda in {1, 2, 3}, G=BatchNorm.
    """

    def __init__(self, lam: float, channels: int):
        super().__init__()
        self.lam = lam
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor, F_x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.lam * x + F_x)


class RecursiveSkipLN(nn.Module):
    """rSkip+LN: literal recurrence y_k = LN(x + y_{k-1}), y_1 = LN(x + F(x,W)).

    Implemented as per plan.md Section 4.3 — computed iteratively for lam
    steps, NOT via the closed-form in Appendix A (which is for interpretability
    only). Each LN at step k is an independent set of learned affine parameters,
    each a channel-wise LayerNorm2d (see module-level DECISION note).

    lam in {1, 2, 3}:
      - lam=1: y = LN(x + F(x, W))          [same as 1xSkip+LN]
      - lam=2: y = LN(x + LN(x + F(x, W)))
      - lam=3: y = LN(x + LN(x + LN(x + F(x, W))))
    """

    def __init__(self, lam: int, channels: int, spatial_size: Optional[int] = None):
        super().__init__()
        assert lam in (1, 2, 3), f"RecursiveSkipLN only supports lam in {{1,2,3}}, got {lam}"
        self.lam = lam
        # One independent LN per recursion step (outermost first in the list)
        self.lns = nn.ModuleList([
            _make_ln(channels, spatial_size) for _ in range(lam)
        ])

    def forward(self, x: torch.Tensor, F_x: torch.Tensor) -> torch.Tensor:
        # Innermost step: y_1 = LN_0(x + F(x, W))
        # Steps proceed outward: y_k = LN_{k-1}(x + y_{k-1})
        # lns[-1] = innermost LN; lns[0] = outermost LN
        y = self.lns[-1](x + F_x)          # y_1
        for ln in reversed(self.lns[:-1]):  # y_2 … y_lam
            y = ln(x + y)
        return y


class RecursiveSkipBN(nn.Module):
    """rSkip+BN: same literal recurrence but with BatchNorm instead of LayerNorm.

    lam in {1, 2, 3}.  Plan.md Section 4.2.
    """

    def __init__(self, lam: int, channels: int):
        super().__init__()
        assert lam in (1, 2, 3), f"RecursiveSkipBN only supports lam in {{1,2,3}}, got {lam}"
        self.lam = lam
        self.bns = nn.ModuleList([nn.BatchNorm2d(channels) for _ in range(lam)])

    def forward(self, x: torch.Tensor, F_x: torch.Tensor) -> torch.Tensor:
        y = self.bns[-1](x + F_x)           # y_1 (innermost)
        for bn in reversed(self.bns[:-1]):   # y_2 … y_lam
            y = bn(x + y)
        return y


class ResidualScaledSkip(nn.Module):
    """LN(x + k*F(x, W)): scales the residual branch, not the skip.

    Table 3, rows LN(x+2F) and LN(x+3F).  k in {2, 3}.
    Plan.md Section 4.2.
    """

    def __init__(self, k: float, channels: int, spatial_size: Optional[int] = None):
        super().__init__()
        self.k = k
        self.ln = _make_ln(channels, spatial_size)

    def forward(self, x: torch.Tensor, F_x: torch.Tensor) -> torch.Tensor:
        return self.ln(x + self.k * F_x)


class LearnableVectorSkip(nn.Module):
    """wSkip+LN: y = LN(w ⊙ x + F(x)), w ∈ ℝ^C learnable, init w=1.

    Plan.md Section 4.2 (critical ablation for LyASC).  w is a per-channel
    learned parameter (broadcast over H,W), initialised to 1.0 so it starts
    as the identity shortcut.  The paper reports this collapses to small norm
    (~0.24 avg per-dim) and gives 8.66% error vs 6.02% for 2rSkip+LN.

    w shape: (C, 1, 1) for broadcastable element-wise multiplication.
    Named `.w` on this module so train.py's checkpoint-time inspection
    (`name.endswith(".skip_variant.w")`) can find and log it per-block.

    DECISION (wskip_ln_eps, added post-hoc after the w_gate_learning_rate
    mitigation was tested and shown NOT to be the primary cause):
    A diagnostic split in train.py's training loop measured wSkip+LN's
    grad_norm~1050 spike at iter 1000 as w_grad_norm=62.1 vs.
    other_grad_norm=1048.7 — i.e. ~99% of the blow-up comes from the REST
    of the model (conv/LN weights), not from w itself. This points at
    LayerNorm2d's backward pass: the 1/sqrt(var+eps) term can become very
    large if some channel's cross-channel variance at a given spatial
    position sits close to zero. Because w is a PER-CHANNEL multiplier
    (unlike a single scalar lambda in the other LN variants), even small
    per-channel divergence in w during training can push a channel's
    relative contribution toward a near-degenerate (low-variance-across-
    channels) configuration at some positions — a route to LN instability
    unique to this variant. eps is raised from the default 1e-5 to 1e-2 for
    LearnableVectorSkip's LN only, reducing the worst-case 1/sqrt(eps)
    multiplier from ~316 to ~10. Every other variant's LN keeps eps=1e-5
    unchanged, so already-validated results (1xSkip, 2xSkip+LN seed0, etc.)
    are unaffected.
    """

    # See "wskip_ln_eps" decision above. Reverted to 1e-5 per instructions.
    _LN_EPS = 1e-5

    def __init__(self, channels: int, spatial_size: Optional[int] = None):
        super().__init__()
        self.ln = _make_ln(channels, spatial_size, eps=self._LN_EPS)
        # Per-channel learnable weight, init=1.0 (identity shortcut start)
        self.w = nn.Parameter(torch.ones(channels, 1, 1))

    def forward(self, x: torch.Tensor, F_x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.w * x + F_x)


# ---------------------------------------------------------------------------
# Factory function — maps config YAML fields to the correct variant class.
# ---------------------------------------------------------------------------

def build_skip_variant(
    variant_cfg: dict,
    channels: int,
    spatial_size: int,
) -> nn.Module:
    """Instantiate the correct skip-connection variant from a YAML variant dict.

    Args:
        variant_cfg: One entry from configs/*.yaml  `variants:` list.
        channels:    Number of output channels of the residual block at this
                     stage (16, 32, or 64 for CIFAR PreAct-ResNets).
        spatial_size: Spatial size of the feature map at this stage
                      (32, 16, or 8 for CIFAR). Passed through to LN-based
                      variants for signature compatibility only — LayerNorm2d
                      does not use it (see module-level DECISION note).

    Returns:
        nn.Module with forward(x, F_x) -> torch.Tensor interface.
    """
    name: str = variant_cfg["name"]
    G: Optional[str] = variant_cfg.get("G")
    lam: float = variant_cfg.get("lambda", 1)
    recursive: bool = variant_cfg.get("recursive", False)
    learnable_w: bool = variant_cfg.get("learnable_w", False)
    scale_F: Optional[float] = variant_cfg.get("scale_residual_instead")

    # wSkip+LN
    if learnable_w:
        return LearnableVectorSkip(channels=channels, spatial_size=spatial_size)

    # LN(x + k*F) variants (Table 3 scale-F rows)
    if scale_F is not None:
        return ResidualScaledSkip(k=float(scale_F), channels=channels, spatial_size=spatial_size)

    # Recursive family
    if recursive:
        lam_int = int(lam)
        if G == "layer_norm":
            return RecursiveSkipLN(lam=lam_int, channels=channels, spatial_size=spatial_size)
        elif G == "batch_norm":
            return RecursiveSkipBN(lam=lam_int, channels=channels)
        else:
            raise ValueError(f"Recursive skip requires G in {{layer_norm, batch_norm}}, got {G}")

    # Non-recursive family
    if G is None:
        if lam == 1:
            return PlainSkip()
        else:
            return ExpandedSkip(lam=float(lam))
    elif G == "layer_norm":
        return ExpandedSkipLN(lam=float(lam), channels=channels, spatial_size=spatial_size)
    elif G == "batch_norm":
        return ExpandedSkipBN(lam=float(lam), channels=channels)
    else:
        raise ValueError(f"Unknown G: {G!r} for variant {name!r}")