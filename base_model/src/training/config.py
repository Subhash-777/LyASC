"""Config loading/parsing for configs/*.yaml, per plan.md Section 6.

Parses the YAML schema shared by all four configs/*.yaml files:
  dataset, data_dir, model, num_seeds, seeds, optimizer, lr_schedule,
  batch_size, augmentation, variants[] each with paper_error,
  acceptance_tolerance_abs_points.

Also handles the Table 6 depth_and_targets matrix.

Validates LR schedule iteration/epoch consistency and logs epoch counts.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# CIFAR training-set sizes (for epoch ↔ iteration conversions)
DATASET_TRAIN_SIZES = {
    "cifar10": 50_000,
    "cifar100": 50_000,
}


@dataclass
class OptimizerConfig:
    type: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 0.0002
    nesterov: bool = False


@dataclass
class LRScheduleConfig:
    initial_lr: float = 0.1
    milestones_iters: List[int] = field(default_factory=lambda: [32000, 48000])
    total_iters: int = 64000
    gamma: float = 0.1


@dataclass
class AugmentationConfig:
    pad: int = 4
    crop_size: int = 32
    random_flip: bool = True


@dataclass
class VariantConfig:
    name: str
    equation: str = ""
    G: Optional[str] = None
    lam: float = 1.0
    recursive: bool = False
    learnable_w: bool = False
    w_init: float = 1.0
    scale_residual_instead: Optional[float] = None
    paper_error: Optional[float] = None
    note: str = ""

    @property
    def raw_dict(self) -> Dict[str, Any]:
        """Return a raw dict suitable for passing to build_skip_variant()."""
        d: Dict[str, Any] = {"name": self.name, "G": self.G, "lambda": self.lam}
        if self.recursive:
            d["recursive"] = True
        if self.learnable_w:
            d["learnable_w"] = True
        if self.scale_residual_instead is not None:
            d["scale_residual_instead"] = self.scale_residual_instead
        return d


@dataclass
class DepthTarget:
    model: str
    depth: int
    baseline_paper_error: Optional[float] = None
    proposal_paper_error: Optional[float] = None


@dataclass
class ExperimentConfig:
    """Parsed representation of a configs/*.yaml file."""
    config_path: str
    config_name: str            # stem of the YAML filename, e.g. "cifar10_table1"
    dataset: str
    data_dir: str               # resolved absolute path
    model: str                  # e.g. "preact_resnet110" (used as default for non-T6)
    num_seeds: int
    seeds: List[int]
    optimizer: OptimizerConfig
    lr_schedule: LRScheduleConfig
    batch_size: int
    augmentation: AugmentationConfig
    variants: List[VariantConfig]
    acceptance_tolerance: float = 0.4

    # Table 6 only (None otherwise)
    depths_and_targets: Optional[List[DepthTarget]] = None

    def get_depth(self, model_str: Optional[str] = None) -> int:
        """Extract depth integer from a model string like 'preact_resnet110'."""
        m = model_str or self.model
        for d in [20, 32, 44, 56, 110]:
            if str(d) in m:
                return d
        raise ValueError(f"Cannot parse depth from model string: {m!r}")

    def iters_per_epoch(self) -> float:
        """Number of training iterations per epoch (may be fractional)."""
        train_size = DATASET_TRAIN_SIZES[self.dataset]
        return train_size / self.batch_size

    def epoch_of_iter(self, iteration: int) -> float:
        """Convert iteration index to equivalent epoch number (1-indexed)."""
        return iteration / self.iters_per_epoch()

    def log_schedule_epochs(self) -> None:
        """Log iteration→epoch mapping for the LR schedule (plan.md Section 8)."""
        ipe = self.iters_per_epoch()
        milestones_epochs = [m / ipe for m in self.lr_schedule.milestones_iters]
        total_epochs = self.lr_schedule.total_iters / ipe
        logger.info(
            "LR schedule (dataset=%s, batch=%d): %.1f iters/epoch | "
            "milestones at iters %s ≈ epochs %s | total %d iters ≈ %.1f epochs",
            self.dataset,
            self.batch_size,
            ipe,
            self.lr_schedule.milestones_iters,
            [f"{e:.1f}" for e in milestones_epochs],
            self.lr_schedule.total_iters,
            total_epochs,
        )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_depth(model_str: str) -> int:
    for d in [20, 32, 44, 56, 110]:
        if str(d) in model_str:
            return d
    raise ValueError(f"Cannot parse depth from '{model_str}'")


def load_config(yaml_path: str, base_dir: Optional[str] = None) -> ExperimentConfig:
    """Parse a configs/*.yaml file into an ExperimentConfig.

    Args:
        yaml_path: Path to the YAML file (absolute or relative to base_dir).
        base_dir:  If given, used to resolve relative data_dir paths in the YAML.
                   Typically the workspace root (LyASC/base_model/).

    Returns:
        ExperimentConfig dataclass instance.
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    config_name = yaml_path.stem

    # ---- Resolve data_dir ----
    # The data_dir in configs/*.yaml is relative to the YAML file's parent
    # directory (configs/ → ../data/ = base_model/data/).
    # Resolve relative to the YAML file location, not base_dir.
    data_dir_raw: str = raw["data_dir"]
    if not Path(data_dir_raw).is_absolute():
        data_dir = str((yaml_path.parent / data_dir_raw).resolve())
    else:
        data_dir = str(Path(data_dir_raw).resolve())

    # ---- Optimizer ----
    opt_raw = raw.get("optimizer", {})
    optimizer = OptimizerConfig(
        type=opt_raw.get("type", "sgd"),
        momentum=opt_raw.get("momentum", 0.9),
        weight_decay=opt_raw.get("weight_decay", 0.0002),
        nesterov=opt_raw.get("nesterov", False),
    )

    # ---- LR schedule ----
    sched_raw = raw.get("lr_schedule", {})
    lr_schedule = LRScheduleConfig(
        initial_lr=sched_raw.get("initial_lr", 0.1),
        milestones_iters=sched_raw.get("milestones_iters", [32000, 48000]),
        total_iters=sched_raw.get("total_iters", 64000),
        gamma=sched_raw.get("gamma", 0.1),
    )

    # ---- Augmentation ----
    aug_raw = raw.get("augmentation", {})
    augmentation = AugmentationConfig(
        pad=aug_raw.get("pad", 4),
        crop_size=aug_raw.get("crop_size", 32),
        random_flip=aug_raw.get("random_flip", True),
    )

    # ---- Variants ----
    variants = []
    for v in raw.get("variants", []):
        variants.append(VariantConfig(
            name=v["name"],
            equation=v.get("equation", ""),
            G=v.get("G"),
            lam=float(v.get("lambda", 1)),
            recursive=bool(v.get("recursive", False)),
            learnable_w=bool(v.get("learnable_w", False)),
            w_init=float(v.get("w_init", 1.0)),
            scale_residual_instead=(
                float(v["scale_residual_instead"])
                if v.get("scale_residual_instead") is not None else None
            ),
            paper_error=v.get("paper_error"),
            note=v.get("note", ""),
        ))

    # ---- Table 6 depth matrix ----
    depths_and_targets: Optional[List[DepthTarget]] = None
    if "depths_and_targets" in raw:
        depths_and_targets = []
        for dt in raw["depths_and_targets"]:
            model_str = dt["model"]
            depths_and_targets.append(DepthTarget(
                model=model_str,
                depth=_parse_depth(model_str),
                baseline_paper_error=dt.get("baseline_paper_error"),
                proposal_paper_error=dt.get("proposal_paper_error"),
            ))

    cfg = ExperimentConfig(
        config_path=str(yaml_path.resolve()),
        config_name=config_name,
        dataset=raw["dataset"],
        data_dir=data_dir,
        model=raw.get("model", "preact_resnet110"),
        num_seeds=raw.get("num_seeds", 5),
        seeds=raw.get("seeds", [0, 1, 2, 3, 4]),
        optimizer=optimizer,
        lr_schedule=lr_schedule,
        batch_size=raw.get("batch_size", 128),
        augmentation=augmentation,
        variants=variants,
        acceptance_tolerance=raw.get("acceptance_tolerance_abs_points", 0.4),
        depths_and_targets=depths_and_targets,
    )

    # Validate and log schedule epoch mapping
    cfg.log_schedule_epochs()

    return cfg
