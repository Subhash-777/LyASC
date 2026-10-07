"""Deterministic seeding helper, per plan.md Section 8 (\"Random seed coverage\" risk).

Sets python random, numpy, torch CPU and CUDA seeds, and configures cudnn
deterministic/benchmark flags. MUST be called with a genuinely different
seed per run (0,1,2,3,4 per plan.md configs) — logs the actual seed used
in every results/logs/ entry for traceability.
"""

import random
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for deterministic reproduction.

    Sets:
      - Python's built-in `random` module
      - NumPy global seed
      - PyTorch CPU seed
      - PyTorch CUDA seed (all GPUs)
      - cudnn deterministic=True, benchmark=False

    Args:
        seed: Integer seed value. Per plan.md Section 8, must be distinct per
              run within a 5-seed sweep (e.g. seeds [0, 1, 2, 3, 4]).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN ops; benchmark=False trades speed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global random seed set to %d", seed)
