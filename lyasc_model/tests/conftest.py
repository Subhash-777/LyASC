"""Conftest for lyasc_model tests, mirroring base_model/tests/conftest.py pattern.

Ensures sys.path includes the lyasc_model root so `from src.models import ...`
imports resolve correctly, and also makes base_model importable for integration
tests that need to reference base_model code.
"""

import sys
from pathlib import Path

import os
import sys
from pathlib import Path

# lyasc_model/ root
lyasc_root = Path(__file__).resolve().parent.parent
# base_model/ root (sibling of lyasc_model/)
base_root = lyasc_root.parent / "base_model"

for r in (str(lyasc_root), str(base_root)):
    if r not in sys.path:
        sys.path.insert(0, r)

import src

for root_path in (base_root, lyasc_root):
    sp = str(root_path / "src")
    if os.path.exists(sp) and sp not in src.__path__:
        src.__path__.append(sp)

for sub in ("models", "training", "utils", "data", "tracking", "evaluation"):
    sub_path_base = str(base_root / "src" / sub)
    sub_path_lyasc = str(lyasc_root / "src" / sub)
    try:
        submod = __import__(f"src.{sub}", fromlist=[sub])
        for sp in (sub_path_base, sub_path_lyasc):
            if os.path.exists(sp) and sp not in submod.__path__:
                submod.__path__.append(sp)
    except ImportError:
        pass
