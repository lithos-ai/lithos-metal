"""Shared fixtures for the oracle tiers (``tests/layers``, ``tests/models``): torch + a checkpoint on disk.

The small same-architecture checkpoint is looked up at ``$MONOLITH_MODELS/<name>`` or ``~/models/<name>``; tests
skip when it (or torch) is missing, so the contract tier stays hermetic.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def checkpoint_dir(name: str) -> Path | None:
    root = os.environ.get("MONOLITH_MODELS", os.path.expanduser("~/models"))
    p = Path(root) / name
    return p if (p / "config.json").exists() else None


def require_torch():
    return pytest.importorskip("torch")


def require_checkpoint(name: str) -> Path:
    p = checkpoint_dir(name)
    if p is None:
        pytest.skip(f"checkpoint {name} not found under $MONOLITH_MODELS or ~/models")
    return p
