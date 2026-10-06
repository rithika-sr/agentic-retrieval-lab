"""Load the project config and seed every random number generator."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def seed_everything(seed: int) -> None:
    """Seed python's `random`, numpy, and torch (CPU and MPS) with the same seed."""
    import torch  # imported here because torch is slow to import

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the MPS generator on Apple Silicon


def load_config(path: str | Path = DEFAULT_CONFIG_PATH, seed: bool = True) -> dict[str, Any]:
    """Read a YAML config into a dict and, by default, seed all RNGs from `project.seed`.

    Relative paths are resolved against the project root, so scripts work from any folder.
    """
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        config = yaml.safe_load(f) or {}

    if seed:
        try:
            seed_value = int(config["project"]["seed"])
        except (KeyError, TypeError) as err:
            raise KeyError(f"{config_path} must define project.seed") from err
        seed_everything(seed_value)
    return config
