"""Tests for arl.config: loading the YAML and making randomness reproducible."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from arl.config import load_config


def _draw() -> tuple[float, float, float]:
    """Draw one random number from each of python, numpy, and torch."""
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_loads_project_config() -> None:
    config = load_config("configs/config.yaml")
    assert config["project"]["seed"] == 42
    assert config["llm"]["model"]
    assert config["fusion"]["rrf_k"] == 60


def test_same_seed_gives_same_random_numbers() -> None:
    load_config()
    first = _draw()
    load_config()
    second = _draw()
    assert first == second


def test_different_seed_gives_different_numbers(tmp_path: Path) -> None:
    other = tmp_path / "other.yaml"
    other.write_text("project:\n  seed: 7\n")
    load_config()
    default_draw = _draw()
    load_config(other)
    assert _draw() != default_draw


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")


def test_missing_seed_raises(tmp_path: Path) -> None:
    bad = tmp_path / "no_seed.yaml"
    bad.write_text("data:\n  n_human_queries: 5\n")
    with pytest.raises(KeyError, match="project.seed"):
        load_config(bad)
