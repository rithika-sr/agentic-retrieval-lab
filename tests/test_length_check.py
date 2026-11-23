"""Toy checks for the length-only re-ranker and the random baseline."""

from __future__ import annotations

import numpy as np
import pytest

from arl.eval.length_check import (
    eligible_queries,
    mean_mrr_over_seeds,
    rerank_by_length,
    word_count,
)

LENGTHS = {"long": 30, "mid": 20, "short": 5, "tiny": 1}


def test_word_count_splits_on_whitespace() -> None:
    assert word_count("one two  three\nfour") == 4


def test_rerank_sorts_shortest_first_for_any_seed() -> None:
    for seed in range(5):
        order = rerank_by_length(
            ["long", "short", "mid", "tiny"], LENGTHS, np.random.default_rng(seed)
        )
        assert order == ["tiny", "short", "mid", "long"]


def test_rerank_breaks_ties_randomly_but_deterministically() -> None:
    lengths = {"a": 10, "b": 10, "c": 3}
    orders = {
        tuple(rerank_by_length(["a", "b", "c"], lengths, np.random.default_rng(s)))
        for s in range(20)
    }
    assert orders == {("c", "a", "b"), ("c", "b", "a")}  # c always first; a/b tie both ways
    run1 = rerank_by_length(["a", "b", "c"], lengths, np.random.default_rng(3))
    assert run1 == rerank_by_length(["a", "b", "c"], lengths, np.random.default_rng(3))


def test_length_mrr_hand_cases() -> None:
    cands = ["long", "mid", "short", "tiny"]
    seeds = list(range(10))
    # positive is the shortest -> rank 1 every time; the longest -> rank 4 every time.
    assert mean_mrr_over_seeds(cands, {"tiny"}, LENGTHS, seeds, base_seed=42) == 1.0
    assert mean_mrr_over_seeds(cands, {"long"}, LENGTHS, seeds, base_seed=42) == 0.25
    # k cuts the list: rank 4 is outside the top 3.
    assert mean_mrr_over_seeds(cands, {"long"}, LENGTHS, seeds, base_seed=42, k=3) == 0.0


def test_random_baseline_matches_expected_value() -> None:
    # One positive among 4 candidates, uniform position: E[1/r] = (1 + 1/2 + 1/3 + 1/4)/4.
    expected = (1 + 1 / 2 + 1 / 3 + 1 / 4) / 4  # 0.5208
    cands = ["long", "mid", "short", "tiny"]
    got = mean_mrr_over_seeds(cands, {"mid"}, None, list(range(4000)), base_seed=42)
    assert got == pytest.approx(expected, abs=0.02)
    again = mean_mrr_over_seeds(cands, {"mid"}, None, list(range(4000)), base_seed=42)
    assert got == again


def test_eligible_queries_keeps_only_positives_within_depth() -> None:
    runs = {"q1": ["a", "b", "c"], "q2": ["x", "y", "z"], "q3": ["m", "n", "p"]}
    qrels = {"q1": {"c"}, "q2": {"nope"}, "q3": {"p"}}
    assert eligible_queries(runs, qrels, depth=3) == {"q1": ["a", "b", "c"], "q3": ["m", "n", "p"]}
    assert eligible_queries(runs, qrels, depth=2) == {}  # c and p sit at rank 3
