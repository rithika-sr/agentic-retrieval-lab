"""Hand-computed toy checks for the retrieval metrics and the paired bootstrap."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arl.eval.metrics import (
    first_positive_rank,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    recall_at_k,
)

RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "e", "z"}  # "z" is relevant but never retrieved


def test_recall_counts_relevant_in_top_k_over_all_relevant() -> None:
    assert recall_at_k(RANKED, RELEVANT, k=1) == 0.0  # top 1 = a
    assert recall_at_k(RANKED, RELEVANT, k=3) == pytest.approx(1 / 3)  # b
    assert recall_at_k(RANKED, RELEVANT, k=5) == pytest.approx(2 / 3)  # b, e (z missing)
    assert recall_at_k(RANKED, {"a"}, k=100) == 1.0  # k longer than the list


def test_first_positive_rank_is_one_based() -> None:
    assert first_positive_rank(RANKED, RELEVANT) == 2
    assert first_positive_rank(RANKED, {"a"}) == 1
    assert first_positive_rank(RANKED, {"z"}) is None


def test_mrr_uses_first_relevant_within_k() -> None:
    assert mrr_at_k(RANKED, RELEVANT, k=10) == 0.5  # first hit at rank 2
    assert mrr_at_k(RANKED, RELEVANT, k=1) == 0.0  # rank 2 is outside the top 1
    assert mrr_at_k(RANKED, {"e"}, k=10) == pytest.approx(1 / 5)
    assert mrr_at_k(RANKED, {"z"}, k=10) == 0.0


def test_ndcg_binary_hand_computed() -> None:
    # Hits at ranks 2 and 5: DCG = 1/log2(3) + 1/log2(6) = 0.63093 + 0.38685 = 1.01778.
    # Ideal (3 relevant, k=5): ranks 1,2,3 -> 1 + 0.63093 + 0.5 = 2.13093.
    expected = (1 / math.log2(3) + 1 / math.log2(6)) / (1 + 1 / math.log2(3) + 1 / math.log2(4))
    assert expected == pytest.approx(0.47762, abs=1e-5)
    assert ndcg_at_k(RANKED, RELEVANT, k=5) == pytest.approx(expected)


def test_ndcg_edge_cases() -> None:
    assert ndcg_at_k(RANKED, {"a"}, k=10) == 1.0  # single positive at rank 1
    assert ndcg_at_k(RANKED, {"b"}, k=10) == pytest.approx(1 / math.log2(3))  # 0.63093
    assert ndcg_at_k(RANKED, {"z"}, k=10) == 0.0
    # Ideal DCG is capped at k: 3 relevant, k=2 -> ideal = 1 + 1/log2(3); one hit at rank 1.
    assert ndcg_at_k(["a", "x"], {"a", "b", "c"}, k=2) == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_metrics_reject_empty_relevant() -> None:
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set())
    with pytest.raises(ValueError):
        ndcg_at_k(RANKED, set())


def test_metrics_agree_with_ranx_on_toy_run() -> None:
    from ranx import Qrels, Run, evaluate

    runs = {"q1": RANKED, "q2": ["x", "y", "a"], "q3": ["c", "a"]}
    rels = {"q1": RELEVANT, "q2": {"a"}, "q3": {"b"}}
    ours = {
        "recall@100": np.mean([recall_at_k(runs[q], rels[q], 100) for q in runs]),
        "mrr@10": np.mean([mrr_at_k(runs[q], rels[q], 10) for q in runs]),
        "ndcg@10": np.mean([ndcg_at_k(runs[q], rels[q], 10) for q in runs]),
    }
    theirs = evaluate(
        Qrels({q: {d: 1 for d in rels[q]} for q in rels}),
        Run({q: {d: float(len(r) - i) for i, d in enumerate(r)} for q, r in runs.items()}),
        list(ours),
    )
    for m, v in ours.items():
        assert theirs[m] == pytest.approx(v)


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.random(50), rng.random(50)
    first = paired_bootstrap(a, b, n_resamples=10_000, seed=42)
    assert first == paired_bootstrap(a, b, n_resamples=10_000, seed=42)
    other = paired_bootstrap(a, b, n_resamples=10_000, seed=7)
    assert other["mean_diff"] == first["mean_diff"]  # the observed mean does not resample
    assert (other["ci_low"], other["ci_high"]) != (first["ci_low"], first["ci_high"])


def test_bootstrap_hand_cases() -> None:
    same = paired_bootstrap([0.1, 0.5, 0.9], [0.1, 0.5, 0.9])
    assert (same["mean_diff"], same["ci_low"], same["ci_high"]) == (0.0, 0.0, 0.0)
    shifted = paired_bootstrap([1.0, 0.5, 0.75], [0.5, 0.0, 0.25])  # every diff = 0.5
    assert shifted["mean_diff"] == pytest.approx(0.5)
    assert shifted["ci_low"] == pytest.approx(0.5)
    assert shifted["ci_high"] == pytest.approx(0.5)
    mixed = paired_bootstrap([1, 0, 1, 1], [0, 0, 0, 1])  # diffs 1,0,1,0 -> mean 0.5
    assert mixed["mean_diff"] == 0.5
    assert 0.0 <= mixed["ci_low"] <= 0.5 <= mixed["ci_high"] <= 1.0
    assert mixed["n"] == 4 and mixed["n_resamples"] == 10_000 and mixed["seed"] == 42


def test_bootstrap_rejects_unpaired_input() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        paired_bootstrap([], [])
