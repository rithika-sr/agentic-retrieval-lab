"""Per-query retrieval metrics with binary relevance, plus a paired bootstrap.

Every metric takes `ranked` (doc ids, best first) and `relevant` (the set of relevant doc
ids for that query). Relevance is binary: a doc is relevant (gain 1) or not (gain 0).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int = 100) -> float:
    """Share of the relevant docs that appear in the top k."""
    if not relevant:
        raise ValueError("recall is undefined for a query with no relevant docs")
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def first_positive_rank(ranked: Sequence[str], relevant: set[str]) -> int | None:
    """1-based rank of the first relevant doc, or None if none was retrieved."""
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return rank
    return None


def mrr_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """Reciprocal rank of the first relevant doc within the top k (0 if none)."""
    rank = first_positive_rank(ranked[:k], relevant)
    return 0.0 if rank is None else 1.0 / rank


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """Binary nDCG@k: DCG = sum of 1/log2(rank+1) over relevant docs in the top k,
    divided by the DCG of a perfect ranking (all relevant docs first)."""
    if not relevant:
        raise ValueError("nDCG is undefined for a query with no relevant docs")
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def paired_bootstrap(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Bootstrap the mean of per-query differences a - b (same queries, same order).

    Resamples query indices with replacement `n_resamples` times and takes the percentile
    interval of the resampled means. Returns the observed mean difference and the CI.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape or a_arr.ndim != 1:
        raise ValueError(f"need two 1-D arrays of equal length, got {a_arr.shape}, {b_arr.shape}")
    if len(a_arr) == 0:
        raise ValueError("need at least one query")
    diff = a_arr - b_arr
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": len(diff),
        "n_resamples": n_resamples,
        "seed": seed,
    }
