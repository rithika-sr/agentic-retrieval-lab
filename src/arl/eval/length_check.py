"""Length-bias check: can passage length alone find the positive among BM25's candidates?

For a query's BM25 top-100 (kept only if it contains a positive), compare MRR@10 of
(a) sorting candidates by word count, shortest first, and (b) a random order. Both are
averaged over the same seeds; in (a) the seed only breaks ties between equal lengths, so
no BM25 rank information leaks into the length ranking.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from arl.eval.metrics import mrr_at_k


def word_count(text: str) -> int:
    """Whitespace word count (same definition as the corpus stats)."""
    return len(text.split())


def shuffled(candidates: Sequence[str], rng: np.random.Generator) -> list[str]:
    """A random permutation of the candidates."""
    return [candidates[i] for i in rng.permutation(len(candidates))]


def rerank_by_length(
    candidates: Sequence[str], lengths: dict[str, int], rng: np.random.Generator
) -> list[str]:
    """Shortest first; ties broken randomly (shuffle, then a stable sort by length)."""
    return sorted(shuffled(candidates, rng), key=lambda d: lengths[d])


def mean_mrr_over_seeds(
    candidates: Sequence[str],
    relevant: set[str],
    lengths: dict[str, int] | None,
    seeds: Sequence[int],
    base_seed: int,
    k: int = 10,
) -> float:
    """Mean MRR@k over seeds: length order if `lengths` is given, else random order."""
    scores = []
    for s in seeds:
        rng = np.random.default_rng((base_seed, s))
        order = (
            rerank_by_length(candidates, lengths, rng)
            if lengths is not None
            else shuffled(candidates, rng)
        )
        scores.append(mrr_at_k(order, relevant, k))
    return float(np.mean(scores))


def eligible_queries(
    runs: dict[str, list[str]], qrels: dict[str, set[str]], depth: int = 100
) -> dict[str, list[str]]:
    """Queries whose top-`depth` contains a positive, mapped to that top-`depth`."""
    return {
        qid: ranked[:depth]
        for qid, ranked in runs.items()
        if qrels.get(qid) and qrels[qid].intersection(ranked[:depth])
    }
