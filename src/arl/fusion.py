"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

score(d) = sum over input rankings of 1 / (k + rank_i(d)), rank 1-based; a ranking that
does not contain d adds nothing. Only ranks are used, so BM25 and cosine scores, which
live on different scales, never need to be normalized.
"""

from __future__ import annotations

from collections.abc import Sequence


def rrf(
    rankings: Sequence[Sequence[str]], k: int = 60, top_n: int | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked doc-id lists into [(doc_id, score)] best first.

    Ties in fused score are broken by doc id (string order) so the output is deterministic.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return fused if top_n is None else fused[:top_n]


def rrf_runs(
    runs: Sequence[dict[str, list[str]]], k: int = 60, top_n: int | None = None
) -> dict[str, list[tuple[str, float]]]:
    """RRF per query over several runs ({query_id: [doc_id, ...]}) with the same queries."""
    query_ids = runs[0].keys()
    for run in runs[1:]:
        if run.keys() != query_ids:
            raise ValueError("all runs must cover the same queries")
    return {q: rrf([run[q] for run in runs], k=k, top_n=top_n) for q in query_ids}
