"""Hand-computed RRF checks."""

from __future__ import annotations

import pytest

from arl.fusion import rrf, rrf_runs


def test_rrf_hand_computed_k60() -> None:
    # A = [a, b, c], B = [b, d, a]
    # a: 1/61 + 1/63 = 0.032266   b: 1/62 + 1/61 = 0.032522
    # c: 1/63        = 0.015873   d: 1/62        = 0.016129
    fused = rrf([["a", "b", "c"], ["b", "d", "a"]], k=60)
    assert [d for d, _ in fused] == ["b", "a", "d", "c"]
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63)
    assert scores["d"] == pytest.approx(1 / 62)


def test_rrf_small_k_changes_order() -> None:
    # k=0: a = 1/1 + 1/3 = 1.333, b = 1/2 + 1/1 = 1.5 -> b still first; c = 1/3, d = 1/2.
    fused = rrf([["a", "b", "c"], ["b", "d", "a"]], k=0)
    assert dict(fused)["a"] == pytest.approx(4 / 3)
    assert [d for d, _ in fused] == ["b", "a", "d", "c"]


def test_rrf_ties_break_by_doc_id_and_top_n() -> None:
    # x and y each appear once at rank 1 -> equal scores; "x" < "y".
    assert [d for d, _ in rrf([["y"], ["x"]], k=60)] == ["x", "y"]
    assert (
        rrf([["a", "b", "c"], ["b", "d", "a"]], k=60, top_n=2)
        == rrf([["a", "b", "c"], ["b", "d", "a"]], k=60)[:2]
    )


def test_rrf_single_ranking_keeps_order() -> None:
    assert [d for d, _ in rrf([["p", "q", "r"]])] == ["p", "q", "r"]


def test_rrf_runs_per_query_and_requires_same_queries() -> None:
    out = rrf_runs([{"q1": ["a", "b"], "q2": ["c"]}, {"q1": ["b"], "q2": ["c", "d"]}], k=60)
    assert [d for d, _ in out["q1"]] == ["b", "a"]
    assert [d for d, _ in out["q2"]] == ["c", "d"]
    with pytest.raises(ValueError):
        rrf_runs([{"q1": ["a"]}, {"q2": ["a"]}])
    with pytest.raises(ValueError):
        rrf([["a"]], k=-1)
