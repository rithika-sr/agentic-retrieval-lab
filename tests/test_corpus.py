"""Tests for subcorpus selection (no archive, no network)."""

from __future__ import annotations

import pytest

from arl.corpus import overlap_stats, pools_by_type, select_corpus_ids

POSITIVES = {"3", "7"}
POOLED = {"7", "10", "11", "12"}


def test_all_positives_present_and_size_equals_target() -> None:
    chosen = select_corpus_ids(POSITIVES, POOLED, n_collection=100, target_size=20, seed=42)
    assert len(chosen) == 20
    assert POSITIVES <= chosen.keys()
    assert POOLED <= chosen.keys()


def test_sources_take_precedence_positive_then_pool() -> None:
    chosen = select_corpus_ids(POSITIVES, POOLED, n_collection=100, target_size=20, seed=42)
    assert chosen["7"] == "positive"  # positive and pooled -> positive
    assert chosen["10"] == "pool"
    assert sum(s == "backfill" for s in chosen.values()) == 20 - 5


def test_no_duplicate_ids_and_backfill_within_collection() -> None:
    chosen = select_corpus_ids(POSITIVES, POOLED, n_collection=30, target_size=30, seed=1)
    assert sorted(chosen, key=int) == [str(i) for i in range(30)]  # every id exactly once


def test_backfill_is_deterministic_and_seed_dependent() -> None:
    a = select_corpus_ids(POSITIVES, POOLED, n_collection=1000, target_size=50, seed=42)
    b = select_corpus_ids(POSITIVES, POOLED, n_collection=1000, target_size=50, seed=42)
    c = select_corpus_ids(POSITIVES, POOLED, n_collection=1000, target_size=50, seed=43)
    assert a == b
    assert a != c


def test_pools_over_target_raise() -> None:
    with pytest.raises(ValueError):
        select_corpus_ids(POSITIVES, POOLED, n_collection=100, target_size=4, seed=42)


def test_pools_by_type_and_overlap() -> None:
    variants = [
        {"query_id": "1", "query_type": "human"},
        {"query_id": "1_p", "query_type": "primary"},
        {"query_id": "1_s1", "query_type": "support"},
    ]
    run = {
        "1": [("a", 1.0), ("b", 0.5)],
        "1_p": [("a", 1.0), ("c", 0.5)],
        "1_s1": [("d", 1.0)],
    }
    pools = pools_by_type(run, variants)
    assert pools == {"human": {"a", "b"}, "primary": {"a", "c"}, "support": {"d"}}
    stats = overlap_stats(pools)
    assert stats["intersection"] == {"human&primary": 1, "human&support": 0, "primary&support": 0}
    assert stats["all_types"] == 0
    assert stats["only_this_type"] == {"human": 1, "primary": 1, "support": 1}
