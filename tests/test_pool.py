"""Tests for pool run I/O and positive-coverage stats (no index, no network)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from arl.pool import pool_hit_rates, read_trec_run, to_run, write_trec_run


def test_to_run_and_trec_round_trip(tmp_path: Path) -> None:
    run = to_run(["q1", "q1_p"], np.array([[7, 3], [3, 9]]), np.array([[2.5, 1.0], [4.0, 0.5]]))
    assert run["q1"] == [("7", 2.5), ("3", 1.0)]
    write_trec_run(run, tmp_path / "run.tsv")
    lines = (tmp_path / "run.tsv").read_text().splitlines()
    assert lines[0] == "q1\tQ0\t7\t1\t2.500000\tbm25_full"
    assert read_trec_run(tmp_path / "run.tsv") == run


def test_pool_hit_rates_per_type_and_per_parent() -> None:
    variants = [
        {"query_id": "1", "parent_id": "1", "query_type": "human"},
        {"query_id": "1_p", "parent_id": "1", "query_type": "primary"},
        {"query_id": "1_s1", "parent_id": "1", "query_type": "support"},
        {"query_id": "2", "parent_id": "2", "query_type": "human"},
        {"query_id": "2_p", "parent_id": "2", "query_type": "primary"},
        {"query_id": "2_s1", "parent_id": "2", "query_type": "support"},
    ]
    run = {
        "1": [("10", 1.0)],  # hit
        "1_p": [("99", 1.0)],
        "1_s1": [("99", 1.0)],
        "2": [("99", 1.0)],
        "2_p": [("99", 1.0)],
        "2_s1": [("20", 1.0)],  # hit
    }
    qrels = {"1": {"10": 1}, "2": {"20": 1, "21": 1}}
    report = pool_hit_rates(run, variants, qrels)
    assert report["human"] == {"n": 2, "hits": 1, "rate": 0.5}
    assert report["primary"]["hits"] == 0
    assert report["support"]["hits"] == 1
    assert report["any_agent_query_per_parent"] == {"n": 2, "hits": 1, "rate": 0.5}
    assert report["any_variant_per_parent"] == {"n": 2, "hits": 2, "rate": 1.0}
