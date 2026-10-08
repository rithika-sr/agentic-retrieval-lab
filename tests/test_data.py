"""Tests for query sampling and the queries/qrels file formats (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from arl.data import (
    positives_summary,
    read_qrels_trec,
    read_queries_jsonl,
    sample_query_ids,
    write_qrels_trec,
    write_queries_jsonl,
)

IDS = [str(i) for i in range(1000)]


def test_sampling_is_deterministic_for_a_seed() -> None:
    assert sample_query_ids(IDS, 50, seed=42) == sample_query_ids(IDS, 50, seed=42)


def test_sampling_changes_with_seed() -> None:
    assert sample_query_ids(IDS, 50, seed=42) != sample_query_ids(IDS, 50, seed=7)


def test_sampling_ignores_input_order() -> None:
    assert sample_query_ids(IDS, 50, seed=42) == sample_query_ids(IDS[::-1], 50, seed=42)


def test_smaller_sample_is_prefix_of_larger() -> None:
    assert sample_query_ids(IDS, 300, seed=42)[:20] == sample_query_ids(IDS, 20, seed=42)


def test_sampling_has_no_duplicates() -> None:
    sample = sample_query_ids(IDS, 300, seed=42)
    assert len(set(sample)) == 300


def test_sampling_too_many_raises() -> None:
    with pytest.raises(ValueError):
        sample_query_ids(IDS[:5], 10, seed=42)


def test_files_round_trip_and_only_keep_sampled_queries(tmp_path: Path) -> None:
    queries = {"1": "what is bm25", "2": "capital of france", "3": "unused"}
    qrels = {"1": {"d1": 1, "d2": 1}, "2": {"d9": 1}, "3": {"d5": 1}}
    sampled = ["2", "1"]

    write_queries_jsonl(sampled, queries, tmp_path / "q.jsonl")
    write_qrels_trec(sampled, qrels, tmp_path / "qrels.tsv")

    records = read_queries_jsonl(tmp_path / "q.jsonl")
    assert [r["query_id"] for r in records] == ["2", "1"]
    assert all(r["query_type"] == "human" for r in records)
    assert read_qrels_trec(tmp_path / "qrels.tsv") == {"2": {"d9": 1}, "1": {"d1": 1, "d2": 1}}


def test_positives_summary() -> None:
    qrels = {"a": {"d1": 1}, "b": {"d1": 1, "d2": 1, "d3": 1}}
    assert positives_summary(["a", "b"], qrels) == {"mean": 2, "min": 1, "max": 3}


def test_fetch_passages_stops_early_and_reports_missing(tmp_path: Path) -> None:
    from arl.data import fetch_passages, read_passages_jsonl, write_passages_jsonl

    seen: list[str] = []

    def stream():
        for i in range(100):
            seen.append(str(i))
            yield str(i), f"text {i}"

    found = fetch_passages(["3", "10"], stream())
    assert found == {"3": "text 3", "10": "text 10"}
    assert seen[-1] == "10"  # stopped as soon as both were found

    write_passages_jsonl(found, tmp_path / "p.jsonl")
    assert read_passages_jsonl(tmp_path / "p.jsonl") == found

    with pytest.raises(KeyError):
        fetch_passages(["3", "missing"], stream())
