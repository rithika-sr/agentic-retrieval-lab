"""Invariants of the built corpus file (skipped when data/ has not been built, e.g. a fresh clone)."""

from __future__ import annotations

import json

import pytest

from arl.config import load_config
from arl.corpus import CORPUS_PATH, SOURCES
from arl.data import QRELS_OFFICIAL_PATH, read_qrels_trec

pytestmark = pytest.mark.skipif(not CORPUS_PATH.exists(), reason="corpus.jsonl not built")


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    with CORPUS_PATH.open() as f:
        return [json.loads(line) for line in f]


def test_size_equals_target(rows: list[dict]) -> None:
    assert len(rows) == load_config()["corpus"]["target_size"]


def test_no_duplicate_ids(rows: list[dict]) -> None:
    assert len({r["doc_id"] for r in rows}) == len(rows)


def test_all_official_positives_present_and_tagged(rows: list[dict]) -> None:
    positives = {d for docs in read_qrels_trec(QRELS_OFFICIAL_PATH).values() for d in docs}
    source = {r["doc_id"]: r["source"] for r in rows}
    assert positives <= source.keys()
    assert {source[d] for d in positives} == {"positive"}


def test_every_row_has_text_and_a_known_source(rows: list[dict]) -> None:
    assert all(r["text"].strip() for r in rows)
    assert {r["source"] for r in rows} <= set(SOURCES)
