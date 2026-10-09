"""Tests for the BM25 wrapper on a toy corpus (no archive, no network)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arl.retrievers.bm25 import (
    batched,
    build_index,
    build_index_reference,
    load_index,
    save_index,
    search,
)

DOCS = [
    ("0", "The Eiffel Tower is in Paris, France."),
    ("1", "Mount Kenya is the second highest mountain in Africa."),
    ("2", "Counties of Virginia include Rockingham County."),
    ("3", "Paris is the capital of France and home to many museums."),
]


def test_batched_splits_and_keeps_order() -> None:
    assert list(batched(range(5), 2)) == [[0, 1], [2, 3], [4]]


def test_build_rejects_ids_that_are_not_positions() -> None:
    with pytest.raises(ValueError):
        build_index([("0", "a doc"), ("5", "another doc")], k1=0.9, b=0.4, chunk_size=1)


def test_search_ranks_matching_doc_first_across_chunks(tmp_path: Path) -> None:
    retriever, tokenizer, stats = build_index(DOCS, k1=0.9, b=0.4, chunk_size=3)
    assert stats["n_docs"] == 4
    docs, scores = search(retriever, tokenizer, ["eiffel tower", "rockingham counties"], k=2)
    assert docs[0][0] == 0
    assert docs[1][0] == 2  # stemming: "counties" matches "County"/"Counties"
    assert scores[0][0] > scores[0][1]


def test_save_load_round_trip(tmp_path: Path) -> None:
    retriever, tokenizer, stats = build_index(DOCS, k1=0.9, b=0.4)
    save_index(retriever, tokenizer, stats, tmp_path / "idx")
    loaded, loaded_tok = load_index(tmp_path / "idx")
    before = search(retriever, tokenizer, ["mountain africa", "unseenword paris"], k=3)
    after = search(loaded, loaded_tok, ["mountain africa", "unseenword paris"], k=3)
    assert (before[0] == after[0]).all()
    assert after[0][0][0] == 1


MORE_DOCS = DOCS + [
    ("4", "Paris Paris Paris: a very repetitive passage about Paris."),
    ("5", "Kenya, Kenya and Virginia counties."),
    ("6", "!!!"),  # tokenizes to nothing
]


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 100])
def test_chunked_builder_matches_stock_bm25s_exactly(chunk_size: int) -> None:
    ref, ref_tok, _ = build_index_reference(MORE_DOCS, k1=0.9, b=0.4)
    new, new_tok, stats = build_index(MORE_DOCS, k1=0.9, b=0.4, chunk_size=chunk_size)
    for key in ("data", "indices", "indptr"):
        assert ref.scores[key].dtype == new.scores[key].dtype
        assert np.array_equal(ref.scores[key], new.scores[key]), key
    assert ref.scores["num_docs"] == new.scores["num_docs"] == stats["n_docs"]
    assert ref.vocab_dict == new.vocab_dict
    assert ref_tok.get_vocab_dict() == new_tok.get_vocab_dict()
