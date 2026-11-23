"""Dense retriever helpers: prefixes, slugs, cache fingerprint, exact search (no model load)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from arl.retrievers import dense


def test_model_prefixes_match_model_cards() -> None:
    assert dense.prefixes("BAAI/bge-small-en-v1.5") == (
        "Represent this sentence for searching relevant passages: ",
        "",
    )
    assert dense.prefixes("intfloat/e5-small-v2") == ("query: ", "passage: ")


def test_unknown_model_has_no_silent_default() -> None:
    with pytest.raises(KeyError):
        dense.prefixes("some/other-model")


def test_add_prefix_applies_to_each_text() -> None:
    q, p = dense.prefixes("intfloat/e5-small-v2")
    assert dense.add_prefix(["how tall is k2", "x"], q) == ["query: how tall is k2", "query: x"]
    assert dense.add_prefix(["K2 is 8,611 m."], p) == ["passage: K2 is 8,611 m."]
    bge_q, bge_p = dense.prefixes("BAAI/bge-small-en-v1.5")
    assert dense.add_prefix(["text"], bge_p) == ["text"]  # bge passages get no prefix
    assert dense.add_prefix(["text"], bge_q)[0].endswith("passages: text")


def test_model_slug() -> None:
    assert dense.model_slug("BAAI/bge-small-en-v1.5") == "bge-small-en-v1.5"
    assert dense.model_slug("intfloat/e5-small-v2") == "e5-small-v2"


def test_fingerprint_changes_with_ids_and_model() -> None:
    a = dense.fingerprint("intfloat/e5-small-v2", 256, ["1", "2"])
    assert a["passage_prefix"] == "passage: " and a["normalized"] is True
    assert a != dense.fingerprint("intfloat/e5-small-v2", 256, ["2", "1"])
    assert a != dense.fingerprint("BAAI/bge-small-en-v1.5", 256, ["1", "2"])


def test_mismatched_cache_is_refused(tmp_path) -> None:
    np.save(tmp_path / "e5-small-v2.npy", np.zeros((2, 4), dtype=np.float32))
    meta = dense.fingerprint("intfloat/e5-small-v2", 256, ["1", "2"])
    (tmp_path / "e5-small-v2.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):  # same model, different doc ids
        dense.corpus_embeddings(
            None,
            "intfloat/e5-small-v2",
            ["1", "3"],
            ["a", "b"],
            256,
            8,
            cache_dir=tmp_path,
            log=lambda _: None,
        )
    emb = dense.corpus_embeddings(
        None,
        "intfloat/e5-small-v2",
        ["1", "2"],
        ["a", "b"],
        256,
        8,
        cache_dir=tmp_path,
        log=lambda _: None,
    )
    assert emb.shape == (2, 4)  # matching cache loads without touching the model


def test_flat_ip_search_is_exact_cosine() -> None:
    corpus = np.array([[1, 0], [0.6, 0.8], [0, 1]], dtype=np.float32)  # unit vectors
    queries = np.array([[0, 1], [1, 0]], dtype=np.float32)
    positions, scores = dense.search(corpus, queries, k=2)
    assert positions.tolist() == [[2, 1], [0, 1]]
    assert scores[0].tolist() == pytest.approx([1.0, 0.8])
    assert scores[1].tolist() == pytest.approx([1.0, 0.6])
