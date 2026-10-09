"""BM25 over MS MARCO passages with bm25s (stemmed, stopwords removed, Lucene scoring).

CLI:
  uv run python -m arl.retrievers.bm25 build --limit 500000 --out data/processed/bm25_slice500k
  uv run python -m arl.retrievers.bm25 build --out data/processed/bm25_full

The collection is streamed from the cached archive and tokenized in chunks; raw text is
dropped after each chunk and only compact (word, tf) postings are kept (see `build_index`,
checked bit-for-bit against stock bm25s indexing in `build_index_reference`). MS MARCO passage ids are
0..8,841,822 in file order, so a passage's index position *is* its doc id (checked while
streaming), and no id table is stored.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path

import bm25s
import bm25s.scoring as bm25s_scoring
import numpy as np
import Stemmer
from bm25s.tokenization import Tokenized, Tokenizer

from arl.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from arl.data import iter_collection

CHUNK_SIZE = 100_000
META_NAME = "arl_meta.json"


def make_tokenizer() -> Tokenizer:
    """Lowercase, English stopwords removed, Snowball English stemming."""
    return Tokenizer(lower=True, stopwords="en", stemmer=Stemmer.Stemmer("english"))


def batched(items: Iterable[tuple[str, str]], size: int) -> Iterator[list[tuple[str, str]]]:
    """Yield lists of up to `size` items (itertools.batched is Python 3.12+)."""
    it = iter(items)
    while chunk := list(islice(it, size)):
        yield chunk


def peak_rss_gb() -> float:
    """Peak resident memory of this process so far, in GB (macOS reports bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def tokenize_collection(
    docs: Iterable[tuple[str, str]], tokenizer: Tokenizer, chunk_size: int = CHUNK_SIZE
) -> list[list[int]]:
    """Token ids per passage, in order. Raises if a doc id differs from its position."""
    all_ids: list[list[int]] = []
    for chunk in batched(docs, chunk_size):
        start = len(all_ids)
        for offset, (doc_id, _) in enumerate(chunk):
            if doc_id != str(start + offset):
                raise ValueError(f"doc id {doc_id} at position {start + offset}")
        texts = [text for _, text in chunk]
        all_ids.extend(
            tokenizer.tokenize(texts, update_vocab=True, return_as="ids", show_progress=False)
        )
        print(f"  tokenized {len(all_ids):,} passages; peak RSS {peak_rss_gb():.2f} GB", flush=True)
    return all_ids


def build_index_reference(
    docs: Iterable[tuple[str, str]], k1: float, b: float, chunk_size: int = CHUNK_SIZE
) -> tuple[bm25s.BM25, Tokenizer, dict[str, float]]:
    """Stock bm25s indexing (all token lists in memory). The exactness baseline for
    `build_index`; too memory-hungry for the full collection on a 16 GB laptop."""
    tokenizer = make_tokenizer()
    t0 = time.perf_counter()
    ids = tokenize_collection(docs, tokenizer, chunk_size)
    t1 = time.perf_counter()
    rss_after_tokenize = peak_rss_gb()
    retriever = bm25s.BM25(k1=k1, b=b, method="lucene")
    retriever.index(Tokenized(ids=ids, vocab=tokenizer.get_vocab_dict()), show_progress=False)
    t2 = time.perf_counter()
    stats = {
        "builder": "reference",
        "n_docs": len(ids),
        "vocab_size": len(tokenizer.get_vocab_dict()),
        "mean_tokens_per_doc": round(sum(map(len, ids)) / max(len(ids), 1), 2),
        "tokenize_s": round(t1 - t0, 1),
        "index_s": round(t2 - t1, 1),
        "peak_rss_gb_after_tokenize": round(rss_after_tokenize, 2),
        "peak_rss_gb_after_index": round(peak_rss_gb(), 2),
    }
    return retriever, tokenizer, stats


@dataclass
class _Chunk:
    """Compact per-chunk postings: one entry per distinct (passage, word) pair, in passage
    order and, within a passage, in word-id order."""

    start: int  # position of the chunk's first passage
    doc_lens: np.ndarray  # int64, tokens per passage
    n_unique: np.ndarray  # int64, distinct words per passage
    terms: np.ndarray  # int32, word id per entry
    tfs: np.ndarray  # uint16, term frequency per entry


def chunk_postings(start: int, token_ids: list[list[int]]) -> _Chunk:
    """Turn a chunk's token-id lists into compact (word, tf) postings per passage."""
    doc_lens = np.fromiter(map(len, token_ids), dtype=np.int64, count=len(token_ids))
    flat = np.fromiter(chain.from_iterable(token_ids), dtype=np.int64, count=int(doc_lens.sum()))
    local_doc = np.repeat(np.arange(len(token_ids), dtype=np.int64), doc_lens)
    keys, tfs = np.unique((local_doc << 32) | flat, return_counts=True)  # sorted: doc, then word
    if tfs.max(initial=0) > np.iinfo(np.uint16).max:
        raise OverflowError("term frequency does not fit in uint16")
    return _Chunk(
        start=start,
        doc_lens=doc_lens,
        n_unique=np.bincount(keys >> 32, minlength=len(token_ids)),
        terms=(keys & 0xFFFFFFFF).astype(np.int32),
        tfs=tfs.astype(np.uint16),
    )


def build_index(
    docs: Iterable[tuple[str, str]], k1: float, b: float, chunk_size: int = CHUNK_SIZE
) -> tuple[bm25s.BM25, Tokenizer, dict[str, float]]:
    """Memory-lean indexing that reproduces `build_index_reference` bit for bit.

    Pass 1 streams the collection, keeping only compact (word, tf) postings per chunk.
    Then document frequencies, avgdl, and IDF are computed with bm25s's own functions.
    Pass 2 scores each chunk with bm25s's tf formula and drops it straight into its CSC
    slot (a counting sort keyed on the known per-word counts), so no global sort is needed.
    """
    tokenizer = make_tokenizer()
    retriever = bm25s.BM25(k1=k1, b=b, method="lucene")
    t0 = time.perf_counter()

    chunks: list[_Chunk] = []
    n_docs = 0
    for batch in batched(docs, chunk_size):
        for offset, (doc_id, _) in enumerate(batch):
            if doc_id != str(n_docs + offset):
                raise ValueError(f"doc id {doc_id} at position {n_docs + offset}")
        ids = tokenizer.tokenize(
            [text for _, text in batch], update_vocab=True, return_as="ids", show_progress=False
        )
        chunks.append(chunk_postings(n_docs, ids))
        n_docs += len(batch)
        del ids
        print(f"  pass 1: {n_docs:,} passages; peak RSS {peak_rss_gb():.2f} GB", flush=True)
    t1 = time.perf_counter()
    rss_after_pass1 = peak_rss_gb()

    # Corpus statistics, computed exactly as bm25s.BM25.build_index_from_ids does.
    vocab_dict = tokenizer.get_vocab_dict()
    n_vocab = len(vocab_dict)
    doc_lens = np.concatenate([c.doc_lens for c in chunks])
    avg_doc_len = doc_lens.mean()
    del doc_lens
    df = np.zeros(n_vocab, dtype=np.int64)
    for c in chunks:
        df += np.bincount(c.terms, minlength=n_vocab)
    doc_frequencies = {int(t): int(df[t]) for t in vocab_dict.values()}
    idf_array = bm25s_scoring._build_idf_array(
        doc_frequencies=doc_frequencies,
        n_docs=n_docs,
        compute_idf_fn=bm25s_scoring._select_idf_scorer(retriever.idf_method),
        dtype=retriever.dtype,
    )
    del doc_frequencies
    calculate_tfc = bm25s_scoring._select_tfc_scorer(retriever.method)

    # CSC layout: column = word, rows (passages) ascending within each column.
    indptr = np.zeros(n_vocab + 1, dtype=np.int64)
    np.cumsum(df, out=indptr[1:])
    nnz = int(indptr[-1])
    data = np.empty(nnz, dtype=retriever.dtype)
    indices = np.empty(nnz, dtype=retriever.int_dtype)
    cursor = indptr[:-1].copy()  # next free slot per word
    while chunks:
        c = chunks.pop(0)  # free each chunk once placed
        rows = c.start + np.repeat(np.arange(len(c.doc_lens), dtype=np.int64), c.n_unique)
        l_d = np.repeat(c.doc_lens, c.n_unique).astype(np.float64)
        tfc = calculate_tfc(
            tf_array=c.tfs.astype(retriever.dtype), l_d=l_d, l_avg=avg_doc_len, k1=k1, b=b
        )
        scores = idf_array[c.terms] * tfc
        order = np.argsort(c.terms, kind="stable")  # group by word, rows stay ascending
        terms_sorted = c.terms[order]
        words, first, counts = np.unique(terms_sorted, return_index=True, return_counts=True)
        dest = cursor[terms_sorted] + (np.arange(len(order)) - np.repeat(first, counts))
        data[dest] = scores[order]
        indices[dest] = rows[order]
        cursor[words] += counts
    t2 = time.perf_counter()

    retriever.scores = {"data": data, "indices": indices, "indptr": indptr, "num_docs": n_docs}
    retriever.nonoccurrence_array = None  # only used by BM25L / BM25+
    if "" not in vocab_dict:  # bm25s adds an empty token when indexing; mirror it
        vocab_dict[""] = max(vocab_dict.values()) + 1
    retriever.vocab_dict = vocab_dict
    retriever.unique_token_ids_set = set(vocab_dict.values())
    stats = {
        "builder": "chunked",
        "n_docs": n_docs,
        "vocab_size": n_vocab,
        "nnz": nnz,
        "mean_tokens_per_doc": round(float(avg_doc_len), 2),
        "tokenize_s": round(t1 - t0, 1),
        "index_s": round(t2 - t1, 1),
        "peak_rss_gb_after_tokenize": round(rss_after_pass1, 2),
        "peak_rss_gb_after_index": round(peak_rss_gb(), 2),
    }
    return retriever, tokenizer, stats


def save_index(retriever: bm25s.BM25, tokenizer: Tokenizer, stats: dict, out_dir: Path) -> None:
    """Save the index, the tokenizer vocab/stopwords, and our build stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(out_dir, show_progress=False)
    tokenizer.save_vocab(out_dir)
    tokenizer.save_stopwords(out_dir)
    (out_dir / META_NAME).write_text(json.dumps(stats, indent=2) + "\n")


def load_index(index_dir: Path) -> tuple[bm25s.BM25, Tokenizer]:
    """Load a saved index (memory-mapped) and its tokenizer."""
    retriever = bm25s.BM25.load(index_dir, mmap=True)
    tokenizer = make_tokenizer()
    tokenizer.load_vocab(index_dir)
    tokenizer.load_stopwords(index_dir)
    return retriever, tokenizer


def search(
    retriever: bm25s.BM25,
    tokenizer: Tokenizer,
    queries: list[str],
    k: int,
    n_threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k (doc positions, scores) per query, shape (n_queries, k).

    `n_threads`: 0 = single thread, -1 = all cores (bm25s convention).
    Query words never seen in the collection are dropped (they cannot match anything).
    """
    query_ids = tokenizer.tokenize(
        queries, update_vocab=False, return_as="ids", show_progress=False
    )
    k = min(k, retriever.scores["num_docs"])
    docs, scores = retriever.retrieve(query_ids, k=k, show_progress=False, n_threads=n_threads)
    return docs, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="index the collection (or its first --limit passages)")
    build.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    build.add_argument("--limit", type=int, default=None, help="first N passages only")
    build.add_argument("--out", required=True, help="output directory for the index")
    args = parser.parse_args()

    config = load_config(args.config)
    bm25_cfg = config["retrievers"]["bm25"]
    out_dir = Path(args.out)
    out_dir = out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; delete it first to rebuild")

    docs = islice(iter_collection(), args.limit)
    retriever, tokenizer, stats = build_index(docs, k1=bm25_cfg["k1"], b=bm25_cfg["b"])
    stats.update({"k1": bm25_cfg["k1"], "b": bm25_cfg["b"], "limit": args.limit})
    t0 = time.perf_counter()
    save_index(retriever, tokenizer, stats, out_dir)
    stats["save_s"] = round(time.perf_counter() - t0, 1)
    stats["peak_rss_gb_final"] = round(peak_rss_gb(), 2)
    (out_dir / META_NAME).write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"Saved index to {out_dir}")


if __name__ == "__main__":
    main()
