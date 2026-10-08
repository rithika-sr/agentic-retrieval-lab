"""Load MS MARCO dev/small, sample human queries with a fixed seed, and save them.

CLI: `uv run python -m arl.data --limit 20`
Writes `data/processed/queries_human.jsonl` and `data/processed/qrels_official.tsv`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import statistics
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import BinaryIO

from arl.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
QUERIES_HUMAN_PATH = PROCESSED_DIR / "queries_human.jsonl"
QRELS_OFFICIAL_PATH = PROCESSED_DIR / "qrels_official.tsv"
POSITIVE_PASSAGES_PATH = PROCESSED_DIR / "positive_passages.jsonl"

Qrels = dict[str, dict[str, int]]  # query_id -> {doc_id: relevance}


def load_msmarco(dataset_id: str) -> tuple[dict[str, str], Qrels]:
    """Return (queries, qrels) for an ir_datasets id. Only positive judgments are kept."""
    import ir_datasets  # imported lazily: tests never touch the network

    dataset = ir_datasets.load(dataset_id)
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    qrels: Qrels = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    return queries, qrels


def sample_query_ids(query_ids: Iterable[str], n: int, seed: int) -> list[str]:
    """Deterministically pick `n` query ids.

    The ids are sorted, shuffled once with `seed`, and the first `n` are taken. Because
    every sample is a prefix of the same shuffle, a smaller sample is always a subset
    of a larger one (so `--limit 20` gives the first 20 of the full 300).
    """
    ids = sorted(set(query_ids))
    if n > len(ids):
        raise ValueError(f"Asked for {n} queries but only {len(ids)} are available")
    random.Random(seed).shuffle(ids)
    return ids[:n]


def write_queries_jsonl(query_ids: list[str], queries: dict[str, str], path: Path) -> None:
    """Write one JSON object per line: query_id, text, query_type="human"."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for qid in query_ids:
            record = {"query_id": qid, "text": queries[qid], "query_type": "human"}
            f.write(json.dumps(record) + "\n")


def write_qrels_trec(query_ids: list[str], qrels: Qrels, path: Path) -> None:
    """Write qrels for `query_ids` in TREC format: `query_id 0 doc_id relevance` (tab-separated)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for qid in query_ids:
            for doc_id, rel in sorted(qrels.get(qid, {}).items()):
                f.write(f"{qid}\t0\t{doc_id}\t{rel}\n")


def read_queries_jsonl(path: Path) -> list[dict[str, str]]:
    """Read a queries JSONL file into a list of dicts."""
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def read_qrels_trec(path: Path) -> Qrels:
    """Read a TREC-format qrels file into {query_id: {doc_id: relevance}}."""
    qrels: Qrels = {}
    with path.open() as f:
        for line in f:
            qid, _, doc_id, rel = line.split()
            qrels.setdefault(qid, {})[doc_id] = int(rel)
    return qrels


def msmarco_archive_path() -> Path:
    """Where ir_datasets cached `collectionandqueries.tar.gz` (respects IR_DATASETS_HOME)."""
    home = Path(os.environ.get("IR_DATASETS_HOME", Path.home() / ".ir_datasets"))
    return home / "msmarco-passage" / "collectionandqueries.tar.gz"


class _LocalFile:
    """Minimal ir_datasets 'streamer' over a local file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @contextlib.contextmanager
    def stream(self) -> Iterator[BinaryIO]:
        with self._path.open("rb") as f:
            yield f


def iter_collection(archive: Path | None = None) -> Iterator[tuple[str, str]]:
    """Stream (doc_id, text) for all 8.8M passages straight from the cached archive.

    Uses ir_datasets' own TarExtract + FixEncoding, so text matches `docs_iter()`, but
    nothing is extracted to disk.
    """
    from ir_datasets.datasets.msmarco_passage import FixEncoding
    from ir_datasets.util import TarExtract

    archive = archive or msmarco_archive_path()
    if not archive.exists():
        raise FileNotFoundError(f"{archive} missing; run `uv run python -m arl.data` first")
    for line in FixEncoding(TarExtract(_LocalFile(archive), "collection.tsv")):
        doc_id, text = line.decode().rstrip("\n").split("\t", 1)
        yield doc_id, text


def fetch_passages(doc_ids: Iterable[str], collection: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Pick out `doc_ids` from a (doc_id, text) stream; stops as soon as all are found."""
    wanted = set(doc_ids)
    found: dict[str, str] = {}
    for doc_id, text in collection:
        if doc_id in wanted:
            found[doc_id] = text
            if len(found) == len(wanted):
                break
    missing = wanted - found.keys()
    if missing:
        raise KeyError(f"{len(missing)} passages not in collection, e.g. {sorted(missing)[:3]}")
    return found


def write_passages_jsonl(passages: dict[str, str], path: Path) -> None:
    """Write {doc_id: text} as JSONL rows sorted by numeric doc_id."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for doc_id in sorted(passages, key=int):
            f.write(json.dumps({"doc_id": doc_id, "text": passages[doc_id]}) + "\n")


def read_passages_jsonl(path: Path) -> dict[str, str]:
    """Read a passages JSONL file into {doc_id: text}."""
    with path.open() as f:
        return {r["doc_id"]: r["text"] for r in map(json.loads, f) if r}


def positives_summary(query_ids: list[str], qrels: Qrels) -> dict[str, float]:
    """Mean/min/max number of positive passages per sampled query."""
    counts = [len(qrels.get(qid, {})) for qid in query_ids]
    return {"mean": statistics.mean(counts), "min": min(counts), "max": max(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--limit", type=int, default=None, help="sample only this many queries")
    args = parser.parse_args()

    config = load_config(args.config)
    n = args.limit or config["data"]["n_human_queries"]
    seed = config["project"]["seed"]

    queries, qrels = load_msmarco(config["data"]["dataset"])
    # Only queries that have at least one positive can be evaluated.
    judged_ids = [qid for qid in queries if qid in qrels]
    print(f"Loaded {len(queries):,} queries; {len(judged_ids):,} have >=1 positive")

    sampled = sample_query_ids(judged_ids, n, seed)
    write_queries_jsonl(sampled, queries, QUERIES_HUMAN_PATH)
    write_qrels_trec(sampled, qrels, QRELS_OFFICIAL_PATH)

    stats = positives_summary(sampled, qrels)
    print(f"Sampled {len(sampled)} queries (seed={seed})")
    print(f"Positives per query: mean={stats['mean']:.3f} min={stats['min']} max={stats['max']}")
    print("Examples:")
    for qid in sampled[:5]:
        print(f"  [{qid}] {queries[qid]}")
    print(f"Wrote {QUERIES_HUMAN_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {QRELS_OFFICIAL_PATH.relative_to(PROJECT_ROOT)}")

    # Text of the official positives (used by the leak judge and the subcorpus).
    start = time.perf_counter()
    positive_ids = {doc_id for qid in sampled for doc_id in qrels[qid]}
    passages = fetch_passages(positive_ids, iter_collection())
    write_passages_jsonl(passages, POSITIVE_PASSAGES_PATH)
    print(
        f"Wrote {POSITIVE_PASSAGES_PATH.relative_to(PROJECT_ROOT)} ({len(passages)} passages, "
        f"streamed from the archive in {time.perf_counter() - start:.0f}s)"
    )


if __name__ == "__main__":
    main()
