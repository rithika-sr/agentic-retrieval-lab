"""Build the pooled subcorpus: official positives + BM25 pools + seeded random backfill.

CLI:
  uv run python -m arl.corpus --limit 20 --ids-only        # stats only, no passage text
  uv run python -m arl.corpus --run-id 2026-10-09_corpus   # full build (~3 min stream)

Writes `data/processed/corpus.jsonl` (doc_id, text, source) and `corpus_stats.json`, and
copies the stats and config to results/foundation/<run_id>/. `source` is the first that
applies of: positive (official qrels), pool (in any query variant's BM25 top-k over the
full collection), backfill (random from the rest of the collection).
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from arl.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from arl.data import (
    PROCESSED_DIR,
    QRELS_OFFICIAL_PATH,
    fetch_passages,
    iter_collection,
    read_qrels_trec,
)
from arl.pool import BM25_FULL_DIR, POOL_RUN_PATH, QUERY_TYPES, query_variants, read_trec_run
from arl.retrievers.bm25 import META_NAME

CORPUS_PATH = PROCESSED_DIR / "corpus.jsonl"
CORPUS_STATS_PATH = PROCESSED_DIR / "corpus_stats.json"
SOURCES = ("positive", "pool", "backfill")


def pools_by_type(
    run: dict[str, list[tuple[str, float]]], variants: list[dict[str, str]]
) -> dict[str, set[str]]:
    """{query_type: set of doc ids in that type's pools}."""
    pools: dict[str, set[str]] = {t: set() for t in QUERY_TYPES}
    for v in variants:
        pools[v["query_type"]].update(doc_id for doc_id, _ in run[v["query_id"]])
    return pools


def select_corpus_ids(
    positives: set[str], pooled: set[str], n_collection: int, target_size: int, seed: int
) -> dict[str, str]:
    """{doc_id: source} with exactly `target_size` ids (collection ids are 0..n-1).

    Raises if positives + pools alone exceed the target (no silent truncation).
    """
    chosen = {d: "positive" for d in positives}
    chosen.update({d: "pool" for d in pooled if d not in chosen})
    n_backfill = target_size - len(chosen)
    if n_backfill < 0:
        raise ValueError(f"positives + pools = {len(chosen):,} > target_size {target_size:,}")
    taken = np.fromiter((int(d) for d in chosen), dtype=np.int64, count=len(chosen))
    candidates = np.setdiff1d(np.arange(n_collection, dtype=np.int64), taken)
    backfill = np.random.default_rng(seed).choice(candidates, size=n_backfill, replace=False)
    chosen.update({str(int(d)): "backfill" for d in np.sort(backfill)})
    return chosen


def overlap_stats(pools: dict[str, set[str]]) -> dict[str, Any]:
    """Pool sizes per query type, pairwise intersections, and passages unique to one type."""
    types = list(pools)
    stats: dict[str, Any] = {"size": {t: len(pools[t]) for t in types}}
    stats["intersection"] = {
        f"{a}&{b}": len(pools[a] & pools[b]) for i, a in enumerate(types) for b in types[i + 1 :]
    }
    stats["all_types"] = len(set.intersection(*pools.values()))
    stats["only_this_type"] = {
        t: len(pools[t] - set().union(*(pools[o] for o in types if o != t))) for t in types
    }
    return stats


def length_stats(texts: Iterable[str]) -> dict[str, float]:
    """Mean/median/max passage length in whitespace words."""
    lengths = [len(t.split()) for t in texts]
    return {
        "mean_words": round(statistics.mean(lengths), 2),
        "median_words": statistics.median(lengths),
        "max_words": max(lengths),
    }


def write_corpus(sources: dict[str, str], texts: dict[str, str], path: Path) -> None:
    """Write corpus rows sorted by numeric doc id."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for doc_id in sorted(sources, key=int):
            row = {"doc_id": doc_id, "text": texts[doc_id], "source": sources[doc_id]}
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--limit", type=int, default=None, help="first N human queries only")
    parser.add_argument("--ids-only", action="store_true", help="select ids + stats, no text")
    parser.add_argument("--run-id", default=None, help="copy stats to results/foundation/")
    args = parser.parse_args()

    config = load_config(args.config)
    target, seed = config["corpus"]["target_size"], config["project"]["seed"]
    results_dir = PROJECT_ROOT / "results" / "foundation" / args.run_id if args.run_id else None
    if results_dir and results_dir.exists():
        raise FileExistsError(f"{results_dir} exists; results are never overwritten")

    variants = query_variants(args.limit)
    parents = {v["parent_id"] for v in variants}
    qrels = read_qrels_trec(QRELS_OFFICIAL_PATH)
    positives = {d for qid in parents for d in qrels[qid]}
    run = read_trec_run(POOL_RUN_PATH)
    pools = pools_by_type(run, variants)
    pooled = set().union(*pools.values())
    n_collection = json.loads((BM25_FULL_DIR / META_NAME).read_text())["n_docs"]
    sources = select_corpus_ids(positives, pooled, n_collection, target, seed)

    counts = {s: sum(v == s for v in sources.values()) for s in SOURCES}
    stats: dict[str, Any] = {
        "n_human_queries": len(parents),
        "n_query_variants": len(variants),
        "pool_slots": sum(len(run[v["query_id"]]) for v in variants),
        "positives": len(positives),
        "positives_in_pools": len(positives & pooled),
        "unique_pooled_passages": len(pooled),
        "pooled_non_positive": counts["pool"],
        "backfill": counts["backfill"],
        "total": len(sources),
        "target_size": target,
        "seed": seed,
        "n_collection": n_collection,
        "pool_overlap_by_query_type": overlap_stats(pools),
    }
    if not args.ids_only:
        texts = fetch_passages(sources, iter_collection())
        stats["passage_length"] = {"all": length_stats(texts.values())} | {
            s: length_stats(texts[d] for d, src in sources.items() if src == s) for s in SOURCES
        }
        write_corpus(sources, texts, CORPUS_PATH)
        CORPUS_STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"Wrote {CORPUS_PATH.relative_to(PROJECT_ROOT)} and corpus_stats.json")
        if results_dir:
            results_dir.mkdir(parents=True)
            shutil.copy(CORPUS_STATS_PATH, results_dir / "corpus_stats.json")
            shutil.copy(args.config, results_dir / "config.yaml")
            print(f"Wrote {results_dir.relative_to(PROJECT_ROOT)}/")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
