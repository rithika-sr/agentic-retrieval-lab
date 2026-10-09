"""Retrieve BM25 top-k from the FULL collection for every query variant (distractor pools).

CLI:
  uv run python -m arl.pool --limit 20                         # first 20 human queries
  uv run python -m arl.pool --run-id 2026-10-08_bm25           # all; save stats to results/

Writes `data/processed/bm25_full_top100.tsv` in TREC run format (tab-separated):
`query_id  Q0  doc_id  rank  score  run_name`, for human, primary, and support queries.
Also reports how often the official positive is in the pool, per query type. That is a
sanity check on the pools, not an evaluation (the full-collection runs are not compared
with anything yet).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from arl.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from arl.data import (
    PROCESSED_DIR,
    QRELS_OFFICIAL_PATH,
    QUERIES_HUMAN_PATH,
    Qrels,
    read_qrels_trec,
    read_queries_jsonl,
)
from arl.leak import read_jsonl
from arl.retrievers.bm25 import load_index, search
from arl.rewrite import QUERIES_AGENT_PATH

BM25_FULL_DIR = PROCESSED_DIR / "bm25_full"
POOL_RUN_PATH = PROCESSED_DIR / "bm25_full_top100.tsv"
RUN_NAME = "bm25_full"
QUERY_TYPES = ("human", "primary", "support")

Run = dict[str, list[tuple[str, float]]]  # query_id -> [(doc_id, score)] best first


def query_variants(n_human: int | None = None) -> list[dict[str, str]]:
    """Human queries plus their agent rewrites: query_id, parent_id, query_type, text."""
    human = read_queries_jsonl(QUERIES_HUMAN_PATH)[:n_human]
    parents = {q["query_id"] for q in human}
    rows = [
        {"query_id": q["query_id"], "parent_id": q["query_id"], "query_type": "human"}
        | {"text": q["text"]}
        for q in human
    ]
    rows += [
        {k: r[k] for k in ("query_id", "parent_id", "query_type", "text")}
        for r in read_jsonl(QUERIES_AGENT_PATH)
        if r["parent_id"] in parents
    ]
    return rows


def to_run(query_ids: list[str], docs: np.ndarray, scores: np.ndarray) -> Run:
    """Search output (positions == MS MARCO doc ids) -> {query_id: [(doc_id, score)]}."""
    return {
        qid: [(str(int(d)), float(s)) for d, s in zip(doc_row, score_row, strict=True)]
        for qid, doc_row, score_row in zip(query_ids, docs, scores, strict=True)
    }


def write_trec_run(run: Run, path: Path, run_name: str = RUN_NAME) -> None:
    """Write a TREC run file: query_id Q0 doc_id rank score run_name (tab-separated)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for qid, ranked in run.items():
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{run_name}\n")


def read_trec_run(path: Path) -> Run:
    """Read a TREC run file back into {query_id: [(doc_id, score)]} in rank order."""
    run: Run = {}
    with path.open() as f:
        for line in f:
            qid, _, doc_id, _, score, _ = line.split()
            run.setdefault(qid, []).append((doc_id, float(score)))
    return run


def pool_hit_rates(
    run: Run, variants: list[dict[str, str]], qrels: Qrels
) -> dict[str, dict[str, Any]]:
    """Share of queries whose pool contains an official positive of their parent.

    Per query type, plus per parent: any agent query (primary or support) hits, and any
    of all five variants hits.
    """
    hit = {
        v["query_id"]: bool(qrels[v["parent_id"]].keys() & {d for d, _ in run[v["query_id"]]})
        for v in variants
    }
    report: dict[str, dict[str, Any]] = {}
    for qtype in QUERY_TYPES:
        ids = [v["query_id"] for v in variants if v["query_type"] == qtype]
        report[qtype] = {"n": len(ids), "hits": sum(hit[q] for q in ids)}
    by_parent: dict[str, list[dict[str, str]]] = {}
    for v in variants:
        by_parent.setdefault(v["parent_id"], []).append(v)
    agent_any = [
        any(hit[v["query_id"]] for v in vs if v["query_type"] != "human")
        for vs in by_parent.values()
    ]
    all_any = [any(hit[v["query_id"]] for v in vs) for vs in by_parent.values()]
    report["any_agent_query_per_parent"] = {"n": len(agent_any), "hits": sum(agent_any)}
    report["any_variant_per_parent"] = {"n": len(all_any), "hits": sum(all_any)}
    for bucket in report.values():
        bucket["rate"] = round(bucket["hits"] / bucket["n"], 4) if bucket["n"] else None
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--limit", type=int, default=None, help="first N human queries only")
    parser.add_argument("--index", default=str(BM25_FULL_DIR))
    parser.add_argument("--out", default=str(POOL_RUN_PATH))
    parser.add_argument("--run-id", default=None, help="save pool stats to results/foundation/")
    parser.add_argument("--logs", nargs="*", default=[], help="run logs to archive")
    args = parser.parse_args()

    config = load_config(args.config)
    k = config["corpus"]["bm25_pool_depth"]
    results_dir = PROJECT_ROOT / "results" / "foundation" / args.run_id if args.run_id else None
    stats_path = results_dir / "pool_stats.json" if results_dir else None
    if stats_path and stats_path.exists():
        raise FileExistsError(f"{stats_path} exists; results are never overwritten")

    variants = query_variants(args.limit)
    t0 = time.perf_counter()
    retriever, tokenizer = load_index(Path(args.index))
    t1 = time.perf_counter()
    docs, scores = search(retriever, tokenizer, [v["text"] for v in variants], k=k, n_threads=-1)
    t2 = time.perf_counter()
    run = to_run([v["query_id"] for v in variants], docs, scores)
    write_trec_run(run, Path(args.out))

    report = {
        "index": str(Path(args.index).relative_to(PROJECT_ROOT)),
        "n_docs_in_index": int(retriever.scores["num_docs"]),
        "pool_depth": k,
        "n_variants": len(variants),
        "load_s": round(t1 - t0, 1),
        "search_s": round(t2 - t1, 1),
        "zero_score_rows": int((scores == 0).sum()),  # padded slots: query matched < k docs
        "official_positive_in_pool": pool_hit_rates(
            run, variants, read_qrels_trec(QRELS_OFFICIAL_PATH)
        ),
    }
    print(json.dumps(report, indent=2))
    print(f"Wrote {Path(args.out).relative_to(PROJECT_ROOT)} ({len(run) * k:,} rows)")
    if stats_path:
        results_dir.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(report, indent=2) + "\n")
        shutil.copy(args.config, results_dir / "config.yaml")
        for log in args.logs:
            text = Path(log).read_text().replace("\r", "\n")
            (results_dir / Path(log).name).write_text(text)
        print(f"Wrote {results_dir.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
