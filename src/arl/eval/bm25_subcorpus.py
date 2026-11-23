"""Evaluate BM25 on the 200k subcorpus for every query variant, plus the length-bias check.

CLI:
  uv run python -m arl.eval.bm25_subcorpus --limit 20                        # print only
  uv run python -m arl.eval.bm25_subcorpus --run-id 2026-11-23_bm25_subcorpus

With --run-id, writes results/m2/<run_id>/: config.yaml, metrics.json, per_query.csv,
length_check.json, run.log. Free and offline: no API calls, no downloads.

Agent queries are scored against their parent's official qrels. The subcorpus's MS MARCO
doc ids are not 0..N-1, so the index is built on positions and mapped back to doc ids.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from arl.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from arl.corpus import CORPUS_PATH
from arl.eval.common import (
    METRICS,
    QRELS_NOTE,
    RunLog,
    build_metrics,
    doc_lists,
    load_queries,
    prepare_out_dir,
    read_corpus,
    run_bm25,
    score_runs,
    write_outputs,
)
from arl.eval.length_check import eligible_queries, mean_mrr_over_seeds, word_count
from arl.eval.metrics import mrr_at_k, paired_bootstrap


def length_check(
    runs: dict[str, list[str]],
    qrels: dict[str, set[str]],
    lengths: dict[str, int],
    depth: int,
    n_seeds: int,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Length-only vs random re-ranking of each eligible human query's BM25 top-`depth`."""
    cands = eligible_queries(runs, qrels, depth)
    qids = sorted(cands)
    seeds = list(range(n_seeds))
    length_mrr = [mean_mrr_over_seeds(cands[q], qrels[q], lengths, seeds, seed) for q in qids]
    random_mrr = [mean_mrr_over_seeds(cands[q], qrels[q], None, seeds, seed) for q in qids]
    bm25_mrr = [mrr_at_k(cands[q], qrels[q], 10) for q in qids]
    pos_len = [lengths[d] for q in qids for d in cands[q] if d in qrels[q]]
    neg_len = [lengths[d] for q in qids for d in cands[q] if d not in qrels[q]]
    return {
        "description": (
            f"Human queries with an official positive in BM25's top {depth} on the subcorpus. "
            "Candidates re-ranked by word count (shortest first, random tie-breaks) vs a "
            f"random order; both averaged over {n_seeds} seeds per query."
        ),
        "n_human_queries_total": len(runs),
        "n_queries_used": len(qids),
        "candidate_depth": depth,
        "n_seeds": n_seeds,
        "mrr@10": {
            "length_only_shorter_first": round(float(np.mean(length_mrr)), 4),
            "random": round(float(np.mean(random_mrr)), 4),
            "bm25_reference": round(float(np.mean(bm25_mrr)), 4),
        },
        "length_minus_random": paired_bootstrap(length_mrr, random_mrr, n_resamples, seed),
        "mean_words_in_candidate_sets": {
            "positives": round(float(np.mean(pos_len)), 2),
            "non_positives": round(float(np.mean(neg_len)), 2),
            "median_positives": float(np.median(pos_len)),
            "median_non_positives": float(np.median(neg_len)),
            "n_positives": len(pos_len),
            "n_non_positives": len(neg_len),
        },
    }


def ranx_cross_check(
    rows: list[dict[str, Any]], runs: dict[str, list[str]], qrels: dict[str, set[str]]
) -> dict[str, float]:
    """Mean metrics from ranx on the same run, for comparison with ours (scores = rank order)."""
    from ranx import Qrels as RQrels
    from ranx import Run as RRun
    from ranx import evaluate

    qids = [r["query_id"] for r in rows if runs[r["query_id"]]]
    rq = RQrels({q: {d: 1 for d in qrels[q]} for q in qids})
    rr = RRun({q: {d: float(len(runs[q]) - i) for i, d in enumerate(runs[q])} for q in qids})
    return {m: float(v) for m, v in evaluate(rq, rr, list(METRICS)).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--limit", type=int, default=None, help="first N human queries")
    parser.add_argument("--run-id", default=None, help="write results/m2/<run_id>/")
    args = parser.parse_args()
    out_dir = prepare_out_dir(args.run_id)

    t_start = time.perf_counter()
    log = RunLog()
    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    bm25_cfg, top_k = config["retrievers"]["bm25"], int(config["retrievers"]["top_k"])
    n_resamples = int(config["eval"]["bootstrap_resamples"])
    lc_cfg = config["eval"]["length_check"]
    log(
        f"BM25 on subcorpus: k1={bm25_cfg['k1']} b={bm25_cfg['b']} top_k={top_k} limit={args.limit}"
    )

    doc_ids, texts, _sources = read_corpus(CORPUS_PATH)
    lengths = {d: word_count(t) for d, t in zip(doc_ids, texts, strict=True)}
    variants, qrels = load_queries(args.limit)
    missing = {d for rel in qrels.values() for d in rel} - set(doc_ids)
    log(
        f"corpus {len(doc_ids):,} passages; {len(variants)} query variants; "
        f"positives missing from corpus: {len(missing)}"
    )
    t_load = time.perf_counter()

    ranked = run_bm25(
        doc_ids, texts, [v["text"] for v in variants], bm25_cfg["k1"], bm25_cfg["b"], top_k
    )
    runs = doc_lists({v["query_id"]: r for v, r in zip(variants, ranked, strict=True)})
    t_bm25 = time.perf_counter()
    short = sum(len(r) < 100 for r in runs.values())
    log(f"index + search: {t_bm25 - t_load:.1f} s; queries with <100 matching passages: {short}")

    rows = score_runs(variants, runs, qrels)
    settings = {
        "k1": bm25_cfg["k1"],
        "b": bm25_cfg["b"],
        "top_k": top_k,
        "limit": args.limit,
        "corpus": str(CORPUS_PATH.relative_to(PROJECT_ROOT)),
        "n_docs": len(doc_ids),
        "qrels": QRELS_NOTE,
    }
    metrics = build_metrics(rows, settings, top_k, n_resamples, seed)
    metrics["queries_with_fewer_than_100_matches"] = short
    ours_human = metrics["by_query_type"]["human"]
    ranx_human = ranx_cross_check([r for r in rows if r["type"] == "human"], runs, qrels)
    log(
        "ranx cross-check (human): "
        + ", ".join(f"{m} ours={ours_human[m]:.4f} ranx={ranx_human[m]:.4f}" for m in METRICS)
    )
    metrics["ranx_cross_check_human"] = {m: round(v, 4) for m, v in ranx_human.items()}
    log(json.dumps(metrics["by_query_type"], indent=2))

    human_runs = {r["query_id"]: runs[r["query_id"]] for r in rows if r["type"] == "human"}
    t_lc0 = time.perf_counter()
    lc = length_check(
        human_runs, qrels, lengths, int(lc_cfg["depth"]), int(lc_cfg["n_seeds"]), n_resamples, seed
    )
    log(f"length check ({time.perf_counter() - t_lc0:.1f} s):")
    log(json.dumps(lc, indent=2))
    log(
        f"wall-clock total: {time.perf_counter() - t_start:.1f} s "
        f"(load {t_load - t_start:.1f} s, BM25 index+search {t_bm25 - t_load:.1f} s)"
    )
    write_outputs(
        out_dir, args.config, rows, {"metrics.json": metrics, "length_check.json": lc}, log
    )


if __name__ == "__main__":
    main()
