"""Evaluate one dense retriever on the 200k subcorpus, compared with BM25 query by query.

CLI:
  uv run python -m arl.eval.dense_subcorpus --model BAAI/bge-small-en-v1.5 --limit 20
  uv run python -m arl.eval.dense_subcorpus --model BAAI/bge-small-en-v1.5 \
      --run-id 2026-11-23_dense_bge-small-en-v1.5

Passage embeddings are encoded once per model and cached under data/processed/embeddings/
(~7 min per small model on MPS). BM25 is re-run in memory (~5 s) for the paired
comparisons and checked against the saved BM25 results. With --run-id, writes
results/m2/<run_id>/: config.yaml, metrics.json (incl. dense - BM25 bootstraps),
per_query.csv, length_top10.json, run.log; full runs also save TREC run files to
data/processed/runs/ for fusion. Free and offline once the model is cached.
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
    BM25_RESULTS_DIR,
    BM25_RUN_PATH,
    METRICS,
    QRELS_NOTE,
    RUNS_DIR,
    RunLog,
    build_metrics,
    doc_lists,
    load_queries,
    paired_vs,
    prepare_out_dir,
    read_corpus,
    read_per_query,
    run_bm25,
    save_run,
    score_runs,
    write_outputs,
)
from arl.eval.length_check import word_count
from arl.eval.metrics import paired_bootstrap
from arl.retrievers import dense


def top10_lengths(runs: dict[str, list[str]], lengths: dict[str, int]) -> dict[str, float]:
    """Mean word count of each query's top-10 passages."""
    return {q: float(np.mean([lengths[d] for d in ranked[:10]])) for q, ranked in runs.items()}


def length_top10(
    human_ids: list[str],
    bm25_runs: dict[str, list[str]],
    dense_runs: dict[str, list[str]],
    lengths: dict[str, int],
    positives: set[str],
    model_name: str,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Mean top-10 passage length, BM25 vs this dense model, on human queries."""
    bm25_len = top10_lengths({q: bm25_runs[q] for q in human_ids}, lengths)
    dense_len = top10_lengths({q: dense_runs[q] for q in human_ids}, lengths)
    return {
        "description": "Mean word count of the top-10 passages per human query, then the mean over queries.",
        "n_queries": len(human_ids),
        "mean_words_top10": {
            "bm25": round(float(np.mean(list(bm25_len.values()))), 2),
            model_name: round(float(np.mean(list(dense_len.values()))), 2),
        },
        "dense_minus_bm25": paired_bootstrap(
            [dense_len[q] for q in human_ids], [bm25_len[q] for q in human_ids], n_resamples, seed
        ),
        "reference_mean_words": {
            "whole_subcorpus": round(float(np.mean(list(lengths.values()))), 2),
            "official_positives": round(float(np.mean([lengths[d] for d in positives])), 2),
        },
    }


def check_bm25_matches_saved(rows: list[dict[str, Any]], log: RunLog) -> None:
    """Compare the in-memory BM25 rows with the saved BM25 results (full runs only)."""
    path = BM25_RESULTS_DIR / "per_query.csv"
    if not path.exists():
        log(f"no saved BM25 results at {path}; skipping consistency check")
        return
    saved = {r["query_id"]: r for r in read_per_query(path)}
    same = sum(
        all(abs(r[m] - saved[r["query_id"]][m]) < 1e-9 for m in METRICS)
        and r["first_pos_rank"] == saved[r["query_id"]]["first_pos_rank"]
        for r in rows
    )
    log(f"BM25 re-run matches saved {path.relative_to(PROJECT_ROOT)}: {same}/{len(rows)} queries")
    if same != len(rows):
        raise RuntimeError("BM25 re-run differs from the saved BM25 results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=sorted(dense.MODEL_PREFIXES))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--limit", type=int, default=None, help="first N human queries")
    parser.add_argument("--run-id", default=None, help="write results/m2/<run_id>/")
    args = parser.parse_args()
    out_dir = prepare_out_dir(args.run_id)

    t0 = time.perf_counter()
    log = RunLog()
    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    top_k = int(config["retrievers"]["top_k"])
    bm25_cfg, dense_cfg = config["retrievers"]["bm25"], config["retrievers"]["dense"]
    n_resamples = int(config["eval"]["bootstrap_resamples"])
    q_prefix, p_prefix = dense.prefixes(args.model)

    doc_ids, texts, sources = read_corpus(CORPUS_PATH)
    lengths = {d: word_count(t) for d, t in zip(doc_ids, texts, strict=True)}
    positives = {d for d, s in zip(doc_ids, sources, strict=True) if s == "positive"}
    variants, qrels = load_queries(args.limit)
    query_texts = [v["text"] for v in variants]
    qids = [v["query_id"] for v in variants]
    log(
        f"{args.model} on subcorpus: {len(doc_ids):,} passages, {len(variants)} queries, "
        f"top_k={top_k}, limit={args.limit}"
    )
    t_load = time.perf_counter()

    bm25_run = dict(
        zip(
            qids,
            run_bm25(doc_ids, texts, query_texts, bm25_cfg["k1"], bm25_cfg["b"], top_k),
            strict=True,
        )
    )
    bm25_runs = doc_lists(bm25_run)
    bm25_rows = score_runs(variants, bm25_runs, qrels)
    t_bm25 = time.perf_counter()
    log(f"BM25 re-run: {t_bm25 - t_load:.1f} s")
    if args.limit is None:
        check_bm25_matches_saved(bm25_rows, log)
        save_run(bm25_run, BM25_RUN_PATH, "bm25_subcorpus")

    device = dense.pick_device(dense_cfg["device"])
    model = dense.load_model(args.model, device, int(dense_cfg["max_seq_length"]))
    t_model = time.perf_counter()
    log(f"device={device}; model load {t_model - t_bm25:.1f} s")
    corpus_emb = dense.corpus_embeddings(
        model,
        args.model,
        doc_ids,
        texts,
        int(dense_cfg["max_seq_length"]),
        int(dense_cfg["batch_size"]),
        log=log,
    )
    t_corpus = time.perf_counter()
    query_emb = dense.encode(model, query_texts, q_prefix, int(dense_cfg["batch_size"]))
    t_query = time.perf_counter()
    positions, scores = dense.search(corpus_emb, query_emb, top_k)
    t_search = time.perf_counter()
    log(
        f"passage embeddings {t_corpus - t_model:.1f} s; query encoding {t_query - t_corpus:.1f} s; "
        f"faiss search {t_search - t_query:.1f} s"
    )

    dense_run = {
        q: [(doc_ids[p], float(s)) for p, s in zip(row_p, row_s, strict=True) if p >= 0]
        for q, row_p, row_s in zip(qids, positions, scores, strict=True)
    }
    dense_runs = doc_lists(dense_run)
    if args.limit is None:
        save_run(
            dense_run,
            RUNS_DIR / f"{dense.model_slug(args.model)}.tsv",
            dense.model_slug(args.model),
        )
    rows = score_runs(variants, dense_runs, qrels)
    settings = {
        "model": args.model,
        "device": device,
        "query_prefix": q_prefix,
        "passage_prefix": p_prefix,
        "normalized_embeddings": True,
        "index": "faiss.IndexFlatIP (exact)",
        "max_seq_length": int(dense_cfg["max_seq_length"]),
        "batch_size": int(dense_cfg["batch_size"]),
        "embedding_dim": int(corpus_emb.shape[1]),
        "top_k": top_k,
        "limit": args.limit,
        "corpus": str(CORPUS_PATH.relative_to(PROJECT_ROOT)),
        "n_docs": len(doc_ids),
        "qrels": QRELS_NOTE,
    }
    metrics = build_metrics(rows, settings, top_k, n_resamples, seed)
    metrics["dense_minus_bm25"] = paired_vs(rows, bm25_rows, n_resamples, seed)
    log(json.dumps(metrics["by_query_type"], indent=2))
    for qtype, comp in metrics["dense_minus_bm25"].items():
        log(
            f"dense - BM25 [{qtype}]: "
            + "; ".join(
                f"{m} {r['mean_diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                for m, r in comp.items()
            )
        )

    human_ids = [v["query_id"] for v in variants if v["query_type"] == "human"]
    lt = length_top10(
        human_ids, bm25_runs, dense_runs, lengths, positives, args.model, n_resamples, seed
    )
    log(json.dumps(lt, indent=2))
    log(f"wall-clock total: {time.perf_counter() - t0:.1f} s")
    write_outputs(
        out_dir, args.config, rows, {"metrics.json": metrics, "length_top10.json": lt}, log
    )


if __name__ == "__main__":
    main()
