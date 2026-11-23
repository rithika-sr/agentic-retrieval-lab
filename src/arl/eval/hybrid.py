"""Evaluate RRF fusion of BM25 + one dense run on the 200k subcorpus.

CLI:
  uv run python -m arl.eval.hybrid --dense bge-small-en-v1.5 --limit 20
  uv run python -m arl.eval.hybrid --dense bge-small-en-v1.5 --run-id 2026-11-23_hybrid_bm25_bge

Reads the saved TREC runs from data/processed/runs/ (written by the full BM25-in-dense and
dense evals), fuses each query's top-k lists with RRF (k = fusion.rrf_k), and writes
results/m2/<run_id>/: config.yaml, metrics.json (incl. hybrid - component bootstraps),
per_query.csv, run.log. Seconds; free and offline.
"""

from __future__ import annotations

import argparse
import json
import time

from arl.config import DEFAULT_CONFIG_PATH, load_config
from arl.eval.common import (
    BM25_RUN_PATH,
    QRELS_NOTE,
    RUNS_DIR,
    RunLog,
    build_metrics,
    doc_lists,
    load_queries,
    load_run,
    paired_vs,
    prepare_out_dir,
    score_runs,
    write_outputs,
)
from arl.fusion import rrf_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dense", required=True, help="dense run name in data/processed/runs/")
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
    rrf_k = int(config["fusion"]["rrf_k"])
    n_resamples = int(config["eval"]["bootstrap_resamples"])
    dense_path = RUNS_DIR / f"{args.dense}.tsv"

    variants, qrels = load_queries(args.limit)
    qids = [v["query_id"] for v in variants]
    components = {
        "bm25": doc_lists(load_run(BM25_RUN_PATH)),
        args.dense: doc_lists(load_run(dense_path)),
    }
    components = {name: {q: run[q][:top_k] for q in qids} for name, run in components.items()}
    log(f"RRF k={rrf_k} over bm25 + {args.dense}, top {top_k} each; {len(qids)} queries")

    fused = doc_lists(rrf_runs(list(components.values()), k=rrf_k, top_n=top_k))
    rows = score_runs(variants, fused, qrels)
    comp_rows = {name: score_runs(variants, run, qrels) for name, run in components.items()}
    settings = {
        "fusion": "reciprocal rank fusion",
        "rrf_k": rrf_k,
        "components": {"bm25": str(BM25_RUN_PATH.name), args.dense: dense_path.name},
        "depth_per_component": top_k,
        "top_k": top_k,
        "limit": args.limit,
        "qrels": QRELS_NOTE,
    }
    metrics = build_metrics(rows, settings, top_k, n_resamples, seed)
    metrics["hybrid_minus_component"] = {
        name: paired_vs(rows, crows, n_resamples, seed) for name, crows in comp_rows.items()
    }
    log(json.dumps(metrics["by_query_type"], indent=2))
    for name, by_type in metrics["hybrid_minus_component"].items():
        for qtype, comp in by_type.items():
            log(
                f"hybrid - {name} [{qtype}]: "
                + "; ".join(
                    f"{m} {r['mean_diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                    for m, r in comp.items()
                )
            )
    log(f"wall-clock total: {time.perf_counter() - t0:.1f} s")
    write_outputs(out_dir, args.config, rows, {"metrics.json": metrics}, log)


if __name__ == "__main__":
    main()
