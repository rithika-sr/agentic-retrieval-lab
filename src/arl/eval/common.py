"""Shared evaluation plumbing: loading, BM25 on the subcorpus, scoring, summaries, output.

Used by `bm25_subcorpus`, `dense_subcorpus`, and `hybrid` so every retriever is scored by
the same code. Agent queries are scored against their parent's official qrels.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from arl.config import PROJECT_ROOT
from arl.data import PROCESSED_DIR, QRELS_OFFICIAL_PATH, read_qrels_trec
from arl.eval.metrics import (
    first_positive_rank,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    recall_at_k,
)
from arl.pool import QUERY_TYPES, Run, query_variants, read_trec_run, write_trec_run

METRICS = ("recall@100", "mrr@10", "ndcg@10")
COMPARE_METRICS = ("recall@100", "mrr@10")  # retriever-vs-retriever bootstraps
PER_QUERY_COLUMNS = ["query_id", "type", "parent_id", *METRICS, "first_pos_rank"]
RUNS_DIR = PROCESSED_DIR / "runs"  # gitignored TREC run files, for fusion and re-analysis
BM25_RUN_PATH = RUNS_DIR / "bm25_subcorpus.tsv"
BM25_RESULTS_DIR = PROJECT_ROOT / "results" / "m2" / "2026-11-23_bm25_subcorpus"
QRELS_NOTE = "official MS MARCO dev/small; agent queries use their parent's"


class RunLog:
    """Print lines and keep them for run.log."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        self.lines.append(msg)


def read_corpus(path: Path) -> tuple[list[str], list[str], list[str]]:
    """doc ids, texts, and sources in file order."""
    ids, texts, sources = [], [], []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["doc_id"])
            texts.append(row["text"])
            sources.append(row["source"])
    return ids, texts, sources


def load_queries(limit: int | None) -> tuple[list[dict[str, str]], dict[str, set[str]]]:
    """Query variants (first `limit` human queries + their rewrites) and per-variant qrels."""
    variants = query_variants(limit)
    parent_qrels = read_qrels_trec(QRELS_OFFICIAL_PATH)
    qrels = {v["query_id"]: set(parent_qrels[v["parent_id"]]) for v in variants}
    return variants, qrels


def run_bm25(
    doc_ids: list[str], texts: list[str], queries: list[str], k1: float, b: float, top_k: int
) -> list[list[tuple[str, float]]]:
    """Index the texts by position, retrieve top_k per query, drop zero-score hits."""
    from arl.retrievers.bm25 import build_index, search

    retriever, tokenizer, _ = build_index(((str(i), t) for i, t in enumerate(texts)), k1=k1, b=b)
    positions, scores = search(retriever, tokenizer, queries, k=top_k)
    return [
        [(doc_ids[p], float(s)) for p, s in zip(row_p, row_s, strict=True) if s > 0]
        for row_p, row_s in zip(positions, scores, strict=True)
    ]


def doc_lists(run: Run) -> dict[str, list[str]]:
    """Drop scores: {query_id: [doc_id, ...]} best first."""
    return {q: [d for d, _ in ranked] for q, ranked in run.items()}


def score_query(ranked: list[str], relevant: set[str]) -> dict[str, Any]:
    """All metrics plus the rank of the first positive (None if not retrieved)."""
    return {
        "recall@100": recall_at_k(ranked, relevant, 100),
        "mrr@10": mrr_at_k(ranked, relevant, 10),
        "ndcg@10": ndcg_at_k(ranked, relevant, 10),
        "first_pos_rank": first_positive_rank(ranked, relevant),
    }


def score_runs(
    variants: list[dict[str, str]], runs: dict[str, list[str]], qrels: dict[str, set[str]]
) -> list[dict[str, Any]]:
    """One row per query variant: ids, type, metrics, first-positive rank."""
    return [
        {"query_id": v["query_id"], "type": v["query_type"], "parent_id": v["parent_id"]}
        | score_query(runs[v["query_id"]], qrels[v["query_id"]])
        for v in variants
    ]


def summarize_by_type(rows: list[dict[str, Any]], top_k: int) -> dict[str, dict[str, Any]]:
    """Mean of each metric per query type."""
    out = {}
    for qtype in QUERY_TYPES:
        sub = [r for r in rows if r["type"] == qtype]
        if not sub:
            continue
        out[qtype] = {"n": len(sub)} | {
            m: round(float(np.mean([r[m] for r in sub])), 4) for m in METRICS
        }
        found = sum(r["first_pos_rank"] is not None for r in sub)
        out[qtype][f"positive_in_top{top_k}"] = round(found / len(sub), 4)
    return out


def paired_type_differences(
    rows: list[dict[str, Any]], n_resamples: int, seed: int
) -> dict[str, Any]:
    """human - primary and human - mean(supports), paired by parent query."""
    by_parent: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in rows:
        by_parent.setdefault(r["parent_id"], {}).setdefault(r["type"], []).append(r)
    parents = sorted(p for p, t in by_parent.items() if set(QUERY_TYPES) <= t.keys())
    out: dict[str, Any] = {}
    for other in ("primary", "support"):
        out[f"human_minus_{other}"] = {
            m: paired_bootstrap(
                [by_parent[p]["human"][0][m] for p in parents],
                [float(np.mean([r[m] for r in by_parent[p][other]])) for p in parents],
                n_resamples=n_resamples,
                seed=seed,
            )
            for m in METRICS
        }
    out["note"] = "paired by parent; support = mean over that parent's 3 support queries"
    return out


def paired_vs(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], n_resamples: int, seed: int
) -> dict[str, dict[str, Any]]:
    """Per query type, bootstrap of A - B for COMPARE_METRICS, paired by query_id."""
    b_by_id = {r["query_id"]: r for r in rows_b}
    out: dict[str, dict[str, Any]] = {}
    for qtype in QUERY_TYPES:
        sub = [r for r in rows_a if r["type"] == qtype and r["query_id"] in b_by_id]
        if not sub:
            continue
        out[qtype] = {
            m: paired_bootstrap(
                [r[m] for r in sub],
                [b_by_id[r["query_id"]][m] for r in sub],
                n_resamples=n_resamples,
                seed=seed,
            )
            for m in COMPARE_METRICS
        }
    return out


def build_metrics(
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
    top_k: int,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """The common metrics.json body: settings, per-type means, human vs agent bootstraps."""
    return {
        "settings": settings,
        "by_query_type": summarize_by_type(rows, top_k),
        "paired_differences": paired_type_differences(rows, n_resamples, seed),
    }


def write_per_query(rows: list[dict[str, Any]], path: Path) -> None:
    """One row per query variant; first_pos_rank is blank when not retrieved."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_QUERY_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: ("" if r[c] is None else r[c]) for c in PER_QUERY_COLUMNS})


def read_per_query(path: Path) -> list[dict[str, Any]]:
    """Read a per_query.csv back (metrics as floats, rank as int or None)."""
    with path.open() as f:
        return [
            {"query_id": r["query_id"], "type": r["type"], "parent_id": r["parent_id"]}
            | {m: float(r[m]) for m in METRICS}
            | {"first_pos_rank": int(r["first_pos_rank"]) if r["first_pos_rank"] else None}
            for r in csv.DictReader(f)
        ]


def save_run(run: Run, path: Path, name: str) -> None:
    """Write a TREC run file (gitignored, under data/)."""
    write_trec_run(run, path, run_name=name)


def load_run(path: Path) -> Run:
    """Read a TREC run file written by `save_run`."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run the retriever's eval first")
    return read_trec_run(path)


def prepare_out_dir(run_id: str | None) -> Path | None:
    """results/m2/<run_id>, refusing to overwrite; None means print-only."""
    if run_id is None:
        return None
    out_dir = PROJECT_ROOT / "results" / "m2" / run_id
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; results are never overwritten")
    return out_dir


def write_outputs(
    out_dir: Path | None,
    config_path: str | Path,
    rows: list[dict[str, Any]],
    json_files: dict[str, Any],
    log: RunLog,
) -> None:
    """config.yaml, per_query.csv, each JSON file, and run.log into out_dir."""
    if out_dir is None:
        print("(no --run-id: nothing written)")
        return
    out_dir.mkdir(parents=True)
    shutil.copy(config_path, out_dir / "config.yaml")
    write_per_query(rows, out_dir / "per_query.csv")
    for name, obj in json_files.items():
        (out_dir / name).write_text(json.dumps(obj, indent=2) + "\n")
    (out_dir / "run.log").write_text("\n".join(log.lines) + "\n")
    shown = out_dir.relative_to(PROJECT_ROOT) if out_dir.is_relative_to(PROJECT_ROOT) else out_dir
    print(f"Wrote {shown}/")
