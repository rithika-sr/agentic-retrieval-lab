"""Structure of metrics.json, per_query.csv, and the retriever-vs-retriever bootstraps."""

from __future__ import annotations

import json

import pytest

from arl.eval.common import (
    COMPARE_METRICS,
    METRICS,
    PER_QUERY_COLUMNS,
    RunLog,
    build_metrics,
    paired_vs,
    read_per_query,
    score_runs,
    write_outputs,
)

VARIANTS = [
    {"query_id": "1", "parent_id": "1", "query_type": "human"},
    {"query_id": "1_p", "parent_id": "1", "query_type": "primary"},
    {"query_id": "1_s1", "parent_id": "1", "query_type": "support"},
    {"query_id": "2", "parent_id": "2", "query_type": "human"},
    {"query_id": "2_p", "parent_id": "2", "query_type": "primary"},
    {"query_id": "2_s1", "parent_id": "2", "query_type": "support"},
]
QRELS = {v["query_id"]: {"d" + v["parent_id"]} for v in VARIANTS}
RUN_A = {"1": ["d1"], "1_p": ["x", "d1"], "1_s1": ["x"], "2": ["d2"], "2_p": ["d2"], "2_s1": []}
RUN_B = {q: [] for q in RUN_A}  # retrieves nothing


def test_score_runs_rows() -> None:
    rows = score_runs(VARIANTS, RUN_A, QRELS)
    by_id = {r["query_id"]: r for r in rows}
    assert by_id["1_p"]["mrr@10"] == 0.5 and by_id["1_p"]["first_pos_rank"] == 2
    assert by_id["1_s1"]["first_pos_rank"] is None
    assert set(rows[0]) == set(PER_QUERY_COLUMNS)


def test_metrics_json_structure() -> None:
    rows = score_runs(VARIANTS, RUN_A, QRELS)
    m = build_metrics(rows, {"model": "toy"}, top_k=1000, n_resamples=200, seed=42)
    assert set(m) == {"settings", "by_query_type", "paired_differences"}
    assert set(m["by_query_type"]) == {"human", "primary", "support"}
    for stats in m["by_query_type"].values():
        assert set(stats) == {"n", *METRICS, "positive_in_top1000"}
    assert m["by_query_type"]["human"]["recall@100"] == 1.0
    assert m["by_query_type"]["primary"]["mrr@10"] == pytest.approx(0.75)  # (1/2 + 1)/2
    diffs = m["paired_differences"]
    assert set(diffs) == {"human_minus_primary", "human_minus_support", "note"}
    hp = diffs["human_minus_primary"]["mrr@10"]
    assert set(hp) == {"mean_diff", "ci_low", "ci_high", "n", "n_resamples", "seed"}
    assert hp["mean_diff"] == pytest.approx(0.25) and hp["n"] == 2


def test_paired_vs_structure_and_values() -> None:
    a = score_runs(VARIANTS, RUN_A, QRELS)
    b = score_runs(VARIANTS, RUN_B, QRELS)
    out = paired_vs(a, b, n_resamples=200, seed=42)
    assert set(out) == {"human", "primary", "support"}
    for comp in out.values():
        assert set(comp) == set(COMPARE_METRICS)
    assert out["human"]["recall@100"]["mean_diff"] == 1.0  # A finds both, B finds none
    assert out["support"]["mrr@10"]["mean_diff"] == 0.0


def test_write_outputs_files_round_trip(tmp_path) -> None:
    rows = score_runs(VARIANTS, RUN_A, QRELS)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("project: {seed: 42}\n")
    log = RunLog()
    log("hello")
    out = tmp_path / "run"
    write_outputs(out, cfg, rows, {"metrics.json": {"a": 1}}, log)
    assert sorted(p.name for p in out.iterdir()) == [
        "config.yaml",
        "metrics.json",
        "per_query.csv",
        "run.log",
    ]
    assert json.loads((out / "metrics.json").read_text()) == {"a": 1}
    assert read_per_query(out / "per_query.csv") == rows
    with pytest.raises(FileExistsError):
        write_outputs(out, cfg, rows, {}, log)
