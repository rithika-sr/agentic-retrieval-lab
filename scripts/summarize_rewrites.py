"""Summarize agent rewrites and leak flags into results/foundation/<run_id>/ (no API calls).

Usage:
  uv run python scripts/summarize_rewrites.py --run-id 2026-10-08_rewrites [--logs a.log b.log]

Writes rewrite_summary.json and a copy of the config (and any given run logs) so every
number quoted in the docs traces to a file in results/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from collections import Counter
from pathlib import Path

from arl.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from arl.data import QUERIES_HUMAN_PATH, read_queries_jsonl
from arl.leak import leak_counts, read_jsonl
from arl.rewrite import FAILURES_PATH, QUERIES_AGENT_PATH, duplicate_counts


def word_stats(texts: list[str]) -> dict[str, float]:
    """Mean/min/max word count."""
    lengths = [len(t.split()) for t in texts]
    return {"mean": round(statistics.mean(lengths), 2), "min": min(lengths), "max": max(lengths)}


def summarize() -> dict:
    """Counts computed from the processed data files."""
    human = read_queries_jsonl(QUERIES_HUMAN_PATH)
    agent = read_jsonl(QUERIES_AGENT_PATH)
    failures = read_jsonl(FAILURES_PATH) if FAILURES_PATH.exists() else []
    by_type = Counter(r["query_type"] for r in agent)
    near = Counter(r["near_dup_of"] for r in agent if r.get("near_dup_of"))
    return {
        "n_human": len(human),
        "n_agent": len(agent),
        "agent_by_type": dict(by_type),
        "n_rewrite_failures": len(failures),
        "exact_duplicates": duplicate_counts(agent),
        "near_duplicate_supports": dict(near),
        "words": {
            "human": word_stats([q["text"] for q in human]),
            "primary": word_stats([r["text"] for r in agent if r["query_type"] == "primary"]),
            "support": word_stats([r["text"] for r in agent if r["query_type"] == "support"]),
        },
        "leak_judge": sorted({r.get("leak_judge") for r in agent if r.get("leak_judge")}),
        "leaks_by_type": leak_counts(agent),
        "parents_with_any_leak": len({r["parent_id"] for r in agent if r.get("leaks_answer")}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--logs", nargs="*", default=[], help="run logs to archive")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "results" / "foundation" / args.run_id
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; results are never overwritten")
    out_dir.mkdir(parents=True)
    summary = summarize()
    (out_dir / "rewrite_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.copy(args.config, out_dir / "config.yaml")
    for log in args.logs:
        text = Path(log).read_text().replace("\r", "\n")  # flatten progress-bar carriage returns
        (out_dir / Path(log).name).write_text(text)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
