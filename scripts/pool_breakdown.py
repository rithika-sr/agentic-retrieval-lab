"""Break down full-collection pool coverage: rewritten primaries vs human, leak vs no leak.

Usage: uv run python scripts/pool_breakdown.py --run-id 2026-10-08_bm25
Writes pool_breakdown.json to results/foundation/<run_id>/ (refuses to overwrite).
"""

from __future__ import annotations

import argparse
import json

from arl.config import PROJECT_ROOT
from arl.data import QRELS_OFFICIAL_PATH, read_qrels_trec
from arl.leak import read_jsonl
from arl.pool import POOL_RUN_PATH, read_trec_run
from arl.rewrite import QUERIES_AGENT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    out = PROJECT_ROOT / "results" / "foundation" / args.run_id / "pool_breakdown.json"
    if out.exists():
        raise FileExistsError(f"{out} exists; results are never overwritten")

    run = read_trec_run(POOL_RUN_PATH)
    qrels = read_qrels_trec(QRELS_OFFICIAL_PATH)

    def hit(query_id: str, parent_id: str) -> bool:
        return bool(qrels[parent_id].keys() & {d for d, _ in run[query_id]})

    rows = read_jsonl(QUERIES_AGENT_PATH)
    primaries = [r for r in rows if r["query_type"] == "primary" and r["duplicate_of"] != "human"]
    human_hit = {r["parent_id"]: hit(r["parent_id"], r["parent_id"]) for r in primaries}
    primary_hit = {r["parent_id"]: hit(r["query_id"], r["parent_id"]) for r in primaries}
    supports = [r for r in rows if r["query_type"] == "support" and r["answer_type"] == "factoid"]

    report = {
        "rewritten_primaries_vs_human": {
            "note": "only primaries that differ from the human query (identical ones excluded)",
            "n": len(primaries),
            "human_hits": sum(human_hit.values()),
            "primary_hits": sum(primary_hit.values()),
            "both": sum(human_hit[p] and primary_hit[p] for p in human_hit),
            "human_only": sum(human_hit[p] and not primary_hit[p] for p in human_hit),
            "primary_only": sum(primary_hit[p] and not human_hit[p] for p in human_hit),
        },
        "factoid_supports_by_leak_v3": {
            str(flag).lower(): {
                "n": len(group := [r for r in supports if r["leaks_answer_v3"] is flag]),
                "hits": sum(hit(r["query_id"], r["parent_id"]) for r in group),
            }
            for flag in (True, False)
        },
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
