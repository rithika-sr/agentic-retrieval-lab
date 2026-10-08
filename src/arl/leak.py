"""Flag agent queries that contain the specific answer stated in the official positive passage.

One yes/no LLM call per agent query. The judge sees the human question, the official
relevant passage(s) from qrels, and the agent query, and decides whether the query
contains the answer as stated in the passage. The verdict is stored as `leaks_answer`
(true / false / null if unparsed) plus `leak_judge` (the judge version) in the same file.

CLI:
  uv run python -m arl.leak --dry-run                      # estimate only
  uv run python -m arl.leak --limit 20 --review            # asks for typed 'yes'

Caveat: the judge is the same model family that wrote the rewrites. Grounding it in the
passage reduces its reliance on its own knowledge, but its agreement with a human must be
measured on held-out queries (see `arl.leak_holdout`).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from arl.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from arl.data import (
    POSITIVE_PASSAGES_PATH,
    QRELS_OFFICIAL_PATH,
    QUERIES_HUMAN_PATH,
    read_passages_jsonl,
    read_qrels_trec,
    read_queries_jsonl,
)
from arl.llm import LLMClient, LLMRequest
from arl.rewrite import QUERIES_AGENT_PATH, print_review, write_jsonl

LEAK_JUDGE_VERSION = "passage_v2"
LEAK_MAX_TOKENS = 5
EXPECTED_OUTPUT_TOKENS = 2  # "YES" / "NO"

LEAK_SYSTEM = (
    "You audit search queries written by a search agent. Answer with exactly one word: YES or NO."
)


def leak_prompt(question: str, passages: list[str], query: str) -> str:
    """The yes/no question for one (question, official passages, agent query) triple."""
    passage_block = "\n".join(
        f"Relevant passage {i}: {text}" for i, text in enumerate(passages, start=1)
    )
    return (
        f"Question: {question}\n"
        f"{passage_block}\n"
        f"Search query: {query}\n\n"
        "Does the search query contain the specific answer to the question as stated in the "
        "relevant passage (a named entity, place, date, number, or name), even as a bare "
        "keyword?"
    )


def leak_request(client: LLMClient, question: str, passages: list[str], query: str) -> LLMRequest:
    """A short-output request at the configured temperature."""
    return LLMRequest(
        prompt=leak_prompt(question, passages, query),
        system=LEAK_SYSTEM,
        max_tokens=LEAK_MAX_TOKENS,
        temperature=client.default_temperature,
    )


def parse_yes_no(text: str) -> bool | None:
    """True for YES, False for NO, None for anything else."""
    words = re.sub(r"[^a-z\s]", " ", text.lower()).split()
    if not words:
        return None
    return {"yes": True, "no": False}.get(words[0])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read any JSONL file into a list of dicts."""
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def positive_texts() -> dict[str, list[str]]:
    """{human query_id: [official positive passage texts]} for the sampled queries."""
    qrels = read_qrels_trec(QRELS_OFFICIAL_PATH)
    passages = read_passages_jsonl(POSITIVE_PASSAGES_PATH)
    return {qid: [passages[d] for d in sorted(docs, key=int)] for qid, docs in qrels.items()}


def leak_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """{query_type: {"leak": n, "no_leak": n, "unparsed": n}} over judged rows."""
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        if "leaks_answer" not in row:
            continue
        bucket = counts.setdefault(row["query_type"], {"leak": 0, "no_leak": 0, "unparsed": 0})
        verdict = row["leaks_answer"]
        bucket["leak" if verdict else "no_leak" if verdict is False else "unparsed"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--input", default=str(QUERIES_AGENT_PATH), help="agent queries JSONL")
    parser.add_argument("--limit", type=int, default=None, help="first N human queries only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--review", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    path = Path(args.input)
    path = path if path.is_absolute() else PROJECT_ROOT / path
    all_human = read_queries_jsonl(QUERIES_HUMAN_PATH)
    human = all_human[: args.limit]
    question = {q["query_id"]: q["text"] for q in human}
    passages = positive_texts()
    rows = [r for r in read_jsonl(path) if r["parent_id"] in question]
    client = LLMClient.from_config(config)

    requests = [
        leak_request(client, question[r["parent_id"]], passages[r["parent_id"]], r["text"])
        for r in rows
    ]
    estimate = client.dry_run(requests, expected_output_tokens=EXPECTED_OUTPUT_TOKENS)
    if args.dry_run:
        n_parents = len({r["parent_id"] for r in rows})
        print(f"Input: {path.relative_to(PROJECT_ROOT)} ({len(rows)} agent queries)")
        print(estimate.summary())
        if n_parents and n_parents < len(all_human):
            scale = len(all_human) / n_parents
            print(
                f"Projected for all {len(all_human)} human queries (scaled x{scale:.0f} from "
                f"{n_parents}): ~{round(len(rows) * scale)} calls, "
                f"~${estimate.est_cost_usd * scale:.4f}"
            )
        return
    if not client.confirm(estimate):
        print("Not confirmed; no calls made.")
        return

    for row, req in zip(rows, requests, strict=True):
        row["leaks_answer"] = parse_yes_no(client.complete(req).text)
        row["leak_judge"] = LEAK_JUDGE_VERSION

    # Rewrite the whole file, updating only the judged rows.
    judged = {r["query_id"]: r for r in rows}
    write_jsonl([judged.get(r["query_id"], r) for r in read_jsonl(path)], path)

    for qtype, c in leak_counts(rows).items():
        print(f"{qtype:>8}: leak={c['leak']} no_leak={c['no_leak']} unparsed={c['unparsed']}")
    print(f"API calls this run: {client.calls_made}; actual cost ${client.actual_cost_usd():.4f}")
    if summary := client.rate_limit_summary():
        print(summary)
    print(f"Updated {path.relative_to(PROJECT_ROOT)}")
    if args.review:
        print_review(human, rows)


if __name__ == "__main__":
    main()
