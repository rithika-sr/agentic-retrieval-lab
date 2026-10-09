"""Leak flag v3: a deterministic rule over LLM-extracted answers (no LLM call here).

For a human question whose official answer is a *factoid*, an agent query leaks if it
contains a content token of the answer that does not appear in the question. Content
tokens are lowercased alphanumeric words, minus scikit-learn's English stopwords, with
simple plural stemming. *Descriptive* answers are not checkable: leaks_answer_v3 = None.

CLI:
  uv run python -m arl.leak_v3 --limit 20                  # print only
  uv run python -m arl.leak_v3 --run-id 2026-10-08_leak_v3 # also write results/foundation/<id>/

Adds answer, answer_type, leaks_answer_v3, and leak_tokens to every row of
`queries_agent.jsonl` whose parent has an extracted answer. The v2 fields are kept.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from arl.answers import read_answers
from arl.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from arl.data import QUERIES_HUMAN_PATH, read_queries_jsonl
from arl.leak import read_jsonl
from arl.rewrite import QUERIES_AGENT_PATH, print_review, write_jsonl

LEAK_RULE_VERSION = "token_v3"
DEV_N = 20  # the first 20 human queries were hand-reviewed: a dev set, not a test set
# Hand-labeled leaks among the dev set's tightened rewrites (agreed with Rithika 2026-10-07).
MANUAL_LEAKS_DEV = ("1003210_s2", "1085008_s3", "1085533_s2", "1085967_s2", "985644_s3")


def stem(token: str) -> str:
    """Simple plural stemming: counties->county, boxes->box, borers->borer (not 'glass')."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("sses", "shes", "ches", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def content_tokens(text: str) -> set[str]:
    """Lowercased, accent-folded, stemmed words minus stopwords and single letters."""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    words = re.findall(r"[a-z0-9]+", folded)
    return {stem(w) for w in words if w not in ENGLISH_STOP_WORDS and (len(w) > 1 or w.isdigit())}


def leak_tokens(question: str, answer: str, query: str) -> list[str]:
    """Answer tokens that are in the query but not in the question (sorted)."""
    new_answer_tokens = content_tokens(answer) - content_tokens(question)
    return sorted(new_answer_tokens & content_tokens(query))


def flag_row(row: dict[str, Any], question: str, answer_row: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of an agent-query row with the v3 fields added."""
    out = dict(row)
    out["answer"] = answer_row["answer"]
    out["answer_type"] = answer_row["answer_type"]
    out["leak_rule"] = LEAK_RULE_VERSION
    if answer_row["answer_type"] == "factoid":
        tokens = leak_tokens(question, answer_row["answer"], row["text"])
        out["leak_tokens"] = tokens
        out["leaks_answer_v3"] = bool(tokens)
    else:  # descriptive or failed extraction: not checkable
        out["leak_tokens"] = []
        out["leaks_answer_v3"] = None
    return out


def leak_counts_v3(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    """{query_type: {answer_type: {"leak", "no_leak", "not_checkable"}}} over flagged rows."""
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        if "leaks_answer_v3" not in row:
            continue
        by_answer = counts.setdefault(row["query_type"], {})
        bucket = by_answer.setdefault(
            row["answer_type"] or "extraction_failed",
            {"leak": 0, "no_leak": 0, "not_checkable": 0},
        )
        verdict = row["leaks_answer_v3"]
        bucket["leak" if verdict else "no_leak" if verdict is False else "not_checkable"] += 1
    return counts


def sanity_checks(rows: list[dict[str, Any]], dev_parent_ids: set[str]) -> dict[str, Any]:
    """(a) identical-to-question primaries must not leak; (b) dev set vs manual labels."""
    identical = [r for r in rows if r.get("duplicate_of") == "human" and "leaks_answer_v3" in r]
    identical_leaks = [r["query_id"] for r in identical if r["leaks_answer_v3"]]
    dev = [r for r in rows if r["parent_id"] in dev_parent_ids and "leaks_answer_v3" in r]
    flagged = {r["query_id"] for r in dev if r["leaks_answer_v3"]}
    manual = set(MANUAL_LEAKS_DEV)
    by_id = {r["query_id"]: r for r in dev}

    def describe(qid: str) -> dict[str, Any]:
        r = by_id[qid]
        return {
            "query_id": qid,
            "text": r["text"],
            "answer": r["answer"],
            "answer_type": r["answer_type"],
            "leak_tokens": r["leak_tokens"],
        }

    return {
        "leak_rule": LEAK_RULE_VERSION,
        "negative_control": {
            "description": "primaries identical to the human question (cannot leak by definition)",
            "n": len(identical),
            "flagged_as_leak": len(identical_leaks),
            "flagged_ids": identical_leaks,
            "passed": not identical_leaks,
        },
        "dev20_vs_manual": {
            "note": "first 20 human queries; hand-reviewed before v3 was written (dev, not test)",
            "n_agent_queries": len(dev),
            "n_not_checkable": sum(r["leaks_answer_v3"] is None for r in dev),
            "manual_leaks": sorted(manual),
            "caught": [describe(q) for q in sorted(manual & flagged)],
            "missed": [describe(q) for q in sorted(manual - flagged) if q in by_id],
            "false_positives": [describe(q) for q in sorted(flagged - manual)],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--limit", type=int, default=None, help="first N human queries only")
    parser.add_argument("--run-id", default=None, help="write results/foundation/<run_id>/")
    parser.add_argument("--review", action="store_true", help="print the dev-set rewrites")
    parser.add_argument("--logs", nargs="*", default=[], help="run logs to archive")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "results" / "foundation" / args.run_id if args.run_id else None
    if out_dir and out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; results are never overwritten")

    all_human = read_queries_jsonl(QUERIES_HUMAN_PATH)
    human = all_human[: args.limit]
    question = {q["query_id"]: q["text"] for q in human}
    answers = read_answers()
    all_rows = read_jsonl(QUERIES_AGENT_PATH)
    updated = [
        flag_row(r, question[r["parent_id"]], answers[r["parent_id"]])
        if r["parent_id"] in question and r["parent_id"] in answers
        else r
        for r in all_rows
    ]
    write_jsonl(updated, QUERIES_AGENT_PATH)
    rows = [r for r in updated if r["parent_id"] in question and "leaks_answer_v3" in r]

    counts = leak_counts_v3(rows)
    sanity = sanity_checks(rows, {q["query_id"] for q in all_human[:DEV_N]})
    answer_types = [answers[q]["answer_type"] for q in question if q in answers]
    report = {
        "n_human_with_answer": len(answer_types),
        "answer_types": {
            "factoid": answer_types.count("factoid"),
            "descriptive": answer_types.count("descriptive"),
            "extraction_failed": answer_types.count(None),
        },
        "parents_with_any_leak": len({r["parent_id"] for r in rows if r["leaks_answer_v3"]}),
        "leaks_by_query_type_and_answer_type": counts,
    }
    print(json.dumps(report, indent=2))
    print(json.dumps(sanity, indent=2))
    print(f"Updated {QUERIES_AGENT_PATH.relative_to(PROJECT_ROOT)} ({len(rows)} rows flagged)")

    if out_dir:
        out_dir.mkdir(parents=True)
        (out_dir / "leak_v3_counts.json").write_text(json.dumps(report, indent=2) + "\n")
        (out_dir / "leak_v3_sanity.json").write_text(json.dumps(sanity, indent=2) + "\n")
        shutil.copy(args.config, out_dir / "config.yaml")
        for log in args.logs:
            shutil.copy(log, out_dir / Path(log).name)
        print(f"Wrote {out_dir.relative_to(PROJECT_ROOT)}/")
    if args.review:
        dev_ids = {q["query_id"] for q in all_human[:DEV_N]}
        print_review(all_human[:DEV_N], [r for r in rows if r["parent_id"] in dev_ids])


if __name__ == "__main__":
    main()
