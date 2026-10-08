"""Rewrite each human query into one primary and N support queries with an LLM.

CLI:
  uv run python -m arl.rewrite --limit 20 --dry-run   # estimate only, no API calls
  uv run python -m arl.rewrite --limit 20             # asks for typed 'yes', then runs
  uv run python -m arl.rewrite --limit 20 --review    # also print queries for hand review

Reads `data/processed/queries_human.jsonl`; writes `data/processed/queries_agent.jsonl`
and `data/processed/rewrite_failures.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator

from arl.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from arl.data import PROCESSED_DIR, QUERIES_HUMAN_PATH, read_queries_jsonl
from arl.llm import LLMClient, LLMRequest

QUERIES_AGENT_PATH = PROCESSED_DIR / "queries_agent.jsonl"
FAILURES_PATH = PROCESSED_DIR / "rewrite_failures.jsonl"
MIN_WORDS, MAX_WORDS = 3, 15
NEAR_DUP_JACCARD = 0.8  # supports at/above this token overlap are flagged (not dropped)
EXPECTED_OUTPUT_TOKENS = 100  # typical JSON reply; used only for the dry-run estimate

SYSTEM_PROMPT = """\
You are the query-planning step of a search agent. Given a user's search request, you \
write the queries the agent will send to a web passage search engine.

Return exactly one JSON object and nothing else:
{"primary": "...", "support": ["...", "...", "..."]}

primary: one query that restates what the user wants to find, cleaned up and unambiguous. \
Keep the same intent; do not narrow or broaden it.

support: exactly 3 more queries that gather evidence for the same need. Each of the \
three must take a different angle; no two may ask for the same thing in different words. \
Across the three:
- rephrase the need with different wording or synonyms,
- look for background or context that a good answer relies on,
- reach a related entity, concept, or attribute connected to the request; the related \
entity must never be the answer itself,
- at least one must be keyword-dense: a compact string of key terms with no question \
words or filler.

Rules for every query:
- 3 to 15 words.
- It is a search query, not an answer. Do not put the answer in the query, even if you \
know it: do not name the place, person, date, number, or name the user is asking for.
- English, no quotation marks, no search operators, no numbering.
- Each support query must differ from the primary and from the other support queries.

Counter-example. Request: "what year did the titanic sink"
- Bad support: "titanic sinking april 1912" (it states the year the user is asking for).
- Good support: "titanic maiden voyage sinking timeline" (it finds pages with the year \
without stating it)."""


class Rewrite(BaseModel):
    """Validated LLM output: one primary query and exactly `n_support` support queries."""

    primary: str
    support: list[str]

    @field_validator("primary")
    @classmethod
    def _check_primary(cls, value: str) -> str:
        return _check_query(value)

    @field_validator("support")
    @classmethod
    def _check_support(cls, values: list[str]) -> list[str]:
        if len(values) != 3:
            raise ValueError(f"support must have exactly 3 queries, got {len(values)}")
        return [_check_query(v) for v in values]


def _check_query(text: str) -> str:
    """Strip whitespace and enforce the 3-15 word rule."""
    text = " ".join(text.split())
    n_words = len(text.split())
    if not MIN_WORDS <= n_words <= MAX_WORDS:
        raise ValueError(f"query must be {MIN_WORDS}-{MAX_WORDS} words, got {n_words}: {text!r}")
    return text


def user_prompt(query: str) -> str:
    """The per-query user message."""
    return f"Search request: {query}"


def retry_prompt(query: str, error: str) -> str:
    """A *different* prompt for the one retry, so it is not served the same cached reply."""
    return (
        f"{user_prompt(query)}\n\nYour previous reply was invalid because: {error}\n"
        "Return only the JSON object, following every rule."
    )


def parse_rewrite(raw: str) -> Rewrite:
    """Extract the JSON object from the reply (tolerating code fences) and validate it.

    Raises ValueError (incl. pydantic ValidationError) on anything invalid.
    """
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in reply")
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as err:
        raise ValueError(f"invalid JSON: {err}") from err
    return Rewrite.model_validate(data)


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace (for duplicate detection)."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def jaccard(a: str, b: str) -> float:
    """Token Jaccard similarity of two normalized strings: |A & B| / |A | B|."""
    set_a, set_b = set(a.split()), set(b.split())
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def rewrite_records(parent_id: str, human_text: str, rewrite: Rewrite) -> list[dict[str, Any]]:
    """Turn one validated rewrite into JSONL rows, flagging duplicates.

    `duplicate_of` (exact match after `normalize`) is "human" if a query equals the human
    query, else "primary" if a support query equals the primary, else "support" if it
    repeats an earlier support query, else None.
    `near_dup_of` (supports only, report-only) uses the same precedence for token Jaccard
    >= NEAR_DUP_JACCARD, and is None when the query is already an exact duplicate.
    """
    human_norm, primary_norm = normalize(human_text), normalize(rewrite.primary)
    rows = [
        {
            "query_id": f"{parent_id}_p",
            "parent_id": parent_id,
            "query_type": "primary",
            "text": rewrite.primary,
            "duplicate_of": "human" if primary_norm == human_norm else None,
            "near_dup_of": None,
        }
    ]
    earlier_support: list[str] = []
    for i, text in enumerate(rewrite.support, start=1):
        norm = normalize(text)
        rows.append(
            {
                "query_id": f"{parent_id}_s{i}",
                "parent_id": parent_id,
                "query_type": "support",
                "text": text,
                "duplicate_of": _match(norm, human_norm, primary_norm, earlier_support, _exact),
                "near_dup_of": None,
            }
        )
        if rows[-1]["duplicate_of"] is None:
            rows[-1]["near_dup_of"] = _match(norm, human_norm, primary_norm, earlier_support, _near)
        earlier_support.append(norm)
    return rows


def _exact(a: str, b: str) -> bool:
    return a == b


def _near(a: str, b: str) -> bool:
    return jaccard(a, b) >= NEAR_DUP_JACCARD


def _match(
    norm: str, human: str, primary: str, earlier: list[str], same: Callable[[str, str], bool]
) -> str | None:
    """Label of the first of human / primary / earlier supports that `same` matches."""
    if same(norm, human):
        return "human"
    if same(norm, primary):
        return "primary"
    if any(same(norm, other) for other in earlier):
        return "support"
    return None


def duplicate_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count flagged duplicates by kind: primary==human, support==human/primary/support."""
    counts = {
        "primary==human": 0,
        "support==human": 0,
        "support==primary": 0,
        "support==support": 0,
    }
    for row in rows:
        if row["duplicate_of"]:
            counts[f"{row['query_type']}=={row['duplicate_of']}"] += 1
    return counts


def duplicate_summary(rows: list[dict[str, Any]]) -> str:
    """One line reporting each duplicate count."""
    counts = duplicate_counts(rows)
    near = {"human": 0, "primary": 0, "support": 0}
    for row in rows:
        if row.get("near_dup_of"):
            near[row["near_dup_of"]] += 1
    return (
        f"Exact duplicates ({sum(counts.values())} total): "
        + ", ".join(f"{k}={v}" for k, v in counts.items())
        + f"\nNear-duplicate supports, Jaccard>={NEAR_DUP_JACCARD} ({sum(near.values())} total): "
        + ", ".join(f"~{k}={v}" for k, v in near.items())
    )


def rewrite_one(client: LLMClient, query: str) -> tuple[Rewrite | None, list[str]]:
    """Call the LLM, retrying once with the error message. Returns (rewrite, errors)."""
    errors: list[str] = []
    for prompt in (user_prompt(query), None):
        if prompt is None:
            prompt = retry_prompt(query, errors[-1])
        response = client.complete(client.request(prompt, system=SYSTEM_PROMPT))
        try:
            return parse_rewrite(response.text), errors
        except (ValueError, ValidationError) as err:
            errors.append(str(err).replace("\n", " ")[:300])
    return None, errors


def first_requests(client: LLMClient, queries: list[dict[str, str]]) -> list[LLMRequest]:
    """The first-attempt request for every query (what the dry run prices)."""
    return [client.request(user_prompt(q["text"]), system=SYSTEM_PROMPT) for q in queries]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows as JSON lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _flags(row: dict[str, Any]) -> str:
    """Review annotations for one row (empty string if none)."""
    flags = []
    if row.get("duplicate_of"):
        flags.append(f"duplicate of {row['duplicate_of']}")
    if row.get("near_dup_of"):
        flags.append(f"near-dup of {row['near_dup_of']}")
    if row.get("leaks_answer") is True:
        flags.append("LEAK")
    elif "leaks_answer" in row and row["leaks_answer"] is None:
        flags.append("leak=unparsed")
    return "   <-- " + "; ".join(flags) if flags else ""


def print_review(human: list[dict[str, str]], agent_rows: list[dict[str, Any]]) -> None:
    """Print each human query with its rewrites for hand review."""
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in agent_rows:
        by_parent.setdefault(row["parent_id"], []).append(row)
    for n, q in enumerate(human, start=1):
        print(f"\n{n:>2}. [{q['query_id']}] HUMAN: {q['text']}")
        for row in by_parent.get(q["query_id"], []):
            tag = "P " if row["query_type"] == "primary" else f"S{row['query_id'][-1]}"
            print(f"    {tag}: {row['text']}{_flags(row)}")
        if q["query_id"] not in by_parent:
            print("    (FAILED: see rewrite_failures.jsonl)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="print the estimate and exit")
    parser.add_argument("--review", action="store_true", help="print rewrites for hand review")
    args = parser.parse_args()

    config = load_config(args.config)
    human = read_queries_jsonl(QUERIES_HUMAN_PATH)[: args.limit]
    client = LLMClient.from_config(config)

    estimate = client.dry_run(
        first_requests(client, human), expected_output_tokens=EXPECTED_OUTPUT_TOKENS
    )
    if args.dry_run:
        print(estimate.summary())
        print("(Retries for invalid JSON are extra: at most one more call per failed query.)")
        return
    if not client.confirm(estimate):
        print("Not confirmed; no calls made.")
        return

    agent_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for q in human:
        rewrite, errors = rewrite_one(client, q["text"])
        if rewrite is None:
            failures.append({"query_id": q["query_id"], "text": q["text"], "errors": errors})
            print(f"  FAILED {q['query_id']}: {errors[-1]}")
            continue
        if errors:
            print(f"  retry fixed {q['query_id']}: {errors[0]}")
        agent_rows.extend(rewrite_records(q["query_id"], q["text"], rewrite))

    write_jsonl(agent_rows, QUERIES_AGENT_PATH)
    write_jsonl(failures, FAILURES_PATH)
    print(
        f"\nRewrote {len(human) - len(failures)}/{len(human)} queries -> {len(agent_rows)} agent "
        f"queries; {len(failures)} failed"
    )
    print(duplicate_summary(agent_rows))
    print(
        f"API calls this run: {client.calls_made}; tokens in/out: "
        f"{client.input_tokens_used:,}/{client.output_tokens_used:,}; "
        f"actual cost ${client.actual_cost_usd():.4f}"
    )
    if summary := client.rate_limit_summary():
        print(summary)
    print(f"Wrote {QUERIES_AGENT_PATH.relative_to(PROJECT_ROOT)}")
    if args.review:
        print_review(human, agent_rows)


if __name__ == "__main__":
    main()
