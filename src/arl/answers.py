"""Extract the answer each official passage gives to its human question (one LLM call each).

The output feeds the deterministic leak rule in `arl.leak_v3`: the LLM only reads off the
answer and its type; it never decides whether an agent query leaks.

CLI:
  uv run python -m arl.answers --limit 20 --dry-run   # estimate only, no API calls
  uv run python -m arl.answers --limit 20             # asks for typed 'yes', then runs

Reads `queries_human.jsonl` + the official positive passages; writes (merges into)
`data/processed/answers.jsonl`, one row per human query.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Literal

from pydantic import BaseModel, ValidationError, field_validator, model_validator

from arl.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from arl.data import PROCESSED_DIR, QUERIES_HUMAN_PATH, read_queries_jsonl
from arl.leak import positive_texts, read_jsonl
from arl.llm import LLMClient, LLMRequest
from arl.rewrite import write_jsonl

ANSWERS_PATH = PROCESSED_DIR / "answers.jsonl"
MAX_ANSWER_WORDS = 6  # factoid answers (the prompt asks for this for all answers)
MAX_DESCRIPTIVE_WORDS = 25  # descriptive gists are never used by the leak rule
ANSWER_MAX_TOKENS = 100
EXPECTED_OUTPUT_TOKENS = 25  # typical JSON reply; used only for the dry-run estimate
EXTRACTOR_VERSION = "answer_v1"

ANSWER_SYSTEM = """\
You read a search question and the passage(s) a human judged relevant to it, and report \
the answer the passage gives.

Return exactly one JSON object and nothing else:
{"answer_type": "factoid", "answer": "..."}

answer_type:
- "factoid": the answer is a specific entity, place, person, organization, date, number, \
or name.
- "descriptive": the answer is a definition, explanation, description, list of reasons, \
or how-to.

answer: at most 6 words, using the passage's own wording where possible. For a \
descriptive answer, give a short gist. If the passage does not really answer the \
question, use "descriptive" and give a short gist of what it says.

Examples:
- Question "who wrote pride and prejudice", passage "...novel by Jane Austen..." -> \
{"answer_type": "factoid", "answer": "Jane Austen"}
- Question "what year did the titanic sink", passage "...sank on 15 April 1912..." -> \
{"answer_type": "factoid", "answer": "15 April 1912"}
- Question "what is photosynthesis", passage "...process plants use to turn light into \
chemical energy..." -> {"answer_type": "descriptive", "answer": "plants turn light into energy"}"""


class Answer(BaseModel):
    """Validated extractor output."""

    answer_type: Literal["factoid", "descriptive"]
    answer: str

    @field_validator("answer")
    @classmethod
    def _clean_answer(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("answer is empty")
        return value

    @model_validator(mode="after")
    def _check_length(self) -> Answer:
        # Only factoid answers feed the leak rule, so only they need the tight cap.
        cap = MAX_ANSWER_WORDS if self.answer_type == "factoid" else MAX_DESCRIPTIVE_WORDS
        n_words = len(self.answer.split())
        if n_words > cap:
            raise ValueError(
                f"{self.answer_type} answer must be at most {cap} words, got {n_words}: "
                f"{self.answer!r}"
            )
        return self


def answer_prompt(question: str, passages: list[str]) -> str:
    """The per-question user message."""
    passage_block = "\n".join(
        f"Relevant passage {i}: {text}" for i, text in enumerate(passages, start=1)
    )
    return f"Question: {question}\n{passage_block}"


def answer_request(client: LLMClient, prompt: str) -> LLMRequest:
    """A short-output request at the configured temperature."""
    return LLMRequest(
        prompt=prompt,
        system=ANSWER_SYSTEM,
        max_tokens=ANSWER_MAX_TOKENS,
        temperature=client.default_temperature,
    )


def parse_answer(raw: str) -> Answer:
    """Extract the JSON object from the reply and validate it. Raises ValueError if invalid."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in reply")
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as err:
        raise ValueError(f"invalid JSON: {err}") from err
    return Answer.model_validate(data)


def extract_one(
    client: LLMClient, question: str, passages: list[str]
) -> tuple[Answer | None, list[str]]:
    """Call the LLM, retrying once with the error appended. Returns (answer, errors)."""
    prompt = answer_prompt(question, passages)
    errors: list[str] = []
    for attempt in range(2):
        if attempt == 1:
            prompt += (
                f"\n\nYour previous reply was invalid because: {errors[-1]}\n"
                "Return only the JSON object, following every rule."
            )
        response = client.complete(answer_request(client, prompt))
        try:
            return parse_answer(response.text), errors
        except (ValueError, ValidationError) as err:
            errors.append(str(err).replace("\n", " ")[:300])
    return None, errors


def read_answers() -> dict[str, dict[str, Any]]:
    """{human query_id: answer row} from answers.jsonl (empty if the file is missing)."""
    if not ANSWERS_PATH.exists():
        return {}
    return {r["query_id"]: r for r in read_jsonl(ANSWERS_PATH)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--limit", type=int, default=None, help="first N human queries only")
    parser.add_argument("--dry-run", action="store_true", help="print the estimate and exit")
    args = parser.parse_args()

    config = load_config(args.config)
    human = read_queries_jsonl(QUERIES_HUMAN_PATH)[: args.limit]
    passages = positive_texts()
    client = LLMClient.from_config(config)

    requests = [
        answer_request(client, answer_prompt(q["text"], passages[q["query_id"]])) for q in human
    ]
    estimate = client.dry_run(requests, expected_output_tokens=EXPECTED_OUTPUT_TOKENS)
    if args.dry_run:
        print(estimate.summary())
        print("(Retries for invalid JSON are extra: at most one more call per failed query.)")
        return
    if not client.confirm(estimate):
        print("Not confirmed; no calls made.")
        return

    rows = read_answers()  # merge, so a --limit run never drops earlier answers
    n_failed = 0
    for q in human:
        answer, errors = extract_one(client, q["text"], passages[q["query_id"]])
        if errors:
            print(f"  {'retry fixed' if answer else 'FAILED'} {q['query_id']}: {errors[0]}")
        n_failed += answer is None
        rows[q["query_id"]] = {
            "query_id": q["query_id"],
            "question": q["text"],
            "answer_type": answer.answer_type if answer else None,
            "answer": answer.answer if answer else None,
            "extractor": EXTRACTOR_VERSION,
            "errors": errors,
        }
    write_jsonl(list(rows.values()), ANSWERS_PATH)

    types = [rows[q["query_id"]]["answer_type"] for q in human]
    print(
        f"\nExtracted {len(human) - n_failed}/{len(human)}: "
        f"factoid={types.count('factoid')} descriptive={types.count('descriptive')} "
        f"failed={n_failed}"
    )
    print(
        f"API calls this run: {client.calls_made}; tokens in/out: "
        f"{client.input_tokens_used:,}/{client.output_tokens_used:,}; "
        f"actual cost ${client.actual_cost_usd():.4f}"
    )
    print(f"Wrote {ANSWERS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
