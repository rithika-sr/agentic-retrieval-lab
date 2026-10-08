"""Tests for the rewrite schema, JSON parsing, retry logic, and duplicate flags (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from arl.llm import LLMCache, LLMClient, LLMRequest, LLMResponse
from arl.rewrite import Rewrite, normalize, parse_rewrite, rewrite_one, rewrite_records

GOOD = {
    "primary": "where was abraham lincoln born",
    "support": [
        "abraham lincoln birthplace location",
        "early life of abraham lincoln family",
        "lincoln birth cabin national historical park",
    ],
}


def test_valid_rewrite_parses() -> None:
    rewrite = Rewrite.model_validate(GOOD)
    assert rewrite.primary == GOOD["primary"]
    assert len(rewrite.support) == 3


@pytest.mark.parametrize("support", [GOOD["support"][:2], GOOD["support"] + ["one more query"]])
def test_wrong_number_of_support_queries_fails(support: list[str]) -> None:
    with pytest.raises(ValidationError):
        Rewrite.model_validate({"primary": GOOD["primary"], "support": support})


@pytest.mark.parametrize("text", ["lincoln", "lincoln born", " ".join(["word"] * 16), "   "])
def test_word_count_limits(text: str) -> None:
    with pytest.raises(ValidationError):
        Rewrite.model_validate({"primary": text, "support": GOOD["support"]})


def test_word_count_boundaries_pass() -> None:
    Rewrite.model_validate({"primary": "a b c", "support": GOOD["support"]})
    Rewrite.model_validate({"primary": " ".join(["w"] * 15), "support": GOOD["support"]})


def test_missing_field_fails() -> None:
    with pytest.raises(ValidationError):
        Rewrite.model_validate({"primary": GOOD["primary"]})


def test_parse_tolerates_code_fences_and_prose() -> None:
    raw = "Here you go:\n```json\n" + json.dumps(GOOD) + "\n```"
    assert parse_rewrite(raw).primary == GOOD["primary"]


@pytest.mark.parametrize("raw", ["no json here", '{"primary": "a b c", "support": [}'])
def test_parse_rejects_bad_json(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_rewrite(raw)


def test_normalize() -> None:
    assert normalize("  Where was Lincoln BORN?! ") == "where was lincoln born"


def test_records_ids_types_and_duplicate_flags() -> None:
    rewrite = Rewrite.model_validate(
        {
            "primary": "Where was Lincoln born?",
            "support": [
                "where was lincoln born",
                "lincoln birthplace kentucky",
                "where was lincoln born",
            ],
        }
    )
    rows = rewrite_records("42", "where was lincoln born", rewrite)
    assert [r["query_id"] for r in rows] == ["42_p", "42_s1", "42_s2", "42_s3"]
    assert [r["query_type"] for r in rows] == ["primary", "support", "support", "support"]
    assert all(r["parent_id"] == "42" for r in rows)
    # human and primary normalize to the same text, so "human" wins for both
    assert [r["duplicate_of"] for r in rows] == ["human", "human", None, "human"]


def test_support_duplicate_of_primary_and_of_support() -> None:
    rewrite = Rewrite.model_validate(
        {
            "primary": "abraham lincoln place of birth",
            "support": [
                "Abraham Lincoln place of birth",
                "lincoln birth cabin",
                "lincoln birth cabin",
            ],
        }
    )
    rows = rewrite_records("7", "where was lincoln born", rewrite)
    assert [r["duplicate_of"] for r in rows] == [None, "primary", None, "support"]


class ScriptedProvider:
    """Returns pre-written replies in order and records the prompts it saw."""

    name = "scripted"
    model = "fake-model"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.prompts.append(request.prompt)
        return LLMResponse(text=self.replies.pop(0), input_tokens=1, output_tokens=1)


def _client(tmp_path: Path, provider: ScriptedProvider) -> LLMClient:
    return LLMClient(
        provider,
        LLMCache(tmp_path / "c.sqlite"),
        max_calls_per_run=10,
        pricing_usd_per_mtok={"fake-model": {"input": 1.0, "output": 5.0}},
    )


def test_retry_uses_a_different_prompt_and_can_recover(tmp_path: Path) -> None:
    provider = ScriptedProvider(["not json", json.dumps(GOOD)])
    rewrite, errors = rewrite_one(_client(tmp_path, provider), "where was lincoln born")
    assert rewrite is not None and len(errors) == 1
    assert len(provider.prompts) == 2 and provider.prompts[0] != provider.prompts[1]
    assert "invalid because" in provider.prompts[1]


def test_two_failures_return_none_without_raising(tmp_path: Path) -> None:
    provider = ScriptedProvider(["not json", '{"primary": "x"}'])
    rewrite, errors = rewrite_one(_client(tmp_path, provider), "where was lincoln born")
    assert rewrite is None and len(errors) == 2


def test_duplicate_counts_separates_primary_equal_to_human() -> None:
    from arl.rewrite import duplicate_counts

    rewrite = Rewrite.model_validate(
        {
            "primary": "Where was Lincoln born?",
            "support": ["lincoln birthplace kentucky", "where was lincoln born", "a b c"],
        }
    )
    counts = duplicate_counts(rewrite_records("1", "where was lincoln born", rewrite))
    assert counts == {
        "primary==human": 1,
        "support==human": 1,
        "support==primary": 0,
        "support==support": 0,
    }


def test_jaccard() -> None:
    from arl.rewrite import jaccard

    assert jaccard("a b c d", "a b c d") == 1.0
    assert jaccard("a b c d", "a b c e") == pytest.approx(3 / 5)
    assert jaccard("a b c d e", "a b c d") == pytest.approx(4 / 5)


def test_near_duplicate_flags_supports_only() -> None:
    rewrite = Rewrite.model_validate(
        {
            "primary": "warwick weather forecast today",
            "support": [
                "warwick weather forecast for today",  # 4/5 = 0.8 vs primary -> near-dup
                "warwick rhode island climate averages",
                "rhode island warwick climate averages",  # same words, reordered -> near-dup
            ],
        }
    )
    rows = rewrite_records("9", "weather in warwick", rewrite)
    assert [r["duplicate_of"] for r in rows] == [None, None, None, None]
    assert [r["near_dup_of"] for r in rows] == [None, "primary", None, "support"]


def test_exact_duplicate_is_not_also_near_duplicate() -> None:
    rewrite = Rewrite.model_validate(
        {
            "primary": "weather in warwick today",
            "support": ["weather in warwick today", "x y z", "p q r"],
        }
    )
    s1 = rewrite_records("9", "weather in warwick", rewrite)[1]
    assert s1["duplicate_of"] == "primary" and s1["near_dup_of"] is None
