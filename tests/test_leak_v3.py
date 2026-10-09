"""Tests for answer extraction parsing and the deterministic v3 leak rule (no network)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from arl.answers import Answer, parse_answer
from arl.leak_v3 import content_tokens, flag_row, leak_counts_v3, leak_tokens, stem


def test_answer_parses_from_fenced_json() -> None:
    answer = parse_answer('```json\n{"answer_type": "factoid", "answer": " Jane   Austen "}\n```')
    assert answer == Answer(answer_type="factoid", answer="Jane Austen")


@pytest.mark.parametrize(
    "data",
    [
        {"answer_type": "list", "answer": "Jane Austen"},
        {"answer_type": "factoid", "answer": "one two three four five six seven"},
        {"answer_type": "factoid", "answer": "  "},
        {"answer_type": "descriptive", "answer": " ".join(["word"] * 26)},
        {"answer": "Jane Austen"},
    ],
)
def test_invalid_answers_fail(data: dict) -> None:
    with pytest.raises(ValidationError):
        Answer.model_validate(data)


def test_descriptive_answers_may_exceed_the_factoid_cap() -> None:
    answer = Answer(answer_type="descriptive", answer="vehicle that can run on E85 ethanol fuel")
    assert len(answer.answer.split()) == 8


def test_parse_answer_without_json_raises() -> None:
    with pytest.raises(ValueError):
        parse_answer("I am not sure what you mean.")


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("counties", "county"),
        ("boxes", "box"),
        ("borers", "borer"),
        ("glass", "glass"),
        ("virus", "virus"),
        ("paris", "paris"),
        ("gas", "gas"),
    ],
)
def test_stem(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_content_tokens_drop_stopwords_punctuation_and_letters() -> None:
    assert content_tokens("Where was Michael B. Jordan born, in 1987?") == {
        "michael",
        "jordan",
        "born",
        "1987",
    }
    assert content_tokens("Café Zürich") == {"cafe", "zurich"}


def test_leak_tokens_ignore_answer_words_already_in_question() -> None:
    question = "what county is grottoes virginia in"
    answer = "Rockingham County"
    assert leak_tokens(question, answer, "rockingham county virginia towns cities") == [
        "rockingham"
    ]
    assert leak_tokens(question, answer, "grottoes virginia county location") == []


def test_plural_in_query_matches_singular_answer() -> None:
    assert leak_tokens("where is the castle", "Hohenzollern", "hohenzollerns castle") == [
        "hohenzollern"
    ]


def test_identical_query_can_never_leak() -> None:
    q = "who played adonis in creed"
    assert leak_tokens(q, "Michael B. Jordan adonis", q) == []


def test_flag_row_descriptive_is_not_checkable() -> None:
    row = {"query_id": "1_s1", "parent_id": "1", "query_type": "support", "text": "x y z"}
    out = flag_row(row, "what is x", {"answer_type": "descriptive", "answer": "x y z"})
    assert out["leaks_answer_v3"] is None and out["leak_tokens"] == []
    assert "leaks_answer_v3" not in row  # input untouched


def test_flag_row_factoid_sets_tokens() -> None:
    row = {"query_id": "1_s1", "parent_id": "1", "query_type": "support", "text": "titanic 1912"}
    out = flag_row(row, "when did titanic sink", {"answer_type": "factoid", "answer": "1912"})
    assert out["leaks_answer_v3"] is True and out["leak_tokens"] == ["1912"]


def test_leak_counts_v3_by_query_and_answer_type() -> None:
    rows = [
        {"query_type": "primary", "answer_type": "factoid", "leaks_answer_v3": False},
        {"query_type": "support", "answer_type": "factoid", "leaks_answer_v3": True},
        {"query_type": "support", "answer_type": "descriptive", "leaks_answer_v3": None},
        {"query_type": "support"},  # not flagged yet
    ]
    assert leak_counts_v3(rows) == {
        "primary": {"factoid": {"leak": 0, "no_leak": 1, "not_checkable": 0}},
        "support": {
            "factoid": {"leak": 1, "no_leak": 0, "not_checkable": 0},
            "descriptive": {"leak": 0, "no_leak": 0, "not_checkable": 1},
        },
    }
