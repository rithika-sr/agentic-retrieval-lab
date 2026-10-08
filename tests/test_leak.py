"""Tests for the answer-leak judge helpers (no network)."""

from __future__ import annotations

import pytest

from arl.leak import leak_counts, leak_prompt, parse_yes_no


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("YES", True), ("Yes.", True), ("no", False), (" NO\n", False), ("Maybe", None), ("", None)],
)
def test_parse_yes_no(reply: str, expected: bool | None) -> None:
    assert parse_yes_no(reply) is expected


def test_leak_prompt_includes_question_every_passage_and_query() -> None:
    prompt = leak_prompt(
        "what year did the titanic sink",
        ["The Titanic sank in April 1912.", "It struck an iceberg."],
        "titanic sinking april 1912",
    )
    assert prompt.startswith("Question: what year did the titanic sink\n")
    assert "Relevant passage 1: The Titanic sank in April 1912." in prompt
    assert "Relevant passage 2: It struck an iceberg." in prompt
    assert "Search query: titanic sinking april 1912" in prompt
    assert "even as a bare keyword?" in prompt


def test_leak_counts_by_type_skips_unjudged_rows() -> None:
    rows = [
        {"query_type": "primary", "leaks_answer": False},
        {"query_type": "support", "leaks_answer": True},
        {"query_type": "support", "leaks_answer": None},
        {"query_type": "support"},  # not judged yet
    ]
    assert leak_counts(rows) == {
        "primary": {"leak": 0, "no_leak": 1, "unparsed": 0},
        "support": {"leak": 1, "no_leak": 0, "unparsed": 1},
    }
