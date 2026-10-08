"""Tests for the LLM client using a fake provider (no network, no cost)."""

from __future__ import annotations

from pathlib import Path

import pytest

from arl.llm import (
    CallCapExceeded,
    LLMCache,
    LLMClient,
    LLMRequest,
    LLMResponse,
    RetryableError,
    cache_key,
)

MODEL = "fake-model"
PRICING = {MODEL: {"input": 1.00, "output": 5.00}}  # USD per 1M tokens


class FakeProvider:
    """Echoes the prompt and counts calls. Can fail the first `fail_times` calls."""

    name = "fake"
    model = MODEL

    def __init__(self, fail_times: int = 0) -> None:
        self.calls = 0
        self.fail_times = fail_times

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RetryableError("429 rate limited")
        return LLMResponse(text=f"echo: {request.prompt}", input_tokens=10, output_tokens=5)


def make_client(tmp_path: Path, provider: FakeProvider, **kwargs) -> LLMClient:
    defaults = {
        "max_calls_per_run": 100,
        "pricing_usd_per_mtok": PRICING,
        "sleep": lambda s: None,
        **kwargs,
    }
    return LLMClient(provider, LLMCache(tmp_path / "cache.sqlite"), **defaults)


def test_cache_miss_then_hit(tmp_path: Path) -> None:
    provider = FakeProvider()
    client = make_client(tmp_path, provider)
    req = LLMRequest(prompt="hello")

    first = client.complete(req)
    second = client.complete(req)

    assert provider.calls == 1  # the second call never reached the provider
    assert client.calls_made == 1
    assert not first.cached and second.cached
    assert first.text == second.text == "echo: hello"


def test_cache_persists_across_clients(tmp_path: Path) -> None:
    make_client(tmp_path, FakeProvider()).complete(LLMRequest(prompt="hi"))
    provider = FakeProvider()
    assert make_client(tmp_path, provider).complete(LLMRequest(prompt="hi")).cached
    assert provider.calls == 0


def test_changing_any_param_is_a_cache_miss(tmp_path: Path) -> None:
    provider = FakeProvider()
    client = make_client(tmp_path, provider)
    client.complete(LLMRequest(prompt="q"))
    client.complete(LLMRequest(prompt="q", temperature=0.7))
    client.complete(LLMRequest(prompt="q", max_tokens=100))
    client.complete(LLMRequest(prompt="q", system="be brief"))
    assert provider.calls == 4


def test_cache_key_depends_on_provider_and_model() -> None:
    req = LLMRequest(prompt="q")
    assert cache_key("anthropic", "m1", req) != cache_key("anthropic", "m2", req)
    assert cache_key("anthropic", "m1", req) != cache_key("ollama", "m1", req)
    assert cache_key("anthropic", "m1", req) == cache_key("anthropic", "m1", LLMRequest("q"))


def test_call_cap_stops_new_calls_but_allows_cache_hits(tmp_path: Path) -> None:
    provider = FakeProvider()
    client = make_client(tmp_path, provider, max_calls_per_run=2)
    client.complete(LLMRequest(prompt="a"))
    client.complete(LLMRequest(prompt="b"))
    with pytest.raises(CallCapExceeded):
        client.complete(LLMRequest(prompt="c"))
    assert provider.calls == 2
    assert client.complete(LLMRequest(prompt="a")).cached  # cached answers are still free


def test_dry_run_math(tmp_path: Path) -> None:
    client = make_client(tmp_path, FakeProvider(), chars_per_token=4.0)
    client.complete(LLMRequest(prompt="x" * 40))  # now cached
    requests = [
        LLMRequest(prompt="x" * 40),  # cache hit
        LLMRequest(prompt="y" * 400),  # 100 tokens
        LLMRequest(prompt="y" * 400),  # duplicate within the batch -> free
        LLMRequest(prompt="z" * 398, system="ab"),  # 400 chars -> 100 tokens
    ]
    est = client.dry_run(requests, expected_output_tokens=50)

    assert est.n_requests == 4
    assert est.n_cache_hits == 2
    assert est.n_calls == 2
    assert est.est_input_tokens == 200
    assert est.est_output_tokens == 100
    # 200 * $1/1M + 100 * $5/1M = 0.0002 + 0.0005
    assert est.est_cost_usd == pytest.approx(0.0007)


def test_dry_run_defaults_to_max_tokens_for_output(tmp_path: Path) -> None:
    client = make_client(tmp_path, FakeProvider())
    est = client.dry_run([LLMRequest(prompt="p", max_tokens=300)])
    assert est.est_output_tokens == 300


def test_dry_run_rejects_unpriced_model(tmp_path: Path) -> None:
    client = make_client(tmp_path, FakeProvider(), pricing_usd_per_mtok={})
    with pytest.raises(KeyError):
        client.dry_run([LLMRequest(prompt="p")])


def test_dry_run_makes_no_calls(tmp_path: Path) -> None:
    provider = FakeProvider()
    make_client(tmp_path, provider).dry_run([LLMRequest(prompt=str(i)) for i in range(10)])
    assert provider.calls == 0


def test_confirm_requires_typed_yes(tmp_path: Path) -> None:
    client = make_client(tmp_path, FakeProvider(), require_confirmation=True)
    est = client.dry_run([LLMRequest(prompt="p")])
    assert client.confirm(est, input_fn=lambda _: "yes")
    assert not client.confirm(est, input_fn=lambda _: "y")
    assert not client.confirm(est, input_fn=lambda _: "")


def test_confirm_refuses_batch_over_call_cap(tmp_path: Path) -> None:
    client = make_client(tmp_path, FakeProvider(), max_calls_per_run=2)
    est = client.dry_run([LLMRequest(prompt=str(i)) for i in range(3)])
    assert not client.confirm(est, input_fn=lambda _: "yes")


def test_retries_with_backoff_then_succeeds(tmp_path: Path) -> None:
    delays: list[float] = []
    provider = FakeProvider(fail_times=2)
    client = make_client(tmp_path, provider, sleep=delays.append, retry_base_delay_s=1.0)
    resp = client.complete(LLMRequest(prompt="p"))
    assert resp.text == "echo: p"
    assert provider.calls == 3
    assert client.calls_made == 1  # retries of one request count as one call
    assert 1.0 <= delays[0] < 2.0 and 2.0 <= delays[1] < 3.0  # exponential + jitter


def test_retries_give_up_after_max(tmp_path: Path) -> None:
    provider = FakeProvider(fail_times=10)
    client = make_client(tmp_path, provider, max_retries=2)
    with pytest.raises(RetryableError):
        client.complete(LLMRequest(prompt="p"))
    assert provider.calls == 3  # 1 try + 2 retries


def test_anthropic_kwargs_send_temperature_via_extra_body() -> None:
    import inspect

    from anthropic.resources.messages import Messages

    from arl.llm import anthropic_create_kwargs

    kwargs = anthropic_create_kwargs("m", LLMRequest(prompt="p", system="s", temperature=0.0))
    assert "temperature" not in kwargs  # anthropic>=1.0 raises TypeError for it
    assert kwargs["extra_body"] == {"temperature": 0.0}
    assert kwargs["system"] == "s"
    accepted = set(inspect.signature(Messages.create).parameters)
    assert set(kwargs) <= accepted  # every key is a real SDK parameter


class FakeClock:
    """A clock that only moves when `sleep` is called (or when advanced manually)."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_throttle_spaces_requests(tmp_path: Path) -> None:
    clock = FakeClock()
    client = make_client(
        tmp_path, FakeProvider(), min_interval_s=1.5, sleep=clock.sleep, clock=clock
    )
    client.complete(LLMRequest(prompt="a"))  # first request: no wait
    clock.now += 0.5  # the "API call" took 0.5 s
    client.complete(LLMRequest(prompt="b"))  # must wait the remaining 1.0 s
    client.complete(LLMRequest(prompt="a"))  # cache hit: no throttle
    assert clock.sleeps == [pytest.approx(1.0)]


def test_retry_respects_server_retry_after(tmp_path: Path) -> None:
    class RetryAfterProvider(FakeProvider):
        def generate(self, request: LLMRequest) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                raise RetryableError("RateLimitError (429): slow down", retry_after=30.0)
            return super().generate(request)

    delays: list[float] = []
    client = make_client(tmp_path, RetryAfterProvider(), sleep=delays.append)
    client.complete(LLMRequest(prompt="p"))
    assert delays == [30.0]  # server's 30 s beats our 1-2 s backoff


def test_retry_log_names_the_error_type(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    client = make_client(tmp_path, FakeProvider(fail_times=1))
    client.complete(LLMRequest(prompt="p"))
    assert "429 rate limited" in capsys.readouterr().out


def test_describe_includes_type_and_status() -> None:
    from arl.llm import _describe

    class FakeStatusError(Exception):
        status_code = 429

    assert _describe(FakeStatusError("too many")).startswith("FakeStatusError (429): too many")
    assert _describe(ValueError("x")) == "ValueError: x"
