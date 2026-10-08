"""Provider-agnostic LLM client with a SQLite cache, dry-run cost estimates, and a call cap.

Every LLM call in the project goes through `LLMClient`:

    client = LLMClient.from_config(config)
    requests = [LLMRequest(prompt=p) for p in prompts]
    if client.confirm(client.dry_run(requests, expected_output_tokens=80)):
        responses = [client.complete(r) for r in requests]

A cache hit never reaches the provider, so re-running a script is free.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from arl.config import PROJECT_ROOT


class CallCapExceeded(RuntimeError):
    """Raised when a run would exceed `llm.max_calls_per_run` real API calls."""


class RetryableError(RuntimeError):
    """A provider error worth retrying (rate limit, server error, network).

    `retry_after` is the server's requested wait in seconds, when it sends one.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(frozen=True)
class LLMRequest:
    """One prompt plus the parameters that change the answer (all part of the cache key)."""

    prompt: str
    system: str | None = None
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass(frozen=True)
class LLMResponse:
    """The model's text plus token usage as reported by the provider."""

    text: str
    input_tokens: int
    output_tokens: int
    cached: bool = False


@dataclass(frozen=True)
class DryRunEstimate:
    """What a batch would cost. Cache hits and in-batch duplicates are free."""

    n_requests: int
    n_cache_hits: int
    n_calls: int
    est_input_tokens: int
    est_output_tokens: int
    est_cost_usd: float
    model: str

    def summary(self) -> str:
        """Human-readable one-block summary for the terminal."""
        return (
            f"Dry run ({self.model}): {self.n_requests} requests, "
            f"{self.n_cache_hits} cached/duplicate -> {self.n_calls} API calls\n"
            f"  est. input tokens:  {self.est_input_tokens:,}\n"
            f"  est. output tokens: {self.est_output_tokens:,}\n"
            f"  est. cost:          ${self.est_cost_usd:.4f}"
        )


class Provider(Protocol):
    """Anything that can turn a request into a response. Tests pass a fake one."""

    name: str
    model: str

    def generate(self, request: LLMRequest) -> LLMResponse: ...


def anthropic_create_kwargs(model: str, request: LLMRequest) -> dict[str, Any]:
    """Arguments for `messages.create()`.

    anthropic>=1.0 removed `temperature` from the SDK signature (a TypeError), but older
    models such as Haiku 4.5 still accept it in the API, so it goes in `extra_body`.
    Newer models (Opus 4.7+, Haiku 5.5, Sonnet 5+) reject or restrict it: revisit if the
    model changes.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": request.max_tokens,
        "messages": [{"role": "user", "content": request.prompt}],
        "extra_body": {"temperature": request.temperature},
    }
    if request.system:
        kwargs["system"] = request.system
    return kwargs


def _describe(exc: Exception) -> str:
    """Short 'ErrorType (status): message' string for retry logs."""
    status = getattr(exc, "status_code", None)
    head = f"{type(exc).__name__} ({status})" if status else type(exc).__name__
    return f"{head}: {str(exc)[:150]}"


class AnthropicProvider:
    """Calls the Anthropic Messages API. Reads ANTHROPIC_API_KEY from `.env`."""

    name = "anthropic"

    def __init__(self, model: str) -> None:
        import anthropic
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        self.model = model
        self._anthropic = anthropic
        # SDK retries are off: LLMClient owns retries so they are visible and testable.
        self._client = anthropic.Anthropic(max_retries=0)
        # Account limits from the first response's headers, e.g. {"requests-limit": "50"}.
        self.rate_limits: dict[str, str] | None = None

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Send one request; translate retryable SDK errors into `RetryableError`."""
        err = self._anthropic
        kwargs = anthropic_create_kwargs(self.model, request)
        try:
            raw = self._client.messages.with_raw_response.create(**kwargs)
        except err.APIConnectionError as exc:
            raise RetryableError(_describe(exc)) from exc
        except err.APIStatusError as exc:
            if exc.status_code == 429 or exc.status_code >= 500:
                retry_after = exc.response.headers.get("retry-after")
                raise RetryableError(
                    _describe(exc), retry_after=float(retry_after) if retry_after else None
                ) from exc
            raise
        if self.rate_limits is None:
            self.rate_limits = {
                k.removeprefix("anthropic-ratelimit-"): v
                for k, v in raw.headers.items()
                if k.startswith("anthropic-ratelimit-") and k.endswith("-limit")
            }
        message = raw.parse()
        text = "".join(block.text for block in message.content if block.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )


class OllamaProvider:
    """Placeholder for a free local model via Ollama (not implemented yet)."""

    name = "ollama"

    def __init__(self, model: str) -> None:
        self.model = model

    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("Ollama support is stubbed; set llm.provider: anthropic")


class LLMCache:
    """SQLite key-value store: request hash -> response text and token usage."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache ("
            " key TEXT PRIMARY KEY, provider TEXT, model TEXT, request_json TEXT,"
            " text TEXT, input_tokens INTEGER, output_tokens INTEGER, created_at REAL)"
        )
        self._conn.commit()

    def get(self, key: str) -> LLMResponse | None:
        """Return the cached response for `key`, or None."""
        row = self._conn.execute(
            "SELECT text, input_tokens, output_tokens FROM llm_cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return LLMResponse(text=row[0], input_tokens=row[1], output_tokens=row[2], cached=True)

    def contains(self, key: str) -> bool:
        """True if `key` is cached."""
        return (
            self._conn.execute("SELECT 1 FROM llm_cache WHERE key = ?", (key,)).fetchone()
            is not None
        )

    def put(
        self, key: str, provider: str, model: str, request: LLMRequest, resp: LLMResponse
    ) -> None:
        """Store a response (committed immediately so a crash loses at most one call)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                provider,
                model,
                json.dumps(asdict(request), sort_keys=True),
                resp.text,
                resp.input_tokens,
                resp.output_tokens,
                time.time(),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


def cache_key(provider: str, model: str, request: LLMRequest) -> str:
    """SHA-256 of provider + model + every request field, with sorted keys for stability."""
    payload = {"provider": provider, "model": model, **asdict(request)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def estimate_tokens(text: str, chars_per_token: float) -> int:
    """Rough offline token count (no API call). Rounds up so estimates err high."""
    return math.ceil(len(text) / chars_per_token)


class LLMClient:
    """Cached, capped, retrying gateway to an LLM provider."""

    def __init__(
        self,
        provider: Provider,
        cache: LLMCache,
        *,
        max_calls_per_run: int,
        pricing_usd_per_mtok: dict[str, dict[str, float]],
        require_confirmation: bool = True,
        max_retries: int = 5,
        retry_base_delay_s: float = 1.0,
        chars_per_token: float = 3.5,
        default_max_tokens: int = 512,
        default_temperature: float = 0.0,
        min_interval_s: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.max_calls_per_run = max_calls_per_run
        self.pricing = pricing_usd_per_mtok
        self.require_confirmation = require_confirmation
        self.max_retries = max_retries
        self.retry_base_delay_s = retry_base_delay_s
        self.chars_per_token = chars_per_token
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.min_interval_s = min_interval_s
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None
        self.calls_made = 0  # real API calls this run (cache hits excluded)
        self.input_tokens_used = 0
        self.output_tokens_used = 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LLMClient:
        """Build the client described by the `llm:` section of the config."""
        llm = config["llm"]
        if llm["provider"] == "anthropic":
            provider: Provider = AnthropicProvider(llm["model"])
        elif llm["provider"] == "ollama":
            provider = OllamaProvider(llm["ollama_model"])
        else:
            raise ValueError(f"Unknown llm.provider: {llm['provider']}")
        return cls(
            provider,
            LLMCache(PROJECT_ROOT / llm["cache_path"]),
            max_calls_per_run=llm["max_calls_per_run"],
            pricing_usd_per_mtok=llm.get("pricing_usd_per_mtok", {}),
            require_confirmation=llm.get("require_confirmation", True),
            max_retries=llm.get("max_retries", 5),
            retry_base_delay_s=llm.get("retry_base_delay_s", 1.0),
            chars_per_token=llm.get("chars_per_token", 3.5),
            default_max_tokens=llm.get("max_tokens", 512),
            default_temperature=llm.get("temperature", 0.0),
            min_interval_s=llm.get("min_interval_s", 0.0),
        )

    def request(self, prompt: str, system: str | None = None) -> LLMRequest:
        """Make an `LLMRequest` with the configured max_tokens and temperature."""
        return LLMRequest(
            prompt=prompt,
            system=system,
            max_tokens=self.default_max_tokens,
            temperature=self.default_temperature,
        )

    def _key(self, request: LLMRequest) -> str:
        return cache_key(self.provider.name, self.provider.model, request)

    def dry_run(
        self, requests: Iterable[LLMRequest], expected_output_tokens: int | None = None
    ) -> DryRunEstimate:
        """Estimate calls, tokens, and cost for a batch without calling the API.

        Requests already cached, or repeated within the batch, are not counted as calls.
        Output tokens default to each request's `max_tokens` (a worst case) unless the
        caller gives a typical `expected_output_tokens`.
        """
        if self.provider.model not in self.pricing:
            raise KeyError(
                f"No pricing for {self.provider.model} in llm.pricing_usd_per_mtok; "
                "add it (verify on the console) before estimating cost"
            )
        price = self.pricing[self.provider.model]
        seen: set[str] = set()
        n_requests = n_hits = in_tok = out_tok = 0
        for req in requests:
            n_requests += 1
            key = self._key(req)
            if key in seen or self.cache.contains(key):
                n_hits += 1
                continue
            seen.add(key)
            in_tok += estimate_tokens((req.system or "") + req.prompt, self.chars_per_token)
            out_tok += (
                expected_output_tokens if expected_output_tokens is not None else req.max_tokens
            )
        cost = in_tok / 1e6 * price["input"] + out_tok / 1e6 * price["output"]
        return DryRunEstimate(
            n_requests=n_requests,
            n_cache_hits=n_hits,
            n_calls=n_requests - n_hits,
            est_input_tokens=in_tok,
            est_output_tokens=out_tok,
            est_cost_usd=cost,
            model=self.provider.model,
        )

    def confirm(self, estimate: DryRunEstimate, input_fn: Callable[[str], str] = input) -> bool:
        """Print the estimate; if confirmation is required, the user must type 'yes'.

        Also refuses a batch that would exceed the remaining call budget.
        """
        print(estimate.summary())
        remaining = self.max_calls_per_run - self.calls_made
        if estimate.n_calls > remaining:
            print(f"Refusing: {estimate.n_calls} calls > {remaining} left under max_calls_per_run")
            return False
        if estimate.n_calls == 0 or not self.require_confirmation:
            return True
        return input_fn("Type 'yes' to spend this: ").strip().lower() == "yes"

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the cached response, or call the provider (with retries) and cache it."""
        key = self._key(request)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if self.calls_made >= self.max_calls_per_run:
            raise CallCapExceeded(f"Reached max_calls_per_run={self.max_calls_per_run}")
        self.calls_made += 1
        response = self._generate_with_retries(request)
        self.input_tokens_used += response.input_tokens
        self.output_tokens_used += response.output_tokens
        self.cache.put(key, self.provider.name, self.provider.model, request, response)
        return response

    def _throttle(self) -> None:
        """Wait so request starts are at least `min_interval_s` apart (client-side rate limit)."""
        if self._last_request_at is not None:
            wait = self._last_request_at + self.min_interval_s - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._clock()

    def _generate_with_retries(self, request: LLMRequest) -> LLMResponse:
        """Exponential backoff with jitter: base * 2**attempt + U(0, 1) seconds.

        If the server sends `retry-after`, we wait at least that long.
        """
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                return self.provider.generate(request)
            except RetryableError as err:
                if attempt == self.max_retries:
                    raise
                delay = self.retry_base_delay_s * 2**attempt + random.uniform(0, 1)
                delay = max(delay, err.retry_after or 0.0)
                print(f"  {err}; waiting {delay:.1f}s (retry {attempt + 1})")
                self._sleep(delay)
        raise AssertionError("unreachable")

    def rate_limit_summary(self) -> str | None:
        """The account rate limits the provider reported, if it reports them."""
        limits = getattr(self.provider, "rate_limits", None)
        if not limits:
            return None
        return "Account rate limits: " + ", ".join(f"{k}={v}" for k, v in sorted(limits.items()))

    def actual_cost_usd(self) -> float:
        """Cost of the real calls made so far, from provider-reported token usage."""
        price = self.pricing[self.provider.model]
        return (
            self.input_tokens_used / 1e6 * price["input"]
            + self.output_tokens_used / 1e6 * price["output"]
        )
