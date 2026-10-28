import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from prometheus_client import REGISTRY
from pydantic import BaseModel

from zipsai.errors import (
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)
from zipsai.integrations import llm as llm_module
from zipsai.observability import JsonFormatter, collect_timings, llm_cost_total
from zipsai.settings import Settings, call_cost_usd, get_settings


class _FakeCompletions:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def create(self, **_kwargs: object) -> None:
        raise self.error


class _FakeClient:
    def __init__(self, error: Exception) -> None:
        self.chat = type("_Chat", (), {"completions": _FakeCompletions(error)})()


def _use_fake_client(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    monkeypatch.setattr(llm_module, "_get_client", lambda *a, **k: _FakeClient(error))
    monkeypatch.setattr(
        llm_module, "get_settings", lambda: Settings("key", None, "model", 30)
    )


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _response(
    status_code: int, headers: dict[str, str] | None = None
) -> httpx.Response:
    return httpx.Response(status_code, request=_request(), headers=headers or {})


def test_rate_limit_error_maps_to_llm_rate_limited_with_retry_after(monkeypatch):
    error = RateLimitError(
        "rate limited", response=_response(429, {"retry-after": "7"}), body=None
    )
    _use_fake_client(monkeypatch, error)

    with pytest.raises(LlmRateLimitedError) as exc_info:
        llm_module.generate_text("system", "user")

    assert exc_info.value.retry_after_seconds == 7


def test_timeout_error_maps_to_llm_timeout(monkeypatch):
    _use_fake_client(monkeypatch, APITimeoutError(_request()))

    with pytest.raises(LlmTimeoutError):
        llm_module.generate_text("system", "user")


def test_upstream_5xx_maps_to_llm_upstream_error(monkeypatch):
    error = APIStatusError("bad gateway", response=_response(502), body=None)
    _use_fake_client(monkeypatch, error)

    with pytest.raises(LlmUpstreamError):
        llm_module.generate_text("system", "user")


def test_client_side_status_error_maps_to_llm_unavailable(monkeypatch):
    error = APIStatusError("bad request", response=_response(400), body=None)
    _use_fake_client(monkeypatch, error)

    with pytest.raises(LlmUnavailableError):
        llm_module.generate_text("system", "user")


def test_connection_error_maps_to_llm_unavailable(monkeypatch):
    _use_fake_client(monkeypatch, APIConnectionError(request=_request()))

    with pytest.raises(LlmUnavailableError):
        llm_module.generate_text("system", "user")


def test_pod_502_retries_text_once_on_openrouter(monkeypatch, caplog):
    pod = Settings("pod-key", "https://pod.example/v1", "pod-model", 30, provider="pod")
    backup = Settings(
        "router-key", "https://router.example/v1", "router-model", 30,
        provider="openrouter",
    )
    calls = []
    failed_urls = []

    def create_client(key, *_args):
        def create(**kwargs):
            calls.append((key, kwargs["model"]))
            if key == "pod-key":
                raise APIStatusError("bad gateway", response=_response(502), body=None)
            return _success_response("ok")

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(llm_module, "get_settings", lambda provider=None: backup if provider else pod)
    monkeypatch.setattr(llm_module, "_get_client", create_client)
    monkeypatch.setattr(llm_module, "mark_pod_unavailable", failed_urls.append)

    with caplog.at_level(logging.INFO, logger="zipsai.integrations.llm"):
        assert llm_module.generate_text("system", "user") == "ok"

    assert calls == [("pod-key", "pod-model"), ("router-key", "router-model")]
    assert failed_urls == [pod.llm_base_url]
    assert _llm_calls(caplog)[-1].provider == "openrouter"
    assert _llm_calls(caplog)[-1].attempt == 2


def test_pod_400_does_not_retry_on_openrouter(monkeypatch):
    pod = Settings("pod-key", "https://pod.example/v1", "pod-model", 30, provider="pod")
    calls = []

    def create_client(key, *_args):
        calls.append(key)
        return _FakeClient(APIStatusError("bad request", response=_response(400), body=None))

    monkeypatch.setattr(llm_module, "get_settings", lambda provider=None: pod)
    monkeypatch.setattr(llm_module, "_get_client", create_client)
    with pytest.raises(LlmUnavailableError):
        llm_module.generate_text("system", "user")
    assert calls == ["pod-key"]


def test_pod_connection_error_retries_structured_on_openrouter(monkeypatch):
    class Answer(BaseModel):
        value: str

    pod = Settings("pod-key", "https://pod.example/v1", "pod-model", 30, provider="pod")
    backup = Settings("router-key", "https://router.example/v1", "router-model", 30, provider="openrouter")
    calls = []

    def client(key, *_args):
        def parse(**kwargs):
            calls.append((key, kwargs["model"]))
            if key == "pod-key":
                raise APIConnectionError(request=_request())
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=Answer(value="ok"), refusal=None))],
                usage=None,
            )

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))

    monkeypatch.setattr(llm_module, "get_settings", lambda provider=None: backup if provider else pod)
    monkeypatch.setattr(llm_module, "_get_client", client)
    monkeypatch.setattr(llm_module, "mark_pod_unavailable", lambda *_: None)

    assert llm_module.generate_structured("system", "user", Answer) == Answer(value="ok")
    assert calls == [("pod-key", "pod-model"), ("router-key", "router-model")]


def test_usage_sink_receives_token_counts(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=usage,
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    monkeypatch.setattr(llm_module, "_get_client", lambda *a, **k: fake_client)
    monkeypatch.setattr(
        llm_module, "get_settings", lambda: Settings("key", None, "model", 30)
    )

    sink: dict[str, object] = {}
    result = llm_module.generate_text("system", "user", usage_sink=sink)

    assert result == "ok"
    assert sink == {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}


def test_usage_sink_includes_reasoning_tokens_when_model_reports_them(monkeypatch):
    # reasoning 모델은 effort를 명시하지 않아도 completion_tokens 안에 보이지 않는
    # reasoning 분량을 섞어 과금한다 — 그 분량을 별도 필드로 복원할 수 있어야 한다.
    usage = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=50,
        total_tokens=62,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=38),
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=usage,
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    monkeypatch.setattr(llm_module, "_get_client", lambda *a, **k: fake_client)
    monkeypatch.setattr(
        llm_module, "get_settings", lambda: Settings("key", None, "model", 30)
    )

    sink: dict[str, object] = {}
    llm_module.generate_text("system", "user", usage_sink=sink)

    assert sink == {
        "input_tokens": 12,
        "output_tokens": 50,
        "total_tokens": 62,
        "reasoning_tokens": 38,
    }


def test_usage_sink_includes_cached_tokens_when_model_reports_them(monkeypatch):
    # 캐시 적중 분량은 prompt_tokens 안에 섞여 들어오고, 적중분은 과금이 싸다 —
    # 교체한 모델의 캐시 적중률을 실측하려면 별도 필드로 복원해야 한다.
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=3,
        total_tokens=103,
        prompt_tokens_details=SimpleNamespace(cached_tokens=80),
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=usage,
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    monkeypatch.setattr(llm_module, "_get_client", lambda *a, **k: fake_client)
    monkeypatch.setattr(
        llm_module, "get_settings", lambda: Settings("key", None, "model", 30)
    )

    sink: dict[str, object] = {}
    llm_module.generate_text("system", "user", usage_sink=sink)

    assert sink == {
        "input_tokens": 100,
        "output_tokens": 3,
        "total_tokens": 103,
        "cached_tokens": 80,
    }


def test_usage_sink_left_untouched_without_sink(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=usage,
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    monkeypatch.setattr(llm_module, "_get_client", lambda *a, **k: fake_client)
    monkeypatch.setattr(
        llm_module, "get_settings", lambda: Settings("key", None, "model", 30)
    )

    # usage_sink를 안 넘기는 기존 호출부는 그대로 동작해야 한다.
    assert llm_module.generate_text("system", "user") == "ok"


class _AsyncStream:
    def __init__(self, chunks: list[object]) -> None:
        self.chunks = chunks

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)


class _AsyncCompletions:
    def __init__(self, chunks: list[object]) -> None:
        self.chunks = chunks
        self.kwargs: dict[str, object] | None = None

    async def create(self, **kwargs: object):
        self.kwargs = kwargs
        return _AsyncStream(self.chunks)


def test_generate_text_stream_yields_only_non_empty_text_chunks(monkeypatch):
    completions = _AsyncCompletions(
        [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="안녕"))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]),
            SimpleNamespace(choices=[]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="하세요"))]),
        ]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    monkeypatch.setattr(llm_module, "_get_async_client", lambda *a, **k: fake_client)
    monkeypatch.setattr(
        llm_module, "get_settings", lambda: Settings("key", None, "model", 30)
    )

    async def collect():
        return [
            chunk
            async for chunk in llm_module.generate_text_stream("system", "user")
        ]

    chunks = asyncio.run(collect())

    assert chunks == ["안녕", "하세요"]
    assert completions.kwargs is not None
    assert completions.kwargs["stream_options"] == {"include_usage": True}


def test_stream_retries_on_openrouter_before_first_text(monkeypatch):
    pod = Settings("pod-key", "https://pod.example/v1", "pod-model", 30, provider="pod")
    backup = Settings("router-key", "https://router.example/v1", "router-model", 30, provider="openrouter")
    calls = []

    def client(key, *_args):
        async def create(**kwargs):
            calls.append((key, kwargs["model"]))
            if key == "pod-key":
                raise APITimeoutError(_request())
            return _AsyncStream([SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="backup"))],
                usage=None,
            )])

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(llm_module, "get_settings", lambda provider=None: backup if provider else pod)
    monkeypatch.setattr(llm_module, "_get_async_client", client)
    monkeypatch.setattr(llm_module, "mark_pod_unavailable", lambda *_: None)

    async def collect():
        return [part async for part in llm_module.generate_text_stream("system", "user")]

    assert asyncio.run(collect()) == ["backup"]
    assert calls == [("pod-key", "pod-model"), ("router-key", "router-model")]


def test_stream_does_not_retry_after_partial_text(monkeypatch):
    pod = Settings("pod-key", "https://pod.example/v1", "pod-model", 30, provider="pod")
    calls = []
    failed_urls = []

    class BrokenStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if not calls:
                calls.append("first chunk")
                return SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="partial"))],
                    usage=None,
                )
            raise APIStatusError("bad gateway", response=_response(502), body=None)

    async def create(**_kwargs):
        return BrokenStream()

    monkeypatch.setattr(llm_module, "get_settings", lambda provider=None: pod)
    monkeypatch.setattr(llm_module, "_get_async_client", lambda *_: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    ))
    monkeypatch.setattr(llm_module, "mark_pod_unavailable", failed_urls.append)

    async def collect():
        return [part async for part in llm_module.generate_text_stream("system", "user")]

    with pytest.raises(LlmUpstreamError):
        asyncio.run(collect())
    assert calls == ["first chunk"]
    assert failed_urls == [pod.llm_base_url]


def test_stream_reads_usage_from_the_trailing_chunk(monkeypatch, caplog):
    usage = SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6)
    completions = _AsyncCompletions(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="답"))],
                usage=None,
            ),
            SimpleNamespace(choices=[], usage=usage),
        ]
    )
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(llm_module, "_get_async_client", lambda *a, **k: fake_client)
    monkeypatch.setattr(
        llm_module, "get_settings", lambda: Settings("key", None, "model", 30)
    )
    sink: dict[str, object] = {}

    async def collect() -> list[str]:
        return [
            chunk
            async for chunk in llm_module.generate_text_stream(
                "system", "user", usage_sink=sink
            )
        ]

    with caplog.at_level(logging.INFO, logger="zipsai.integrations.llm"):
        assert asyncio.run(collect()) == ["답"]

    recorded = _llm_calls(caplog)
    assert sink["input_tokens"] == 4
    assert sink["output_tokens"] == 2
    assert recorded[-1].input_tokens == 4
    assert recorded[-1].output_tokens == 2
    assert not hasattr(recorded[-1], "usage_missing")


def test_stream_without_usage_marks_tokens_missing(monkeypatch, caplog):
    completions = _AsyncCompletions(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="답"))],
                usage=None,
            )
        ]
    )
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(llm_module, "_get_async_client", lambda *a, **k: fake_client)
    monkeypatch.setattr(
        llm_module, "get_settings", lambda: Settings("key", None, "model", 30)
    )

    async def collect() -> list[str]:
        return [
            chunk
            async for chunk in llm_module.generate_text_stream("system", "user")
        ]

    with caplog.at_level(logging.INFO, logger="zipsai.integrations.llm"):
        assert asyncio.run(collect()) == ["답"]

    recorded = _llm_calls(caplog)
    assert recorded[-1].usage_missing is True
    assert recorded[-1].input_tokens is None
    assert recorded[-1].output_tokens is None


def _duration_count(outcome: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "zipsai_llm_call_duration_seconds_count",
            {"purpose": "unknown", "model": "model", "outcome": outcome},
        )
        or 0.0
    )


def _llm_calls(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.message == "llm_call"]


def test_unknown_exception_is_logged_as_failure(monkeypatch, caplog):
    class BoomError(RuntimeError):
        pass

    _use_fake_client(monkeypatch, BoomError("주민 원문"))
    before = _duration_count("fail")

    with (
        caplog.at_level(logging.INFO, logger="zipsai.integrations.llm"),
        pytest.raises(BoomError),
    ):
        llm_module.generate_text("system", "user")

    recorded = _llm_calls(caplog)
    assert recorded[-1].outcome == "fail"
    assert recorded[-1].status_code is None
    assert recorded[-1].error_code == "BoomError"
    assert "주민 원문" not in recorded[-1].getMessage()
    assert _duration_count("fail") == before + 1


def test_closed_stream_is_logged_as_cancelled(monkeypatch, caplog):
    completions = _AsyncCompletions(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="앞"))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="뒤"))]
            ),
        ]
    )
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(llm_module, "_get_async_client", lambda *a, **k: fake_client)
    monkeypatch.setattr(
        llm_module, "get_settings", lambda: Settings("key", None, "model", 30)
    )
    before = _duration_count("cancelled")

    async def close_early() -> None:
        stream = llm_module.generate_text_stream("system", "user")
        await anext(stream)
        await stream.aclose()

    with caplog.at_level(logging.INFO, logger="zipsai.integrations.llm"):
        asyncio.run(close_early())

    recorded = _llm_calls(caplog)
    assert recorded[-1].outcome == "cancelled"
    assert recorded[-1].status_code is None
    assert _duration_count("cancelled") == before + 1


def test_successful_call_is_logged_as_ok(monkeypatch, caplog):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kwargs: response))
    )
    monkeypatch.setattr(llm_module, "_get_client", lambda *a, **k: fake_client)
    monkeypatch.setattr(
        llm_module, "get_settings", lambda: Settings("key", None, "model", 30)
    )
    before = _duration_count("ok")

    with caplog.at_level(logging.INFO, logger="zipsai.integrations.llm"):
        assert llm_module.generate_text("system", "user") == "ok"

    recorded = _llm_calls(caplog)
    assert recorded[-1].outcome == "ok"
    assert recorded[-1].status_code == 200
    assert recorded[-1].error_code is None
    assert recorded[-1].cost_usd is None
    assert _duration_count("ok") == before + 1


def test_cost_uses_a_price_pair_and_keeps_cached_tokens() -> None:
    tokens = {"input": 1_000_000, "output": 2_000_000, "cached": 500_000}
    priced = Settings(
        "key",
        None,
        "model",
        30,
        llm_price_input_per_1m=3.0,
        llm_price_output_per_1m=6.0,
    )
    assert call_cost_usd(priced, "turn_reply", tokens) == 15.0

    one_side = Settings("key", None, "model", 30, llm_price_input_per_1m=3.0)
    assert call_cost_usd(one_side, "turn_reply", tokens) is None

    vlm = Settings(
        "key",
        None,
        "model",
        30,
        llm_price_input_per_1m=9.0,
        llm_price_output_per_1m=9.0,
        vlm_price_input_per_1m=1.0,
        vlm_price_output_per_1m=2.0,
    )
    image_tokens = {"input": 1_000_000, "output": 1_000_000, "cached": 1_000_000}
    assert call_cost_usd(vlm, "image_analysis", image_tokens) == 3.0
    assert call_cost_usd(vlm, "turn_reply", {"input": 1_000_000, "output": 0}) == 9.0


def test_blank_price_env_stays_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_MODEL", "model")
    monkeypatch.setenv("LLM_PRICE_INPUT_PER_1M", " ")
    monkeypatch.delenv("LLM_PRICE_OUTPUT_PER_1M", raising=False)
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.llm_price_input_per_1m is None
        assert settings.llm_price_output_per_1m is None
    finally:
        get_settings.cache_clear()


def test_invalid_price_env_is_unset_and_settings_still_load(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    bad = "not-a-number"
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_MODEL", "model")
    monkeypatch.setenv("LLM_PRICE_INPUT_PER_1M", bad)
    monkeypatch.setenv("LLM_PRICE_OUTPUT_PER_1M", "2.5")
    get_settings.cache_clear()
    try:
        with caplog.at_level(logging.WARNING, logger="zipsai.settings"):
            settings = get_settings()
        assert settings.llm_price_input_per_1m is None
        assert settings.llm_price_output_per_1m == 2.5
        warnings = [
            record
            for record in caplog.records
            if record.getMessage() == "invalid_price_setting"
        ]
        assert len(warnings) == 1
        assert warnings[0].setting == "LLM_PRICE_INPUT_PER_1M"
        line = JsonFormatter().format(warnings[0])
        assert bad not in line
        assert "LLM_PRICE_INPUT_PER_1M" in line
    finally:
        get_settings.cache_clear()


def test_priced_call_logs_cost_and_adds_it_to_the_request(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: Settings(
            "key",
            None,
            "model",
            30,
            llm_price_input_per_1m=1.0,
            llm_price_output_per_1m=2.0,
        ),
    )
    call = llm_module.LlmCall(
        purpose="turn_reply",
        model="model",
        base_url=None,
        streaming=False,
        started=0.0,
        structured=False,
        usage=SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000),
    )
    labels = {"purpose": "turn_reply", "model": "model"}
    before = REGISTRY.get_sample_value("zipsai_llm_cost_usd_total", labels) or 0.0

    with (
        collect_timings(),
        caplog.at_level(logging.INFO, logger="zipsai.integrations.llm"),
    ):
        call.finish(outcome="ok", status_code=200, output_text="ok")
        assert llm_cost_total() == 3.0

    assert _llm_calls(caplog)[-1].cost_usd == 3.0
    assert (REGISTRY.get_sample_value("zipsai_llm_cost_usd_total", labels) or 0.0) == (
        before + 3.0
    )


def test_one_price_does_not_increment_the_cost_counter(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: Settings("key", None, "model", 30, llm_price_input_per_1m=1.0),
    )
    call = llm_module.LlmCall(
        purpose="turn_reply",
        model="model",
        base_url=None,
        streaming=False,
        started=0.0,
        structured=False,
        usage=SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000),
    )
    labels = {"purpose": "turn_reply", "model": "model"}
    before = REGISTRY.get_sample_value("zipsai_llm_cost_usd_total", labels) or 0.0

    with (
        collect_timings(),
        caplog.at_level(logging.INFO, logger="zipsai.integrations.llm"),
    ):
        call.finish(outcome="ok", status_code=200, output_text="ok")
        assert llm_cost_total() is None

    assert _llm_calls(caplog)[-1].cost_usd is None
    assert (REGISTRY.get_sample_value("zipsai_llm_cost_usd_total", labels) or 0.0) == before


_LLM_CALL_FIELDS = (
    "purpose",
    "provider_host",
    "model",
    "attempt",
    "streaming",
    "latency_ms",
    "ttft_ms",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "finish_reason",
    "status_code",
    "error_code",
    "structured_output_ok",
    "retry_reason",
    "image_count",
    "image_pixels_total",
    "cost_usd",
    "think_tag_present",
    "hangul_ratio",
)
_SYSTEM_SECRET = "SYSTEM_SECRET_TEXT"
_USER_SECRET = "USER_SECRET_TEXT"
_RESPONSE_SECRET = "RESPONSE_SECRET_TEXT"


def _success_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=5),
    )


def test_successful_llm_call_line_has_the_documented_fields(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    response = _success_response(_RESPONSE_SECRET)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    monkeypatch.setattr(llm_module, "_get_client", lambda *a, **k: fake_client)
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: Settings("key", "https://api.example.com/v1", "model", 30),
    )

    with caplog.at_level(logging.INFO, logger="zipsai.integrations.llm"):
        assert (
            llm_module.generate_text(_SYSTEM_SECRET, _USER_SECRET) == _RESPONSE_SECRET
        )

    line = JsonFormatter().format(_llm_calls(caplog)[-1])
    payload = json.loads(line)
    for name in _LLM_CALL_FIELDS:
        assert name in payload
    assert payload["provider_host"] == "api.example.com"
    assert payload["status_code"] == 200
    assert payload["image_pixels_total"] is None
    assert _SYSTEM_SECRET not in line
    assert _USER_SECRET not in line
    assert _RESPONSE_SECRET not in line


@pytest.mark.parametrize(
    ("error", "status_code", "error_code"),
    [
        (
            RateLimitError("rate limited", response=_response(429), body=None),
            429,
            "MODEL_RATE_LIMITED",
        ),
        (APITimeoutError(_request()), None, "MODEL_TIMEOUT"),
        (
            APIStatusError("upstream", response=_response(500), body=None),
            500,
            "MODEL_UPSTREAM_ERROR",
        ),
        (
            APIStatusError("bad request", response=_response(400), body=None),
            400,
            "DEPENDENCY_NOT_READY",
        ),
    ],
)
def test_http_errors_log_status_and_error_code(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    status_code: int | None,
    error_code: str,
) -> None:
    _use_fake_client(monkeypatch, error)

    with (
        caplog.at_level(logging.INFO, logger="zipsai.integrations.llm"),
        pytest.raises(
            (LlmRateLimitedError, LlmTimeoutError, LlmUpstreamError, LlmUnavailableError)
        ),
    ):
        llm_module.generate_text(_SYSTEM_SECRET, _USER_SECRET)

    record = _llm_calls(caplog)[-1]
    assert record.status_code == status_code
    assert record.error_code == error_code
    line = JsonFormatter().format(record)
    assert _SYSTEM_SECRET not in line
    assert _USER_SECRET not in line


def test_runpod_provider_host_keeps_only_the_hostname(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    endpoint = "endpoint-SHOULD-NOT-LEAK"
    base_url = f"https://api.runpod.ai/v2/{endpoint}/openai/v1"
    assert llm_module.provider_host(base_url) == "api.runpod.ai"
    response = _success_response(_RESPONSE_SECRET)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    monkeypatch.setattr(llm_module, "_get_client", lambda *a, **k: fake_client)
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: Settings("key", base_url, "model", 30),
    )

    with caplog.at_level(logging.INFO, logger="zipsai.integrations.llm"):
        llm_module.generate_text(_SYSTEM_SECRET, _USER_SECRET)

    line = JsonFormatter().format(_llm_calls(caplog)[-1])
    assert json.loads(line)["provider_host"] == "api.runpod.ai"
    assert endpoint not in line
    assert _RESPONSE_SECRET not in line
