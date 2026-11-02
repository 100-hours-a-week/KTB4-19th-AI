import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError, OpenAI

from zipsai.errors import LlmUnavailableError
from zipsai.integrations import llm
from zipsai.orchestration.tool_contracts import TOOL_CONTRACTS
from zipsai.settings import Settings


def _response(content=None, tool_calls=None, finish_reason=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason or ("tool_calls" if tool_calls else "stop"),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7),
    )


def test_tool_decision_replays_assistant_and_tool_result(monkeypatch):
    calls = []
    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(
            name="search_building_docs", arguments='{"query":"주차"}'
        ),
    )
    responses = iter(
        [_response(tool_calls=[tool_call]), _response(content="주차 규정입니다")]
    )

    def create(**kwargs):
        calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return next(responses)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(llm, "_get_client", lambda *a, **k: client)
    monkeypatch.setattr(
        llm, "get_settings", lambda provider=None: Settings("key", None, "model", 30)
    )
    specs = [contract.model_spec() for contract in TOOL_CONTRACTS]
    messages = [{"role": "user", "content": "주차 규정 알려줘"}]

    first = llm.generate_tool_decision(messages, specs)
    assert first.content is None
    assert first.tool_calls == (
        llm.ToolCall("call_1", "search_building_docs", {"query": "주차"}),
    )

    messages.extend(
        [
            first.message,
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status":"ok"}'},
        ]
    )
    second = llm.generate_tool_decision(messages, specs)
    assert second.content == "주차 규정입니다"
    assert second.tool_calls == ()
    assert calls[0]["tools"] == calls[1]["tools"] == specs
    assert len(calls[0]["messages"]) == 1
    assert calls[1]["messages"][-2] is first.message
    assert calls[1]["messages"][-1]["tool_call_id"] == "call_1"


def test_tool_decision_serializes_sdk_message_on_follow_up(monkeypatch):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_building_docs",
                            "arguments": '{"query":"주차"}',
                        },
                    }
                ],
            }
            reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": "주차 규정입니다"}
            reason = "stop"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "model",
                "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7,
                },
            },
        )

    client = OpenAI(
        api_key="key",
        base_url="https://mock.example/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(llm, "_get_client", lambda *a, **k: client)
    monkeypatch.setattr(
        llm, "get_settings", lambda provider=None: Settings("key", None, "model", 30)
    )
    monkeypatch.setattr(llm, "get_tracing_client", lambda: None)
    messages = [{"role": "user", "content": "주차 규정 알려줘"}]
    tools = [TOOL_CONTRACTS[0].model_spec()]

    first = llm.generate_tool_decision(messages, tools)
    messages.extend(
        [
            first.message,
            {
                "role": "tool",
                "tool_call_id": first.tool_calls[0].id,
                "content": '{"status":"ok"}',
            },
        ]
    )
    assert llm.generate_tool_decision(messages, tools).content == "주차 규정입니다"
    assert requests[1]["messages"][1]["tool_calls"][0]["id"] == "call_1"
    assert requests[1]["messages"][2]["tool_call_id"] == "call_1"


def test_tool_decision_falls_back_with_same_messages_and_tools(monkeypatch, caplog):
    pod = Settings("pod-key", "https://pod.example/v1", "pod-model", 30, provider="pod")
    backup = Settings(
        "router-key",
        "https://router.example/v1",
        "router-model",
        30,
        provider="openrouter",
    )
    attempts = []

    def client(key, *_args):
        def create(**kwargs):
            attempts.append((key, kwargs))
            if key == "pod-key":
                response = httpx.Response(
                    502,
                    request=httpx.Request(
                        "POST", "https://pod.example/v1/chat/completions"
                    ),
                )
                raise APIStatusError("bad gateway", response=response, body=None)
            return _response(content="완료")

        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

    monkeypatch.setattr(llm, "_get_client", client)
    monkeypatch.setattr(
        llm, "get_settings", lambda provider=None: backup if provider else pod
    )
    monkeypatch.setattr(llm, "mark_pod_unavailable", lambda _url: None)
    messages = [{"role": "user", "content": "질문"}]
    specs = [TOOL_CONTRACTS[0].model_spec()]
    sink = {}

    with caplog.at_level(logging.INFO, logger="zipsai.integrations.llm"):
        result = llm.generate_tool_decision(messages, specs, usage_sink=sink)

    assert result.content == "완료"
    assert [item[1]["model"] for item in attempts] == ["pod-model", "router-model"]
    assert all(
        item[1]["messages"] is messages and item[1]["tools"] is specs
        for item in attempts
    )
    assert sink == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
    record = [record for record in caplog.records if record.message == "llm_call"][-1]
    assert record.provider == "openrouter"
    assert record.attempt == 2


@pytest.mark.parametrize("arguments", ["not-json", "[]"])
def test_tool_decision_rejects_invalid_arguments(monkeypatch, arguments):
    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="search_building_docs", arguments=arguments),
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: _response(tool_calls=[tool_call])
            )
        )
    )
    monkeypatch.setattr(llm, "_get_client", lambda *a, **k: client)
    monkeypatch.setattr(
        llm, "get_settings", lambda provider=None: Settings("key", None, "model", 30)
    )

    with pytest.raises(LlmUnavailableError, match="invalid tool arguments"):
        llm.generate_tool_decision(
            [{"role": "user", "content": "질문"}], [TOOL_CONTRACTS[0].model_spec()]
        )


def test_tool_decision_rejects_truncated_response(monkeypatch):
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: _response(
                    content="미완성 답변", finish_reason="length"
                )
            )
        )
    )
    monkeypatch.setattr(llm, "_get_client", lambda *a, **k: client)
    monkeypatch.setattr(
        llm, "get_settings", lambda provider=None: Settings("key", None, "model", 30)
    )

    with pytest.raises(LlmUnavailableError, match="incomplete response"):
        llm.generate_tool_decision(
            [{"role": "user", "content": "질문"}], [TOOL_CONTRACTS[0].model_spec()]
        )


def test_tool_decision_rejects_non_function_call(monkeypatch):
    tool_call = SimpleNamespace(
        id="call_1",
        type="other",
        function=SimpleNamespace(name="search_building_docs", arguments="{}"),
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: _response(tool_calls=[tool_call])
            )
        )
    )
    monkeypatch.setattr(llm, "_get_client", lambda *a, **k: client)
    monkeypatch.setattr(
        llm, "get_settings", lambda provider=None: Settings("key", None, "model", 30)
    )

    with pytest.raises(LlmUnavailableError, match="invalid tool arguments"):
        llm.generate_tool_decision(
            [{"role": "user", "content": "질문"}], [TOOL_CONTRACTS[0].model_spec()]
        )
