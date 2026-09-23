import pytest
from qdrant_client import models

import zipsai.knowledge.node as node_module
from zipsai.contracts.converse import ConverseRequest
from zipsai.knowledge.node import (
    EMPTY_QUESTION_REPLY,
    NO_EVIDENCE_REPLY,
    handle_knowledge,
)
from zipsai.settings import EMBEDDING_DIM


def _request(text: str | None = "세탁실은 몇 시까지 쓸 수 있나요?") -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 1,
            "room_no": "101",
            "resident_id": "r-1",
            "conversation_id": "c-1",
            "trace_id": "t-1",
            "current_route": "knowledge",
            "current_complaint_state": None,
            "message": {
                "message_id": "m-1",
                "text": text,
                "image_urls": [] if text else ["https://example.com/a.jpg"],
            },
            "conversation_history": [],
            "complaint_draft": None,
        }
    )


def _chunk(text: str) -> models.ScoredPoint:
    return models.ScoredPoint(
        id=1, version=0, score=0.9, payload={"title": "세탁실 이용", "text": text}
    )


class Spy:
    """각 단계가 몇 번 불렸는지 센다. 근거가 없을 때 건너뛰는지 보려면 필요하다."""

    def __init__(self) -> None:
        self.encode_calls = 0
        self.search_calls = 0
        self.generate_calls = 0


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> Spy:
    counter = Spy()

    def encode(_question: str, *, encoder: object):
        counter.encode_calls += 1
        return [0.0] * EMBEDDING_DIM, {"7": 0.5}

    monkeypatch.setattr(node_module, "get_client", lambda: object())
    monkeypatch.setattr(node_module, "query_encoder", lambda: object())
    monkeypatch.setattr(node_module, "encode_question", encode)
    return counter


def _stub_search(
    monkeypatch: pytest.MonkeyPatch, spy: Spy, chunks: list[models.ScoredPoint]
) -> None:
    def search(_vector: object, _building_id: int, **_kwargs: object):
        spy.search_calls += 1
        return chunks

    monkeypatch.setattr(node_module, "search_chunks", search)


def _stub_llm(monkeypatch: pytest.MonkeyPatch, spy: Spy, answer: str) -> None:
    def generate(**_kwargs: object) -> str:
        spy.generate_calls += 1
        return answer

    monkeypatch.setattr(node_module, "generate_text", generate)


def test_answers_from_retrieved_evidence(
    monkeypatch: pytest.MonkeyPatch, spy: Spy
) -> None:
    _stub_search(monkeypatch, spy, [_chunk("세탁실은 22시까지입니다.")])
    _stub_llm(monkeypatch, spy, "22시까지 이용할 수 있습니다.")

    result = handle_knowledge(_request())

    assert result["reply"] == "22시까지 이용할 수 있습니다."
    assert result["complaint_state"] is None


def test_skips_the_model_when_search_finds_nothing(
    monkeypatch: pytest.MonkeyPatch, spy: Spy
) -> None:
    # 실패가 성공보다 싸야 한다. 근거가 0건이면 LLM 요금이 나가면 안 된다.
    _stub_search(monkeypatch, spy, [])
    _stub_llm(monkeypatch, spy, "이 답변은 나오면 안 된다")

    result = handle_knowledge(_request())

    assert result["reply"] == NO_EVIDENCE_REPLY
    assert spy.generate_calls == 0


def test_replaces_the_internal_token_with_the_fallback_reply(
    monkeypatch: pytest.MonkeyPatch, spy: Spy
) -> None:
    _stub_search(monkeypatch, spy, [_chunk("주차는 2대까지입니다.")])
    _stub_llm(monkeypatch, spy, "  no_evidence\n")

    assert handle_knowledge(_request())["reply"] == NO_EVIDENCE_REPLY


def test_keeps_an_answer_that_merely_contains_the_token(
    monkeypatch: pytest.MonkeyPatch, spy: Spy
) -> None:
    # 부분 문자열로 판정하면 건물 문서에 그 단어가 있는 정상 답변이 회피로 뒤집힌다.
    _stub_search(monkeypatch, spy, [_chunk("와이파이 비밀번호는 no_evidence 입니다.")])
    _stub_llm(monkeypatch, spy, "비밀번호는 no_evidence 입니다.")

    assert handle_knowledge(_request())["reply"] == "비밀번호는 no_evidence 입니다."


def test_asks_for_the_question_instead_of_filing_an_empty_card(
    monkeypatch: pytest.MonkeyPatch, spy: Spy
) -> None:
    # 근거 없음 문구는 "질문을 관리자에게 전달했다"고 말한다. 전달할 질문이 없으면 빈 카드가 된다.
    _stub_search(monkeypatch, spy, [_chunk("아무거나")])
    _stub_llm(monkeypatch, spy, "이 답변은 나오면 안 된다")

    result = handle_knowledge(_request(text=None))

    assert result["reply"] == EMPTY_QUESTION_REPLY
    assert result["reply"] != NO_EVIDENCE_REPLY
    assert spy.encode_calls == 0
    assert spy.search_calls == 0
    assert spy.generate_calls == 0
