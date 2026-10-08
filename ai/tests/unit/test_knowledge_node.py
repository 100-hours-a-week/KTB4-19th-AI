import pytest
from qdrant_client import models

import zipsai.knowledge.node as node_module
from zipsai.contracts.converse import ConverseRequest, ImageAnalysis, ImageObservation
from zipsai.knowledge.node import (
    EMPTY_QUESTION_REPLY,
    NO_EVIDENCE_REPLY,
    SNIPPET_LENGTH,
    handle_knowledge,
)
from zipsai.settings import EMBEDDING_DIM


def _request(
    text: str | None = "세탁실은 몇 시까지 쓸 수 있나요?",
    *,
    history: list[dict[str, object]] | None = None,
    images: list[dict[str, object]] | None = None,
    current_route: str = "knowledge",
) -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 1,
            "room_no": "101",
            "resident_id": "r-1",
            "conversation_id": "c-1",
            "turn_id": "t-1",
            "trace_id": "trace-1",
            "current_route": current_route,
            "current_complaint_state": None,
            "message": {
                "message_id": "m-1",
                "text": text,
                "images": images or [],
            },
            "conversation_history": history or [],
            "complaint_draft": None,
        }
    )


def _chunk(
    text: str, *, doc_id: str = "doc-laundry", title: str = "세탁실 이용"
) -> models.ScoredPoint:
    return models.ScoredPoint(
        id=1,
        version=0,
        score=0.9,
        payload={"doc_id": doc_id, "title": title, "text": text},
    )


class Spy:
    """각 단계가 몇 번 불렸는지 센다. 근거가 없을 때 건너뛰는지 보려면 필요하다."""

    def __init__(self) -> None:
        self.encode_calls = 0
        self.search_calls = 0
        self.generate_calls = 0
        self.encoded_questions: list[str] = []
        self.prompt_inputs: list[dict[str, object]] = []


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> Spy:
    counter = Spy()

    def encode(_question: str, *, encoder: object, trace_id: str):
        counter.encode_calls += 1
        counter.encoded_questions.append(_question)
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
    def generate(**kwargs: object) -> str:
        spy.generate_calls += 1
        spy.prompt_inputs.append(kwargs)
        return answer

    monkeypatch.setattr(node_module, "generate_text", generate)


def _stub_llm_sequence(
    monkeypatch: pytest.MonkeyPatch, spy: Spy, outputs: list[str]
) -> None:
    remaining = iter(outputs)

    def generate(**kwargs: object) -> str:
        spy.generate_calls += 1
        spy.prompt_inputs.append(kwargs)
        return next(remaining)

    monkeypatch.setattr(node_module, "generate_text", generate)


def test_answers_from_retrieved_evidence(
    monkeypatch: pytest.MonkeyPatch, spy: Spy
) -> None:
    _stub_search(monkeypatch, spy, [_chunk("세탁실은 22시까지입니다.")])
    _stub_llm(monkeypatch, spy, "22시까지 이용할 수 있습니다.")

    result = handle_knowledge(_request())
    route_result = result["result"]

    assert result["reply"] == "22시까지 이용할 수 있습니다."
    assert result["complaint_state"] is None
    assert route_result.has_sufficient_evidence is True
    assert route_result.qa_card_draft is None
    # 완료 조건이 "문서 제목 + 본문 일부"를 요구한다.
    assert [c.title for c in route_result.citations] == ["세탁실 이용"]
    assert route_result.citations[0].snippet == "세탁실은 22시까지입니다."
    assert route_result.citations[0].source_type == "building_document"


def test_folds_citations_of_the_same_document(
    monkeypatch: pytest.MonkeyPatch, spy: Spy
) -> None:
    # 한 문서에서 청크가 여럿 걸린다. 같은 문서가 근거 목록에 반복되면 안 된다.
    _stub_search(
        monkeypatch,
        spy,
        [_chunk("세탁실은 22시까지입니다."), _chunk("세탁기는 4대입니다.")],
    )
    _stub_llm(monkeypatch, spy, "22시까지입니다.")

    citations = handle_knowledge(_request())["result"].citations

    assert [c.source_id for c in citations] == ["doc-laundry"]


def test_keeps_separate_documents_that_share_a_title(
    monkeypatch: pytest.MonkeyPatch, spy: Spy
) -> None:
    # "공지사항"처럼 제목이 겹치는 문서가 실제로 있다. 제목으로 접으면 근거가 사라진다.
    _stub_search(
        monkeypatch,
        spy,
        [
            _chunk("9월 점검", doc_id="doc-09", title="공지사항"),
            _chunk("10월 점검", doc_id="doc-10", title="공지사항"),
        ],
    )
    _stub_llm(monkeypatch, spy, "점검 일정입니다.")

    citations = handle_knowledge(_request())["result"].citations

    assert [c.source_id for c in citations] == ["doc-09", "doc-10"]


def test_cuts_a_long_chunk_down_to_a_snippet(
    monkeypatch: pytest.MonkeyPatch, spy: Spy
) -> None:
    _stub_search(monkeypatch, spy, [_chunk("가" * 400)])
    _stub_llm(monkeypatch, spy, "답변")

    snippet = handle_knowledge(_request())["result"].citations[0].snippet

    assert len(snippet) == SNIPPET_LENGTH


def test_skips_the_model_when_search_finds_nothing(
    monkeypatch: pytest.MonkeyPatch, spy: Spy
) -> None:
    # 실패가 성공보다 싸야 한다. 근거가 0건이면 LLM 요금이 나가면 안 된다.
    _stub_search(monkeypatch, spy, [])
    _stub_llm(monkeypatch, spy, "이 답변은 나오면 안 된다")

    result = handle_knowledge(_request())
    route_result = result["result"]

    assert result["reply"] == NO_EVIDENCE_REPLY
    assert spy.generate_calls == 0
    assert route_result.has_sufficient_evidence is False
    assert route_result.citations == []
    assert route_result.qa_card_draft.question == "세탁실은 몇 시까지 쓸 수 있나요?"


def test_replaces_the_internal_token_with_the_fallback_reply(
    monkeypatch: pytest.MonkeyPatch, spy: Spy
) -> None:
    _stub_search(monkeypatch, spy, [_chunk("주차는 2대까지입니다.")])
    _stub_llm(monkeypatch, spy, "  no_evidence\n")

    result = handle_knowledge(_request())

    assert result["reply"] == NO_EVIDENCE_REPLY
    assert result["result"].has_sufficient_evidence is False
    # 모델이 거절했어도 질문은 관리자에게 넘어가야 한다.
    assert result["result"].qa_card_draft.question == "세탁실은 몇 시까지 쓸 수 있나요?"


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
    assert result["result"].qa_card_draft is None
    assert result["result"].has_sufficient_evidence is False
    assert spy.encode_calls == 0
    assert spy.search_calls == 0
    assert spy.generate_calls == 0


def test_uses_new_image_analysis_as_search_context(
    monkeypatch: pytest.MonkeyPatch, spy: Spy
) -> None:
    _stub_search(
        monkeypatch,
        spy,
        [_chunk("음식물 쓰레기는 전용 수거함에 버립니다.", title="음식물 배출 안내")],
    )
    _stub_llm_sequence(
        monkeypatch,
        spy,
        ["돼지껍데기는 음식물 쓰레기로 어디에 버리나요?", "음식물 쓰레기 전용 수거함에 버리면 됩니다."],
    )
    analysis = ImageAnalysis(
        images=[
            ImageObservation(
                attachmentId=123, summary="돼지껍데기 조각", ocrText="음식물"
            )
        ]
    )

    result = handle_knowledge(
        _request(
            "이거 어디다 버려요?",
            history=[
                {
                    "message_id": "old-image",
                    "role": "user",
                    "text": "건조기가 멈췄어요.",
                    "images": [
                        {"attachmentId": 9, "summary": "건조기 화면 E3", "ocrText": "E3"}
                    ],
                }
            ],
            images=[{"attachmentId": 123, "url": "https://example.com/photo.jpg"}],
        ),
        image_analysis=analysis,
    )

    assert spy.encoded_questions == [
        "돼지껍데기는 음식물 쓰레기로 어디에 버리나요?"
    ]
    rewrite_prompt = str(spy.prompt_inputs[0]["user_prompt"])
    assert "질문: 이거 어디다 버려요?" in rewrite_prompt
    assert "사진 분석 맥락: 돼지껍데기 조각; 음식물" in rewrite_prompt
    assert "건조기 화면 E3" not in rewrite_prompt
    answer_prompt = str(spy.prompt_inputs[1]["user_prompt"])
    assert "질문: 돼지껍데기는 음식물 쓰레기로 어디에 버리나요?" in answer_prompt
    assert "사진 분석 맥락:" not in answer_prompt
    assert result["reply"] == "음식물 쓰레기 전용 수거함에 버리면 됩니다."


def test_uses_prior_user_question_for_photo_only_followup(
    monkeypatch: pytest.MonkeyPatch, spy: Spy
) -> None:
    _stub_search(monkeypatch, spy, [])
    _stub_llm_sequence(
        monkeypatch, spy, ["돼지껍데기는 어디에 버리나요?"]
    )
    history = [
        {"message_id": "m1", "role": "user", "text": "돼지껍데기는 어디 버려요?"},
        {"message_id": "m2", "role": "assistant", "text": "사진을 보내주시겠어요?"},
    ]
    analysis = ImageAnalysis(
        images=[ImageObservation(attachmentId=123, summary="돼지껍데기", ocrText=None)]
    )

    result = handle_knowledge(
        _request(
            None,
            history=history,
            images=[{"attachmentId": 123, "url": "https://example.com/photo.jpg"}],
        ),
        image_analysis=analysis,
    )

    assert spy.encoded_questions == [
        "돼지껍데기는 어디에 버리나요?"
    ]
    assert result["result"].qa_card_draft.question == spy.encoded_questions[0]


def test_uses_images_from_the_latest_user_turn_for_a_later_question(
    monkeypatch: pytest.MonkeyPatch, spy: Spy
) -> None:
    _stub_search(monkeypatch, spy, [])
    _stub_llm_sequence(monkeypatch, spy, ["건조기 E3 오류는 무엇을 뜻하나요?"])
    history = [
        {
            "message_id": "m1",
            "role": "user",
            "text": "이거 어디 버려요?",
            "images": [{"attachmentId": 1, "summary": "돼지껍데기"}],
        },
        {"message_id": "m2", "role": "assistant", "text": "음식물로 배출하세요."},
        {
            "message_id": "m3",
            "role": "user",
            "text": "건조기가 멈췄어요.",
            "images": [{"attachmentId": 2, "summary": "건조기 화면 E3", "ocrText": "E3"}],
        },
        {"message_id": "m4", "role": "assistant", "text": "어떤 점이 궁금하신가요?"},
    ]

    result = handle_knowledge(
        _request("이 오류가 무슨 뜻이에요?", history=history, current_route="clarify")
    )

    assert spy.encoded_questions == ["건조기 E3 오류는 무엇을 뜻하나요?"]
    rewrite_prompt = str(spy.prompt_inputs[0]["user_prompt"])
    assert "사진 첨부 때 발화: 건조기가 멈췄어요." in rewrite_prompt
    assert "건조기 화면 E3; E3" in rewrite_prompt
    assert "돼지껍데기" not in rewrite_prompt
    assert result["result"].qa_card_draft.question == spy.encoded_questions[0]


def test_ignores_older_images_when_the_latest_user_turn_has_no_image(
    monkeypatch: pytest.MonkeyPatch, spy: Spy
) -> None:
    _stub_search(monkeypatch, spy, [])
    history = [
        {
            "message_id": "m1",
            "role": "user",
            "text": "이거 어디 버려요?",
            "images": [{"attachmentId": 1, "summary": "돼지껍데기"}],
        },
        {"message_id": "m2", "role": "assistant", "text": "음식물로 배출하세요."},
        {"message_id": "m3", "role": "user", "text": "주차 등록은 어디서 해요?"},
    ]

    result = handle_knowledge(_request("등록 방법을 알려주세요.", history=history))

    assert spy.encoded_questions == ["등록 방법을 알려주세요."]
    assert spy.generate_calls == 0
    assert result["result"].qa_card_draft.question == "등록 방법을 알려주세요."


def test_does_not_use_history_images_without_a_clarification_followup(
    monkeypatch: pytest.MonkeyPatch, spy: Spy
) -> None:
    _stub_search(monkeypatch, spy, [])
    history = [
        {
            "message_id": "m1",
            "role": "user",
            "text": "이거 어디 버려요?",
            "images": [{"attachmentId": 1, "summary": "돼지껍데기"}],
        },
        {"message_id": "m2", "role": "assistant", "text": "음식물로 배출하세요."},
    ]

    result = handle_knowledge(
        _request("주차 등록은 어디서 해요?", history=history, current_route="knowledge")
    )

    assert spy.encoded_questions == ["주차 등록은 어디서 해요?"]
    assert spy.generate_calls == 0
    assert result["result"].qa_card_draft.question == "주차 등록은 어디서 해요?"
