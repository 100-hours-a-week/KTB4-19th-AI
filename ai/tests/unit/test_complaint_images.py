import pytest

import zipsai.complaint.node as node_module
from zipsai.complaint.node import _TurnInterpretation, handle_complaint
from zipsai.contracts.converse import (
    ComplaintDraft,
    ComplaintState,
    ConverseRequest,
    ImageAnalysis,
    ImageObservation,
)
from zipsai.errors import EmbeddingError


@pytest.fixture(autouse=True)
def _stub_interpreter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        node_module,
        "_interpret_with_stage",
        lambda _request, _image_analysis=None: _TurnInterpretation(),
    )


def _request(
    *,
    text: str | None,
    images: list[dict[str, object]] | None = None,
    history: list[dict[str, object]] | None = None,
    draft: ComplaintDraft | None = None,
) -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 1,
            "room_no": "301",
            "resident_id": "linda",
            "conversation_id": "conv-001",
            "turn_id": "turn-001",
            "trace_id": "trace-001",
            "current_route": "complaint",
            "current_complaint_state": ComplaintState.COLLECTING,
            "message": {
                "message_id": "msg-current",
                "text": text,
                "images": images or [],
            },
            "conversation_history": history or [],
            "complaint_draft": draft,
        }
    )


def test_photo_only_uses_intent_summary_and_keeps_attachment_id():
    analysis = ImageAnalysis(
        images=[
            ImageObservation(
                attachmentId=123,
                summary="세탁기 아래 물이 고임",
                ocrText="모델명 ABC",
            )
        ]
    )

    result = handle_complaint(
        _request(
            text=None,
            images=[
                {"attachmentId": 123, "url": "https://example.com/washer.png"}
            ],
        ),
        image_analysis=analysis,
    )["result"]

    assert result.complaint_draft.symptom == "세탁기 아래 물이 고임"
    assert result.complaint_draft.location is None
    assert result.complaint_draft.attachment_ids == [123]
    assert result.image_analysis == analysis


def test_photo_ocr_is_passed_to_complaint_extraction_for_symptom(
    monkeypatch: pytest.MonkeyPatch,
):
    analysis = ImageAnalysis(
        images=[
            ImageObservation(
                attachmentId=123,
                summary="세탁기 조작부가 보임",
                ocrText="LE",
            )
        ]
    )
    received: list[ImageAnalysis | None] = []

    def interpret(_request, image_analysis=None):
        received.append(image_analysis)
        return _TurnInterpretation(
            fields=ComplaintDraft(symptom="세탁기에 LE 오류가 표시됨")
        )

    monkeypatch.setattr(node_module, "_interpret_with_stage", interpret)
    monkeypatch.setattr(
        node_module,
        "_generate_turn_finalization",
        lambda *_args: node_module._TurnFinalization(
            symptom="세탁기에 LE 오류가 표시됨", reply=""
        ),
    )

    result = handle_complaint(
        _request(
            text=None,
            images=[
                {"attachmentId": 123, "url": "https://example.com/washer.png"}
            ],
        ),
        image_analysis=analysis,
    )["result"]

    assert received == [analysis]
    assert result.complaint_draft.symptom == "세탁기에 LE 오류가 표시됨"


def test_complaint_extraction_prompt_receives_summary_and_ocr(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, str] = {}
    analysis = ImageAnalysis(
        images=[
            ImageObservation(
                attachmentId=123,
                summary="세탁기 조작부가 보임",
                ocrText="LE",
            )
        ]
    )

    def capture_prompt(**kwargs):
        captured["user"] = kwargs["user_prompt"]
        captured["system"] = kwargs["system_prompt"]
        return node_module._InterpretationOutput(
            complaint_switch="same",
            issue_type="facility",
            location=None,
            symptom="세탁기에 LE 오류가 표시됨",
            occurred_at=None,
        )

    monkeypatch.setattr(node_module, "generate_structured", capture_prompt)
    node_module._interpret_turn(
        _request(
            text=None,
            images=[
                {"attachmentId": 123, "url": "https://example.com/washer.png"}
            ],
        ),
        image_analysis=analysis,
    )

    assert "세탁기 조작부가 보임" in captured["user"]
    assert '"ocrText": "LE"' in captured["user"]
    assert "Never infer location, date, or cause from an image" in captured["system"]


def test_photo_summary_is_acknowledged_while_asking_for_location():
    result = handle_complaint(
        _request(
            text=None,
            images=[
                {"attachmentId": 123, "url": "https://example.com/washer.png"}
            ],
        ),
        image_analysis=ImageAnalysis(
            images=[ImageObservation(attachmentId=123, summary="세탁기 아래 물이 고임")]
        ),
    )

    assert result["result"].missing_fields == ["location"]
    assert result["reply"] == (
        "사진은 확인했습니다 — 세탁기 아래 물이 고임. 어디에서 생긴 문제인가요?"
    )


def test_photo_without_analysis_keeps_attachment_without_fabricating_symptom():
    result = handle_complaint(
        _request(
            text=None,
            images=[
                {"attachmentId": 123, "url": "https://example.com/washer.png"}
            ],
        )
    )["result"]

    assert result.complaint_draft.symptom is None
    assert result.complaint_draft.attachment_ids == [123]
    assert result.image_analysis is None


def test_multiple_images_deduplicate_ids_and_do_not_copy_raw_ocr_to_fallback():
    analysis = ImageAnalysis(
        images=[
            ImageObservation(
                attachmentId=123,
                summary="세탁기 아래 물이 고임",
                ocrText="모델명 ABC",
            ),
            ImageObservation(attachmentId=123, summary="바닥이 젖음"),
            ImageObservation(attachmentId=456, summary="배수구 주변 물기"),
        ]
    )

    result = handle_complaint(
        _request(
            text=None,
            images=[
                {"attachmentId": 123, "url": "https://example.com/washer.png"},
                {"attachmentId": 456, "url": "https://example.com/drain.jpg"},
            ],
        ),
        image_analysis=analysis,
    )["result"]

    assert result.complaint_draft.attachment_ids == [123, 456]
    assert result.complaint_draft.symptom == "세탁기 아래 물이 고임 바닥이 젖음 배수구 주변 물기"
    assert result.image_analysis.images[0].ocr_text == "모델명 ABC"
    assert "모델명 ABC" not in result.complaint_draft.symptom


def test_unanalyzed_attachment_ids_are_retained():
    result = handle_complaint(
        _request(
            text=None,
            images=[
                {"attachmentId": 123, "url": "https://example.com/washer.png"},
                {"attachmentId": 456, "url": "https://example.com/drain.jpg"},
            ],
        ),
        image_analysis=ImageAnalysis(
            images=[ImageObservation(attachmentId=123, summary="세탁기 아래 물이 고임")]
        ),
    )["result"]

    assert result.complaint_draft.attachment_ids == [123, 456]


def test_resident_text_correction_is_used_when_it_disagrees_with_image(
    monkeypatch: pytest.MonkeyPatch,
):
    analysis = ImageAnalysis(
        images=[ImageObservation(attachmentId=123, summary="물이 샘")]
    )
    monkeypatch.setattr(
        node_module,
        "_interpret_with_stage",
        lambda _request, _image_analysis=None: _TurnInterpretation(
            fields=ComplaintDraft(location="욕실", symptom="배수구가 막힘")
        ),
    )
    monkeypatch.setattr(
        node_module,
        "_generate_turn_finalization",
        lambda *_args: node_module._TurnFinalization(
            symptom="욕실 배수구가 막힘", reply=""
        ),
    )

    result = handle_complaint(
        _request(
            text="물이 새는 게 아니라 배수구가 막혔어요",
            images=[
                {"attachmentId": 123, "url": "https://example.com/bathroom.jpg"}
            ],
        ),
        image_analysis=analysis,
    )["result"]

    assert result.complaint_draft.symptom == "욕실 배수구가 막힘"


def test_switch_question_returns_image_analysis_without_attaching_to_old_draft(
    monkeypatch: pytest.MonkeyPatch,
):
    analysis = ImageAnalysis(
        images=[ImageObservation(attachmentId=123, summary="세탁기 아래 물이 고임")]
    )
    draft = ComplaintDraft(
        issue_type="leak",
        location="보일러실",
        symptom="보일러에서 물이 샘",
        attachmentIds=[88],
    )
    monkeypatch.setattr(
        node_module,
        "_interpret_with_stage",
        lambda _request, _image_analysis=None: _TurnInterpretation(
            fields=ComplaintDraft(
                issue_type="leak", symptom="세탁기 아래 물이 고임"
            ),
            switch="ask",
        ),
    )

    result = handle_complaint(
        _request(
            text="세탁기 아래로 물이 새요",
            images=[
                {
                    "attachmentId": 123,
                    "url": "https://example.com/washer.png",
                }
            ],
            draft=draft,
        ),
        image_analysis=analysis,
    )["result"]

    assert result.complaint_draft.attachment_ids == [88]
    assert result.image_analysis == analysis


def test_accepting_switch_attaches_only_the_pending_user_turn_image(
    monkeypatch: pytest.MonkeyPatch,
):
    history = [
        {
            "message_id": "old-photo",
            "role": "user",
            "text": "보일러 사진이에요",
            "images": [
                {"attachmentId": 11, "summary": "보일러"},
            ],
        },
        {
            "message_id": "old-reply",
            "role": "assistant",
            "text": "보일러 사진을 확인했습니다.",
        },
        {
            "message_id": "new-issue",
            "role": "user",
            "text": "세탁기 아래로 물이 새요",
            "images": [
                {"attachmentId": 123, "summary": "세탁기 아래 물이 고임"},
            ],
        },
        {
            "message_id": "switch-question",
            "role": "assistant",
            "text": "보일러 누수 건은 접어두고 「세탁기 아래 물이 고임」으로 전환할까요?",
        },
    ]
    monkeypatch.setattr(
        node_module,
        "_interpret_with_stage",
        lambda _request: _TurnInterpretation(
            fields=ComplaintDraft(symptom="세탁기 아래 물이 고임"), switch="accept"
        ),
    )
    draft = ComplaintDraft(
        issue_type="leak",
        location="보일러실",
        symptom="보일러 누수",
        attachmentIds=[11],
    )

    result = handle_complaint(
        _request(text="네, 그걸로 해주세요", history=history, draft=draft)
    )["result"]

    assert result.complaint_draft.attachment_ids == [123]
    assert result.complaint_draft.symptom == "세탁기 아래 물이 고임"


def test_no_attachment_ids_skips_representative_selection_even_with_history_images(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "query_encoder",
        lambda: pytest.fail("no candidates means no embedding request"),
    )
    request = _request(
        text="세탁기가 고장 났어요.",
        history=[
            {
                "message_id": "old-photo",
                "role": "user",
                "text": "참고 사진",
                "images": [{"attachmentId": 11, "summary": "세탁기"}],
            }
        ],
        draft=ComplaintDraft(
            issue_type="facility", location="세탁실", symptom="세탁기가 고장 남"
        ),
    )

    result = handle_complaint(request)["result"]

    assert result.missing_fields == []
    assert result.complaint_draft.attachment_ids == []
    assert result.complaint_draft.representative_attachment_id is None


def test_single_attachment_is_recommended_without_embedding(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "query_encoder",
        lambda: pytest.fail("a single candidate does not need embedding"),
    )
    request = _request(
        text="세탁기가 고장 났어요.",
        draft=ComplaintDraft(
            issue_type="facility",
            location="세탁실",
            symptom="세탁기가 고장 남",
            attachmentIds=[123],
        ),
    )

    result = handle_complaint(request)["result"]

    assert result.complaint_draft.representative_attachment_id == 123


def test_multiple_attachments_select_highest_cosine_summary_match(
    monkeypatch: pytest.MonkeyPatch,
):
    encoded_texts: list[list[str]] = []

    class FakeEncoder:
        def encode(self, texts: list[str], trace_id: str):
            assert trace_id == "trace-001"
            encoded_texts.append(texts)
            return [
                [1.0, 0.0],
                [0.98, 0.2],
                [0.98, 0.2],
                [0.0, 1.0],
            ], [{}, {}, {}, {}]

    monkeypatch.setattr(node_module, "query_encoder", lambda: FakeEncoder())
    request = _request(
        text="세탁기 아래 누수",
        history=[
            {
                "message_id": "photos",
                "role": "user",
                "text": None,
                "images": [
                    {"attachmentId": 789, "summary": "세탁기 아래 바닥이 젖어 있음"},
                    {"attachmentId": 123, "summary": "세탁기 아래 바닥이 젖어 있음"},
                    {"attachmentId": 456, "summary": "보일러가 보임"},
                    {"attachmentId": 999, "summary": "민원과 무관한 사진"},
                ],
            }
        ],
        draft=ComplaintDraft(
            issue_type="leak",
            location="세탁실",
            symptom="세탁기 아래 누수",
            attachmentIds=[789, 123, 456],
        ),
    )

    result = handle_complaint(request)["result"]

    assert result.complaint_draft.representative_attachment_id == 789
    assert encoded_texts == [
        [
            "leak 세탁실 세탁기 아래 누수",
            "세탁기 아래 바닥이 젖어 있음",
            "세탁기 아래 바닥이 젖어 있음",
            "보일러가 보임",
        ]
    ]


def test_missing_summary_falls_back_to_first_attachment(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "query_encoder",
        lambda: pytest.fail("a missing summary should short-circuit before embedding"),
    )
    request = _request(
        text="누수예요.",
        history=[
            {
                "message_id": "photo-1",
                "role": "user",
                "text": None,
                "images": [{"attachmentId": 123, "summary": "세탁기 아래 물"}],
            },
        ],
        draft=ComplaintDraft(
            issue_type="leak",
            location="세탁실",
            symptom="누수",
            attachmentIds=[123, 456],
        ),
    )

    result = handle_complaint(request)["result"]

    assert result.missing_fields == []
    assert result.complaint_draft.representative_attachment_id == 123


def test_embedding_failure_falls_back_to_first_attachment(
    monkeypatch: pytest.MonkeyPatch,
):
    class FailedEncoder:
        def encode(self, _texts: list[str], _trace_id: str):
            raise EmbeddingError("embedding unavailable")

    monkeypatch.setattr(node_module, "query_encoder", lambda: FailedEncoder())
    request = _request(
        text="누수예요.",
        history=[
            {
                "message_id": "photo-1",
                "role": "user",
                "text": None,
                "images": [{"attachmentId": 123, "summary": "세탁기 아래 물"}],
            },
            {
                "message_id": "photo-2",
                "role": "user",
                "text": None,
                "images": [{"attachmentId": 456, "summary": "배수구 주변 물"}],
            },
        ],
        draft=ComplaintDraft(
            issue_type="leak",
            location="세탁실",
            symptom="누수",
            attachmentIds=[123, 456],
        ),
    )

    result = handle_complaint(request)["result"]

    assert result.missing_fields == []
    assert result.complaint_draft.representative_attachment_id == 123
