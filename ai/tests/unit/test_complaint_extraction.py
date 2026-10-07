import threading
from datetime import datetime

import pytest

import zipsai.complaint.node as node_module
from zipsai import observability
from zipsai.complaint.node import (
    _interpret_turn,
    _TurnInterpretation,
    handle_complaint,
)
from zipsai.contracts.converse import (
    ComplaintDraft,
    ComplaintState,
    ConverseRequest,
    ImageAnalysis,
    ImageObservation,
)
from zipsai.errors import (
    ComplaintExtractionError,
    ImageAnalysisError,
    LlmUnavailableError,
)
from zipsai.observability import bind, collect_timings


def _node_records(caplog: pytest.LogCaptureFixture) -> list:
    """caplog 핸들러는 root에 붙어 다른 로거 기록까지 담는다. 이 모듈 것만 남긴다."""
    return [r for r in caplog.records if r.name == node_module.__name__]


@pytest.fixture(autouse=True)
def _stub_structured_llm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(node_module, "generate_structured", lambda **_: None)


def _make_request(
    text: str | None,
    image_urls: list[str] | None = None,
    current_complaint_state: str = "collecting",
    conversation_history: list[dict[str, object]] | None = None,
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
            "current_complaint_state": current_complaint_state,
            "message": {
                "message_id": "msg-001",
                "text": text,
                "image_urls": image_urls or [],
            },
            "conversation_history": conversation_history or [],
            "complaint_draft": None,
        }
    )


def _interpretation_output(**updates: object) -> node_module._InterpretationOutput:
    return node_module._InterpretationOutput(
        complaint_switch=updates.get("complaint_switch", "same"),
        issue_type=updates.get("issue_type"),
        location=updates.get("location"),
        symptom=updates.get("symptom"),
        occurred_at=updates.get("occurred_at"),
    )


def test_complaint_prompt_carries_formatted_history_without_model_repr(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, str] = {}

    def fake_generate_structured(**kwargs: object):
        captured["user_prompt"] = str(kwargs["user_prompt"])
        return _interpretation_output(location="화장실")

    monkeypatch.setattr(node_module, "generate_structured", fake_generate_structured)

    _interpret_turn(
        _make_request(
            "화장실이요",
            conversation_history=[
                {
                    "message_id": "msg-h1",
                    "role": "user",
                    "text": "물이 새요",
                    "image_urls": [],
                }
            ],
        )
    )

    assert "user: 물이 새요" in captured["user_prompt"]
    assert "msg-h1" not in captured["user_prompt"]


def test_interpret_turn_maps_structured_output_to_current_turn_draft(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "generate_structured",
        lambda **_: _interpretation_output(
            issue_type="leak",
            location="화장실",
            symptom="천장에서 물이 떨어져요",
            occurred_at=datetime(2026, 9, 22),  # noqa: DTZ001
        ),
    )

    result = _interpret_turn(_make_request("화장실 천장에서 물이 계속 떨어져요"))

    assert result.fields == ComplaintDraft(
        issue_type="leak",
        location="화장실",
        symptom="천장에서 물이 떨어져요",
        occurred_at=datetime(2026, 9, 22),  # noqa: DTZ001
    )


def test_interpret_turn_retries_empty_structured_output_once(
    monkeypatch: pytest.MonkeyPatch,
):
    outputs = iter([None, _interpretation_output(location="화장실")])
    monkeypatch.setattr(node_module, "generate_structured", lambda **_: next(outputs))

    result = _interpret_turn(_make_request("화장실이요"))

    assert result.fields.location == "화장실"


def test_interpret_turn_gives_up_after_empty_structured_outputs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    calls: list[int] = []

    def empty_output(**_: object):
        calls.append(1)

    monkeypatch.setattr(node_module, "generate_structured", empty_output)

    with (
        caplog.at_level("INFO", logger=node_module.__name__),
        pytest.raises(ComplaintExtractionError),
    ):
        handle_complaint(_make_request("아무 말"))

    assert len(calls) == node_module._INTERPRETATION_ATTEMPTS
    records = _node_records(caplog)
    retries = [r for r in records if r.getMessage() == "interpretation_retry"]
    assert [(r.attempt, r.error_type) for r in retries] == [
        (1, "ComplaintExtractionError"),
        (2, "ComplaintExtractionError"),
    ]
    failed = [
        r for r in records if r.getMessage() == "stage_done" and r.outcome == "fail"
    ]
    assert [(r.stage, r.error_type) for r in failed] == [
        ("text_interpretation", "ComplaintExtractionError")
    ]


def test_interpret_turn_passes_todays_kst_date_to_prompt(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, str] = {}

    def fake_generate_structured(**kwargs: object):
        captured["user_prompt"] = str(kwargs["user_prompt"])
        return _interpretation_output()

    monkeypatch.setattr(node_module, "generate_structured", fake_generate_structured)

    _interpret_turn(_make_request("어제부터 그랬어요"))

    today = datetime.now(node_module._KST).date().isoformat()
    assert f"오늘 날짜(Asia/Seoul): {today}" in captured["user_prompt"]


def test_handle_complaint_merges_new_values_without_erasing_existing_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    request = _make_request("화장실로 정정할게요.")
    request.complaint_draft = ComplaintDraft(
        issue_type="water_supply",
        location="주방",
        symptom="온수가 나오지 않음",
    )
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(ComplaintDraft(location="화장실")),
    )

    result = handle_complaint(request)["result"]

    assert result.complaint_draft == ComplaintDraft(
        issue_type="water_supply",
        location="화장실",
        symptom="온수가 나오지 않음",
    )
    assert result.missing_fields == []


def test_advance_complaint_merges_collected_evidence_without_model_calls(
    monkeypatch: pytest.MonkeyPatch,
):
    request = _make_request("화장실이요")
    request.complaint_draft = ComplaintDraft(symptom="천장에서 물이 떨어짐")
    evidence = node_module._TurnEvidence(
        interpretation=_TurnInterpretation(ComplaintDraft(location="화장실")),
        image_analysis=None,
    )
    monkeypatch.setattr(
        node_module,
        "generate_structured",
        lambda *_args, **_kwargs: pytest.fail("state update called the text LLM"),
    )
    monkeypatch.setattr(
        node_module,
        "analyze_images",
        lambda *_args, **_kwargs: pytest.fail("state update called the VLM"),
    )

    progress = node_module._advance_complaint(request, evidence)

    assert progress.draft == ComplaintDraft(
        location="화장실",
        symptom="천장에서 물이 떨어짐",
    )
    assert progress.missing_fields == []


def test_handle_complaint_includes_image_summary_in_the_draft(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(ComplaintDraft(location="욕실")),
    )
    monkeypatch.setattr(
        node_module,
        "analyze_images",
        lambda _, __: ImageAnalysis(
            images=[
                ImageObservation(
                    url="https://example.com/leak.jpg",
                    summary="바닥에 물이 고여 있음",
                    ocr_text="E1",
                )
            ]
        ),
        raising=False,
    )

    result = handle_complaint(
        _make_request("욕실 바닥이 젖었어요", ["https://example.com/leak.jpg"])
    )["result"]

    assert result.complaint_draft.location == "욕실"
    assert result.complaint_draft.symptom == "바닥에 물이 고여 있음"
    assert result.complaint_draft.image_urls == ["https://example.com/leak.jpg"]
    assert result.image_analysis.images[0].ocr_text == "E1"


def test_handle_complaint_uses_photo_summary_as_symptom_when_text_has_none(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(ComplaintDraft(location="욕실")),
    )
    monkeypatch.setattr(
        node_module,
        "analyze_images",
        lambda _, __: ImageAnalysis(
            images=[
                ImageObservation(
                    url="https://example.com/leak.jpg",
                    summary="바닥에 물이 고여 있음",
                    ocr_text=None,
                )
            ]
        ),
        raising=False,
    )

    result = handle_complaint(
        _make_request("이거 보세요", ["https://example.com/leak.jpg"])
    )["result"]

    assert result.complaint_draft.symptom == "바닥에 물이 고여 있음"


def test_handle_complaint_merges_text_and_photo_symptoms(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(
            ComplaintDraft(location="욕실", symptom="물이 새요")
        ),
    )
    monkeypatch.setattr(
        node_module,
        "analyze_images",
        lambda _, __: ImageAnalysis(
            images=[
                ImageObservation(
                    url="https://example.com/leak.jpg",
                    summary="천장 모서리에서 물이 떨어짐",
                    ocr_text=None,
                )
            ]
        ),
        raising=False,
    )
    monkeypatch.setattr(
        node_module,
        "_generate_turn_finalization",
        lambda progress, _evidence, _next_action: node_module._TurnFinalization(
            symptom="욕실 천장 모서리에서 물이 떨어짐",
            reply="",
        ),
    )

    result = handle_complaint(
        _make_request("물이 새요", ["https://example.com/leak.jpg"])
    )["result"]

    assert result.complaint_draft.symptom == "욕실 천장 모서리에서 물이 떨어짐"


def test_handle_complaint_keeps_text_flow_when_vlm_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    def raise_image_analysis_error(_: list[str], __: str) -> ImageAnalysis:
        raise ImageAnalysisError("VLM returned an invalid image analysis")

    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(ComplaintDraft(location="욕실")),
    )
    monkeypatch.setattr(
        node_module,
        "analyze_images",
        raise_image_analysis_error,
        raising=False,
    )

    result = handle_complaint(
        _make_request("욕실 바닥이 젖었어요", ["https://example.com/leak.jpg"])
    )["result"]

    assert result.image_analysis is None
    assert result.complaint_draft.image_urls == ["https://example.com/leak.jpg"]


def test_handle_complaint_distinguishes_photo_analysis_failure_from_no_photo(
    monkeypatch: pytest.MonkeyPatch,
):
    def raise_image_analysis_error(_: list[str], __: str) -> ImageAnalysis:
        raise ImageAnalysisError("VLM returned an invalid image analysis")

    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(ComplaintDraft(location="욕실")),
    )
    monkeypatch.setattr(
        node_module, "analyze_images", raise_image_analysis_error, raising=False
    )

    reply = handle_complaint(
        _make_request("욕실 바닥이 젖었어요", ["https://example.com/leak.jpg"])
    )["reply"]

    assert reply == "사진을 받았지만 분석에 실패했어요. 어떤 불편 증상인지 알려주세요."


def test_handle_complaint_acknowledges_photo_when_fields_still_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(ComplaintDraft()),
    )
    monkeypatch.setattr(
        node_module,
        "analyze_images",
        lambda _, __: ImageAnalysis(
            images=[
                ImageObservation(
                    url="https://example.com/leak.jpg",
                    summary="천장에서 물이 흐르는 흔적",
                    ocr_text=None,
                )
            ]
        ),
        raising=False,
    )

    reply = handle_complaint(
        _make_request("이거 보세요", ["https://example.com/leak.jpg"])
    )["reply"]

    assert (
        reply
        == "사진은 확인했습니다 — 천장에서 물이 흐르는 흔적. 어디에서 생긴 문제인가요?"
    )


def test_handle_complaint_uses_generic_reply_without_photo(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(ComplaintDraft()),
    )

    reply = handle_complaint(_make_request("음.."))["reply"]

    assert reply == "어디에서 생긴 문제인가요?"


def test_handle_complaint_uses_llm_generated_reply(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(ComplaintDraft()),
    )
    monkeypatch.setattr(
        node_module,
        "generate_structured",
        lambda **_: node_module._TurnFinalization(
            symptom=None,
            reply="화장실 세면대인지 변기 쪽인지 알려주시겠어요?",
        ),
    )

    reply = handle_complaint(_make_request("화장실이 좀 이상해요"))["reply"]

    assert reply == "화장실 세면대인지 변기 쪽인지 알려주시겠어요?"


def test_handle_complaint_uses_finalizer_to_merge_symptoms(
    monkeypatch: pytest.MonkeyPatch,
):
    request = _make_request("탈수할 때는 LE 오류도 떠요")
    request.complaint_draft = ComplaintDraft(
        issue_type="facility",
        location="세탁실",
        symptom="세탁기가 작동하지 않음",
    )
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: node_module._TurnInterpretation(
            ComplaintDraft(symptom="탈수할 때 LE 오류가 표시됨")
        ),
    )
    monkeypatch.setattr(
        node_module,
        "generate_structured",
        lambda **_: node_module._TurnFinalization(
            symptom="세탁기가 작동하지 않고 탈수할 때 LE 오류가 표시됨",
            reply="",
        ),
    )

    outcome = handle_complaint(request)

    assert outcome["result"].complaint_draft == ComplaintDraft(
        issue_type="facility",
        location="세탁실",
        symptom="세탁기가 작동하지 않고 탈수할 때 LE 오류가 표시됨",
    )
    assert outcome["reply"] == "민원 정보를 확인했습니다. 접수할 내용을 확인해 주세요."


def test_handle_complaint_uses_finalizer_reply_for_the_decided_missing_field(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, str] = {}

    def finalize(**kwargs):
        captured["user_prompt"] = kwargs["user_prompt"]
        return node_module._TurnFinalization(
            symptom=None,
            reply="화장실에서 어떤 불편이 생겼는지 알려주시겠어요?",
        )

    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: node_module._TurnInterpretation(
            ComplaintDraft(location="화장실")
        ),
    )
    monkeypatch.setattr(
        node_module,
        "generate_structured",
        finalize,
    )

    outcome = handle_complaint(_make_request("화장실이요"))

    assert outcome["result"].missing_fields == ["symptom"]
    assert outcome["reply"] == "화장실에서 어떤 불편이 생겼는지 알려주시겠어요?"
    assert "다음 행동: ask_symptom" in captured["user_prompt"]


def test_handle_complaint_does_not_rewrite_a_single_symptom_while_asking_location(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: node_module._TurnInterpretation(
            ComplaintDraft(symptom="물이 새요")
        ),
    )
    monkeypatch.setattr(
        node_module,
        "generate_structured",
        lambda **_: node_module._TurnFinalization(
            symptom="배관 파손으로 물이 새요",
            reply="어디에서 물이 새는지 알려주시겠어요?",
        ),
    )

    outcome = handle_complaint(_make_request("물이 새요"))

    assert outcome["result"].complaint_draft.symptom == "물이 새요"


def test_handle_complaint_falls_back_to_current_symptom_and_fixed_question(
    monkeypatch: pytest.MonkeyPatch,
):
    request = _make_request("이제는 물 대신 소리가 나요")
    request.complaint_draft = ComplaintDraft(symptom="물이 새요")
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: node_module._TurnInterpretation(
            ComplaintDraft(symptom="소리가 남")
        ),
    )

    def unavailable(**_):
        raise LlmUnavailableError("finalizer unavailable")

    monkeypatch.setattr(node_module, "generate_structured", unavailable)

    outcome = handle_complaint(request)

    assert outcome["result"].complaint_draft.symptom == "소리가 남"
    assert outcome["reply"] == "어디에서 생긴 문제인가요?"


def test_handle_complaint_skips_finalizer_for_complete_single_symptom(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: node_module._TurnInterpretation(
            ComplaintDraft(location="화장실", symptom="물이 새요")
        ),
    )
    monkeypatch.setattr(
        node_module,
        "generate_structured",
        lambda **_: pytest.fail("complete turn called the finalizer"),
    )

    outcome = handle_complaint(_make_request("화장실에서 물이 새요"))

    assert outcome["result"].complaint_draft.symptom == "물이 새요"
    assert outcome["reply"] == "민원 정보를 확인했습니다. 접수할 내용을 확인해 주세요."


def test_handle_complaint_prefixes_llm_reply_when_photo_analyzed(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(ComplaintDraft()),
    )
    monkeypatch.setattr(
        node_module,
        "generate_structured",
        lambda **_: node_module._TurnFinalization(
            symptom=None,
            reply="정확히 어디쯤인지 알려주시겠어요?",
        ),
    )
    monkeypatch.setattr(
        node_module,
        "analyze_images",
        lambda _, __: ImageAnalysis(
            images=[
                ImageObservation(
                    url="https://example.com/leak.jpg",
                    summary="바닥에 물이 고여 있음",
                    ocr_text=None,
                )
            ]
        ),
        raising=False,
    )

    reply = handle_complaint(
        _make_request("이거 보세요", ["https://example.com/leak.jpg"])
    )["reply"]

    assert (
        reply
        == "사진은 확인했습니다 — 바닥에 물이 고여 있음. 정확히 어디쯤인지 알려주시겠어요?"
    )


def test_handle_complaint_clears_state_when_fields_complete(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(
            ComplaintDraft(issue_type="leak", location="화장실", symptom="물이 새요"),
        ),
    )

    result = handle_complaint(_make_request("화장실에서 물이 새요"))

    # 상태를 비우고 missing_fields를 빈 배열로 내보내면 백엔드가 민원 카드를 만든다.
    assert result["complaint_state"] is None
    assert result["result"].missing_fields == []


def test_handle_complaint_keeps_collecting_when_fields_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(ComplaintDraft(location="화장실")),
    )

    result = handle_complaint(_make_request("화장실이 이상해요"))

    assert result["complaint_state"] == ComplaintState.COLLECTING


def test_handle_complaint_asks_only_about_missing_symptom(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(ComplaintDraft(location="화장실")),
    )

    reply = handle_complaint(_make_request("화장실이 이상해요"))["reply"]

    assert reply == "어떤 불편 증상인지 알려주세요."


def test_handle_complaint_asks_only_about_missing_location(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(ComplaintDraft(symptom="물이 새요")),
    )

    reply = handle_complaint(_make_request("물이 새요"))["reply"]

    assert reply == "어디에서 생긴 문제인가요?"


def test_handle_complaint_defaults_issue_type_to_other_when_unclassified(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(
            ComplaintDraft(location="화장실", symptom="이상해요"),
        ),
    )

    result = handle_complaint(_make_request("화장실이 이상해요"))

    assert result["complaint_state"] is None
    assert result["result"].complaint_draft.issue_type == "other"


def test_handle_complaint_merges_occurred_at_without_erasing_existing_value(
    monkeypatch: pytest.MonkeyPatch,
):
    request = _make_request("화장실로 정정할게요.")
    request.complaint_draft = ComplaintDraft(
        issue_type="water_supply",
        location="주방",
        symptom="온수가 나오지 않음",
        occurred_at=datetime(2026, 9, 20),  # noqa: DTZ001
    )
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(ComplaintDraft(location="화장실")),
    )

    result = handle_complaint(request)["result"]

    assert result.complaint_draft.occurred_at == datetime(2026, 9, 20)  # noqa: DTZ001


def test_handle_complaint_asks_one_field_at_a_time_when_both_are_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    # 둘을 한 문장으로 같이 물으면 "몰라"가 어느 필드에 대한 답인지 알 수 없다.
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(ComplaintDraft()),
    )

    outcome = handle_complaint(_make_request("좀 이상해요"))

    assert outcome["reply"] == node_module._MISSING_FIELD_REPLY["location"]
    assert "증상" not in outcome["reply"]


def test_handle_complaint_falls_back_to_plain_prefix_when_photo_has_no_summary(
    monkeypatch: pytest.MonkeyPatch,
):
    # VLM이 요약을 못 준 경우까지 사진 내용을 노출하려 들면 빈 문장이 붙는다.
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(ComplaintDraft()),
    )
    monkeypatch.setattr(
        node_module,
        "analyze_images",
        lambda _, __: ImageAnalysis(
            images=[
                ImageObservation(
                    url="https://example.com/blur.jpg", summary=None, ocr_text=None
                )
            ]
        ),
        raising=False,
    )

    reply = handle_complaint(
        _make_request("이것 좀 봐주세요", image_urls=["https://example.com/blur.jpg"])
    )["reply"]

    assert reply == "사진은 확인했습니다. 어디에서 생긴 문제인가요?"


def test_handle_complaint_logs_stages_and_turn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    request = _make_request("몰라")
    request.complaint_draft = ComplaintDraft(symptom="온수가 나오지 않음")
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        # 추출 프롬프트가 "모른다"는 답을 location="모름"으로 채워 올려보낸 상황.
        lambda _, **__: _TurnInterpretation(ComplaintDraft(location="모름")),
    )

    with caplog.at_level("INFO", logger=node_module.__name__):
        handle_complaint(request)

    records = _node_records(caplog)
    # 외부 호출이 있는 단계만 잰다. 순수 계산 구간은 항상 0ms라 줄만 늘린다.
    stages = [r for r in records if r.getMessage() == "stage_done"]
    assert [(r.stage, r.outcome) for r in stages] == [
        ("text_interpretation", "ok"),
        ("image_analysis", "skipped"),
        ("turn_finalization", "skipped"),
    ]
    assert all(r.duration_ms >= 0 for r in stages)

    turn = records[-1]
    assert turn.getMessage() == "complaint_turn"
    assert turn.location_unknown is True
    # "모름"이 위치를 채웠으므로 더 물을 항목이 없다 — 수집이 여기서 끝난다.
    assert turn.asked is None
    assert turn.reply_source == "complete"
    assert turn.total_ms >= 0


def test_handle_complaint_logs_image_failure_and_keeps_text_flow(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(ComplaintDraft(location="욕실")),
    )
    monkeypatch.setattr(
        node_module,
        "analyze_images",
        lambda *_: (_ for _ in ()).throw(ImageAnalysisError("invalid image")),
    )

    with caplog.at_level("INFO", logger=node_module.__name__):
        result = handle_complaint(
            _make_request("물이 새요", ["https://example.com/leak.jpg"])
        )

    assert result["result"].image_analysis is None
    failed = [
        r
        for r in _node_records(caplog)
        if r.getMessage() == "stage_done" and r.outcome == "fail"
    ]
    assert [(r.stage, r.error_type, r.error) for r in failed] == [
        ("image_analysis", "ImageAnalysisError", "invalid image")
    ]
    assert failed[0].duration_ms >= 0


def test_handle_complaint_photo_only_logs_image_analysis(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    image_url = "https://example.com/leak.jpg"
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda *_, **__: _TurnInterpretation(ComplaintDraft()),
    )
    monkeypatch.setattr(
        node_module,
        "analyze_images",
        lambda *_: ImageAnalysis(
            images=[
                ImageObservation(
                    url=image_url, summary="바닥에 물이 고여 있음", ocr_text=None
                )
            ]
        ),
    )

    with caplog.at_level("INFO", logger=node_module.__name__):
        outcome = handle_complaint(_make_request(None, [image_url]))

    assert outcome["result"].complaint_draft.image_urls == [image_url]
    assert outcome["result"].missing_fields == ["location"]
    assert outcome["reply"].startswith("사진은 확인했습니다 — 바닥에 물이 고여 있음.")
    records = _node_records(caplog)
    analyzed = [
        r
        for r in records
        if r.getMessage() == "stage_done" and r.stage == "image_analysis"
    ]
    assert [r.outcome for r in analyzed] == ["ok"]
    assert analyzed[0].duration_ms >= 0
    assert records[-1].photo == "analyzed"


def test_handle_complaint_keeps_confirmed_location_against_unknown(
    monkeypatch: pytest.MonkeyPatch,
):
    # 증상을 물은 턴에 "모르겠어요"가 오면 추출이 location="모름"을 올려보내기도 한다.
    # 이미 확인된 위치를 그 값으로 덮으면 관리자가 쓸 수 없는 카드가 된다.
    request = _make_request("모르겠어요")
    request.complaint_draft = ComplaintDraft(location="주방")
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(ComplaintDraft(location="모름")),
    )

    outcome = handle_complaint(request)

    assert outcome["result"].complaint_draft.location == "주방"
    assert outcome["result"].missing_fields == ["symptom"]


def _switching_draft() -> ComplaintDraft:
    """전환 판정이 실제로 도달하는 초안 — 증상은 있고 위치는 아직 없다.

    필수 항목이 다 차면 그 턴에 카드가 나가고 대화가 끝나므로, 완성된 초안을
    들고 다음 턴이 오는 경우는 없다.
    """
    return ComplaintDraft(
        issue_type="leak",
        symptom="천장에서 물이 떨어져요",
        image_urls=["https://example.com/old.jpg"],
    )


def test_handle_complaint_asks_before_replacing_a_draft_with_a_new_complaint(
    monkeypatch: pytest.MonkeyPatch,
):
    request = _make_request("세탁기가 안 돌아가요")
    request.complaint_draft = _switching_draft()
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(
            ComplaintDraft(symptom="세탁기가 안 돌아감"), switch="ask"
        ),
    )

    outcome = handle_complaint(request)

    # 아직 전환된 게 아니다. 거절당하면 되돌릴 방법이 없으므로 초안을 그대로 둔다.
    assert outcome["result"].complaint_draft == _switching_draft()
    assert outcome["complaint_state"] is ComplaintState.COLLECTING
    # 돌려주는 초안이 옛 민원이므로 빈 칸도 옛 민원 기준이다.
    assert outcome["result"].missing_fields == ["location"]
    # 새 증상이 낫표 안에 들어가는 것이 계약이다. 수락 턴의 추출 프롬프트가 이 문구에서
    # 증상을 되읽으므로, 낫표가 빠지면 회수가 조용히 깨진다.
    assert "「세탁기가 안 돌아감」" in outcome["reply"]


def test_handle_complaint_does_not_attach_photos_to_the_draft_while_asking(
    monkeypatch: pytest.MonkeyPatch,
):
    request = _make_request(
        "세탁기가 안 돌아가요", image_urls=["https://example.com/new.jpg"]
    )
    request.complaint_draft = _switching_draft()
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(
            ComplaintDraft(symptom="세탁기가 안 돌아감"), switch="ask"
        ),
    )
    image_calls: list[list[str]] = []

    def record_image_analysis(urls, _prompt):
        image_calls.append(urls)
        return ImageAnalysis(images=[])

    monkeypatch.setattr(node_module, "analyze_images", record_image_analysis)

    outcome = handle_complaint(request)

    # 새 민원의 사진을 옛 초안에 붙이면 관리자가 엉뚱한 사진을 보게 된다.
    assert outcome["result"].complaint_draft.image_urls == [
        "https://example.com/old.jpg"
    ]
    assert image_calls == [["https://example.com/new.jpg"]]


def test_handle_complaint_clears_the_old_draft_once_the_switch_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
):
    request = _make_request("네")
    request.complaint_draft = _switching_draft()
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(
            ComplaintDraft(symptom="세탁기가 안 돌아감"), switch="accept"
        ),
    )

    draft = handle_complaint(request)["result"].complaint_draft

    assert draft.symptom == "세탁기가 안 돌아감"
    # 위치는 옛 민원의 것이다. 남으면 세탁기 민원이 화장실에 있다고 접수된다.
    assert draft.location is None
    assert draft.issue_type is None
    # 사진도 같이 비워져야 한다. 텍스트만 비우면 옛 사진이 새 카드에 딸려간다.
    assert draft.image_urls == []


def test_handle_complaint_keeps_this_turns_photo_after_accepting_a_switch(
    monkeypatch: pytest.MonkeyPatch,
):
    request = _make_request("네", image_urls=["https://example.com/new.jpg"])
    request.complaint_draft = _switching_draft()
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(
            ComplaintDraft(symptom="세탁기가 안 돌아감"), switch="accept"
        ),
    )
    monkeypatch.setattr(
        node_module,
        "analyze_images",
        lambda _, __: ImageAnalysis(
            images=[
                ImageObservation(
                    url="https://example.com/new.jpg", summary=None, ocr_text=None
                )
            ]
        ),
    )

    draft = handle_complaint(request)["result"].complaint_draft

    assert draft.image_urls == ["https://example.com/new.jpg"]


@pytest.mark.parametrize(
    ("current", "extracted"),
    [
        # 비교할 기존 증상이 없다. 전환할 대상이 아예 없는 상태다.
        (None, ComplaintDraft(symptom="세탁기가 안 돌아감")),
        # 새 증상을 못 읽었다. 확인 문구를 만들 재료가 없다.
        (ComplaintDraft(symptom="물이 새요"), ComplaintDraft()),
    ],
)
def test_settle_switch_drops_ask_without_the_pieces_it_needs(
    current: ComplaintDraft | None, extracted: ComplaintDraft
):
    settled = node_module._settle_switch(
        _TurnInterpretation(extracted, switch="ask"), current
    )
    assert settled.switch == "same"


def test_settle_switch_discards_a_contradictory_ask_extraction():
    # "다른 민원"이라면서 증상이 없다. 값을 병합하면 다른 민원의 issue_type이
    # 지금 초안에 얹혀 짜깁기 카드가 된다.
    settled = node_module._settle_switch(
        _TurnInterpretation(ComplaintDraft(issue_type="electricity"), switch="ask"),
        ComplaintDraft(symptom="물이 새요"),
    )
    assert settled.switch == "same"
    assert settled.fields.issue_type is None


def test_settle_switch_keeps_extraction_when_there_is_nothing_to_switch_from():
    # 전환할 대상이 없으면 평범한 수집 턴이다. 추출값을 버리면 증상을 잃는다.
    settled = node_module._settle_switch(
        _TurnInterpretation(ComplaintDraft(symptom="물이 새요"), switch="ask"), None
    )
    assert settled.switch == "same"
    assert settled.fields.symptom == "물이 새요"


def test_settle_switch_never_demotes_accept():
    # accept를 same으로 돌리면 방금 접어두기로 한 민원이 완성 상태로 카드까지 나간다.
    settled = node_module._settle_switch(
        _TurnInterpretation(ComplaintDraft(), switch="accept"),
        ComplaintDraft(symptom="물이 새요"),
    )
    assert settled.switch == "accept"


def test_handle_complaint_does_not_file_the_abandoned_draft_when_recovery_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    # 입주민이 전환을 수락했는데 추출이 새 증상을 못 읽은 턴. 옛 초안은 완성돼 있다.
    request = _make_request("네")
    request.complaint_draft = _switching_draft()
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(ComplaintDraft(), switch="accept"),
    )

    outcome = handle_complaint(request)

    # 접어두기로 한 민원이 카드로 나가면 안 된다. 빈 초안으로 다시 물어야 한다.
    assert outcome["complaint_state"] is ComplaintState.COLLECTING
    assert outcome["result"].missing_fields == ["location", "symptom"]
    assert outcome["result"].complaint_draft.symptom is None
    assert outcome["result"].complaint_draft.image_urls == []


def test_handle_complaint_logs_the_switch_decision(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    # 확인 턴을 유지할지 빼고 판정만 남길지는 이 로그의 분포를 보고 정한다.
    request = _make_request("세탁기가 안 돌아가요")
    request.complaint_draft = _switching_draft()
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda _, **__: _TurnInterpretation(
            ComplaintDraft(symptom="세탁기가 안 돌아감"), switch="ask"
        ),
    )

    with caplog.at_level("INFO", logger=node_module.__name__):
        handle_complaint(request)

    turns = [r for r in _node_records(caplog) if r.getMessage() == "complaint_turn"]
    assert [(r.switch, r.reply_source) for r in turns] == [("ask", "switch_ask")]


def test_handle_complaint_runs_text_and_image_calls_concurrently(
    monkeypatch: pytest.MonkeyPatch,
):
    """전환 대상이 없는 턴에서는 텍스트 해석과 VLM 호출이 겹쳐 실행돼야 한다.

    둘 다 "상대가 먼저 시작했다"는 신호를 기다리게 해서, 순차 실행이면
    타임아웃으로 멈춘다.
    """
    text_started = threading.Event()
    image_started = threading.Event()

    def slow_interpret(_request, **_kwargs):
        text_started.set()
        assert image_started.wait(timeout=1), "image analysis never started"
        return _TurnInterpretation(ComplaintDraft(location="욕실"))

    def slow_analyze(_urls, _prompt):
        image_started.set()
        assert text_started.wait(timeout=1), "text interpretation never started"
        return ImageAnalysis(
            images=[
                ImageObservation(
                    url="https://example.com/leak.jpg", summary=None, ocr_text=None
                )
            ]
        )

    monkeypatch.setattr(node_module, "_interpret_turn", slow_interpret)
    monkeypatch.setattr(node_module, "analyze_images", slow_analyze, raising=False)

    result = handle_complaint(
        _make_request("욕실 바닥이 젖었어요", ["https://example.com/leak.jpg"])
    )["result"]

    assert result.complaint_draft.location == "욕실"
    assert result.image_analysis is not None


def test_handle_complaint_collects_text_and_images_before_settling_switch(
    monkeypatch: pytest.MonkeyPatch,
):
    """전환 관계 판단 전에도 텍스트와 이미지 근거를 함께 수집한다."""
    started = {"text": threading.Event(), "image": threading.Event()}
    release = threading.Event()

    def recording_interpret(_request, **_kwargs):
        started["text"].set()
        assert started["image"].wait(timeout=1)
        release.wait(timeout=1)
        return _TurnInterpretation(
            ComplaintDraft(symptom="세탁기가 안 돌아감"), switch="ask"
        )

    def recording_analyze(_urls, _prompt):
        started["image"].set()
        assert started["text"].wait(timeout=1)
        release.set()
        return ImageAnalysis(
            images=[
                ImageObservation(
                    url="https://example.com/new.jpg", summary=None, ocr_text=None
                )
            ]
        )

    monkeypatch.setattr(node_module, "_interpret_turn", recording_interpret)
    monkeypatch.setattr(node_module, "analyze_images", recording_analyze, raising=False)

    request = _make_request(
        "탈수할 때 LE 오류가 떠요", image_urls=["https://example.com/new.jpg"]
    )
    request.complaint_draft = ComplaintDraft(symptom="세탁기가 작동하지 않음")

    outcome = handle_complaint(request)

    assert outcome["result"].complaint_draft.symptom == "세탁기가 작동하지 않음"


def test_handle_complaint_skips_text_interpretation_for_photo_only_turn(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda *_, **__: pytest.fail("photo-only turn called the text LLM"),
    )
    monkeypatch.setattr(
        node_module,
        "analyze_images",
        lambda *_: ImageAnalysis(
            images=[
                ImageObservation(
                    url="https://example.com/leak.jpg",
                    summary="바닥에 물이 고여 있음",
                    ocr_text=None,
                )
            ]
        ),
    )

    outcome = handle_complaint(_make_request(None, ["https://example.com/leak.jpg"]))

    assert outcome["result"].complaint_draft.image_urls == [
        "https://example.com/leak.jpg"
    ]


def test_handle_complaint_treats_whitespace_only_text_as_photo_only(
    monkeypatch: pytest.MonkeyPatch,
):
    """api/converse.py의 has_text, knowledge/node.py의 question과 같은 strip() 기준.

    공백만 있는 메시지는 사진 전용 턴으로 취급해 텍스트 LLM을 부르지 않아야 한다.
    """
    monkeypatch.setattr(
        node_module,
        "_interpret_turn",
        lambda *_, **__: pytest.fail("whitespace-only text called the text LLM"),
    )
    monkeypatch.setattr(
        node_module,
        "analyze_images",
        lambda *_: ImageAnalysis(
            images=[
                ImageObservation(
                    url="https://example.com/leak.jpg",
                    summary="바닥에 물이 고여 있음",
                    ocr_text=None,
                )
            ]
        ),
    )

    outcome = handle_complaint(_make_request("   ", ["https://example.com/leak.jpg"]))

    assert outcome["result"].complaint_draft.image_urls == [
        "https://example.com/leak.jpg"
    ]


def test_handle_complaint_propagates_interpretation_failure_from_concurrent_path(
    monkeypatch: pytest.MonkeyPatch,
):
    """동시 실행 경로에서도 텍스트 해석 실패가 순차 실행과 같이 그대로 올라와야 한다."""
    monkeypatch.setattr(
        node_module,
        "analyze_images",
        lambda *_: ImageAnalysis(images=[]),
        raising=False,
    )

    with pytest.raises(ComplaintExtractionError):
        handle_complaint(
            _make_request("욕실 바닥이 젖었어요", ["https://example.com/leak.jpg"])
        )


def test_handle_complaint_concurrent_path_keeps_bound_context_and_timings(
    monkeypatch: pytest.MonkeyPatch,
):
    """copy_context()로 넘긴 스레드에서도 bind()의 필드와 collect_timings()가 보여야 한다.

    `_ContextFilter`는 `configure_logging()`이 붙인 핸들러에서만 동작해 caplog로는
    검증할 수 없다. 대신 각 스레드 안에서 ContextVar를 직접 읽어 비교한다.
    """
    seen_turn_ids: list[object] = []

    def recording_interpret(_request, **_kwargs):
        seen_turn_ids.append((observability._context.get() or {}).get("turn_id"))
        return _TurnInterpretation(ComplaintDraft(location="욕실"))

    def recording_analyze(*_args):
        seen_turn_ids.append((observability._context.get() or {}).get("turn_id"))
        return ImageAnalysis(
            images=[
                ImageObservation(
                    url="https://example.com/leak.jpg", summary=None, ocr_text=None
                )
            ]
        )

    monkeypatch.setattr(node_module, "_interpret_turn", recording_interpret)
    monkeypatch.setattr(node_module, "analyze_images", recording_analyze, raising=False)

    with bind(turn_id="turn-concurrent"), collect_timings() as timings:
        handle_complaint(
            _make_request("욕실 바닥이 젖었어요", ["https://example.com/leak.jpg"])
        )

    assert seen_turn_ids == ["turn-concurrent", "turn-concurrent"]
    assert {"text_interpretation", "image_analysis"} <= set(timings)
