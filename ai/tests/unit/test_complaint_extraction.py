from datetime import datetime

import pytest

import zipsai.complaint.node as node_module
from zipsai.complaint.node import (
    _extract_complaint_fields_and_reply,
    extract_complaint_fields,
    handle_complaint,
)
from zipsai.contracts.converse import (
    ComplaintDraft,
    ComplaintState,
    ConverseRequest,
    ImageAnalysis,
    ImageObservation,
)
from zipsai.errors import ComplaintExtractionError, ImageAnalysisError


def _node_records(caplog: pytest.LogCaptureFixture) -> list:
    """caplog 핸들러는 root에 붙어 다른 로거 기록까지 담는다. 이 모듈 것만 남긴다."""
    return [r for r in caplog.records if r.name == node_module.__name__]


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


def test_complaint_prompt_carries_formatted_history_without_model_repr(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, str] = {}

    def fake_generate_text(system_prompt: str, user_prompt: str) -> str:
        captured["user_prompt"] = user_prompt
        return '{"location": "화장실"}'

    monkeypatch.setattr(node_module, "generate_text", fake_generate_text)

    extract_complaint_fields(
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


def test_extract_complaint_fields_parses_llm_json(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        node_module,
        "generate_text",
        lambda system_prompt, user_prompt: (
            '{"issue_type": "leak", "location": "화장실", "symptom": "천장에서 물이 떨어져요"}'
        ),
    )

    result = extract_complaint_fields(
        _make_request("화장실 천장에서 물이 계속 떨어져요")
    )

    assert result == ComplaintDraft(
        issue_type="leak", location="화장실", symptom="천장에서 물이 떨어져요"
    )


def test_extract_complaint_fields_accepts_json_code_fence(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "generate_text",
        lambda system_prompt, user_prompt: (
            '```json\n{"issue_type": "leak", "location": "화장실", '
            '"symptom": "천장에서 물이 떨어져요"}\n```'
        ),
    )

    result = extract_complaint_fields(
        _make_request("화장실 천장에서 물이 계속 떨어져요")
    )

    assert result.issue_type == "leak"


def test_extract_complaint_fields_defaults_missing_keys_to_none(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "generate_text",
        lambda system_prompt, user_prompt: (
            '{"issue_type": null, "location": null, "symptom": null}'
        ),
    )

    result = extract_complaint_fields(_make_request("음.."))

    assert result == ComplaintDraft()


def test_extract_complaint_fields_raises_on_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module, "generate_text", lambda system_prompt, user_prompt: "not json"
    )

    with pytest.raises(ComplaintExtractionError):
        extract_complaint_fields(_make_request("아무 말"))


def test_extract_complaint_fields_raises_on_invalid_issue_type(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "generate_text",
        lambda system_prompt, user_prompt: (
            '{"issue_type": "bogus", "location": null, "symptom": null}'
        ),
    )

    with pytest.raises(ComplaintExtractionError):
        extract_complaint_fields(_make_request("아무 말"))


def test_extract_complaint_fields_retries_once_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[int] = []

    def flaky_generate_text(system_prompt: str, user_prompt: str) -> str:
        calls.append(1)
        if len(calls) == 1:
            return "not json"
        return '{"issue_type": "leak", "location": "화장실", "symptom": "물이 새요"}'

    monkeypatch.setattr(node_module, "generate_text", flaky_generate_text)

    result = extract_complaint_fields(_make_request("화장실에서 물이 새요"))

    assert len(calls) == 2
    assert result == ComplaintDraft(
        issue_type="leak", location="화장실", symptom="물이 새요"
    )


def test_extract_complaint_fields_gives_up_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    calls: list[int] = []

    def always_broken(system_prompt: str, user_prompt: str) -> str:
        calls.append(1)
        return "not json"

    monkeypatch.setattr(node_module, "generate_text", always_broken)

    # 단계 로그가 붙은 실제 경로(handle_complaint)로 확인한다.
    with (
        caplog.at_level("INFO", logger=node_module.__name__),
        pytest.raises(ComplaintExtractionError),
    ):
        handle_complaint(_make_request("아무 말"))

    assert len(calls) == node_module._EXTRACTION_ATTEMPTS
    records = _node_records(caplog)
    retries = [r for r in records if r.getMessage() == "extraction_retry"]
    assert [(r.attempt, r.error_type) for r in retries] == [
        (1, "JSONDecodeError"),
        (2, "JSONDecodeError"),
    ]
    failed = [
        r for r in records if r.getMessage() == "stage_done" and r.outcome == "fail"
    ]
    assert [(r.stage, r.error_type) for r in failed] == [
        ("text_extraction", "ComplaintExtractionError")
    ]
    # 원인(JSONDecodeError)은 logger.exception이 실은 트레이스백에 남는다.
    assert "JSONDecodeError" in failed[0].exc_text


def test_extract_complaint_fields_parses_occurred_at(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        node_module,
        "generate_text",
        lambda system_prompt, user_prompt: (
            '{"issue_type": null, "location": null, "symptom": null, "occurred_at": "2026-09-22"}'
        ),
    )

    result = extract_complaint_fields(_make_request("어제부터 그랬어요"))

    # ComplaintDraft.occurred_at은 naive datetime이다(LLM이 "2026-09-22" 날짜만 준다).
    # tzinfo를 붙이면 실제 파싱 결과와 달라져 비교가 깨진다.
    assert result.occurred_at == datetime(2026, 9, 22)  # noqa: DTZ001


def test_extract_complaint_fields_passes_todays_kst_date_to_prompt(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, str] = {}

    def fake_generate_text(system_prompt: str, user_prompt: str) -> str:
        captured["user_prompt"] = user_prompt
        return '{"issue_type": null, "location": null, "symptom": null, "occurred_at": null}'

    monkeypatch.setattr(node_module, "generate_text", fake_generate_text)

    extract_complaint_fields(_make_request("어제부터 그랬어요"))

    today = datetime.now(node_module._KST).date().isoformat()
    assert f"오늘 날짜(Asia/Seoul): {today}" in captured["user_prompt"]


def test_extract_complaint_fields_and_reply_parses_reply_and_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "generate_text",
        lambda system_prompt, user_prompt: (
            '{"issue_type": null, "location": "화장실", "symptom": null, "occurred_at": null, '
            '"missing": ["symptom"], "reply": "어떤 증상인지 알려주시겠어요?"}'
        ),
    )

    draft, reply, missing = _extract_complaint_fields_and_reply(
        _make_request("화장실이 이상해요")
    )

    assert draft == ComplaintDraft(location="화장실")
    assert reply == "어떤 증상인지 알려주시겠어요?"
    assert missing == {"symptom"}


def test_extract_complaint_fields_and_reply_defaults_missing_keys_to_empty(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "generate_text",
        lambda system_prompt, user_prompt: (
            '{"issue_type": null, "location": null, "symptom": null}'
        ),
    )

    _, reply, missing = _extract_complaint_fields_and_reply(_make_request("음.."))

    assert reply == ""
    assert missing == set()


def test_extract_complaint_fields_and_reply_ignores_invalid_missing_entries(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "generate_text",
        lambda system_prompt, user_prompt: (
            '{"issue_type": null, "location": null, "symptom": null, '
            '"missing": ["location", "issue_type", "bogus"], "reply": "위치를 알려주세요"}'
        ),
    )

    _, _reply, missing = _extract_complaint_fields_and_reply(_make_request("음.."))

    assert missing == {"location"}


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
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(location="화장실"), "", set()),
    )

    result = handle_complaint(request)["result"]

    assert result.complaint_draft == ComplaintDraft(
        issue_type="water_supply",
        location="화장실",
        symptom="온수가 나오지 않음",
    )
    assert result.missing_fields == []


def test_handle_complaint_returns_image_analysis_without_changing_text_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(location="욕실"), "", set()),
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
    assert result.complaint_draft.image_urls == ["https://example.com/leak.jpg"]
    assert result.image_analysis.images[0].ocr_text == "E1"


def test_handle_complaint_keeps_text_flow_when_vlm_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    def raise_image_analysis_error(_: list[str], __: str) -> ImageAnalysis:
        raise ImageAnalysisError("VLM returned an invalid image analysis")

    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(location="욕실"), "", set()),
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
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(location="욕실"), "", set()),
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
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(), "", set()),
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
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(), "", set()),
    )

    reply = handle_complaint(_make_request("음.."))["reply"]

    assert reply == "어디에서 생긴 문제인가요?"


def test_handle_complaint_uses_llm_generated_reply_when_missing_matches(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (
            ComplaintDraft(),
            "화장실 세면대인지 변기 쪽인지 알려주시겠어요?",
            {"location"},
        ),
    )

    reply = handle_complaint(_make_request("화장실이 좀 이상해요"))["reply"]

    assert reply == "화장실 세면대인지 변기 쪽인지 알려주시겠어요?"


def test_handle_complaint_falls_back_when_llm_missing_disagrees_with_actual(
    monkeypatch: pytest.MonkeyPatch,
):
    # LLM은 location만 빠졌다고(잘못) 생각하지만, 실제 missing_fields는 symptom이다.
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (
            ComplaintDraft(location="화장실"),
            "정확한 위치를 알려주시겠어요?",
            {"location"},
        ),
    )

    reply = handle_complaint(_make_request("화장실이 이상해요"))["reply"]

    assert reply == "어떤 불편 증상인지 알려주세요."


def test_handle_complaint_prefixes_llm_reply_when_photo_analyzed(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (
            ComplaintDraft(),
            "정확히 어디쯤인지 알려주시겠어요?",
            {"location"},
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
        "_extract_complaint_fields_and_reply",
        lambda _: (
            ComplaintDraft(issue_type="leak", location="화장실", symptom="물이 새요"),
            "",
            set(),
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
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(location="화장실"), "", set()),
    )

    result = handle_complaint(_make_request("화장실이 이상해요"))

    assert result["complaint_state"] == ComplaintState.COLLECTING


def test_handle_complaint_asks_only_about_missing_symptom(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(location="화장실"), "", set()),
    )

    reply = handle_complaint(_make_request("화장실이 이상해요"))["reply"]

    assert reply == "어떤 불편 증상인지 알려주세요."


def test_handle_complaint_asks_only_about_missing_location(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(symptom="물이 새요"), "", set()),
    )

    reply = handle_complaint(_make_request("물이 새요"))["reply"]

    assert reply == "어디에서 생긴 문제인가요?"


def test_handle_complaint_defaults_issue_type_to_other_when_unclassified(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (
            ComplaintDraft(location="화장실", symptom="이상해요"),
            "",
            set(),
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
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(location="화장실"), "", set()),
    )

    result = handle_complaint(request)["result"]

    assert result.complaint_draft.occurred_at == datetime(2026, 9, 20)  # noqa: DTZ001


def test_handle_complaint_asks_one_field_at_a_time_when_both_are_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    # 둘을 한 문장으로 같이 물으면 "몰라"가 어느 필드에 대한 답인지 알 수 없다.
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(), "", set()),
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
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(), "", set()),
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
        "_extract_complaint_fields_and_reply",
        # 추출 프롬프트가 "모른다"는 답을 location="모름"으로 채워 올려보낸 상황.
        lambda _: (ComplaintDraft(location="모름"), "", set()),
    )

    with caplog.at_level("INFO", logger=node_module.__name__):
        handle_complaint(request)

    records = _node_records(caplog)
    # 외부 호출이 있는 단계만 잰다. 순수 계산 구간은 항상 0ms라 줄만 늘린다.
    stages = [r for r in records if r.getMessage() == "stage_done"]
    assert [(r.stage, r.outcome) for r in stages] == [
        ("text_extraction", "ok"),
        ("image_analysis", "skipped"),
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
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(location="욕실"), "", set()),
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
        "generate_text",
        lambda *_: (
            '{"location": null, "symptom": null, "missing": ["location", "symptom"]}'
        ),
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
    assert outcome["result"].missing_fields == ["location", "symptom"]
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
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(location="모름"), "", set()),
    )

    outcome = handle_complaint(request)

    assert outcome["result"].complaint_draft.location == "주방"
    assert outcome["result"].missing_fields == ["symptom"]
