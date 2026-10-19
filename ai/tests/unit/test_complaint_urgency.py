from zipsai.complaint.urgency import _grade, score_urgency
from zipsai.contracts.converse import ComplaintDraft, ImageAnalysis, ImageObservation


def _draft(symptom: str | None = None, location: str | None = None) -> ComplaintDraft:
    return ComplaintDraft(location=location, symptom=symptom)


def test_strong_safety_signal_gives_full_score_and_escalates():
    result = score_urgency(_draft(symptom="주방에서 가스 누출 냄새가 나요"), None)

    assert result.escalated is True
    assert result.level == "urgent"
    assert result.is_urgent is True


def test_weak_safety_signal_gives_half_score_without_escalation():
    result = score_urgency(
        _draft(symptom="여러 세대에서 단수가 되고 점점 심해지고 있어요. 위험해 보여요."),
        None,
    )

    assert result.escalated is False
    assert result.score == 22 + 25 + 15 + 15  # safety half + progression/function/scope full
    assert result.level == "urgent"  # 77점으로 임계값(70) 통과, 상향 규칙과 무관
    assert result.unknown_axes == []
    assert result.confidence == "high"


def test_no_signal_axis_is_unknown_not_zero():
    result = score_urgency(_draft(symptom="그냥 좀 불편해요"), None)

    assert set(result.unknown_axes) == {"safety", "progression", "function", "scope"}
    assert result.score == 0
    # 안전/진행 축이 unknown이면 합산이 낮아도 최소 priority로 올린다.
    assert result.level == "priority"
    assert result.confidence == "low"


def test_known_safety_and_progression_below_threshold_is_normal():
    result = score_urgency(_draft(symptom="위험해 보이고 아직도 그래요"), None)

    assert result.unknown_axes == ["function", "scope"]
    assert result.score == 22 + 12  # safety 절반 + progression 절반
    assert result.level == "normal"
    assert result.confidence == "low"  # function/scope는 unknown


def test_priority_range_without_unknown_axes():
    result = score_urgency(
        _draft(symptom="단수가 되고 아직도 그래요. 복도에도 영향이 있고 위험해 보여요"),
        None,
    )

    assert result.unknown_axes == []
    assert 40 <= result.score < 70
    assert result.level == "priority"


def test_scope_axis_considers_location_field():
    result = score_urgency(_draft(symptom="물이 샙니다", location="공용 현관"), None)

    assert "scope" not in result.unknown_axes


def test_duplicate_evidence_in_symptom_and_image_does_not_double_count():
    image_analysis = ImageAnalysis(
        images=[ImageObservation(attachmentId=1, summary="화재 위험이 보입니다")]
    )
    result = score_urgency(
        _draft(symptom="화재 화재 화재 위험해 보여요"), image_analysis
    )

    assert result.score <= 45 + 25 + 15 + 15  # 전체 배점 합을 넘지 않는다
    # safety 축은 강한 신호 만점(45)만 반영되고 중복 가산되지 않는다.
    assert result.escalated is True


def test_grade_boundaries_directly():
    assert _grade(70, []) == "urgent"
    assert _grade(69, []) == "priority"
    assert _grade(40, []) == "priority"
    assert _grade(39, []) == "normal"
    assert _grade(0, ["safety"]) == "priority"
    assert _grade(0, ["function", "scope"]) == "normal"
