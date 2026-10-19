import pytest
from pydantic import ValidationError

from zipsai.complaint.urgency_llm import DimensionJudgment, UrgencyEvidence


def _evidence(**overrides: object) -> UrgencyEvidence:
    base = {
        "severity": DimensionJudgment(status="unknown"),
        "imminence": DimensionJudgment(status="unknown"),
        "scope": DimensionJudgment(status="unknown"),
    }
    base.update(overrides)
    return UrgencyEvidence(**base)


def test_unknown_status_accepts_all_nulls():
    evidence = _evidence()
    assert evidence.severity.status == "unknown"
    assert evidence.severity.value is None


def test_unknown_status_rejects_a_value():
    with pytest.raises(ValidationError):
        _evidence(severity=DimensionJudgment(status="unknown", value=2))


def test_supported_status_requires_value_in_range():
    with pytest.raises(ValidationError):
        _evidence(
            severity=DimensionJudgment(
                status="supported", value=5, source="symptom", quote="가스 누출"
            )
        )


def test_supported_status_requires_quote():
    with pytest.raises(ValidationError):
        _evidence(
            severity=DimensionJudgment(status="supported", value=4, source="symptom")
        )


def test_explicitly_absent_requires_value_zero():
    with pytest.raises(ValidationError):
        _evidence(
            severity=DimensionJudgment(
                status="explicitly_absent",
                value=1,
                source="symptom",
                quote="불은 안 났어요",
            )
        )


def test_image_source_requires_attachment_id():
    with pytest.raises(ValidationError):
        _evidence(
            severity=DimensionJudgment(
                status="supported", value=4, source="image", quote="검은 연기"
            )
        )


def test_valid_supported_dimension_with_image_source():
    evidence = _evidence(
        severity=DimensionJudgment(
            status="supported",
            value=4,
            source="image",
            attachment_id=1,
            quote="검은 연기가 보임",
        )
    )
    assert evidence.severity.value == 4


from zipsai.complaint.urgency_llm import (
    DEFAULT_SCORING_CONFIG,
    score_urgency,
)


def _supported(value: int, *, quote: str = "증상") -> DimensionJudgment:
    return DimensionJudgment(
        status="supported", value=value, source="symptom", quote=quote
    )


def _absent(*, quote: str = "끝났어요") -> DimensionJudgment:
    return DimensionJudgment(
        status="explicitly_absent", value=0, source="symptom", quote=quote
    )


def _unknown() -> DimensionJudgment:
    return DimensionJudgment(status="unknown")


def test_all_dimensions_known_matches_design_doc_worst_case():
    evidence = UrgencyEvidence(
        severity=_supported(4), imminence=_supported(4), scope=_supported(2)
    )
    result = score_urgency(evidence, DEFAULT_SCORING_CONFIG)
    assert result.score == 100
    assert result.is_urgent is True


def test_severity_three_imminence_three_hits_threshold_exactly():
    evidence = UrgencyEvidence(
        severity=_supported(3), imminence=_supported(3), scope=_unknown()
    )
    result = score_urgency(evidence, DEFAULT_SCORING_CONFIG)
    assert result.score == 60
    assert result.is_urgent is True


def test_moderate_case_is_not_urgent():
    evidence = UrgencyEvidence(
        severity=_supported(2), imminence=_absent(), scope=_supported(1)
    )
    result = score_urgency(evidence, DEFAULT_SCORING_CONFIG)
    assert result.score == 35
    assert result.is_urgent is False


def test_severity_four_overrides_low_score_when_imminence_resolved():
    evidence = UrgencyEvidence(
        severity=_supported(4), imminence=_absent(quote="지금은 꺼졌어요"), scope=_unknown()
    )
    result = score_urgency(evidence, DEFAULT_SCORING_CONFIG)
    assert result.score == 50
    assert result.is_urgent is True  # override, score alone would not reach 60


def test_severity_four_overrides_when_imminence_unclear():
    evidence = UrgencyEvidence(severity=_supported(4), imminence=_unknown(), scope=_unknown())
    result = score_urgency(evidence, DEFAULT_SCORING_CONFIG)
    assert result.score == 58
    assert result.is_urgent is True


def test_all_unknown_dimensions_yield_null_score():
    evidence = UrgencyEvidence(severity=_unknown(), imminence=_unknown(), scope=_unknown())
    result = score_urgency(evidence, DEFAULT_SCORING_CONFIG)
    assert result.score is None
    assert result.is_urgent is None
    assert result.dimensions["severity"]["status"] == "unknown"


def test_unknown_severity_and_imminence_map_to_one_not_zero():
    evidence = UrgencyEvidence(severity=_unknown(), imminence=_unknown(), scope=_supported(2))
    result = score_urgency(evidence, DEFAULT_SCORING_CONFIG)
    # severity=1, imminence=1, scope=2 -> 50*1/4 + 30*1/4 + 20*2/2 = 12.5+7.5+20 = 40
    assert result.score == 40
    assert result.is_urgent is False
