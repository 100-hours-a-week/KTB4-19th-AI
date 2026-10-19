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
