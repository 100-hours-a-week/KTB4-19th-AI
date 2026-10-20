"""Task 1(DimensionJudgment/UrgencyEvidence) 검증 규칙을 눈으로 확인하는 스크립트.

주의: DimensionJudgment 자체엔 검증기가 없다. status/value/source/quote 조합 규칙은
UrgencyEvidence로 감쌀 때(model_validator)만 걸린다 — 그래서 아래 모든 케이스를
severity 자리에 넣어 UrgencyEvidence(severity=..., imminence=_unknown(), scope=_unknown())
로 테스트한다.

실행: ai/.venv/bin/python ai/temp_urgency_schema_demo.py
"""

from pydantic import ValidationError

from zipsai.complaint.urgency_llm import DimensionJudgment, UrgencyEvidence


def _unknown() -> DimensionJudgment:
    return DimensionJudgment(status="unknown")


def _check(label: str, severity: DimensionJudgment, *, expect_valid: bool) -> None:
    try:
        UrgencyEvidence(severity=severity, imminence=_unknown(), scope=_unknown())
        ok = True
        detail = "통과"
    except ValidationError as error:
        ok = False
        # 긴 pydantic 트레이스 대신 우리가 raise한 메시지만 뽑는다.
        detail = str(error.errors()[0]["ctx"]["error"])

    mark = "✅" if ok == expect_valid else "❌(기대와 다름!)"
    print(f"{mark} {label}")
    print(f"   severity={severity!r}")
    print(f"   -> {detail}\n")


def main() -> None:
    print("=" * 70)
    print("정상 케이스 (전부 통과해야 함)")
    print("=" * 70 + "\n")

    _check("unknown: 아무 근거 없음", _unknown(), expect_valid=True)

    _check(
        "explicitly_absent: 위험 없음을 명시적으로 확인",
        DimensionJudgment(
            status="explicitly_absent",
            value=0,
            source="symptom",
            quote="불은 안 났어요",
        ),
        expect_valid=True,
    )

    _check(
        "supported + symptom 출처",
        DimensionJudgment(
            status="supported", value=4, source="symptom", quote="가스 누출"
        ),
        expect_valid=True,
    )

    _check(
        "supported + image 출처 (attachment_id 포함)",
        DimensionJudgment(
            status="supported",
            value=4,
            source="image",
            attachment_id=7,
            quote="검은 연기가 보임",
        ),
        expect_valid=True,
    )

    print("=" * 70)
    print("잘못된 케이스 (전부 ValidationError 발생해야 함)")
    print("=" * 70 + "\n")

    _check(
        "unknown인데 value가 섞여 있음",
        DimensionJudgment(status="unknown", value=2),
        expect_valid=False,
    )

    _check(
        "explicitly_absent인데 value가 0이 아님",
        DimensionJudgment(
            status="explicitly_absent", value=1, source="symptom", quote="끝났어요"
        ),
        expect_valid=False,
    )

    _check(
        "supported인데 severity 범위(1~4)를 벗어난 값(5)",
        DimensionJudgment(
            status="supported", value=5, source="symptom", quote="가스 누출"
        ),
        expect_valid=False,
    )

    _check(
        "supported인데 quote가 없음",
        DimensionJudgment(status="supported", value=4, source="symptom"),
        expect_valid=False,
    )

    _check(
        "supported + image 출처인데 attachment_id가 없음",
        DimensionJudgment(
            status="supported", value=4, source="image", quote="검은 연기"
        ),
        expect_valid=False,
    )


if __name__ == "__main__":
    main()
