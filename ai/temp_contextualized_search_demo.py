"""작동 순서를 눈으로 확인하는 검색 fallback 데모. 실제 Qdrant/LLM은 호출하지 않음."""


def demo(original: str, contextualized: str, results_by_query: dict[str, list[str]]) -> None:
    searched: list[str] = []

    def search(question: str) -> list[str]:
        searched.append(question)
        return results_by_query.get(question, [])

    # 실제 knowledge/node.py 흐름: 원문부터 검색하고, 결과가 없을 때만 보강문으로 재검색.
    chunks = search(original)
    final_question = original
    if not chunks and contextualized and contextualized != original:
        final_question = contextualized
        chunks = search(contextualized)

    print(f"검색 순서: {searched}")
    print(f"최종 결과: {chunks or '근거 없음 응답'}")
    print(f"QA 초안 질문: {final_question}")
    print()


print("1) 원문 검색 성공: 맥락 보강문은 사용하지 않음")
demo(
    "분리수거는 언제 해요?",
    "분리수거 배출 요일은 언제인가요?",
    {"분리수거는 언제 해요?": ["화요일 저녁 배출"]},
)

print("2) 원문 검색 실패: 맥락 보강문으로 한 번 더 검색")
demo(
    "그거 어디다 해?",
    "분리수거는 어디에 배출하나요?",
    {"분리수거는 어디에 배출하나요?": ["건물 앞 지정 배출대"]},
)

print("3) 둘 다 실패: 근거 없음 응답")
demo("그거 어디다 해?", "분리수거는 어디에 배출하나요?", {})
