# Issue #180 구현 및 검증 보고서

## 결과 요약

민원 유형 계약, 증상/위치 추출 지침, 발생 시점 표현을 보강하고 민원 Tool 경계의 응답 일관성 테스트를 추가했다. 기존 민원 파이프라인과 API 필드는 유지했다. 코드 태스크별 커밋은 네 개이며, 이 문서는 검증과 남은 위험을 정리한다.

| 태스크 | 변경 및 이유 | 커밋 |
| --- | --- | --- |
| 1. taxonomy 계약 | 신규 민원 draft는 9개 서비스 taxonomy만 허용한다. 과거 댓글의 `facility` 등은 의미를 추측해 재분류하지 않고, 댓글 입력 경계에서만 레거시 값을 계속 수용한다. 프롬프트, fixture, 테스트, 데모가 신규 계약을 사용하도록 맞췄다. | `248cdb4` `feat(complaint): 민원 taxonomy 계약 반영` |
| 2. 증상·위치 추출 | 민원 작성 요청 문장 자체를 증상으로 간주하지 않고, 위치에 이미 기록한 방/영역을 증상에 반복하지 않도록 프롬프트 계약을 구체화했다. 기존 `handle_complaint` 병합/최종화 흐름은 변경하지 않았다. | `969bf40` `fix(complaint): 증상 및 위치 추출 보정` |
| 3. 발생 시점 | `이번 주`/`지난주`의 월~일 경계, 한정어 없는 요일, 날짜 없는 시간, 날짜와 시간이 함께 명시된 경우의 KST ISO 오프셋을 정의했다. API의 `datetime | None` 타입은 유지했다. | `5b23ba8` `fix(complaint): 발생 시점 추출 규칙 보정` |
| 4. 응답 경로 | 실제 파이프라인은 필수 필드 누락 시 접수 카드를 만들지 않고 해당 필드 질문과 collecting 상태를 반환했다. Tool 경계에서도 reply·누락 필드·초안이 함께 유지되는 테스트를 추가하고, 불필요한 응답 파이프라인 변경은 피했다. | `ba57809` `fix(complaint): 응답 및 SSE 결과 정합성 보정` |

## 검증

- Task 1: 전체 Git 추적 테스트 `723 passed, 10 warnings`.
- Task 2: 민원 추출 테스트 `42 passed`; 전체 추적 테스트 `724 passed, 10 warnings`; Ruff 통과.
- Task 3: 민원 추출 테스트 `43 passed`; 전체 추적 테스트 `725 passed, 10 warnings`; Ruff 통과.
- Task 4: 민원 추출/Tool/graph/Converse 회귀 `118 passed, 1 warning`; Ruff 통과.
- 최종: 전체 Git 추적 테스트 `726 passed, 10 warnings`.
- 기본 `uv run pytest -q`: 수집 실패. 미추적 `tests/unit/test_modal_vllm.py`가 존재하지 않는 `_build_vllm_command`를, 미추적 `tests/unit/test_web_tool.py`가 존재하지 않는 `WebSearchRateLimitedError`를 import한다. 두 파일은 이 브랜치 변경이 아니어서 수정하지 않았다.
- 실제 Qwen Pod 검증: **미완료**. 실행 시 upstream이 `503 no upstreams available`을 반환해 모델 출력 비교를 할 수 없었다. 위 단위 테스트 통과를 실모델 추출 품질 통과로 해석하면 안 된다.

## 호환성과 남은 위험

- `ComplaintDraft.issue_type` 신규 응답 계약은 `plumbing`, `heating_cooling`, `interior_repair`, `electrical`, `appliance`, `hygiene`, `common_area`, `noise`, `other`다. 신규 요청/응답을 저장하는 백엔드도 이 값 집합을 지원해야 한다.
- 댓글 입력은 저장 데이터 이관이 확인될 때까지 레거시 enum을 허용한다. 이는 읽기 호환만 보장하며, 레거시 값을 신규 taxonomy로 자동 변환하지 않는다.
- 발생 시점 날짜/요일/시간 해석은 여전히 모델 프롬프트에 의존한다. KST 주 경계 규칙을 테스트로 고정했지만, 실제 Qwen 준수율은 Pod 복구 후 날짜·요일·시각 사례로 검증해야 한다.
- 증상 프롬프트 변경 또한 모델 품질 개선의 보장은 아니다. `욕실 천장에서 물이 새요`, 부정/대조, 수정, 후속 답변 사례의 실모델 결과를 확인해야 한다.

## 근거

- Python 공식 문서의 [`datetime`](https://docs.python.org/3.12/library/datetime.html)은 naive datetime이 시간대 의미를 자체적으로 갖지 않는다고 명시한다. 따라서 날짜/시간 근거를 KST로 한정하고, 명시된 시각에는 오프셋을 요구했다.
- Python 표준 라이브러리 [`zoneinfo`](https://docs.python.org/3/library/zoneinfo.html)를 기준으로 애플리케이션의 기존 `Asia/Seoul` 기준 날짜 주입을 유지했다.
- 프롬프트 변경은 기존 구조화 추출과 파이프라인 경계를 유지하는 최소 수정이다. 모델 출력 정답셋이나 실측 baseline은 추가/생성하지 않았다.
