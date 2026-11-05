# Issue #180 구현 및 검증 보고서

## 결과 요약

민원 유형 계약, 증상/위치 추출 지침, 발생 시점 표현을 보강하고 복합 요청에서 민원 후속 질문이 빠지지 않도록 응답 조립을 보정했다. 위치 정정 시 이전 위치가 증상에 남는 문제도 보정했다. 기존 민원 파이프라인과 API 필드는 유지했다.

| 태스크 | 변경 및 이유 | 커밋 |
| --- | --- | --- |
| 1. taxonomy 계약 | 신규 민원 draft는 9개 서비스 taxonomy만 허용한다. 과거 댓글의 `facility` 등은 의미를 추측해 재분류하지 않고, 댓글 입력 경계에서만 레거시 값을 계속 수용한다. 프롬프트, fixture, 테스트, 데모가 신규 계약을 사용하도록 맞췄다. | `248cdb4` `feat(complaint): 민원 taxonomy 계약 반영` |
| 2. 증상·위치 추출 | 작성 요청만으로 증상을 만들어내지 않도록 프롬프트를 구체화했다. 위치 정정 시 증상에 남은 기존 위치 문자열도 새 위치로 교체해 draft 내부 모순을 줄였다. | `969bf40`, `0739b52` |
| 3. 발생 시점 | `이번 주`/`지난주`의 월~일 경계, 한정어 없는 요일, 날짜 없는 시간, 명시된 날짜·시간의 KST 처리를 정의했다. `datetime | None` API 필드는 유지하고 날짜 전용 naive 값과 명시 시간의 KST aware 값 직렬화를 테스트로 고정했다. | `5b23ba8` |
| 4. 응답 경로 | 민원 필수 필드가 비어 있는 복합 요청에서 composer가 후속 질문을 누락하면 민원 Tool의 authoritative follow-up을 응답에 보존한다. 프롬프트에도 누락 필드가 있으면 접수가 완료된 것처럼 답하지 않도록 명시하고, Tool 경계의 상태·초안·질문 일관성을 테스트했다. | `ba57809`, `0739b52` |

## 검증

- Task 1: 전체 Git 추적 테스트 `723 passed, 10 warnings`.
- Task 2: 민원 추출 테스트 `42 passed`; 전체 추적 테스트 `724 passed, 10 warnings`; Ruff 통과.
- Task 3: 민원 추출 테스트 `43 passed`; 전체 추적 테스트 `725 passed, 10 warnings`; Ruff 통과.
- Task 4: 민원 추출/Tool/graph/Converse 회귀 `118 passed, 1 warning`; Ruff 통과.
- 리뷰 수정 후 최종: 전체 Git 추적 테스트 `732 passed, 10 warnings`; 민원 추출/Tool/graph/계약 회귀 `130 passed`; Ruff 통과.
- 기본 `uv run pytest -q`: 수집 실패. 미추적 `tests/unit/test_modal_vllm.py`가 존재하지 않는 `_build_vllm_command`를, 미추적 `tests/unit/test_web_tool.py`가 존재하지 않는 `WebSearchRateLimitedError`를 import한다. 두 파일은 이 브랜치 변경이 아니어서 수정하지 않았다.
- 실제 Qwen Pod 검증: **미완료**. 실행 시 upstream이 `503 no upstreams available`을 반환해 모델 출력 비교를 할 수 없었다. 위 단위 테스트 통과를 실모델 추출 품질 통과로 해석하면 안 된다.

## 호환성과 남은 위험

- `ComplaintDraft.issue_type` 신규 응답 계약은 `plumbing`, `heating_cooling`, `interior_repair`, `electrical`, `appliance`, `hygiene`, `common_area`, `noise`, `other`다. 신규 요청/응답을 저장하는 백엔드도 이 값 집합을 지원해야 한다.
- 댓글 입력은 저장 데이터 이관이 확인될 때까지 레거시 enum을 허용한다. 이는 읽기 호환만 보장하며, 레거시 값을 신규 taxonomy로 자동 변환하지 않는다.
- 발생 시점 날짜/요일/시간 해석은 여전히 모델 프롬프트에 의존한다. KST 주 경계 규칙을 테스트로 고정했지만, 실제 Qwen 준수율은 Pod 복구 후 날짜·요일·시각 사례로 검증해야 한다.
- 날짜만 있는 추출은 naive `datetime`으로 직렬화되고, 명시 시각은 `+09:00`을 포함한 aware `datetime`으로 직렬화된다. 이 기존 혼합 동작은 테스트로 문서화했지만 BE의 해석 계약을 확인하지 못해 변경하지 않았다. BE가 naive 값을 UTC로 취급한다면 날짜가 어긋날 수 있으므로 API 소비자와 별도 확인이 필요하다.
- 복합 응답은 누락 필드 후속 질문을 서버에서 보존하지만, 자연어 composer가 그와 모순되는 완료 문구를 동시에 생성하는 모든 표현을 의미적으로 판별/제거하지는 않는다. 프롬프트로 억제하고 질문 보존으로 사용자에게 필요한 다음 행동을 전달하되, 실제 모델 출력은 Pod 복구 후 검증해야 한다.
- 도어락은 승인된 전용 분류가 없어 `other`로 명시했다. 도메인에서 별도 택소노미를 결정하면 프롬프트/분류 계약을 함께 갱신해야 한다.
- 증상 프롬프트 변경 또한 모델 품질 개선의 보장은 아니다. `욕실 천장에서 물이 새요`, 부정/대조, 수정, 후속 답변 사례의 실모델 결과를 확인해야 한다.

## 근거

- Python 공식 문서의 [`datetime`](https://docs.python.org/3.12/library/datetime.html)은 naive datetime이 시간대 의미를 자체적으로 갖지 않는다고 명시한다. 따라서 날짜/시간 근거를 KST로 한정하고, 명시된 시각에는 오프셋을 요구했다.
- Python 표준 라이브러리 [`zoneinfo`](https://docs.python.org/3/library/zoneinfo.html)를 기준으로 애플리케이션의 기존 `Asia/Seoul` 기준 날짜 주입을 유지했다.
- 프롬프트 변경은 기존 구조화 추출과 파이프라인 경계를 유지하는 최소 수정이다. 모델 출력 정답셋이나 실측 baseline은 추가/생성하지 않았다.

## 리뷰 후 보정

독립 코드 리뷰에서 확인된 복합 응답의 후속 질문 누락, 위치 정정 후 증상 내 구 위치 잔존, 도어락 분류 경계, taxonomy payload fixture 및 과거 프롬프트 비교 스키마 문제를 회귀 테스트로 고정하고 수정했다. 리뷰 수정 커밋은 `0739b52`다. 날짜 프롬프트 경계 테스트와 실제 Pydantic 직렬화 테스트도 추가했다. 실 Qwen 호출은 upstream `503` 때문에 여전히 확인하지 못했다.
