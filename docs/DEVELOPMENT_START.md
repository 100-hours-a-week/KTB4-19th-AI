# 개발 시작 가이드

## 핵심 원칙

개발을 시작하지 못하는 문제는 지식 부족보다 **첫 구현 단위가 너무 큰 문제**인 경우가 많다. API와 파이프라인은 전체 지도이고, 실제 구현은 검증 가능한 사용자 행동 하나에서 시작한다.

```text
사용자 행동 하나 → 입력·출력 예시 → 테스트 → 최소 구현 → 외부 연동 → 실패 처리 → 다음 기능
```

## zipsAI의 권장 착수 순서

### 1. FastAPI 실행과 Health Baseline

첫 Issue는 서버가 실행되고 `GET /health`가 팀과 합의한 응답을 반환하는 것으로 제한한다.

- 서버가 실행된다.
- `/health`가 약속한 JSON과 상태 코드를 반환한다.
- 정상 응답 테스트가 있다.
- `uv run ruff format --check .`, `uv run ruff check .`, `uv run pytest`가 통과한다.

프로젝트에는 이미 `pyproject.toml`, `uv.lock`, `src/zipsai/`가 있으므로 프로젝트를 새로 초기화하지 않는다.

### 2. `/converse` 계약 확정

코드보다 먼저 정상 요청, 정상 응답, 필수 필드 누락 요청을 각각 JSON 예시로 작성한다. `route`, `state`, `intent`는 Enum으로 확정한 뒤 `contracts/`의 Pydantic 모델로 옮긴다.

코딩 전에는 아래 세 문장을 쓴다.

```text
입력: 어떤 요청인가?
출력: 성공하면 어떤 구조가 반환되는가?
실패: 가장 중요한 실패는 어떻게 표현되는가?
```

이 문장을 쓸 수 없다면 구현을 시작하지 않고 Backend와 API 계약을 먼저 맞춘다.

### 3. 텍스트 민원 한 개의 세로 흐름

첫 의미 있는 기능은 이미지, RAG, 멀티턴을 제외한 텍스트 민원 하나다.

```text
POST /converse
→ 요청 검증
→ 민원 분석
→ complaint draft 응답
```

단위 테스트는 실제 LLM 대신 고정된 분석 결과를 사용한다. 실제 모델 호출은 `integrations/`에서만 수행하고, API 라우터·기능 로직·외부 호출을 분리한다.

### 4. RAG와 라우팅은 기능이 생긴 뒤 추가

다음 순서는 문서 청킹과 검색, `no_evidence` 처리, 근거가 있는 답변, 최소 라우팅이다. 민원과 지식 질의 경로가 둘 다 생기기 전에는 LangGraph나 복잡한 상태 그래프를 만들지 않는다. V1에서는 Enum과 단순 분기로 충분하며, 멀티턴 상태 전이는 V2에서 필요해질 때 도입한다.

## 현재 초안의 취급

루트의 `converse.py`와 `model.py`는 탐색용 초안으로 보관할 수 있지만, 서비스 구현의 기반으로 계속 확장하지 않는다.

- 요청·응답 모델은 `src/zipsai/contracts/`로 옮긴다.
- 모델 호출은 import 시 실행하거나 `print()`하지 않고 `src/zipsai/integrations/`의 함수에서 수행한다.
- 모델명, URL, 타임아웃은 settings에서 읽는다.
- 단위 테스트는 실제 LLM·VLM·Vector DB를 호출하지 않는다.

## 작업 습관

- 코딩 전 `입력 / 출력 / 주요 실패`를 한 줄씩 쓴다.
- 같은 패턴이 있는지 `rg`로 저장소를 먼저 검색한다.
- 한 번에 한 동작만 구현하고 즉시 실행한다.
- 모르는 API 이름은 공식 문서에서 찾고, 작은 예제로 실제 반환값을 확인한다.
- 테스트는 정상 사례 하나와 중요한 실패 사례 하나부터 작성한다.
- 중복이 실제로 생기기 전에는 공통 모듈·Factory·Interface를 만들지 않는다.
- 커밋 전 `git diff`를 리뷰어 관점에서 읽고, 하나의 논리적 변경만 커밋한다.
- 작성한 함수의 책임과 분리 이유를 말로 설명할 수 있어야 한다.

## 공식 문서 사용 순서

문서는 처음부터 끝까지 읽는 교재보다, 현재 Issue의 질문을 푸는 참조 자료로 사용한다.

1. [FastAPI First Steps](https://fastapi.tiangolo.com/tutorial/first-steps/)
2. [Pydantic Models](https://pydantic.dev/docs/validation/latest/concepts/models/)
3. [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
4. [FastAPI Bigger Applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
5. 필요한 시점의 OpenAI, LangGraph, Vector DB 문서

## 오늘의 한 가지 작업

`[AI] FastAPI 실행 및 Health Baseline` Issue를 만들고 `/health` 하나를 테스트까지 통과시킨다. 완료 뒤에는 `/converse`의 정상 요청 JSON과 실패 요청 JSON을 하나씩 작성한다.
