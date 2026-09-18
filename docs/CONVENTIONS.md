# zipsAI 개발 컨벤션

## 1. 기본 원칙

- 모든 작업은 GitHub Issue에서 시작한다.
- Issue 하나는 담당자 한 명이 1~2일 안에 완료할 수 있는 크기로 작성한다.
- `main`은 릴리스 가능한 안정 버전, `dev`는 다음 릴리스를 통합하는 버전으로 사용한다.
- `main`과 `dev`는 항상 실행 가능한 상태를 유지한다.
- `main`과 `dev`에 직접 push하지 않는다. 모든 변경은 Pull Request로 병합한다.
- 작성자가 아닌 AI 팀원 한 명이 코드를 확인하고 승인한 뒤 병합한다.

---

## 2. GitHub 작업 흐름

### 2.1 Project 상태

GitHub Project에서 아래 상태를 사용한다.


| 상태                  | 의미                                  |
| ------------------- | ----------------------------------- |
| `Backlog`           | 해야 하지만 아직 일정이 정해지지 않은 작업            |
| `Ready`             | 요구사항과 완료 조건이 정리되어 시작할 수 있는 작업       |
| `In Progress`       | 담당자가 개발 중인 작업                       |
| `In Review`         | PR이 Ready for review 상태인 작업         |
| `Ready for Release` | PR이 `dev`에 병합되어 `main` 릴리스를 기다리는 작업 |
| `Done`              | PR이 병합되고 Issue가 닫힌 작업               |


### 2.2 작업 순서

1. Issue를 생성한다.
2. Issue를 GitHub Project에 추가하고 담당자와 우선순위를 지정한다.
3. 작업 시작 시 상태를 `In Progress`로 변경한다.
4. 최신 `dev`에서 Issue 번호를 포함한 브랜치를 생성한다.
5. 작업이 하루 이상 걸리면 Draft PR을 먼저 열어 진행 상황을 공유한다.
6. `dev`를 대상으로 PR을 만들고 Issue를 수동 연결한다.
7. 검증이 끝나면 PR을 Ready for review로 바꾸고 Project 상태를 `In Review`로 변경한다.
8. 상대 팀원의 승인을 받은 뒤 `dev`에 Squash merge한다.
9. 브랜치를 삭제하고 Project 상태를 `Ready for Release`로 변경한다.
10. 통합 검증 후 `dev`에서 `main`으로 릴리스 PR을 만든다.
11. 릴리스 PR에 포함된 Issue를 `Closes #번호`로 나열한다.
12. 릴리스 PR이 병합되면 Issue와 Project 상태가 `Done`인지 확인한다.

---

## 3. Issue 규칙

### 3.1 제목

```text
[과정명] 기능 및 키워드
```

예시:

```text
[AI] 문서 인덱싱
[AI] Main Agent Baseline
[AI] 민원 처리 Baseline 
[AI] RAG 질의 처리 
```

- `과정명`은 팀 또는 트랙을 나타낸다. 현재 AI 레포에서는 `[AI]`를 사용한다.
- 기능과 키워드는 작업 결과가 드러나도록 작성한다.
- 담당자는 GitHub Assignee 기능을 활용한다.
- 작업 종류는 제목에 반복하지 않고 Issue Label의 `feat`, `fix`, `refactor`, `test`, `docs`, `chore`로 구분한다.

### 3.2 Type


| Type       | 용도                  |
| ---------- | ------------------- |
| `feat`     | 새로운 기능              |
| `fix`      | 버그 수정               |
| `refactor` | 동작 변경 없는 코드 구조 개선   |
| `test`     | 테스트 및 평가 데이터        |
| `docs`     | 문서 변경               |
| `chore`    | 설정, 의존성, 빌드 등 기타 작업 |


### 3.3 본문

```md
## 목적

이 작업이 필요한 이유를 작성한다.

## 작업 목록

- [ ] [인덱싱] 문서 파싱하기 (2026/09/14 13:00~14:00)
- [ ] [인덱싱] 문서 청킹하기 (0.5시간)
- [ ] [인덱싱] 임베딩 입력 형식 확인하기 (0.5시간)

## 완료 조건

- [ ] 예시 문서가 파싱과 청킹을 거쳐 검색 가능한 형태로 변환된다.
- [ ] 관련 테스트와 Ruff 검사가 통과한다.

## 참고

- 관련 API 문서, 설계 문서 또는 선행 Issue
```

### 3.4 작업 목록 작성 규칙

- 작업은 완료 여부를 체크할 수 있는 원자 단위로 나눈다.
- 예상 시간은 `0.5시간` 단위로 작성한다.
- 한 작업은 가능하면 `0.5~1시간`, 최대 `2시간`을 넘지 않게 나눈다.
- 일정이 정해졌다면 `(2026/09/14 13:00~14:00)` 형식으로 작성한다.
- 일정이 아직 정해지지 않았다면 `(0.5시간)`, `(1시간)`처럼 예상 소요 시간만 작성한다.
- 체크박스 앞에는 `[인덱싱]`, `[민원]`, `[RAG]`, `[Agent]`처럼 작업 영역을 표시한다.
- 작업 과정이 아니라 결과가 드러나는 동사를 사용한다. 예: `검토하기`, `구현하기`, `검증하기`.
- 작업이 끝나면 즉시 체크하고, 예상보다 커지면 남은 작업을 별도 체크박스나 Issue로 분리한다.
- 완료 조건이 불명확하면 Issue를 `Ready`로 옮기지 않는다.

---

## 4. 브랜치 규칙

### 4.1 형식

```text
<type>/<issue-number>/<keyword>
```

- 영문 소문자와 숫자를 사용한다.
- 단어는 하이픈으로 구분한다.
- 키워드는 작업 내용을 알아볼 수 있을 정도로 짧게 작성한다.

예시:

```text
feat/12/complaint-baseline
feat/18/knowledge-rag
feat/21/document-indexing
fix/27/conversation-state
refactor/31/vector-store
docs/35/api-contract
```

### 4.2 생성

```bash
git switch dev
git pull --ff-only
git switch -c feat/12/complaint-baseline
```

하나의 브랜치에서는 하나의 Issue만 처리한다. 작업 중 다른 문제가 발견되면 현재 PR에 섞지 않고 별도 Issue를 생성한다.

---

## 5. 커밋 규칙

### 5.1 형식

```text
<type>(<scope>): <summary>
```

예시:

```text
feat(complaint): 민원 필수 정보 추출 추가
fix(agent): route와 state 정합성 검사
test(knowledge): 근거 없는 질의 사례 추가
refactor(indexing): 청킹 로직 분리
docs(api): converse 응답 계약 수정
chore: Ruff 설정 추가
```

- `type`은 Issue Type과 같은 값을 사용한다.
- `scope`는 `agent`, `complaint`, `knowledge`, `indexing`, `api`처럼 변경 영역을 작성한다.
- 제목은 변경 결과가 드러나게 작성한다.
- 마침표를 붙이지 않는다.
- 하나의 커밋에는 하나의 논리적 변경만 포함한다.
- Issue 번호는 브랜치와 PR에서 관리하므로 커밋마다 반복하지 않아도 된다.

---

## 6. Pull Request 규칙

### 6.1 제목

```text
[type] #<issue-number> 작업 설명
```

예시:

```text
[feat] #12 민원 Baseline 구현
[fix] #27 대화 상태 정합성 오류 수정
```

### 6.2 본문

```md
## 관련 Issue

Refs #12

## 변경 내용

- 무엇을 변경했는지 작성한다.

## 확인 방법

```bash
uv run pytest
uv run ruff check .
```

## 확인 결과

- 정상 시나리오:
- 실패 시나리오:
- 프롬프트 변경 전후 결과 또는 평가 결과:

## 리뷰 요청 사항

- 리뷰어가 집중해서 확인할 부분을 작성한다.

## 체크리스트

- [ ] Issue 범위에 해당하는 변경만 포함했다.
- [ ] API 및 상태 계약을 지켰다.
- [ ] 관련 테스트를 추가하거나 수정했다.
- [ ] pytest와 Ruff가 통과한다.
- [ ] 새 환경변수를 `.env.example`에 반영했다.
- [ ] 비밀값과 개인정보가 포함되지 않았다.
```

- 기능 PR은 `dev`를 대상으로 만들고 `Refs #번호`를 사용한다.
- 기능 PR의 Development 사이드바에서 관련 Issue를 수동으로 연결한다.
- `dev` 대상 PR에서는 `Closes #번호`가 Issue를 자동 종료하지 않는다.
- Issue 자동 종료는 `dev`에서 `main`으로 보내는 릴리스 PR에 `Closes #번호`를 작성해 처리한다.
- 선행 PR이 있다면 `Depends on #PR번호`를 작성한다.
- 미완성 작업은 Draft PR로 공유하되 리뷰를 요청하지 않는다.
- PR 작성자는 제출 전 변경 파일을 직접 한 번 검토한다.

### 6.3 리뷰와 병합

- 최소 한 명의 승인이 필요하다.
- 리뷰어는 기능 동작, API 계약, 오류 처리, 테스트를 우선 확인한다.
- 수정 요청은 반영 여부와 이유를 댓글로 남긴다.
- 모든 대화가 해결되고 검사에 통과한 뒤 작성자가 병합한다.
- 기능 브랜치는 `dev`에 Squash merge한다.
- 병합 후 원격 브랜치를 삭제한다.

### 6.4 릴리스 PR

통합 테스트를 통과한 `dev`를 `main`으로 병합한다.

```md
## 릴리스 내용

- Main Agent Baseline
- 민원 Baseline
- RAG 인덱싱 및 질의 Baseline

## 포함 Issue

Closes #12
Closes #18
Closes #21

## 통합 검증

- [ ] 민원 대화 흐름
- [ ] 건물 문서 질의 흐름
- [ ] 문서 인덱싱 후 검색 흐름
- [ ] Ruff 및 pytest 통과
```

- `dev`에서 `main`으로 향하는 릴리스 PR은 Squash하지 않고 Merge commit을 사용한다.
- 릴리스 PR에는 새로운 기능 수정이나 리팩터링을 추가하지 않는다.
- 릴리스 중 발견한 문제는 별도 Issue와 `fix/*` 브랜치로 처리한다.
- 릴리스 후 필요하면 `v0.1.0`과 같이 태그를 생성한다.

### 6.5 긴급 수정

- 운영 중인 `main`의 긴급 수정은 `main`에서 `hotfix/<issue-number>/<keyword>` 브랜치를 생성한다.
- Hotfix PR은 `main`으로 보내고 상대 팀원 리뷰 후 병합한다.
- 병합된 Hotfix는 `main`에서 `dev`로 다시 병합해 두 브랜치를 동기화한다.

---

## 7. Python 환경

- Python `3.12`를 사용한다.
- 로컬, CI, Docker에서 같은 Python minor 버전을 사용한다.
- 패키지와 가상환경은 `uv`로 관리한다.
- 프로젝트 설정은 `pyproject.toml`에서 관리한다.
- `uv.lock`을 커밋한다.
- 포맷과 린트는 Ruff, 테스트는 pytest를 사용한다.
- `.env`는 커밋하지 않고 필요한 키만 `.env.example`에 작성한다.

기본 확인 명령:

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

---

## 8. 디렉터리 규칙(선택)

```text
repository/
├── src/
│   └── zipsai/
│       ├── main.py
│       ├── settings.py
│       ├── errors.py
│       ├── api/
│       ├── contracts/
│       ├── orchestration/
│       ├── complaint/
│       ├── knowledge/
│       ├── indexing/
│       ├── integrations/
│       └── observability.py
├── tests/
│   ├── unit/
│   └── integration/
├── evals/
├── .env.example
├── .python-version
├── pyproject.toml
├── uv.lock
├── CONVENTIONS.md
├── CLAUDE.md
└── README.md
```

### 8.1 책임


| 경로               | 책임                                       |
| ---------------- | ---------------------------------------- |
| `api/`           | FastAPI 요청 수신, 입력 검증, 서비스 호출, HTTP 응답 변환 |
| `contracts/`     | Backend와 공유하는 Pydantic 요청·응답 모델          |
| `orchestration/` | Main Agent의 요청 라우팅과 대화 상태 전이             |
| `complaint/`     | 민원 정보 추출, 이미지 분석, 초안 보완, 필수 필드 검증        |
| `knowledge/`     | 건물 문서 검색, 근거 기반 답변, citation 구성          |
| `indexing/`      | 문서 파싱, 정제, 청킹, 임베딩, 색인                   |
| `integrations/`  | LLM, VLM, 임베딩 모델, Vector DB, Docling 호출  |
| `tests/`         | 상태, 계약, 오류 처리와 통합 흐름 검증                  |
| `evals/`         | 민원 추출 및 RAG 품질 평가 사례                     |


### 8.2 의존 방향

```text
api -> orchestration -> complaint | knowledge -> integrations
api -> indexing -> integrations
```

- 반대 방향으로 import하지 않는다.
- 기능 코드가 FastAPI 객체나 HTTP 상태 코드에 의존하지 않게 한다.
- 기능별 프롬프트는 해당 기능 디렉터리에 둔다.
- 실제 코드가 생길 때 디렉터리를 만들고 빈 폴더를 미리 생성하지 않는다.
- 후속 기능인 `insights/`는 관련 Issue가 시작될 때 추가한다.

---

## 9. 파일 및 이름 규칙


| 대상              | 규칙                      | 예시                      |
| --------------- | ----------------------- | ----------------------- |
| Python 파일       | `snake_case.py`         | `vector_store.py`       |
| 디렉터리·패키지        | `snake_case`            | `document_indexing/`    |
| 테스트 파일          | `test_<대상>.py`          | `test_router.py`        |
| 클래스·Pydantic 모델 | `PascalCase`            | `ConverseRequest`       |
| 함수·변수           | `snake_case`            | `retrieve_documents`    |
| 상수·환경변수         | `UPPER_SNAKE_CASE`      | `MODEL_TIMEOUT_SECONDS` |
| 내부 전용 함수        | 앞에 `_` 사용               | `_validate_state`       |
| 평가 데이터          | `<feature>_cases.jsonl` | `complaint_cases.jsonl` |


- 기능별 프롬프트는 해당 기능 폴더의 `prompts.py`에 둔다.
- 요청·응답 모델은 `contracts/`에 둔다.
- 외부 모델과 도구 호출은 `integrations/`에 둔다.
- 역할이 불분명한 `utils.py`, `common.py`는 만들지 않는다.
- 파일 하나에 여러 클래스가 있어도 책임이 같으면 억지로 분리하지 않는다.

---

## 10. 코드 작성 규칙

- API 라우터는 요청 검증, 서비스 호출, HTTP 응답 변환만 담당한다.
- 기능 로직은 `complaint/`, `knowledge/`, `indexing/`에 둔다.
- Main Agent의 라우팅과 상태 전이는 `orchestration/`에 둔다.
- 외부에 노출되는 함수와 요청·응답 모델에는 타입 힌트를 작성한다.
- `route`, `state`, `intent`의 고정 값은 Enum으로 관리한다.
- 모델명, 외부 URL, 타임아웃은 코드에 직접 쓰지 않고 설정에서 관리한다.
- `print()` 대신 공통 로거를 사용하고 `trace_id`를 유지한다.
- 예외를 무시하지 않고 호출자가 처리할 수 있는 오류로 변환한다.
- 표준 라이브러리와 기존 의존성을 먼저 확인한 뒤 새 패키지를 추가한다.
- 구현이 하나뿐인 Interface, Base 클래스, Factory는 만들지 않는다.
- 실제 중복이 생기기 전에 공통 모듈을 만들지 않는다.
- 주석은 코드가 무엇을 하는지가 아니라 왜 그렇게 했는지를 설명할 때만 작성한다.

---

## 11. 테스트와 AI 평가

- `tests/`는 상태 전이, 스키마, 오류 처리처럼 결과가 고정된 코드를 검증한다.
- `evals/`는 민원 추출 정확도와 RAG 답변 품질처럼 모델 결과를 평가한다.
- 단위 테스트에서는 실제 LLM, VLM, Vector DB를 호출하지 않는다.
- 기능 PR에는 최소한 정상 사례와 주요 실패 사례를 포함한다.
- 프롬프트나 모델을 변경하면 같은 평가 데이터로 변경 전후를 비교한다.
- RAG가 근거를 찾지 못하면 추측하지 않고 `no_evidence`를 반환하는 사례를 검증한다.

---

## 12. 팀 소통 규칙

- 작업 시작 시 Issue 담당자와 Project 상태를 갱신한다.
- API 계약, 상태, 환경변수 변경은 코드 작성 전에 팀원에게 공유한다.
- 반나절 이상 막히면 Issue나 팀 채널에 원인과 필요한 도움을 남긴다.
- 설계 결정은 대화로만 끝내지 않고 관련 Issue 또는 문서에 기록한다.
- 리뷰는 가능하면 요청받은 당일 확인한다.

