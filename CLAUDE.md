# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

`zipsAI` AI 레포 — 민원(complaint) 처리 및 건물 문서 RAG 질의를 담당하는 FastAPI 기반 서비스.
As of this writing the repo contains only `README.md`, `docs/CONVENTIONS.md`, `docs/PROJECT.md`, and `.gitignore` — no `src/`, `pyproject.toml`, or code exists yet. The directory layout and rules below are the team's agreed target structure (from `docs/CONVENTIONS.md`); scaffold into it rather than inventing a different layout.

For product context (why this exists, target users, AI module boundaries, API contract) see [docs/PROJECT.md](docs/PROJECT.md).

## Commands

Once the project is scaffolded (`pyproject.toml` present), use `uv`:

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

Run a single test: `uv run pytest tests/unit/test_<name>.py::test_case`.

Python 3.12, dependencies/venv managed by `uv`, `uv.lock` is committed.

## Architecture (target layout)

```
src/zipsai/
├── main.py, settings.py, errors.py, observability.py
├── api/            # FastAPI routers: request validation, service calls, HTTP response conversion only
├── contracts/      # Pydantic request/response models shared with backend
├── orchestration/  # Main Agent routing + conversation state transitions
├── complaint/      # 민원 정보 추출, 이미지 분석, 초안 보완, 필수 필드 검증
├── knowledge/      # 건물 문서 검색, 근거 기반 답변, citation 구성
├── indexing/       # 문서 파싱 → 정제 → 청킹 → 임베딩 → 색인
└── integrations/   # LLM/VLM/embedding/Vector DB/Docling calls — the only place external model calls live
tests/{unit,integration}/
evals/              # 민원 추출 정확도 / RAG 답변 품질 — model-output evaluation, not pass/fail unit tests
```

Dependency direction (one-way, do not import backwards):

```
api -> orchestration -> complaint | knowledge -> integrations
api -> indexing -> integrations
```

- Feature code must not depend on FastAPI objects or HTTP status codes.
- Each feature's prompts live in that feature's own `prompts.py`, not centralized.
- `route` / `state` / `intent` fixed values are Enums.
- No `utils.py`/`common.py` grab-bags; no shared module until real duplication exists.
- No single-implementation Interface/Base class/Factory.
- Don't pre-create empty directories for future features (e.g. `insights/`) — add them when the issue actually starts.
- Model names, external URLs, timeouts belong in settings, not hardcoded.
- Use the common logger with `trace_id` preserved; never `print()`.
- Unit tests never call real LLM/VLM/Vector DB — that's what `evals/` and integration tests are for.
- RAG that finds no evidence returns `no_evidence` rather than guessing — this is a required test case for any RAG change.

## Naming

| Target | Rule | Example |
|---|---|---|
| Python files/dirs | `snake_case` | `vector_store.py` |
| Test files | `test_<subject>.py` | `test_router.py` |
| Classes/Pydantic models | `PascalCase` | `ConverseRequest` |
| Functions/variables | `snake_case` | `retrieve_documents` |
| Constants/env vars | `UPPER_SNAKE_CASE` | `MODEL_TIMEOUT_SECONDS` |
| Internal-only functions | leading `_` | `_validate_state` |
| Eval data | `<feature>_cases.jsonl` | `complaint_cases.jsonl` |

## Git workflow

- `main` and `dev` are always releasable/runnable; never push directly — PRs only, one non-author AI reviewer approves.
- Branch: `<type>/<issue-number>/<keyword>` off latest `dev`, e.g. `feat/12/complaint-baseline`. One issue per branch.
- Commit: `<type>(<scope>): <summary>` (Korean summary, no trailing period), e.g. `fix(agent): route와 state 정합성 검사`. `type` ∈ feat/fix/refactor/test/docs/chore/build. `build`는 Dockerfile/compose.yaml/의존성 등 빌드·배포 설정 변경 전용.
- PR title: `[type] #<issue-number> 작업 설명`. PR target is `dev` (uses `Refs #n`, not `Closes #n` — `Closes` only auto-closes on the `dev`→`main` release PR). Squash-merge feature PRs into `dev`; the `dev`→`main` release PR uses a merge commit, not squash.
- Full PR/issue body templates (목적/작업 목록/완료 조건, checklist including `.env.example` and no-secrets checks) are in `docs/CONVENTIONS.md` §3 and §6 — follow them when opening issues/PRs in this repo.
