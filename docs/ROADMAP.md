# zipsAI AI 로드맵 (V1 → V3)

3개 버전(V1/V2/V3), 각 3주 스프린트로 진행한다. 트랙 구분은 [CLAUDE.md](../CLAUDE.md)의 모듈 경계(`complaint/` · `knowledge/` · `orchestration`+`integrations`)와 대응한다. 배경은 [PROJECT.md](PROJECT.md) 참고.

## 한눈에 보기

| 주차 | 버전 | 제목 | 상태 |
|---|---|---|---|
| week4 (1주차) | v1 | V1 설계 / Baseline | Not started |
| week5 (2주차) | v1 | V1 핵심 기능 연결 | Not started |
| week6 (3주차, 배포) | v1 | V1 MVP E2E / 평가 / 배포 | Not started |
| week7 (1주차) | v2 | V2 Agent 판단 구조 설계 | Not started |
| week8 (2주차) | v2 | V2 Multi-turn / Tool Calling | Not started |
| week9 (3주차, 배포) | v2 | V2 Agentic 민원 처리 E2E | Not started |
| week10 (1주차) | v3 | V3 축적 데이터 활용 시작 | Not started |
| week11 (2주차) | v3 | V3 관리자 Intelligence 고도화 | Not started |
| week12 (3주차, 배포) | v3 | V3 관리자 Agent 배포 | Not started |

각 주차 표의 트랙 구성:
- **민원 분석 (Complaint/VLM)** — `complaint/`
- **지식 검색 (RAG/Knowledge)** — `knowledge/`
- **Agent / 오케스트레이션** — `orchestration/` + `integrations/` + Backend 연동
- **주간 산출물** — 그 주에 실제로 나와야 하는 결과물

---

## V1 — 대화형 민원 접수 MVP (week4–6)

### week4 · 1주차 · V1 설계 / Baseline

**목표**: 대화형 민원 접수의 입력·출력 계약 확정

| 민원 분석 | 지식 검색 (RAG) | Agent / 오케스트레이션 |
|---|---|---|
| 민원 Label / Schema 정의 | RAG 대상 범위 정의 | Main Agent 최소 범위 정의 |
| 사진+텍스트 입력 구조화 기준 설계 | 공지·규칙·매뉴얼 Ingest baseline | Intent: 민원 / RAG / 일반대화 |
| VLM baseline 구현 | Parsing / Chunking 전략 정의 | Agent State / Interface 정의 |
| issue_type, location, description 추출 | Embedding / Vector DB 선정 | VLM / RAG Tool Interface |
| | Retrieval baseline 구현 | AI↔Backend API Schema 확정 |

**주간 산출물**: VLM 구조화 baseline · 간단 RAG baseline · Agent State / API Schema

**상태**: Not started

---

### week5 · 2주차 · V1 핵심 기능 연결

**목표**: 대화에서 민원 접수와 건물 질의까지 연결

| 민원 분석 | 지식 검색 (RAG) | Agent / 오케스트레이션 |
|---|---|---|
| 이미지+텍스트 민원 분석 고도화 | Ingest pipeline 완성 | 민원 / RAG / 일반대화 최소 Routing |
| unknown / 판단 불가 처리 | Retrieval → Generation 연결 | Agent에서 VLM·RAG Tool 연결 |
| Output validation | 근거 포함 답변 구성 | 민원 DB 등록 API 연결 |
| 민원 등록용 Backend output 확정 | 검색 결과 부족 시 fallback 처리 | 기능별 테스트셋 구축 |

**주간 산출물**: 대화형 민원 접수 E2E · 대화형 건물 질의응답 · 민원 DB 등록 연결

**상태**: Not started

---

### week6 · 3주차 · 배포 · V1 MVP E2E / 평가 / 배포

**목표**: 대화형 민원 접수 MVP 완성

| 민원 분석 | 지식 검색 (RAG) | Agent / 오케스트레이션 / 배포 |
|---|---|---|
| 실패 케이스 분석 | Retrieval / Generation 품질 평가 | 민원 접수 → DB → 관리자 확인 E2E |
| Prompt / 모델 조정 | 답변 근거·누락 케이스 점검 | Routing 평가 |
| 민원 구조화 품질 평가 | 건물별 Filtering 검증 | AI→Backend→UI 검증 |
| Dashboard 연동 확인 | RAG 오류 수정 | 오류 처리 / 로그, Interface / Schema 정리, 배포 및 운영 안정화 |

**주간 산출물**: **V1 대화형 민원 접수 MVP 배포** — VLM 구조화 + 간단 RAG + 최소 Routing

**상태**: Not started

---

## V2 — Agentic 민원 처리 (week7–9)

### week7 · 1주차 · V2 Agent 판단 구조 설계

**목표**: 단순 Routing에서 민원 triage workflow로 확장

| 민원 Triage | Tool 설계 | Agent Workflow |
|---|---|---|
| 민원 긴급도 / 심각도 판단 기준 설계 | Complaint Agent Workflow 설계 | Understand → Triage → Retrieve → Decide → Act 설계 |
| 정보 충분성 판단 요소 정의 | RAG / History / RDB / Web Tool 역할 정의 | Complaint State / Triage Schema 정의 |
| 추가 이미지 / 정보 필요 여부 | Tool 선택 정책 | Multi-turn 상태 관리 설계 |
| 이미지-텍스트 불일치 기준 | Tool 실패 / fallback 정책 설계 | Conversation State / Complaint DB 분리 |
| 추가 질문 필요 여부 설계 | | Workflow와 Backend 상태 전이 정렬 |

**주간 산출물**: V2 Agent 판단 구조 · 민원 Triage 기준 · Tool Orchestration 설계

**상태**: Not started

---

### week8 · 2주차 · V2 Multi-turn / Tool Calling

**목표**: 상황에 맞는 정보 수집과 도구 선택 연결

| 민원 분석 | Tool 연결 | Agent / Backend 연동 |
|---|---|---|
| Multi-turn 민원 Agent 구현 | RAG / RDB / Complaint History / Web Tool 연결 | Multi-turn Conversation State 연결 |
| 부족한 정보 추가 질문 | Query별 Tool 선택 | Tool Calling 구현 |
| 새 이미지·텍스트 반영 | Tool fallback | 민원 Workflow와 Backend API 연동 |
| 민원 정보 업데이트 | Tool 결과 통합 | 민원 상태 조회 / 변경 연결 |
| 긴급도 재평가 | Service Tool Interface 연결 | |

**주간 산출물**: Agent 기반 추가 질문 · RAG·Web·History·DB Orchestration · 민원 정보 업데이트

**상태**: Not started

---

### week9 · 3주차 · 배포 · V2 Agentic 민원 처리 E2E

**목표**: 판단·도구 사용·상태 반영 안정화

| 민원 Triage 평가 | Tool / RAG 품질 | E2E / 배포 |
|---|---|---|
| 긴급도 평가 | Tool Routing / Fallback 평가 | Complaint Workflow E2E |
| 정보 충분성 / 불일치 평가 | RAG·Web·History·DB 결과 품질 점검 | 민원 상태 전이 검증 |
| 실패 케이스 분석 | Knowledge Routing 개선 | 관리자 알림 연결 |
| Triage 기준 개선 | Agent 응답 안정화 | Service Tool E2E, 통합 평가 / 배포 안정화 |

**주간 산출물**: **V2 Agentic 민원 처리 배포** — 긴급도 판단 + Multi-turn + Tool Calling + 상태 반영

**상태**: Not started

---

## V3 — 축적 데이터 기반 관리자 Intelligence (week10–12)

### week10 · 1주차 · V3 축적 데이터 활용 시작

**목표**: 유사·반복 민원 관계 분석 기반 마련

| 유사 민원 분석 | Operational Knowledge | 연결 / Interface |
|---|---|---|
| 유사 민원 검색 baseline | Operational Knowledge 구조 설계 | 민원 데이터 / Embedding / Metadata 연결 |
| 민원 데이터셋 정리 | 과거 민원·처리 결과 Retrieval | History / Knowledge / Agent 연결 |
| 민원 Embedding + Metadata 결합 | 건물 정보 검색 구조 확장 | 분석 결과 Interface 설계 |
| 반복 문제 탐지 기준 설계 | Cold Start 전략 설계, 분석 결과를 Agent Context로 연결 | Dashboard 연동 계약 정의 |

**주간 산출물**: 과거 유사 사례 조회 · 반복 민원 탐지 baseline · Operational Knowledge 연결

**상태**: Not started

---

### week11 · 2주차 · V3 관리자 Intelligence 고도화

**목표**: 건물 단위 반복 문제와 운영 Insight 제공

| 반복 문제 분석 | Knowledge 계층화 | 관리자 연결 |
|---|---|---|
| 유사 민원 검색 고도화 | Building / Operational / Common Knowledge 계층화 | Dashboard Insight 연결 |
| 민원 Clustering | 건물 단위 Knowledge Context 확장 | 관리자용 결과 Schema 정의 |
| 시간·위치·유형 기반 반복 문제 분석 | Insight 생성용 RAG / Agent 고도화 | 과거 사례를 Agent 판단에 활용 |
| 건물 단위 문제 후보 탐지 | 관리자 질문·분석 Workflow 연결, Tool Orchestration 고도화 | 추천 / 알림 Interface 설계 |

**주간 산출물**: 건물 단위 문제 탐지 · 반복·유사 민원 Insight · 관리자 질문형 분석

**상태**: Not started

---

### week12 · 3주차 · 배포 · V3 관리자 Agent 배포

**목표**: Insight를 예방적 추천과 실제 action으로 연결

| 성능 개선 | Action Tool | E2E / 평가 / 배포 |
|---|---|---|
| VLM / Similarity 성능 개선 | Operational Knowledge 기반 Agent / RAG 최적화 | Insight → 추천 → Action E2E |
| 반복 문제 탐지 평가 | Action Tool 연결 | 관리자 승인 기반 Action 연결 |
| 예방 가능성 판단 | Tool 선택 / 응답 품질 튜닝 | Dashboard / 알림 / 업무 API 연결 |
| Confidence / Latency 최적화 | Action 실패 / Fallback 분석 | Agent Evaluation Pipeline, Task Success / Tool Selection 평가 |
| 분석 실패 케이스 개선 | Cold Start 평가 | Token / Cost / Latency 최적화, 최종 배포 안정화 |

**주간 산출물**: **V3 관리자 Intelligence / Action Agent 배포** — 반복 문제 탐지 → 예방적 추천 → 관리자 승인 기반 Action

**상태**: Not started
