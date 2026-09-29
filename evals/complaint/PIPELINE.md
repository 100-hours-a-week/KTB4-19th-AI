# 민원 평가 파이프라인

> 현재 실행 가능한 명령과 구현 범위는 [README.md](README.md)를 따른다. 아래는 확장 설계이며 일부 파일·지표·CI 연결은 아직 구현되지 않았다.

`METRICS.md`가 무엇을 재는지 정하고, 이 문서는 그것을 어떻게 반복 실행하고 쌓는지를 정한다.

목표는 점수를 한 번 내는 것이 아니라 **다음 개선이 효과가 있었는지 판정 가능한 상태**를 유지하는 것이다.

---

## 1. 설계 결정 네 개

이 네 개가 나머지를 결정한다.

### 1.1 실행과 채점을 분리한다

```text
run.py   모델 호출 → predictions.jsonl 만 남기고 끝. 점수 계산 안 함
score.py predictions.jsonl + 정답셋 → scores.json. 모델 호출 안 함
```

채점 기준은 계속 바뀐다. 동의어셋이 늘고, judge 프롬프트가 고쳐지고, 게이트 수치가 조정된다. 분리해두면 **과거 run 전부를 새 채점기로 다시 채점**할 수 있다. LLM 재호출 비용 0.

합쳐두면 채점기를 고칠 때마다 이전 이력과 비교가 불가능해진다. 이력이 쌓이지 않는다는 뜻이고, 그러면 파이프라인의 존재 이유가 사라진다.

### 1.2 저장소는 git이다

```text
evals/complaint/runs/2026-10-02T11-30Z-d73b94a/
├── predictions.jsonl   모델 원본 출력 (수백 KB)
├── meta.json           모델명·프롬프트 해시·git sha·실행 시각·비용
└── scores.json         채점 결과
```

둘 다 커밋한다. DB·대시보드·실험 추적 SaaS 없다. 이력 조회는 `git log`, 추세표는 `report.py`가 `runs/*/scores.json`을 훑어 `HISTORY.md`를 재생성한다 — `evals/synthetic/2-reproduce/make_report.py`와 같은 패턴.

210턴 규모에서 이보다 무거운 저장소는 관리 비용만 늘린다.

### 1.3 배포본이 아니라 노드 함수를 직접 호출한다

버킷 1~3은 `handle_complaint()`·`analyze_images()`를 직접 부른다. HTTP를 거치면 인증·네트워크·배포 버전이 변수로 섞여 **모델 품질 변화를 격리할 수 없다**.

배포본 검증은 별도 스모크 하나로 분리한다. `evals/e2e/converse_smoke.py` — `indexing_smoke.py`와 같은 형태로 응답 스키마와 상태 전이만 확인하고 품질은 재지 않는다.

| 대상 | 무엇을 잡나 |
| --- | --- |
| `evals/complaint/run.py` | 모델 품질 회귀 |
| `evals/e2e/converse_smoke.py` | 배포·계약 파손 |

### 1.4 응답 캐시를 둔다

`cache/<sha256(model + system + user)>.json`. 같은 모델·같은 프롬프트면 재호출하지 않는다.

한 run의 호출량:

| 항목 | 콜 수 |
| --- | --- |
| 턴 210 × (intent + extraction) | 420 |
| 사진 30 | 30 |
| **합계 / 1회** | **450** |
| 3회 반복 | 1,350 |

캐시가 있으면 `complaint/prompts.py`만 고친 경우 extraction 210콜만 새로 나가고 VLM 30장과 intent 210콜은 무료다. 프롬프트를 자주 만지는 단계에서 이게 실행 가능 여부를 가른다.

flip rate(비결정성)를 잴 때만 `--no-cache`로 우회한다.

---

## 2. 흐름

```text
complaint_cases.jsonl ─┐
routing_cases.jsonl ────┼→ run.py ─→ predictions.jsonl ─┐
photo_cases.jsonl* ─────┘                                │
                                                        ├→ score.py ─→ scores.json ─┐
동의어셋 · judge 프롬프트 ──────────────────────────────┘                            │
                                                                                     ├→ compare.py ─→ PASS / FAIL
                                                              baseline.json ─────────┘
                                                                                     │
                                                    runs/*/scores.json ─→ report.py ─→ HISTORY.md
```

`*` 사진 정답셋은 현재 없다. 이미지 후보 검수 후 `photo_cases.jsonl`을 추가한다. 현재 `--bucket all`은 민원과 라우팅만 실행한다.

---

## 3. 명령

```bash
cd ai
uv sync --group evals

# 예측 생성 (모델 호출)
uv run python -m evals.complaint.run --bucket all --repeat 3

# 채점 (모델 호출 없음)
uv run python -m evals.complaint.score runs/2026-10-02T11-30Z-d73b94a

# baseline 대비 판정 — 미달이면 exit 1
uv run python -m evals.complaint.compare runs/2026-10-02T11-30Z-d73b94a

# 추세표 갱신
uv run python -m evals.complaint.report
```

`--bucket`은 `classification` / `extraction` / `vlm` / `all`. 프롬프트 하나만 고쳤으면 해당 버킷만 돌린다.

채점기를 고친 뒤 전 이력 재채점:

```bash
for d in evals/complaint/runs/*/; do uv run python -m evals.complaint.score "$d"; done
uv run python -m evals.complaint.report
```

별도 도구 없이 쉘 루프로 끝난다. 1.1의 분리가 주는 이득이 여기서 나온다.

---

## 4. baseline과 회귀 판정

`baseline.json`은 **현재 승인된 점수**다. 자동 갱신하지 않는다.

```text
compare.py 판정:
  주지표가 baseline − tolerance 미달  → FAIL (exit 1)
  게이트 절대값 미달                  → FAIL
  그 외                               → PASS
```

- `tolerance`는 초기 0.03 고정. 3회 반복의 표준편차가 쌓이면 지표별로 바꾼다. 비결정성 폭보다 작게 잡으면 매번 헛경보가 난다.
- 개선이 확인되면 수동 승급: `cp runs/<id>/scores.json baseline.json` 후 커밋.

**`baseline.json`의 git diff가 곧 "개선이 보인다"의 실체다.** 언제 무엇이 몇 점 올랐는지가 커밋 이력에 남고, 그 커밋이 프롬프트 변경 커밋과 짝을 이룬다.

---

## 5. CI 연결

매 PR에 돌리면 안 된다. 비용이 들고, 비결정성 때문에 무관한 PR에서 헛실패가 난다.

기존 `ai-ci.yml`의 `paths-filter`에 필터를 하나 추가하는 것으로 시작한다.

| 트리거 | 하는 일 |
| --- | --- |
| **프롬프트 경로 변경 PR** (`complaint/prompts.py`, `orchestration/prompts.py`) | run + score + compare, 결과를 PR 코멘트. FAIL이면 체크 실패 |
| PR에 `eval` 라벨 | 위와 동일 (프롬프트 외 변경을 수동으로 평가할 때) |
| `schedule` 야간 1회 | 전체 버킷 3회 반복, `runs/`에 커밋 |

프롬프트 경로 필터가 핵심이다. "프롬프트를 바꿨으면 평가가 자동으로 돈다"가 되면 평가를 건너뛸 수가 없어진다. `CONVENTIONS.md` 11절의 "프롬프트나 모델을 변경하면 같은 평가 데이터로 변경 전후를 비교한다"를 사람 규율이 아니라 CI로 강제하는 것이다.

시크릿은 기존 `AWS_AI_CI_ROLE_ARN` 패턴을 따르고, LLM 키는 Actions secret으로 추가한다.

---

## 6. 착수 순서

한 번에 다 만들면 끝나지 않는다. 각 단계가 끝날 때마다 쓸 수 있는 상태가 되게 쪼갠다.

| 단계 | 산출물 | 이 단계가 끝나면 |
| --- | --- | --- |
| **1** | 정답셋 20턴 + 사진 10장, `run.py` + `score.py`, 게이트 없음 | 숫자를 처음 본다. 지표 정의의 허점이 여기서 드러난다 |
| **2** | 정답셋 전량(210턴·30장), `baseline.json` 고정, `compare.py` | 회귀 판정이 된다. 개선 여부를 말할 수 있다 |
| **3** | `report.py` + `HISTORY.md`, CI 프롬프트 경로 트리거 | 개선이 누적으로 보인다. 평가를 건너뛸 수 없다 |
| **4** | judge(E15·E16) + 인간 라벨 50건 κ 검증 | symptom 품질을 판정할 수 있다 |

judge를 마지막에 두는 이유: κ 검증에 인간 라벨링이 필요해서 먼저 착수하면 1단계에서 막힌다. 그동안 symptom은 E13(하드룰)·E14(내용어 재현율)로만 잰다. 둘 다 자동이고 회귀 감시에는 충분하다.

1단계의 정답셋 20턴은 `METRICS.md` §5의 준비물 중 `origin` 라벨만 갖추면 시작할 수 있다. 동의어셋·음성셋·refs는 2단계에서 채운다.

---

## 7. 만들지 않는 것

| 안 만드는 것 | 대신 |
| --- | --- |
| 대시보드 | `HISTORY.md` 마크다운 표 |
| 실험 추적 SaaS (W&B, Langfuse) | `runs/` + git |
| 병렬 실행기 | 순차 실행. 450콜은 캐시 없이도 수 분이다 |
| 결과 저장 DB | jsonl |
| 평가용 별도 프롬프트 버전 관리 | `meta.json`의 프롬프트 해시 + git sha |

규모가 열 배로 커지고 순차 실행이 실제로 막힐 때 다시 본다.
