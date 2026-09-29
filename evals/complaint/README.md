# 민원 AI 평가 실행

현재 seed 정답셋은 단독 텍스트 110턴, 대화 11건(28턴), 라우팅 20건이다. 작성·계약 검증은 끝났지만 도메인 담당자의 정답 검수는 아직 필요하다. VLM 골든셋은 이미지 검수 후 작성한다.

별도 [생성 이미지 후보 120장](IMAGE-CANDIDATES.md)은 `photos/candidates/`와 `image_candidates.jsonl`에 접수했다. 아직 픽셀 기준 정답·독립 검수가 없어 이 `run.py`의 VLM 평가셋에 포함하지 않는다.

저장소 루트에서 실행한다. 실제 모델을 호출하므로 `.env`의 `LLM_API_KEY`·`LLM_MODEL` 설정이 필요하다.

```bash
PYTHONPATH=.:ai/src ai/.venv/bin/python -m evals.complaint.run --validate-only
```

이 검사는 모델 호출 없이 158개 텍스트·라우팅 사례의 계약과 필수 필드를 확인한다.

```bash
PYTHONPATH=.:ai/src ai/.venv/bin/python -m evals.complaint.run --bucket complaint
PYTHONPATH=.:ai/src ai/.venv/bin/python -m evals.complaint.run --bucket routing
# 현재 두 영역을 한 baseline으로 비교할 때
PYTHONPATH=.:ai/src ai/.venv/bin/python -m evals.complaint.run --bucket all
```

`--bucket vlm`은 검수된 사진을 `photo_cases.jsonl`에 등록한 뒤 사용할 수 있다. 지금 실행하면 정답셋 부재 오류를 낸다.

각 명령이 출력한 `RUN_DIR`에 `predictions.jsonl`과 `meta.json`이 생긴다. 모델 오류가 있는 행은 `error`를 기록하며, 채점은 실패한다. 원인을 고친 뒤 새 실행으로 다시 측정한다.

```bash
PYTHONPATH=.:ai/src ai/.venv/bin/python -m evals.complaint.score RUN_DIR
PYTHONPATH=.:ai/src ai/.venv/bin/python -m evals.complaint.baseline init RUN_DIR
PYTHONPATH=.:ai/src ai/.venv/bin/python -m evals.complaint.baseline compare NEW_RUN_DIR
```

`baseline init`은 기존 파일을 덮어쓰지 않는다. 현재 `--bucket all`은 민원과 라우팅만 실행한다. 정답 검수 후 실제 모델 결과를 사용하고 점수를 임의로 넣지 않는다. `compare`는 같은 범위·정답셋·채점기의 지표를 0.03 허용폭으로 비교하며 악화 시 종료 코드 1을 반환한다. 정답 라벨만 변경됐다면 이전 예측을 다시 채점하고, 입력 문장이나 사진을 변경했다면 모델도 다시 실행한다.

`score.py`의 `final_draft_exact`는 자유 서술 증상의 글자가 다르면 같은 뜻도 오답으로 세는 엄격한 보조 지표다. baseline은 구조 필드 정확도와 OCR 핵심 토큰 F1 등을 비교한다. 증상 핵심어 점수와 `forbidden_phrase_rate`는 긍정·부정을 이해하지 못하므로 환각 판정이나 자동 게이트로 쓰지 않는다. 실제 후속 질문의 의미도 자동 채점하지 않는다. 사용자에게 노출할 VLM 요약·증상·질문은 별도 사례 검토가 필요하다. 대화 점수는 이전 **정답** 초안을 입력하는 teacher-forced 결과이며 실제 사용자와의 반복 대화를 재현하지 않는다.

평가지표의 정의·분모·현재 구현 여부는 [`METRICS.md`](METRICS.md)의 「현재 평가 계약」을 따른다. 자유서술의 사람·LLM 판정 기준 초안은 [`JUDGE-PROMPTS.md`](JUDGE-PROMPTS.md)에 있다. 그 아래 상세 설계와 `PIPELINE.md`에는 확장 목표가 포함되어 있다. 210턴·30장, flip rate, judge, CI 연동은 현재 실행 코드에 없다.
여러 사진을 한 번에 보냈을 때의 내용 순서와 사진 분석 실패 후 대화 진행은 현재 seed 품질 평가 범위에 없다. 제품 단위 테스트의 계약 검증과 구분한다.
