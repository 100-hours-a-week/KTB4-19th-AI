---
name: complaint-evaluation
description: 민원 AI 평가셋, run.py, score.py, baseline을 만들거나 다시 실행·보완할 때 사용한다. 일반 단위 테스트 작성에는 사용하지 않는다.
---

# 민원 AI 평가 하네스

## 실행 순서

1. `evals/complaint/METRICS.md`, `PIPELINE.md`, 실제 제품 코드를 읽고 차이를 기록한다.
2. 초안 추출 → issue_type → 대화 진행 → 의도 라우팅 → VLM 순서로 각 정답셋을 작성한다. 정답 작성은 `complaint-eval-curator`(opus), 실행·채점 코드는 `complaint-eval-engineer`(sonnet)가 담당한다. 각 단계의 데이터·채점 결과는 `complaint-eval-reviewer`(opus)가 읽기 전용으로 독립 검토한다. 한 단계의 검증을 마친 뒤 다음 단계로 간다.
   - 새 이미지가 들어오면 curator가 파일 수·해시·보이는 사실·실제 OCR을 먼저 확인한다. 생성 프롬프트와 파일명만으로 `visible_anomaly`나 `gold_ocr`을 확정하지 않는다. 불확실하면 `unknown` 또는 판독 불가로 남긴다.
   - `complaint-eval-advisor`(opus)는 단계 종료와 baseline 확정 전에 지표 타당성·표본 편향·데이터 누출·점수 해석을 읽기 전용으로 점검하고 우선순위별 조언을 남긴다. 사례별 정답 검토를 맡는 reviewer와 역할을 구분한다.
3. 실행기는 예측과 메타데이터를 저장하고, 채점기는 저장된 예측만 읽는다. 실제 모델 점수가 없으면 baseline을 만들지 않는다.
4. 각 단계에서 정상 사례 하나와 오류 사례 하나를 검증한다. 정답 ID 중복, 예측 누락, 모델 호출 실패는 명시적으로 실패시킨다.
5. 기존 baseline과 비교한 결과를 보고 개선을 판단한다. 정답셋 자체가 바뀌면 기존 점수도 다시 계산한다.

## 오류 처리

모델 호출이 실패하면 해당 사례 ID와 오류를 예측 파일에 남기고, 그 실행의 채점·baseline 생성을 중단한다. 제품의 텍스트 JSON 재시도 정책은 제품 코드를 그대로 따른다. 인증·사용량 한도 오류는 평가 실행을 중단하고 원인을 보고한다.

## 테스트 시나리오

- 정상: 턴 정답 JSONL과 예측 JSONL의 ID가 모두 맞으면 점수 JSON을 낸다.
- 오류: 예측 한 줄이 빠지거나 JSON이 잘못되면 성공 점수 대신 오류와 ID를 반환한다.
