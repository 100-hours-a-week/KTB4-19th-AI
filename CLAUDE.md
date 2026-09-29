## Commit preparation

When the user says `커밋준비` or asks for add commands, a commit message, and issue/PR drafts, use the `commit-prep` skill. Give copy-paste commands only; never stage, commit, stash, push, or otherwise mutate Git state, even if the user asks to do so.

## 하네스: 민원 AI 평가

**목표:** 민원 모델의 분류·추출·대화·라우팅·이미지 품질을 같은 정답셋으로 반복 측정한다.

**호출 조건:** 민원 AI 평가셋 작성, 평가 실행, 채점, baseline 변경 요청에는 `complaint-evaluation` 스킬을 사용한다.

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
| --- | --- | --- | --- |
| 2026-09-28 | 평가 하네스 구성 | 에이전트·스킬 | 정답 검토와 반복 측정을 분리 |
| 2026-09-29 | 시니어 평가 자문 역할 추가 | complaint-eval-advisor·스킬 | 지표 타당성과 점수 해석을 독립 점검 |
