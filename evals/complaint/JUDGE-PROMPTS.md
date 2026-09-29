# 민원 자유서술 판정 프롬프트 초안

이 문서는 `issue_type`·위치·날짜·OCR·라우팅의 **정답 비교를 대체하지 않는다**. 사람 판정이 필요한 증상 문장, 사진 캡션, 후속 질문에만 쓴다. 판정 모델은 아직 선택되지 않았으므로 아래는 모델 중립 JSON 계약이다. 모델을 고른 뒤 별도 실행·파싱 검증과 사람 라벨 일치도 검증을 거쳐 사용한다. 현재 `score.py`는 이 프롬프트를 호출하지 않는다.

## 사용 전 공통 규칙

- 평가 대상 모델이 만든 문장이나 캡션을 정답 생성의 근거로 쓰지 않는다. 정답 사실은 원본 발화·사진을 보고 사람이 먼저 작성한다.
- 판정기는 평가 대상 모델과 독립된 호출로 실행한다. 판정 모델명, 프롬프트 해시, 입력 ID, 출력 원문을 저장한다.
- 정답 사실 목록에 없는 새로운 사실이 출력에 있더라도 이미지·원문에서 확인될 가능성이 있으면 `needs_review`로 보낸다. 무조건 환각으로 확정하지 않는다.
- 판정 결과는 사람 라벨 표본에서 일치도를 확인하기 전까지 **진단용**이다. 불일치와 `needs_review`는 사람이 재판정한다.

## 1. `complaint_draft.symptom`: 사실 보존·근거 없는 추가

입력은 평가 턴의 원문, 이전 대화와 확인된 초안, **사람이 확인한** 사진 관찰, 사실 ID 목록, 모델이 작성한 최종 증상 문장이다. 이전 발화의 명시적 정정은 이전 값보다 우선한다. 사진을 직접 보지 않는 판정기에는 VLM이 스스로 만든 미검수 캡션을 사진 사실로 제공하지 않는다.

`{{INPUT_JSON}}` 필수 키: `case_id`, `current_user_text`, `conversation_history`, `confirmed_draft`, `verified_image_observations`, `gold_facts`(각 항목 `id`·`fact`), `candidate_symptom`.

```text
너는 민원 초안의 증상 문장만 판정한다. 아래 JSON 입력에 적힌 현재 사용자 발화, 이전 대화, 확인된 초안, verified_image_observations만 증거로 사용한다. gold_facts는 사람이 원본을 보고 확정한 사실 ID 목록이다. 모델이 생성한 문장을 증거로 삼지 마.

판정 규칙:
1. candidate_symptom이 보존한 gold_facts의 ID를 covered_fact_ids에 넣는다. 동의어와 자연스러운 바꿔쓰기는 허용하고 단순 단어 포함만으로 보존을 인정하지 마.
2. 사용자가 명시하지 않았고 확인된 사진 관찰에도 없는 구체적인 상태·원인·정도·시각·위치를 단정한 구절은 unsupported_claims에 넣는다. 부정, 추측, 인용의 차이를 구별한다.
3. 현재 턴의 명시적 정정과 충돌하는 구절은 contradicted_claims에 넣는다.
4. candidate_symptom이 비어 있거나 '모름' 같은 플레이스홀더뿐이면 usable_symptom=false다. 그렇지 않으면 관리자가 무엇이 불편한지 이해할 만큼 구체적인지를 판정한다. 위치·유형·날짜 슬롯의 정확성은 여기서 판정하지 마.
5. 제공된 증거만으로 확정할 수 없는 구절은 needs_review에 넣고 억지로 참·거짓을 결정하지 마. 골드 사실 목록이 불완전해 보이면 그 이유를 needs_review에 적는다.

입력:
{{INPUT_JSON}}

출력은 설명 문장이나 코드 펜스 없이 다음 형태의 JSON 객체 하나만 반환해:
{"covered_fact_ids":[],"unsupported_claims":[{"span":"","reason":""}],"contradicted_claims":[{"span":"","reason":""}],"usable_symptom":true,"needs_review":[{"span":"","reason":""}]}
```

집계: 사실 재현율은 `covered_fact_ids / gold_facts`다. 근거 없는 추가 사례율은 검토한 증상 중 `unsupported_claims` 또는 `contradicted_claims`가 하나 이상인 비율이다. `needs_review`가 있으면 자동 합격·불합격으로 확정하지 않는다.

## 2. VLM `summary`: 보이는 사실·근거 없는 주장

판정기는 **원본 이미지를 볼 수 있어야** 한다. 이미지와 함께, 사람이 작성한 `visible_facts`(ID·문장), `visible_anomaly`(`true`/`false`/`unknown`), `legibility`(`clear`/`partial`/`unreadable`), 판독 불가 영역을 제공한다. 신고 내용이나 `issue_type`은 사진에 보이는 사실을 늘리는 근거가 아니다. `ocr_text`의 문자 일치와 핵심 토큰 F1은 별도 결정적 채점기로 계산한다.

`{{INPUT_JSON}}` 필수 키: `case_id`, `visible_facts`(각 항목 `id`·`fact`), `visible_anomaly`, `legibility`, `illegible_regions`, `candidate_summary`. 같은 요청에 해당 원본 이미지 파일을 첨부한다.

```text
너는 민원 사진 한 장과 모델 캡션 candidate_summary를 판정한다. 원본 이미지와 사람이 확인한 visible_facts를 함께 보되, 사진에서 확인되지 않는 고장 원인·작동 상태·소리·냄새·위치는 사실로 인정하지 마. visible_anomaly와 legibility는 참고 라벨이며 이미지에 없는 세부 정보를 만들어내는 근거가 아니다.

판정 규칙:
1. 캡션이 의미를 보존한 visible_facts의 ID를 covered_fact_ids에 넣는다. 사실이 보이지 않거나 gold에 기록되지 않았는데 이미지에서 확인 가능해 보이면 needs_review에 보낸다.
2. 캡션이 사진으로 확인할 수 없는 기기 고장 원인, 증상의 심각도, 작동 실패, 소음 등을 단정하면 unsupported_claims에 그 구절과 이유를 넣는다. '확인하기 어렵다'처럼 불확실성을 명시한 문장은 단정으로 세지 마.
3. visible_anomaly=false인 사진에서 캡션이 이상·고장을 단정하면 false_alarm=true다. visible_anomaly=unknown이면 false_alarm을 null로 두고 사람이 검토할 구절을 needs_review에 넣는다.
4. 흐림·어두움·가림으로 판독할 수 없는 사실을 맞혔다고 주장해도 사진 근거로 인정하지 마. 반대로 판독 불가를 솔직히 밝힌 캡션을 누락된 이상 사실 때문에 벌점 주지 마.
5. `ocr_text`의 전체 전사 점수는 여기서 계산하지 마. 다만 **사용자에게 보이는 캡션**이 코드·수치를 구체적으로 주장했다면 사진의 실제 문자와 비교한다. 예를 들어 사진·정답 OCR은 E1인데 캡션이 E2라고 하면 `unsupported_claims`에 넣는다. 판독 불가 문자에 대해 자신 있게 특정 코드를 쓴 경우도 근거 없는 주장이다.

입력(원본 이미지 첨부 + JSON):
{{INPUT_JSON}}

출력은 설명 문장이나 코드 펜스 없이 다음 형태의 JSON 객체 하나만 반환해:
{"covered_fact_ids":[],"unsupported_claims":[{"span":"","reason":""}],"false_alarm":false,"needs_review":[{"span":"","reason":""}]}
```

집계: 사실 재현율은 `covered_fact_ids / visible_facts`다. 근거 없는 주장 사례율은 `unsupported_claims`가 하나 이상인 캡션 비율이다. 정상 오경보율은 `visible_anomaly=false`인 사진 중 `false_alarm=true`인 비율이다. `unknown`·`unreadable`은 정상/비정상 분모에 억지로 넣지 않고 별도 조건으로 보고한다.

## 3. 민원 대화 `reply`: 다음 질문의 의미

`missing_fields`와 `asked_field`의 구조 일치는 코드로 채점한다. 아래 판정은 사용자에게 보이는 한국어 답변이 실제로 무엇을 요구하는지 확인할 때만 쓴다.

`{{INPUT_JSON}}` 필수 키: `case_id`, `confirmed_draft`, `expected_missing_fields`, `previous_assistant_questions`, `candidate_reply`.

```text
너는 민원 수집 대화의 사용자에게 보이는 답변을 판정한다. JSON 입력의 confirmed_draft, expected_missing_fields, previous_assistant_questions, candidate_reply만 사용한다. 답변 안의 사진 요약 문장은 질문이 아니지만, 그 요약이 새 사실을 단정했는지는 이 판정의 대상이 아니다.

판정 규칙:
1. candidate_reply가 사용자에게 새로 요구하는 정보를 requested_slots에 적는다. location, symptom뿐 아니라 issue_type, occurred_at, 기타 정보는 `other`로 기록하고 그 내용을 reason에 적는다. 문장의 물음표 개수가 아니라 의미를 보고 판단한다.
2. requested_slots에 둘 이상 있으면 asks_one_slot=false다. 질문이 없으면 빈 배열과 asks_one_slot=true를 반환한다.
3. confirmed_draft에 이미 확인된 슬롯을 다시 요구하면 reasks_known_slot=true다. 사용자의 명시적 정정 요청은 중복 질문으로 세지 마.
4. expected_missing_fields가 비어 있지 않은데 다른 슬롯을 요구하거나 아무것도 묻지 않으면 asks_next_missing_slot=false다. 비어 있을 때에는 추가 정보 요청이 없어야 true다.
5. 이전 질문과 같은 문장을 되풀이했더라도 사용자가 아직 답하지 않았다면 중복 질문으로 단정하지 마. 증거가 불충분하면 needs_review에 적어.

입력:
{{INPUT_JSON}}

출력은 설명 문장이나 코드 펜스 없이 다음 형태의 JSON 객체 하나만 반환해:
{"requested_slots":[],"other_requested_information":[],"asks_one_slot":true,"reasks_known_slot":false,"asks_next_missing_slot":true,"needs_review":[{"span":"","reason":""}]}
```

집계: 한 턴 한 질문 준수율과 중복 질문률은 **후속 질문이 필요한 턴**에서 계산한다. 완료 턴은 별도로 추가 질문이 없었는지 검사한다. 의미 판정 결과가 구조적 `asked_field`와 다르면 사람이 확인한다.
