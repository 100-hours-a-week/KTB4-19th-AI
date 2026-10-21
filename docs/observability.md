# 로그·메트릭

AI(`ai-api`)와 임베딩은 stderr로 JSON 한 줄을 낸다. 파일 핸들러는 없다. Qdrant는 `QDRANT__LOGGER__FORMAT=json`으로 stdout 로그만 JSON이다. 기동 배너는 평문이라 `jq`에 걸리지 않는다.

공통 필드: `timestamp`(UTC, 밀리초), `level`, `service`(`ai-api`/`embedding`), `env`(`LANGFUSE_TRACING_ENVIRONMENT`, 없으면 null), `version`(`ZIPSAI_VERSION`, 없으면 null), `logger`, `event`. 요청 묶음 키가 있으면 `trace_id`, `turn_id`, `job_id`, `conversation_id`, `building_id`를 붙인다.

## 남기지 않는 것

입주민 원문, 답변, 프롬프트, 문서·청크·OCR 본문, 사진 요약, 민원 위치·증상 문자열, API 키, 비밀번호, presigned URL. 포매터는 `sk-`, `Bearer`, 32자 이상 hex, 슬래시 없는 40자 이상 base64, 전화번호, 차량번호, 쿼리가 붙은 URL, 띄어쓴 한글 문장을 `[REDACTED]`로 바꾼다. `doc_title`과 `file_key_path`는 문장·hex 패턴을 건너뛰고 `sk-`, `Bearer`, `api_key=`, presigned 쿼리만 가린다. 띄어쓴 한글 문장은 가려지니 에러 메시지는 영어로 쓴다. 원인은 `error_code`, `error_type`, `stage`로 남긴다. 민감 키의 문자열·목록·객체는 통째로 가리고, 숫자·bool·null은 남긴다. `texts`는 개수다. 한 글자라도 가리면 `redacted=true`.

## 이벤트

기존 이름(`request_done`, `stage_done`, `index_done`, `job_accepted`, `agent_error`, `fallback`, `complaint_turn`)은 유지하고 필드만 더한다.

| event | 추가 필드 |
|---|---|
| `http_access` | `method`, `path`(템플릿), `status`, `duration_ms`. `/health`·`/metrics`는 제외 |
| `llm_call` | `purpose`, `provider_host`(base_url 호스트만), `model`, `attempt`, `streaming`, `latency_ms`, `ttft_ms`, 토큰 4종, `finish_reason`, `status_code`, `error_code`, `structured_output_ok`, `retry_reason`, `image_count`, `image_pixels_total`(항상 null), `cost_usd`(단가 쌍이 없으면 null), `think_tag_present`, `hangul_ratio` |
| `stage_done` parse | `empty_pages`, `min_chars_per_page`, `tables_found`, `tables_parsed`, `tables_failed`(null), `table_fail_reasons`(null), `images_found`, `parser_status`, `ocr_used`(null), `failed_pages` |
| `index_done` | `doc_bytes`, `pages`, `chunks`, `embed_batches`, `stage_ms`, `error_code` |
| `job_accepted` | `doc_title`, `file_key_path`(쿼리 제거), `file_key_ext`, `valid_doc_ids`(쉼표 문자열. 빈 목록은 `""`, 없으면 null), `valid_doc_count` |
| `index_result` | 문서 요청이 200으로 끝날 때 한 줄. `http_status`, `index_status`, `index_reason`, `index_retryable`, `cleanup_status`, `cleanup_reason`, `cleanup_retryable`, `total_ms`, `pages`, `chunks`, `points`, `reconcile_count`. 없는 쪽은 null |
| `comment_accepted` | `kind`(`upsert`/`delete`), `items`, `building_ids`, `complaint_detail_ids`(정수만, 쉼표) |
| `comment_item_done` | 답변 한 건. `kind`, `status`, `reason_code`, `retryable`, `duration_ms`, `comment_chars`, `symptom_chars`, `validation_fields`, `complaint_type`, `issue_type`, `has_location`, `has_occurred_time`. 본문·위치·증상 문자열은 없다 |
| `comment_done` | `kind`, `items`, `succeeded`, `failed`, `skipped`, `total_ms`, `http_status`, `indexing_in_flight`, `embed_batches`, `embed_ms` |
| `indexing_rejected` | 429 `IN_PROGRESS` 또는 503 `BUSY`. 거절된 요청의 유일한 줄. `job_id`, `route`, `error_code`, `doc_id`, `building_id`, `items`, `indexing_in_flight`(거절된 요청은 빠짐), `retry_after_s`. `job_accepted`·`comment_accepted`는 접수가 된 뒤에만 남는다 |
| `stage_done` gate | 기존 `top_score`, `threshold`, `passed` |
| `stage_done` hybrid | `score_min`, `score_max`, `doc_ids`(상위 5), `scores`(상위 5), `building_filter_applied`, `invalid_doc_hits`(null) |
| `complaint_turn` | `asked_field`, `ask_repeat_count`, `draft_filled`(필드 이름), `missing_fields`, `urgency_level`, `image_count`, `images_analyzed`, `summary_chars`, `ocr_chars` |
| `complaint_ask_repeated` | 같은 항목을 이미 2번 물었으면 WARNING. `asked_field`, `ask_repeat_count` |
| `request_done` | 기존 필드 + `http_status`, `streaming`, `stage_ms`, `tokens_total`, `cost_usd_total`(계산된 호출 비용의 합, 없으면 null), `model` |
| `embed_request` / `embed_done` / `embed_failed` | `trace_id`, `texts`, `total_chars`, `max_chars`, `priority`, `queue_wait_ms`, `infer_ms`, `encode_ms`, `queue_len`, `duration_ms`, `model`, `device`, `load_avg`, `outcome`, `status_code`, `error_code` |
| `model_loaded` / `warmup_done` | `duration_ms`, `model` |

`ask_repeat_count`는 서버에 저장하지 않는다. 요청의 `conversation_history`에서 바로 앞 어시스턴트 줄을 거꾸로 본다. 그 항목의 고정 질문이 들어 있거나, 위치는 `어디|위치|장소|어느 (곳|방|공간)`, 증상은 `증상|어떤 (문제|불편)|어떻게 (되|안)`에 맞으면 센다. 둘 다 아니면 멈춘다. 이번이 세 번째 위치 질문이면 2다. `reply_source`는 이미 `complaint_turn`에 있다.

위치를 3번 이상 물은 대화 수. `conversation_id`마다 시계열이 생기므로 하루 점검에만 쓴다. 알람은 `zipsai_complaint_ask_repeats_total`을 쓴다.

```logql
count(
  sum by (conversation_id) (
    count_over_time({service="ai-api"} | json | event=`complaint_turn` | asked_field=`location` [24h])
  ) >= 3
)
```

표 실패 이유는 이 docling이 구분하지 않는다. `tables_found`·`tables_parsed`는 문서에 남은 표 수이고, `tables_failed`와 `table_fail_reasons`는 null이다. `ocr_used`도 null이다. `do_ocr=True`는 설정이지 페이지에서 OCR이 돌았다는 증거가 아니다.

`urgency_level`은 수집 중 null이다. 완료 카드도 지금 채점기(`urgency-llm`)가 level을 안 줘서 null이다. 점수로 level을 다시 만들지 않는다.

`cost_usd`는 입력·출력 단가가 둘 다 있을 때 `input/1e6*입력단가 + output/1e6*출력단가`다. 캐시 토큰은 빼지 않는다. `image_analysis`만 `VLM_PRICE_INPUT_PER_1M`·`VLM_PRICE_OUTPUT_PER_1M`을 쓰고, 다른 purpose는 `LLM_PRICE_INPUT_PER_1M`·`LLM_PRICE_OUTPUT_PER_1M`을 쓴다. 한 쌍 중 하나만 있거나 비어 있으면 null이고 `zipsai_llm_cost_usd_total`은 올리지 않는다. RunPod는 GPU 시간 과금이라 단가를 비운다. 비용은 RunPod 청구액으로 본다. OpenRouter처럼 토큰 과금일 때만 단가를 넣는다. `cost_usd_total`은 그 요청 안에서 계산된 `llm_call` 비용의 합이다.

임베딩 대기는 #126 이후 락이 아니라 우선순위 큐다. 로그 필드는 `queue_wait_ms`다. 메트릭 `zipsai_embed_queue_wait_seconds`에 그 대기 시간을 넣고, `zipsai_embed_queue_length`에 그 요청의 `queue_len`을 넣는다.

문서 색인·답변 반영·답변 삭제는 끝날 때까지 기다린 뒤 HTTP 200 본문으로 항목별 결과를 돌려준다. SQS 통지는 없다. 같은 문서를 처리 중이면 429 `IN_PROGRESS`(`Retry-After: 30`), 동시 처리 상한(`INDEXING_MAX_CONCURRENCY`, 기본 2)을 넘으면 503 `BUSY`(`Retry-After: 10`)다. 잠금과 상한은 프로세스 안이라 워커 1개 기준이다. 목록은 Loki `| json`이 배열을 버리므로 쉼표 문자열로 남긴다. 문서 색인 실측은 20.7초, 45.2초였다. 백엔드 읽기 타임아웃은 지금 60초이고 180초로 늘려 달라고 요청한 상태다. 답변 100건의 임베딩만 약 2분이라는 값은 실측이 아니라 추정이다. `BATCH_SIZE`는 2라 호출이 50번이면 60초에 닿을 수 있다.

## 메트릭

경로 `GET /metrics`, 포트 8000, 인증 없음. `/health`와 `/metrics`는 요청 카운터에서 뺀다. uvicorn 워커는 1개라 multiprocess 모드는 쓰지 않는다. 라벨에 `trace_id`, `building_id`, `doc_id`, 원문을 넣지 않는다.

요청 히스토그램 경계(초): 0.5, 1, 2, 4, 6, 12, 15, 18, 20, 25, 30, 60, 120.

단계 경계(초): 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 10, 12, 18, 30.

TTFT 경계(초): 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 5, 10.

| 이름 | 라벨 |
|---|---|
| `zipsai_build_info` | `service`, `version`. 값은 항상 1 |
| `zipsai_http_requests_total` | `service`, `method`, `path`, `status` |
| `zipsai_http_request_duration_seconds` | `service`, `method`, `path` |
| `zipsai_requests_in_flight` | `service` |
| `zipsai_converse_duration_seconds` | `route`, `outcome` |
| `zipsai_stage_duration_seconds` | `stage`, `outcome` |
| `zipsai_stage_failures_total` | `stage`, `error_code` |
| `zipsai_llm_tokens_total` | `purpose`, `model`, `kind` |
| `zipsai_llm_cost_usd_total` | `purpose`, `model` |
| `zipsai_llm_call_duration_seconds` | `purpose`, `model`, `outcome` |
| `zipsai_llm_ttft_seconds` | `purpose`, `model` |
| `zipsai_llm_structured_output_failures_total` | `purpose`, `model` |
| `zipsai_gate_decisions_total` | `passed` |
| `zipsai_fallbacks_total` | `reason` |
| `zipsai_complaint_ask_repeats_total` | `field` |
| `zipsai_index_jobs_total` | `outcome`, `failed_stage` |
| `zipsai_indexing_in_flight` | 없음. 접수된 색인·답변 작업 수 |
| `zipsai_indexing_results_total` | `kind`(`document_index`, `document_cleanup`, `comment_upsert`, `comment_delete`), `status`, `reason_code`. 성공은 `none`, 표에 없는 코드는 `other` |
| `zipsai_parse_tables_total` | `result` |
| `zipsai_embed_requests_total` | `outcome` |
| `zipsai_embed_duration_seconds` | `outcome` |
| `zipsai_embed_queue_wait_seconds` | 없음 |
| `zipsai_embed_queue_length` | 없음 |
| `zipsai_embed_texts_total` | 없음 |

`service`는 `ai-api` 또는 `embedding`이다. 임베딩 전용 다섯 개는 임베딩 프로세스에만 있다. `zipsai_build_info`의 `version`은 이미지 빌드 인자 `ZIPSAI_VERSION`이고, 없으면 `unknown`이다. 로그 필드 `version`은 같은 환경 변수가 없으면 null이다.

Qdrant 메트릭은 `GET :6333/metrics`이고 `api-key` 헤더가 필요하다. compose가 API 키를 켜 두기 때문이다.

## 운영 지표를 메트릭으로 바꿀 수 있는 것

| 지표 | 메트릭 |
|---|---|
| S1 | Prometheus `up` |
| S6 | `zipsai_embed_requests_total`, `zipsai_embed_duration_seconds`, `zipsai_embed_queue_wait_seconds`, `zipsai_embed_queue_length` |
| Q1·Q3 | `zipsai_converse_duration_seconds`, `zipsai_stage_duration_seconds` |
| Q4 | `zipsai_stage_failures_total`, `zipsai_http_requests_total`의 5xx |
| Q7·Q9 | `zipsai_requests_in_flight`, `zipsai_http_requests_total` |
| Q13 | `zipsai_llm_tokens_total`, `zipsai_llm_cost_usd_total` |

Q1k·Q2·I2·S5의 판정 정의는 여전히 로그다. 운영 로그는 Loki에 쌓인다. 스트림 라벨은 `service`(`ai-api`/`embedding`/`qdrant`)와 `env`만 쓴다. `trace_id`, `turn_id`, `conversation_id`는 JSON 필드라 `| json`으로 꺼낸다.

```logql
# Q1k knowledge p95. 목표 20000ms 이하
quantile_over_time(0.95,
  {service="ai-api"} | json | event=~"request_done|stream_request_done" | intent_route=`knowledge` | unwrap total_ms [1h]
)
```

```logql
# Q2 60초 초과. 목표 0
count_over_time({service="ai-api"} | json | event=~"request_done|stream_request_done" | total_ms > 60000 [1h])
```

```logql
# Q4 에러 응답. 두 done 이벤트의 실패
count_over_time({service="ai-api"} | json | event=~"request_done|stream_request_done" | outcome=`fail` [1h])
```

```logql
# I2 단계 실패. 목표 0
sum by (stage) (
  count_over_time({service="ai-api"} | json | event=`stage_done` | outcome=`fail` [1h])
)
```

```logql
# S5 요청별 stage_done 줄 수. knowledge·complaint는 4 이상
sum by (turn_id) (
  count_over_time({service="ai-api"} | json | event=`stage_done` | turn_id != `` [1h])
)
```

```logql
# S5 trace_id가 빈 stage_done. 목표 0
count_over_time({service="ai-api"} | json | event=`stage_done` | trace_id=`` [1h])
```
