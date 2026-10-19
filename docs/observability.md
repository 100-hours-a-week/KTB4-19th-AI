# 로그·메트릭

AI(`ai-api`)와 임베딩은 stderr로 JSON 한 줄을 낸다. 파일 핸들러는 없다. Qdrant는 `QDRANT__LOGGER__FORMAT=json`으로 stdout 로그만 JSON이다. 기동 배너는 평문이라 `jq`에 걸리지 않는다.

공통 필드: `timestamp`(UTC, 밀리초), `level`, `service`(`ai-api`/`embedding`), `env`(`LANGFUSE_TRACING_ENVIRONMENT`, 없으면 null), `version`(`ZIPSAI_VERSION`, 없으면 null), `logger`, `event`. 요청 묶음 키가 있으면 `trace_id`, `turn_id`, `job_id`, `conversation_id`, `building_id`를 붙인다.

## 남기지 않는 것

입주민 원문, 답변, 프롬프트, 문서·청크·OCR 본문, 사진 요약, 민원 위치·증상 문자열, API 키, 비밀번호, presigned URL. 포매터는 `sk-`, `Bearer`, 32자 이상 hex, 슬래시 없는 40자 이상 base64, 전화번호, 차량번호, 쿼리가 붙은 URL, 띄어쓴 한글 문장을 `[REDACTED]`로 바꾼다. 민감 키의 문자열·목록·객체는 통째로 가리고, 숫자·bool·null은 남긴다. `texts`는 개수다. 한 글자라도 가리면 `redacted=true`.

## 이벤트

기존 이름(`request_done`, `stage_done`, `index_done`, `job_accepted`, `agent_error`, `fallback`, `complaint_turn`)은 유지하고 필드만 더한다.

| event | 추가 필드 |
|---|---|
| `http_access` | `method`, `path`(템플릿), `status`, `duration_ms`. `/health`·`/metrics`는 제외 |
| `llm_call` | `purpose`, `provider_host`(base_url 호스트만), `model`, `attempt`, `streaming`, `latency_ms`, `ttft_ms`, 토큰 4종, `finish_reason`, `status_code`, `error_code`, `structured_output_ok`, `retry_reason`, `image_count`, `image_pixels_total`(항상 null), `cost_usd`(단가 없으면 null), `think_tag_present`, `hangul_ratio` |
| `stage_done` parse | `empty_pages`, `min_chars_per_page`, `tables_found`, `tables_parsed`, `tables_failed`(null), `table_fail_reasons`(null), `images_found`, `parser_status`, `ocr_used`(null), `failed_pages` |
| `index_done` | `doc_bytes`, `pages`, `chunks`, `embed_batches`, `stage_ms`, `error_code` |
| `stage_done` gate | 기존 `top_score`, `threshold`, `passed` |
| `stage_done` hybrid | `score_min`, `score_max`, `doc_ids`(상위 5), `scores`(상위 5), `building_filter_applied`, `invalid_doc_hits`(null) |
| `complaint_turn` | `asked_field`, `ask_repeat_count`, `draft_filled`(필드 이름), `missing_fields`, `urgency_level`, `image_count`, `images_analyzed`, `summary_chars`, `ocr_chars` |
| `complaint_ask_repeated` | 같은 항목을 이미 2번 물었으면 WARNING. `asked_field`, `ask_repeat_count` |
| `request_done` | 기존 필드 + `http_status`, `streaming`, `stage_ms`, `tokens_total`, `cost_usd_total`(null), `model` |
| `embed_request` / `embed_done` / `embed_failed` | `trace_id`, `texts`, `total_chars`, `max_chars`, `priority`, `queue_wait_ms`, `infer_ms`, `encode_ms`, `queue_len`, `duration_ms`, `model`, `device`, `load_avg`, `outcome`, `status_code`, `error_code` |
| `model_loaded` / `warmup_done` | `duration_ms`, `model` |

`ask_repeat_count`는 서버에 저장하지 않는다. 요청의 `conversation_history`에서, 바로 앞 어시스턴트 줄이 그 항목의 고정 질문이면 센다. 이번이 세 번째 위치 질문이면 2다.

표 실패 이유는 이 docling이 구분하지 않는다. `tables_found`·`tables_parsed`는 문서에 남은 표 수이고, `tables_failed`와 `table_fail_reasons`는 null이다. `ocr_used`도 null이다. `do_ocr=True`는 설정이지 페이지에서 OCR이 돌았다는 증거가 아니다.

`urgency_level`은 수집 중 null이다. 완료 카드도 지금 채점기(`urgency-llm`)가 level을 안 줘서 null이다. 점수로 level을 다시 만들지 않는다.

`cost_usd`는 단가 설정이 없어 항상 null이다. 비용 카운터는 0으로 둔다.

임베딩 대기는 #126 이후 락이 아니라 우선순위 큐다. 로그 필드는 `queue_wait_ms`다. 메트릭 `zipsai_embed_lock_wait_seconds`에 그 대기 시간을 넣는다.

## 메트릭

경로 `GET /metrics`, 포트 8000, 인증 없음. `/health`와 `/metrics`는 요청 카운터에서 뺀다. uvicorn 워커는 1개라 multiprocess 모드는 쓰지 않는다. 라벨에 `trace_id`, `building_id`, `doc_id`, 원문을 넣지 않는다.

요청 히스토그램 경계(초): 0.5, 1, 2, 4, 6, 12, 15, 18, 20, 25, 30, 60, 120.

단계 경계(초): 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 10, 12, 18, 30.

TTFT 경계(초): 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 5, 10.

| 이름 | 라벨 |
|---|---|
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
| `zipsai_parse_tables_total` | `result` |
| `zipsai_embed_requests_total` | `outcome` |
| `zipsai_embed_duration_seconds` | `outcome` |
| `zipsai_embed_lock_wait_seconds` | 없음 |
| `zipsai_embed_texts_total` | 없음 |

`service`는 `ai-api` 또는 `embedding`이다. 임베딩 전용 네 개는 임베딩 프로세스에만 있다.

Qdrant 메트릭은 `GET :6333/metrics`이고 `api-key` 헤더가 필요하다. compose가 API 키를 켜 두기 때문이다.

## 운영 지표를 메트릭으로 바꿀 수 있는 것

| 지표 | 메트릭 |
|---|---|
| S1 | Prometheus `up` |
| S6 | `zipsai_embed_requests_total`, `zipsai_embed_duration_seconds`, `zipsai_embed_lock_wait_seconds` |
| Q1·Q3 | `zipsai_converse_duration_seconds`, `zipsai_stage_duration_seconds` |
| Q4 | `zipsai_stage_failures_total`, `zipsai_http_requests_total`의 5xx |
| Q7·Q9 | `zipsai_requests_in_flight`, `zipsai_http_requests_total` |
| Q13 | `zipsai_llm_tokens_total`, `zipsai_llm_cost_usd_total` |

Q1k·Q2·I2·S5의 판정 정의는 여전히 로그다. 아래 쿼리는 Loki로 옮겨도 같은 값을 내기 위한 예시다. 스트림 라벨 이름은 클라우드 수집 설정에 맞춘다.

```logql
# Q1k knowledge p95. 목표 20000ms 이하
quantile_over_time(0.95,
  {service="ai-api"} | json | event=`request_done` | intent_route=`knowledge` | unwrap total_ms [1h]
)
```

```logql
# Q2 60초 초과. 목표 0
count_over_time({service="ai-api"} | json | event=`request_done` | total_ms > 60000 [1h])
```

```logql
# Q4 에러 응답. agent_error 기준
count_over_time({service="ai-api"} | json | event=`agent_error` [1h])
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
