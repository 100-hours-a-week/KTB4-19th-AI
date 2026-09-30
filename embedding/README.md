# embedding

BGE-M3 임베딩 서비스. 외부에 공개하지 않고 `ai-api`가 `EMBEDDING_API_URL`로만 호출한다.

`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`을 설정하면
`/embed` 요청마다 Langfuse에 모델, 배치 크기, 처리 시간, 결과 벡터 수가 기록된다.
원문과 벡터 값은 전송하지 않는다. `LANGFUSE_TRACING_ENABLED=false`로 끌 수 있다.
