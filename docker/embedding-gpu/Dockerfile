# syntax=docker/dockerfile:1
# Build context: docker/embedding-gpu. Named context: embedding-src=./embedding.
FROM python:3.12-slim AS base
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS dependencies
ENV HF_HOME=/app/.cache/huggingface
RUN pip install --no-cache-dir uv==0.13.7
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev \
    && /app/.venv/bin/python -c 'import torch; assert torch.version.cuda == "12.6", torch.version.cuda'

FROM dependencies AS models
# GitHub-hosted build runners have no GPU. Load on CPU only to cache model files.
RUN /app/.venv/bin/python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', devices='cpu', use_fp16=False)"

FROM dependencies AS application
COPY --from=embedding-src /README.md ./README.md
COPY --from=embedding-src /src ./src
RUN uv sync --locked --no-dev

FROM base AS runtime
ARG ZIPSAI_VERSION=unknown
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    EMBEDDING_DEVICE=cuda \
    ZIPSAI_VERSION=$ZIPSAI_VERSION
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app app
COPY --link --from=application --chown=10001:10001 /app/.venv /app/.venv
COPY --link --from=models --chown=10001:10001 /app/.cache/huggingface /app/.cache/huggingface
COPY --from=application --chown=10001:10001 /app/src /app/src
USER app
EXPOSE 8000
ENTRYPOINT ["uvicorn", "embedding.main:app", "--host", "0.0.0.0", "--port", "8000"]
