# syntax=docker/dockerfile:1
# Build context: embedding/ (same as the existing CI matrix).
FROM python:3.12-slim AS dependencies
WORKDIR /app
ENV HF_HOME=/app/.cache/huggingface
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

FROM dependencies AS models
# Keep the existing model download/load validation and CPU-only dependency lock.
RUN /app/.venv/bin/python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)"

FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/app/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --system app && useradd --system --gid app app
# Large, stable contents are copied before application source, in separate layers.
COPY --link --from=dependencies --chown=app:app /app/.venv /app/.venv
COPY --link --from=models --chown=app:app /app/.cache/huggingface /app/.cache/huggingface
COPY --from=dependencies --chown=app:app /app/pyproject.toml /app/uv.lock /app/README.md ./
COPY --chown=app:app src ./src
# Install the project after source COPY, preserving its editable /app/src import path.
# uv is mounted only for this step; it is not added to the runtime image.
# No package cache is retained in this final installation layer.
RUN --mount=type=bind,from=dependencies,source=/usr/local/bin/uv,target=/usr/local/bin/uv \
    UV_NO_CACHE=1 uv sync --frozen --no-dev
ARG ZIPSAI_VERSION=unknown
ENV ZIPSAI_VERSION=$ZIPSAI_VERSION
USER app
EXPOSE 8000
ENTRYPOINT ["uvicorn", "embedding.main:app", "--host", "0.0.0.0", "--port", "8000"]
