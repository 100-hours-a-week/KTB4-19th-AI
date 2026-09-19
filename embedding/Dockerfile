FROM python:3.12-slim AS build
WORKDIR /app
ENV HF_HOME=/app/.cache/huggingface
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev
RUN /app/.venv/bin/python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)"
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/app/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --system app && useradd --system --gid app app
COPY --from=build --chown=app:app /app /app
USER app
EXPOSE 8000
ENTRYPOINT ["uvicorn", "embedding.main:app", "--host", "0.0.0.0", "--port", "8000"]
