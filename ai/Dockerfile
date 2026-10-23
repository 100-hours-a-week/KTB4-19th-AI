FROM python:3.12-slim AS base
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS build
WORKDIR /app
ENV HF_HOME=/app/.cache/huggingface
ENV TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken
RUN pip install --no-cache-dir uv
COPY ai/pyproject.toml ai/uv.lock ai/README.md ./
RUN uv sync --frozen --no-install-project --no-dev
# Keep the model cache reusable across unrelated application source changes.
COPY ai/scripts/prefetch_models.py ./scripts/prefetch_models.py
COPY ai/src/zipsai/__init__.py ./src/zipsai/__init__.py
COPY ai/src/zipsai/errors.py ./src/zipsai/errors.py
COPY ai/src/zipsai/indexing/__init__.py ./src/zipsai/indexing/__init__.py
COPY ai/src/zipsai/indexing/parse.py ./src/zipsai/indexing/parse.py
COPY ai/src/zipsai/indexing/pdf_text.py ./src/zipsai/indexing/pdf_text.py
RUN PYTHONPATH=/app/src /app/.venv/bin/python scripts/prefetch_models.py
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
COPY ai/src ./src
COPY ai/scripts ./scripts
RUN uv sync --frozen --no-dev

FROM base
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/app/.cache/huggingface HF_HUB_OFFLINE=1 TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken
RUN groupadd --system app && useradd --system --gid app app
COPY --from=build --chown=app:app /app /app
ARG ZIPSAI_VERSION=unknown
ENV ZIPSAI_VERSION=$ZIPSAI_VERSION
USER app
EXPOSE 8000
ENTRYPOINT ["uvicorn", "zipsai.main:app", "--host", "0.0.0.0", "--port", "8000"]
