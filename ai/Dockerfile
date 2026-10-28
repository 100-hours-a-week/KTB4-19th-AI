# syntax=docker/dockerfile:1
# Build context: repository root (same as the existing CI matrix).
FROM python:3.12-slim AS base
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS dependencies
WORKDIR /app
ENV HF_HOME=/app/.cache/huggingface
ENV TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken
RUN pip install --no-cache-dir uv
COPY ai/pyproject.toml ai/uv.lock ai/README.md ./
RUN uv sync --frozen --no-install-project --no-dev

FROM dependencies AS models
# Preserve the current prefetch behavior. Decoupling it from app imports is phase 2.
COPY ai/scripts/prefetch_models.py ./scripts/prefetch_models.py
COPY ai/src/zipsai/__init__.py ./src/zipsai/__init__.py
COPY ai/src/zipsai/errors.py ./src/zipsai/errors.py
COPY ai/src/zipsai/indexing/__init__.py ./src/zipsai/indexing/__init__.py
COPY ai/src/zipsai/indexing/parse.py ./src/zipsai/indexing/parse.py
COPY ai/src/zipsai/indexing/pdf_text.py ./src/zipsai/indexing/pdf_text.py
RUN PYTHONPATH=/app/src /app/.venv/bin/python scripts/prefetch_models.py
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
# RapidOCR downloads inside site-packages, not HF_HOME. Preserve those artifacts too.
RUN /app/.venv/bin/python -c "import pathlib, shutil, rapidocr; target=pathlib.Path('/model-artifacts/rapidocr-models'); target.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(pathlib.Path(rapidocr.__file__).parent / 'models', target)"

FROM base AS runtime
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/app/.cache/huggingface HF_HUB_OFFLINE=1 TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken
RUN groupadd --system app && useradd --system --gid app app
COPY --link --from=dependencies --chown=app:app /app/.venv /app/.venv
COPY --link --from=models --chown=app:app /app/.cache/huggingface /app/.cache/huggingface
COPY --link --from=models --chown=app:app /app/.cache/tiktoken /app/.cache/tiktoken
COPY --from=models --chown=app:app /model-artifacts/rapidocr-models /app/.venv/lib/python3.12/site-packages/rapidocr/models
COPY --from=dependencies --chown=app:app /app/pyproject.toml /app/uv.lock /app/README.md ./
COPY --chown=app:app ai/src ./src
COPY --chown=app:app ai/scripts ./scripts
# Only project installation follows source changes; large dependency/model COPYs stay separate.
RUN --mount=type=bind,from=dependencies,source=/usr/local/bin/uv,target=/usr/local/bin/uv \
    UV_NO_CACHE=1 uv sync --frozen --no-dev
ARG ZIPSAI_VERSION=unknown
ENV ZIPSAI_VERSION=$ZIPSAI_VERSION
USER app
EXPOSE 8000
ENTRYPOINT ["uvicorn", "zipsai.main:app", "--host", "0.0.0.0", "--port", "8000"]
