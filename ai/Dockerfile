FROM python:3.12-slim AS base
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS build
WORKDIR /app
ENV HF_HOME=/app/.cache/huggingface
RUN pip install --no-cache-dir uv
COPY ai/pyproject.toml ai/uv.lock ai/README.md ./
RUN uv sync --frozen --no-install-project --no-dev
COPY ai/src ./src
COPY ai/scripts ./scripts
RUN uv sync --frozen --no-dev
RUN /app/.venv/bin/python scripts/prefetch_models.py

FROM base
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/app/.cache/huggingface HF_HUB_OFFLINE=1
RUN groupadd --system app && useradd --system --gid app app
COPY --from=build --chown=app:app /app /app
USER app
EXPOSE 8000
ENTRYPOINT ["uvicorn", "zipsai.main:app", "--host", "0.0.0.0", "--port", "8000"]
