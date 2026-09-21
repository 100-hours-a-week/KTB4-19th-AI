FROM python:3.12-slim AS build
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
RUN groupadd --system app && useradd --system --gid app app
COPY --from=build --chown=app:app /app /app
USER app
EXPOSE 8000
ENTRYPOINT ["uvicorn", "zipsai.main:app", "--host", "0.0.0.0", "--port", "8000"]
