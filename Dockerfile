FROM python:3.12-slim AS build
WORKDIR /app
ENV HF_HOME=/app/.cache/huggingface
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev
COPY src ./src
COPY scripts ./scripts
RUN uv sync --frozen --no-dev
# 모델을 여기서 받아 이미지에 넣는다. 런타임에 내려받게 두면 첫 색인이 느려지고,
# 아웃바운드가 막힌 서브넷에서는 아예 실패한다.
RUN /app/.venv/bin/python scripts/prefetch_models.py

FROM python:3.12-slim
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/app/.cache/huggingface HF_HUB_OFFLINE=1
# docling이 OpenCV를 거쳐 이미지를 읽는다. 없으면 임포트 단계에서 죽는다.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --system app && useradd --system --gid app app
COPY --from=build --chown=app:app /app /app
USER app
EXPOSE 8000
ENTRYPOINT ["uvicorn", "zipsai.main:app", "--host", "0.0.0.0", "--port", "8000"]
