FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-ind libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY backend ./backend
RUN uv sync --frozen --extra ml --no-dev
COPY alembic.ini ./
ENV AURORA_DATA_DIR=/data PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
EXPOSE 8101
HEALTHCHECK --interval=20s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8101/health')"
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8101"]
