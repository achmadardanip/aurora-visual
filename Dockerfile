FROM python:3.12.13-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-ind libgomp1 && rm -rf /var/lib/apt/lists/*
RUN groupadd --system --gid 10001 aurora && useradd --system --uid 10001 --gid aurora --home /app aurora
WORKDIR /app
COPY --chown=aurora:aurora pyproject.toml uv.lock ./
COPY --chown=aurora:aurora backend ./backend
COPY --chown=aurora:aurora scripts/backup.py scripts/restore_backup.py scripts/verify_backup.py ./scripts/
RUN uv sync --frozen --extra ml --no-dev
COPY --chown=aurora:aurora alembic.ini ./
RUN mkdir -p /data && chown aurora:aurora /data
ENV AURORA_DATA_DIR=/data PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER aurora
EXPOSE 8101
HEALTHCHECK --interval=20s --timeout=5s CMD python -c \
    "import os, urllib.request; host = os.getenv('AURORA_ALLOWED_HOSTS', '127.0.0.1').split(',')[0].strip(); urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8101/health', headers={'Host': host}))"
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.api.main:app --host 0.0.0.0 --port 8101 --no-server-header"]
