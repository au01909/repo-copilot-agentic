# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# git is required at runtime (repository cloning); build-essential covers the
# rare native-extension dependency that occasionally needs compiling on
# arm64/uncommon platforms.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Non-root user — the container clones untrusted third-party repositories
# into its own filesystem; running as root has no benefit here and is a real
# risk given repository content is explicitly untrusted input.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/qdrant_data \
    && chown -R appuser:appuser /app
USER appuser

ENV SQLITE_PATH=/app/data/repo_copilot.db \
    QDRANT_PATH=/app/qdrant_data \
    MLFLOW_TRACKING_URI=sqlite:////app/data/mlflow.db \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
