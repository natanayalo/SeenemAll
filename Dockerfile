# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS builder

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt requirements-openvino.txt ./
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip,sharing=locked \
    pip wheel --wheel-dir /app/wheels -r requirements-openvino.txt

# Final stage
FROM python:3.11-slim

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy wheels from builder stage
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt /app/requirements-openvino.txt ./

# Install dependencies
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-index --find-links=/wheels -r requirements-openvino.txt

# Pre-download default sentence-transformers model and cross-encoder to eliminate first-request cold start
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy application code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
