# syntax=docker/dockerfile:1.7
ARG OPENVINO_RUNTIME_IMAGE=openvino/model_server:2026.4.0-gpu

FROM python:3.12-slim AS builder

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
COPY requirements.txt requirements-openvino.txt requirements-docker.txt ./
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip,sharing=locked \
    pip wheel --wheel-dir /app/wheels -r requirements-docker.txt

# Final stage includes the Intel GPU runtime and OpenVINO version used by the
# WSL2 GPU path. The API runs in this image instead of the model server binary.
FROM ${OPENVINO_RUNTIME_IMAGE} AS runtime

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH
ENV HOME=/root

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy wheels from builder stage
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt /app/requirements-openvino.txt /app/requirements-docker.txt ./

# Install dependencies into the application environment
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-index --find-links=/wheels -r requirements-docker.txt

# Pre-download default sentence-transformers model and cross-encoder to eliminate first-request cold start
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy application code
COPY . .

EXPOSE 8000
ENTRYPOINT ["/opt/venv/bin/uvicorn"]
CMD ["api.main:app", "--host", "0.0.0.0", "--port", "8000"]
