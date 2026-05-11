ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/cache/huggingface \
    NEMO_HOME=/cache/nemo \
    XDG_CACHE_HOME=/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libsndfile1 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN pip install --upgrade pip setuptools wheel uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src

RUN uv pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0 \
        torchaudio==2.8.0 \
    && uv sync --frozen --no-dev --extra server --extra nemo

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /cache/huggingface /cache/nemo /tmp/parakeet-stt \
    && chown -R appuser:appuser /app /cache /tmp/parakeet-stt

USER appuser

ENV HOST=0.0.0.0 \
    PORT=5092 \
    PYTHONPATH=/app/src \
    PARAKEET_STT_MODEL=nvidia/parakeet-tdt-0.6b-v3 \
    PARAKEET_STT_DEVICE=auto \
    PARAKEET_STT_MAX_CONCURRENT=1 \
    PARAKEET_STT_WARMUP_ON_LOAD=true \
    PARAKEET_STT_MAX_FILE_MB=100 \
    PARAKEET_STT_TEMP_DIR=/tmp/parakeet-stt

EXPOSE 5092

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:5092/health || exit 1

CMD ["parakeet-stt-server"]
