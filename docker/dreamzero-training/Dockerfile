# DreamZero Training Container
# For GH200 (arm64): use nvcr.io/nvidia/pytorch:24.12-py3-igpu
# For x86 H100:      use nvcr.io/nvidia/pytorch:24.12-py3
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what's needed for pip install (source code is mounted at runtime)
COPY pyproject.toml /app/
COPY groot/ /app/groot/

# Install Python dependencies
RUN pip install --no-cache-dir -e . --extra-index-url https://download.pytorch.org/whl/cu129

# flash-attn: the base image may ship a version compiled against a different
# PyTorch ABI. Remove any pre-installed copy, then build from source.
# Building takes ~10-20 min; if it fails (e.g. no nvcc), training falls back
# to ATTENTION_BACKEND=torch (PyTorch SDPA) automatically.
RUN pip uninstall -y flash-attn 2>/dev/null; \
    MAX_JOBS=8 pip install --no-cache-dir --no-build-isolation flash-attn || \
    echo "flash-attn build failed, will use SDPA fallback"

# Install transformer engine for GH200/GB200
RUN pip install --no-cache-dir --no-build-isolation transformer_engine[pytorch] || \
    echo "transformer_engine not available, skipping"

# Entrypoint (small, rarely changes — bake it in)
COPY docker/dreamzero-training/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
