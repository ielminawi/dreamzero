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

# The base image ships PyTorch + flash-attn + transformer_engine compiled
# together. Our pyproject.toml installs torch==2.8.0 which breaks the ABI
# for the other two. Remove them all before installing our deps cleanly.
RUN pip uninstall -y flash-attn transformer-engine transformer_engine 2>/dev/null || true

# Copy only what's needed for pip install (source code is mounted at runtime)
COPY pyproject.toml /app/
COPY groot/ /app/groot/

# Install Python dependencies (upgrades PyTorch to 2.8.0)
RUN pip install --no-cache-dir -e . --extra-index-url https://download.pytorch.org/whl/cu129

# Rebuild flash-attn from source against torch 2.8.0.
# Takes ~10-20 min; if it fails, training uses ATTENTION_BACKEND=torch (SDPA).
RUN MAX_JOBS=8 pip install --no-cache-dir --no-build-isolation flash-attn || \
    echo "flash-attn build failed, will use SDPA fallback"

# Rebuild transformer engine against torch 2.8.0.
# Optional — only needed for TE attention backend.
RUN pip install --no-cache-dir --no-build-isolation transformer_engine[pytorch] || \
    echo "transformer_engine build failed, skipping"

# Entrypoint (small, rarely changes — bake it in)
COPY docker/dreamzero-training/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
