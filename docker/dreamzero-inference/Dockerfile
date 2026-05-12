# DreamZero Inference Server
# For GH200 (arm64): use nvcr.io/nvidia/pytorch:24.12-py3-igpu
# For x86 H100:      use pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
# syntax=docker/dockerfile:1
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    ffmpeg \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Dependency layer (only invalidated when pyproject.toml changes) ----
COPY pyproject.toml /app/
# Stub package so pip can resolve and install all deps before copying real code
RUN mkdir -p groot && touch groot/__init__.py

# Install DreamZero dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129

# Install flash-attn: try pre-built wheel first (avoids 20-40 min C++ compile),
# fall back to building from source if no wheel matches this CUDA/PyTorch version.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install flash-attn \
        --find-links https://github.com/Dao-AILab/flash-attention/releases/ \
    || MAX_JOBS=8 pip install --no-build-isolation flash-attn \
    || echo "flash-attn not available for this platform, skipping"

# Install transformer engine for GH200/GB200
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation transformer_engine[pytorch] || \
    echo "transformer_engine not available, skipping"

# ---- Code layer (only invalidated when groot/ source changes) ----
COPY groot/ /app/groot/

EXPOSE ${DREAMZERO_PORT:-8000}

ENTRYPOINT ["torchrun"]
CMD ["--nproc_per_node=1", "--standalone", \
     "socket_test_optimized_AR.py", \
     "--port", "8000", \
     "--model-path", "/checkpoints/franka_orca_lora", \
     "--enable-dit-cache"]
