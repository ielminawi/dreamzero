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

# Install flash-attn (succeeds on H100 x86, fails on GH200 arm64)
RUN MAX_JOBS=8 pip install --no-cache-dir --no-build-isolation flash-attn || \
    echo "flash-attn not available for this platform, skipping"

# Install transformer engine for GH200/GB200
RUN pip install --no-cache-dir --no-build-isolation transformer_engine[pytorch] || \
    echo "transformer_engine not available, skipping"

# Entrypoint (small, rarely changes — bake it in)
COPY docker/dreamzero-training/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
