# DreamZero Inference Server
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
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY pyproject.toml /app/
COPY groot/ /app/groot/
COPY eval_utils/ /app/eval_utils/
COPY socket_test_optimized_AR.py /app/
COPY configs/ /app/configs/

# Install DreamZero
RUN pip install --no-cache-dir -e . --extra-index-url https://download.pytorch.org/whl/cu129

# Install flash-attn (skip on arm64 if not available)
RUN MAX_JOBS=8 pip install --no-cache-dir --no-build-isolation flash-attn || \
    echo "flash-attn not available for this platform, skipping"

# Install transformer engine for GH200/GB200
RUN pip install --no-cache-dir --no-build-isolation transformer_engine[pytorch] || \
    echo "transformer_engine not available, skipping"

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${DREAMZERO_PORT:-8000}/healthz || exit 1

EXPOSE ${DREAMZERO_PORT:-8000}

ENTRYPOINT ["torchrun"]
CMD ["--nproc_per_node=1", "--standalone", \
     "socket_test_optimized_AR.py", \
     "--port", "8000", \
     "--model-path", "/checkpoints/dreamzero_franka_orca_lora", \
     "--enable-dit-cache"]
