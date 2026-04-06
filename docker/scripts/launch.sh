#!/bin/bash
# Launch DreamZero + Isaac Sim pipeline
#
# Usage:
#   bash docker/scripts/launch.sh                    # Start both services
#   bash docker/scripts/launch.sh --build            # Rebuild and start
#   bash docker/scripts/launch.sh --inference-only   # Start only inference server
#   bash docker/scripts/launch.sh --down             # Stop all services
#
# Remote viewing (from your local machine):
#   ssh -L 8211:localhost:8211 -L 49100:localhost:49100 ubuntu@<lambda-ip>
#   Then connect with Omniverse Streaming Client to localhost:8211

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$DOCKER_DIR")"

cd "$DOCKER_DIR"

# Parse arguments
BUILD_FLAG=""
SERVICES=""
case "${1:-}" in
    --build)
        BUILD_FLAG="--build"
        ;;
    --inference-only)
        SERVICES="dreamzero-inference"
        ;;
    --down)
        docker compose down
        echo "All services stopped."
        exit 0
        ;;
    --logs)
        docker compose logs -f ${2:-}
        exit 0
        ;;
esac

# Ensure checkpoint directory exists
if [ ! -d "${REPO_DIR}/checkpoints" ]; then
    echo "WARNING: checkpoints/ directory not found at ${REPO_DIR}/checkpoints"
    echo "Make sure your model weights are available before the inference server starts."
fi

# Create output directories
mkdir -p "${REPO_DIR}/output/inference" "${REPO_DIR}/output/sim"

# Detect architecture for base image selection
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "Detected arm64 (GH200/Grace Hopper). Using arm64 container images."
    export BASE_IMAGE="nvcr.io/nvidia/pytorch:24.12-py3-igpu"
else
    echo "Detected x86_64. Using standard container images."
    export BASE_IMAGE="nvcr.io/nvidia/pytorch:24.12-py3"
fi

echo "Starting DreamZero + Isaac Sim pipeline..."
echo "  Inference port: ${DREAMZERO_PORT:-8000}"
echo "  Streaming port: ${LIVESTREAM_PORT:-8211}"
echo ""
echo "To view remotely, run on your local machine:"
echo "  ssh -L 8211:localhost:8211 -L 49100:localhost:49100 ubuntu@<lambda-ip>"
echo "  Then open Omniverse Streaming Client -> localhost:8211"
echo ""

docker compose up -d $BUILD_FLAG $SERVICES

echo ""
echo "Services started. Use 'docker compose logs -f' to follow logs."
echo "Use 'bash docker/scripts/launch.sh --down' to stop."
