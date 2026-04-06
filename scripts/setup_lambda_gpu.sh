#!/usr/bin/env bash
# Setup script for Lambda Labs H100 instances
# Installs NVIDIA graphics/Vulkan libraries required by Isaac Sim
# Run once after launching a fresh instance, then reboot.

set -euo pipefail

echo "=== Lambda H100 GPU Setup for Isaac Sim ==="

# Install NVIDIA GL/Vulkan userspace libraries
echo "[1/3] Installing NVIDIA graphics libraries..."
sudo apt-get update
sudo apt-get install -y libnvidia-gl-580 nvidia-utils-580

# Ensure Vulkan ICD file exists
echo "[2/3] Verifying Vulkan ICD configuration..."
sudo mkdir -p /usr/share/vulkan/icd.d
if [ ! -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3"}}' \
        | sudo tee /usr/share/vulkan/icd.d/nvidia_icd.json
    echo "  Created nvidia_icd.json"
else
    echo "  nvidia_icd.json already exists"
fi

# Verify
echo "[3/3] Verifying installation..."
ldconfig -p | grep libGLX_nvidia && echo "  libGLX_nvidia found" || echo "  WARNING: libGLX_nvidia not found"
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader && echo "  nvidia-smi OK" || echo "  WARNING: nvidia-smi failed"

echo ""
echo "=== Setup complete. Reboot now: sudo reboot ==="
