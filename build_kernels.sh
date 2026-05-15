#!/bin/bash
# ============================================================
# build_kernels.sh
# Run this INSIDE the container after docker compose run.
# nvcc is only available at runtime (CUDA is a bind-mount).
#
# Usage:
#   docker compose run --rm inference-engine /build_kernels.sh
# ============================================================
set -e

# Verify nvcc is accessible
if ! command -v nvcc &> /dev/null; then
    echo "[ERROR] nvcc not found. Is the container running with --runtime=nvidia?"
    echo "        Check: docker compose run --rm inference-engine nvcc --version"
    exit 1
fi

echo "[OK] nvcc: $(nvcc --version | grep release)"
echo "[OK] CUDA_HOME: ${CUDA_HOME}"

cd /workspace

mkdir -p build && cd build

cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DCUTLASS_PATH=${CUTLASS_PATH} \
    -DCMAKE_CUDA_FLAGS="--use_fast_math -O3 --generate-line-info" \
    -DCMAKE_CUDA_COMPILER=$(which nvcc)

ninja -j$(nproc)

echo ""
echo "Build complete. Artifacts in /workspace/build/"
ls /workspace/build/
