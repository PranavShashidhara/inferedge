# ============================================================
# LLM Inference Engine – Jetson Orin Nano (Ampere sm_87)
# Base: NVIDIA L4T base (JetPack 6.x / CUDA 12.6, ARM64)
# ============================================================

FROM nvcr.io/nvidia/l4t-base:r36.2.0

# ── System deps ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    cmake \
    ninja-build \
    wget \
    curl \
    ca-certificates \
    libssl-dev \
    libopenblas-dev \
    openssh-client \
    procps \
    lsb-release \
    libglib2.0-0 \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# ── CUDA paths (runtime mount — not available during build) ───
ENV PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/lib/aarch64-linux-gnu:/opt/cusparselt/lib
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.7"
ENV CUDAARCHS="87"

# ── cuSPARSELt ────────────────────────────────────────────────
RUN CUSPARSELT_VER="0.7.1.0" && \
    CUSPARSELT_NAME="libcusparse_lt-linux-aarch64-${CUSPARSELT_VER}-archive" && \
    wget -q "https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/${CUSPARSELT_NAME}.tar.xz" && \
    tar xf "${CUSPARSELT_NAME}.tar.xz" && \
    mkdir -p /opt/cusparselt/include /opt/cusparselt/lib && \
    cp -a "${CUSPARSELT_NAME}/include/"* /opt/cusparselt/include/ && \
    cp -a "${CUSPARSELT_NAME}/lib/"* /opt/cusparselt/lib/ && \
    ldconfig && \
    rm -rf "${CUSPARSELT_NAME}" "${CUSPARSELT_NAME}.tar.xz"

# ── CUTLASS headers ───────────────────────────────────────────
RUN git clone --depth 1 --branch v3.5.0 \
    https://github.com/NVIDIA/cutlass.git /opt/cutlass && \
    find /opt/cutlass -mindepth 1 -maxdepth 1 \
         ! -name include ! -name tools -exec rm -rf {} +

ENV CUTLASS_PATH=/opt/cutlass

# ── pip base tooling ──────────────────────────────────────────
RUN pip3 install --upgrade pip setuptools wheel

# ── PyTorch + torchvision (JetPack 6 / CUDA 12.6 / aarch64) ──
RUN pip3 install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0 \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# ── Triton ────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    triton==3.4.0 \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    || echo "[WARN] Triton unavailable — skipping."

# ── Project dependencies ──────────────────────────────────────
WORKDIR /workspace
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ── Copy project source ───────────────────────────────────────
COPY . .

# ── Runtime environment ───────────────────────────────────────
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0
ENV CUBLAS_WORKSPACE_CONFIG=:16:8
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV TRITON_CACHE_DIR=/workspace/.triton_cache

RUN mkdir -p /workspace/profiling \
             /workspace/.triton_cache \
             /workspace/results/plots \
             /workspace/results/csv_logs \
             /workspace/profiling/nsight_reports \
             /root/.cache/huggingface

# ── write build_kernels.sh into the image ─────────────────────
RUN printf '%s\n' \
    '#!/bin/bash' \
    'set -e' \
    'if ! command -v nvcc &> /dev/null; then' \
    '    echo "[ERROR] nvcc not found — run with --runtime=nvidia"' \
    '    exit 1' \
    'fi' \
    'echo "[OK] nvcc: $(nvcc --version | grep release)"' \
    'cd /workspace' \
    'mkdir -p build && cd build' \
    'cmake .. -G Ninja \' \
    '    -DCMAKE_BUILD_TYPE=Release \' \
    '    -DCMAKE_CUDA_ARCHITECTURES=87 \' \
    '    -DCUTLASS_PATH=/opt/cutlass \' \
    '    -DCMAKE_CUDA_COMPILER=$(which nvcc) \' \
    '    -DCMAKE_CUDA_FLAGS="--use_fast_math -O3 --generate-line-info"' \
    'ninja -j$(nproc)' \
    'echo "Done. Artifacts in /workspace/build/"' \
    > /build_kernels.sh && chmod +x /build_kernels.sh

CMD ["/bin/bash"]