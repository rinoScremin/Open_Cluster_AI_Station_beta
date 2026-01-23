#!/usr/bin/env bash
set -e

./install_libtorch.sh

echo "=== Installing system build dependencies ==="
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    libopenblas-dev \
    libzmq3-dev \
    libpthread-stubs0-dev

echo "🔍 Detecting system capabilities..."
OS="$(uname -s)"

# -------------------------------
# Detect NVIDIA GPU
# -------------------------------
HAS_NVIDIA_GPU=0
if command -v lspci >/dev/null 2>&1 && lspci | grep -qi nvidia; then
    HAS_NVIDIA_GPU=1
    echo "✅ NVIDIA GPU detected"
else
    echo "ℹ️ No NVIDIA GPU detected"
fi

# -------------------------------
# Detect CUDA
# -------------------------------
HAS_CUDA=0
if command -v nvcc >/dev/null 2>&1; then
    HAS_CUDA=1
    echo "✅ CUDA toolkit detected"
else
    echo "ℹ️ CUDA toolkit not found"
fi

# -------------------------------
# Detect Vulkan
# -------------------------------
HAS_VULKAN=0
if command -v vulkaninfo >/dev/null 2>&1; then
    HAS_VULKAN=1
    echo "✅ Vulkan supported"
else
    echo "ℹ️ Vulkan not available"
fi

# -------------------------------
# Detect Metal (macOS only)
# -------------------------------
HAS_METAL=0
if [[ "$OS" == "Darwin" ]]; then
    if system_profiler SPDisplaysDataType 2>/dev/null | grep -qi "metal"; then
        HAS_METAL=1
        echo "✅ Metal supported"
    else
        echo "ℹ️ Metal not supported"
    fi
fi

# -------------------------------
# Resolve paths
# -------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GGML_DIR="$PROJECT_ROOT/cluster_matrix/ggml"

echo "📂 Project root: $PROJECT_ROOT"
echo "📂 GGML dir:     $GGML_DIR"
cd "$GGML_DIR"

# -------------------------------
# Base flags (CPU ALWAYS ENABLED)
# -------------------------------
CMAKE_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_BLAS=ON
    -DGGML_BLAS_VENDOR=OpenBLAS
    -DGGML_OPENCL=OFF
    -DGGML_CUDA=OFF
    -DGGML_VULKAN=OFF
    -DGGML_METAL=OFF
)

# -------------------------------
# Backend selection (CPU + accel)
# -------------------------------
if [[ $HAS_NVIDIA_GPU -eq 1 && $HAS_CUDA -eq 1 ]]; then
    echo "🚀 Enabling CUDA backend (CPU + OpenBLAS + CUDA)"
    CMAKE_FLAGS+=(-DGGML_CUDA=ON)

elif [[ $HAS_VULKAN -eq 1 && "$OS" == "Linux" ]]; then
    echo "🎮 Enabling Vulkan backend (CPU + OpenBLAS + Vulkan)"
    CMAKE_FLAGS+=(-DGGML_VULKAN=ON)

elif [[ $HAS_METAL -eq 1 ]]; then
    echo "🍎 Enabling Metal backend (CPU + OpenBLAS + Metal)"
    CMAKE_FLAGS+=(-DGGML_METAL=ON)

else
    echo "🧠 CPU-only build (OpenBLAS)"
fi

# -------------------------------
# Build
# -------------------------------
echo "🛠️ Configuring CMake with flags:"
printf '   %s\n' "${CMAKE_FLAGS[@]}"

cmake -B build "${CMAKE_FLAGS[@]}"

echo "⚙️ Building matrix_zmq_server..."
cmake --build build --target matrix_zmq_server -j"$(nproc)"

# -------------------------------
# Return to project root
# -------------------------------
cd "$PROJECT_ROOT"
echo "📂 Returned to project root: $PROJECT_ROOT"
echo "🎉 ggml backend installation complete!"