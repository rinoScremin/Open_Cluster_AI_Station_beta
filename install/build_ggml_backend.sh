#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
trap 'echo "❌ ggml backend build failed at line $LINENO. See logs above." >&2' ERR

# -------------------------------
# Paths
# -------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GGML_DIR="$PROJECT_ROOT/cluster_matrix/ggml"

if [[ ! -d "$GGML_DIR" ]]; then
    echo "❌ GGML directory not found: $GGML_DIR"
    exit 1
fi

cd "$GGML_DIR"

# -------------------------------
# Download libtorch
# -------------------------------
OS="$(uname -s)"
ARCH="$(uname -m)"
LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.2.2+cpu}"
LIBTORCH_URL="${LIBTORCH_URL:-}"
LIBTORCH_DIR="${LIBTORCH_DIR:-$GGML_DIR/libtorch}"

download_file() {
    local url="$1"
    local out="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -L -o "$out" "$url"
    elif command -v python3 >/dev/null 2>&1; then
        python3 - <<PY
import urllib.request
urllib.request.urlretrieve("$url", "$out")
PY
    else
        echo "❌ Neither curl nor python3 found for downloading."
        exit 1
    fi
}

extract_zip() {
    local zip_path="$1"
    local dest="$2"
    if command -v unzip >/dev/null 2>&1; then
        unzip -q "$zip_path" -d "$dest"
    elif command -v python3 >/dev/null 2>&1; then
        python3 - <<PY
import zipfile
with zipfile.ZipFile("$zip_path", "r") as zf:
    zf.extractall("$dest")
PY
    else
        echo "❌ Neither unzip nor python3 found for extracting zip."
        exit 1
    fi
}

if [[ -d "$LIBTORCH_DIR" && "${FORCE_LIBTORCH:-0}" != "1" ]]; then
    echo "ℹ️ libtorch already present at $LIBTORCH_DIR (set FORCE_LIBTORCH=1 to re-download)"
else
    if [[ -z "$LIBTORCH_URL" ]]; then
        if [[ "$OS" == "Linux" && ( "$ARCH" == "x86_64" || "$ARCH" == "amd64" ) ]]; then
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.2.2%2Bcpu.zip"
        else
            echo "❌ No default libtorch URL for ${OS}-${ARCH}."
            echo "   Provide LIBTORCH_URL or preinstall libtorch at $LIBTORCH_DIR."
            exit 1
        fi
    fi

    echo "🟢 Downloading libtorch $LIBTORCH_VERSION..."
    TMP_DIR="$(mktemp -d)"
    ZIP_PATH="$TMP_DIR/libtorch.zip"
    download_file "$LIBTORCH_URL" "$ZIP_PATH"

    echo "📦 Extracting libtorch..."
    extract_zip "$ZIP_PATH" "$TMP_DIR"

    if [[ -d "$LIBTORCH_DIR" ]]; then
        rm -rf "$LIBTORCH_DIR"
    fi

    if [[ -d "$TMP_DIR/libtorch" ]]; then
        mv "$TMP_DIR/libtorch" "$LIBTORCH_DIR"
    elif [[ -d "$TMP_DIR/libtorch/libtorch" ]]; then
        mv "$TMP_DIR/libtorch/libtorch" "$LIBTORCH_DIR"
    else
        echo "❌ libtorch directory not found after extraction."
        exit 1
    fi

    rm -rf "$TMP_DIR"
    echo "✅ libtorch installed to $LIBTORCH_DIR"
fi

# -------------------------------
# Install system build dependencies
# -------------------------------
INSTALL_DEPS="${INSTALL_DEPS:-1}"
SUDO=""
if [[ "${EUID:-$(id -u)}" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
    echo "=== Installing system build dependencies ==="
    if command -v apt-get >/dev/null 2>&1; then
        $SUDO apt-get update
        $SUDO apt-get install -y \
            build-essential \
            cmake \
            pkg-config \
            git \
            libopenblas-dev \
            libzmq3-dev \
            libpthread-stubs0-dev
    elif command -v dnf >/dev/null 2>&1; then
        $SUDO dnf install -y \
            gcc \
            gcc-c++ \
            make \
            cmake \
            pkgconfig \
            git \
            openblas-devel \
            zeromq-devel
    elif command -v pacman >/dev/null 2>&1; then
        $SUDO pacman -S --needed --noconfirm \
            base-devel \
            cmake \
            pkgconf \
            git \
            openblas \
            zeromq
    elif [[ "$OS" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
        brew install cmake pkg-config git openblas zeromq
    else
        echo "⚠️ No supported package manager detected. Install dependencies manually."
        echo "   Needed: cmake, pkg-config, git, OpenBLAS, ZeroMQ headers"
    fi
else
    echo "ℹ️ Skipping dependency installation (INSTALL_DEPS=$INSTALL_DEPS)."
fi

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "❌ Required command not found: $1"
        exit 1
    fi
}

require_cmd cmake

# -------------------------------
# Detect system capabilities
# -------------------------------
echo "🔍 Detecting system capabilities..."

# NVIDIA GPU
HAS_NVIDIA_GPU=0
if command -v nvidia-smi >/dev/null 2>&1; then
    HAS_NVIDIA_GPU=1
    echo "✅ NVIDIA GPU detected (nvidia-smi)"
elif command -v lspci >/dev/null 2>&1 && lspci | grep -qi nvidia; then
    HAS_NVIDIA_GPU=1
    echo "✅ NVIDIA GPU detected"
else
    echo "ℹ️ No NVIDIA GPU detected"
fi

# CUDA
HAS_CUDA=0
if command -v nvcc >/dev/null 2>&1; then
    HAS_CUDA=1
    echo "✅ CUDA toolkit detected"
else
    echo "ℹ️ CUDA toolkit not found"
fi

# Vulkan
HAS_VULKAN=0
if command -v vulkaninfo >/dev/null 2>&1; then
    HAS_VULKAN=1
    echo "✅ Vulkan supported"
elif command -v pkg-config >/dev/null 2>&1 && pkg-config --exists vulkan; then
    HAS_VULKAN=1
    echo "✅ Vulkan supported (pkg-config)"
else
    echo "ℹ️ Vulkan not available"
fi

# Metal (macOS only)
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
# Configure CMake
# -------------------------------
echo "📂 Project root: $PROJECT_ROOT"
echo "📂 GGML dir:     $GGML_DIR"

CMAKE_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_CPU=ON
    -DGGML_BLAS=ON
    -DGGML_BLAS_VENDOR=OpenBLAS
    -DGGML_OPENCL=OFF
    -DGGML_CUDA=OFF
    -DGGML_VULKAN=OFF
    -DGGML_METAL=OFF
)

if [[ -d "$LIBTORCH_DIR" ]]; then
    CMAKE_FLAGS+=(-DCMAKE_PREFIX_PATH="$LIBTORCH_DIR")
fi

# Backend selection
FORCE_BACKEND="${FORCE_BACKEND:-auto}"
if [[ "$FORCE_BACKEND" == "cuda" ]]; then
    echo "🚀 Forcing CUDA backend"
    CMAKE_FLAGS+=(-DGGML_CUDA=ON)
elif [[ "$FORCE_BACKEND" == "vulkan" ]]; then
    echo "🎮 Forcing Vulkan backend"
    CMAKE_FLAGS+=(-DGGML_VULKAN=ON)
elif [[ "$FORCE_BACKEND" == "metal" ]]; then
    echo "🍎 Forcing Metal backend"
    CMAKE_FLAGS+=(-DGGML_METAL=ON)
elif [[ "$FORCE_BACKEND" == "cpu" ]]; then
    echo "🧠 Forcing CPU-only build"
else
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
fi

# -------------------------------
# Build
# -------------------------------
echo "🛠️ Configuring CMake with flags:"
printf '   %s\n' "${CMAKE_FLAGS[@]}"

if [[ "${CLEAN_BUILD:-0}" == "1" ]]; then
    rm -rf build
fi

cmake -S . -B build "${CMAKE_FLAGS[@]}"

echo "⚙️ Building matrix_zmq_server..."
CPU_CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"
cmake --build build --target matrix_zmq_server -j"$CPU_CORES"

# -------------------------------
# Return to project root
# -------------------------------
cd "$PROJECT_ROOT"
echo "📂 Returned to project root: $PROJECT_ROOT"
echo "🎉 ggml backend installation complete!"
