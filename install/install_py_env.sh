#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
trap 'echo "❌ Python env install failed at line $LINENO. See logs above." >&2' ERR

# -------------------------------
# Resolve paths
# -------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
echo "📂 Project root: $PROJECT_ROOT"

# -------------------------------
# Python environment
# -------------------------------
ENV_NAME="${ENV_NAME:-open_cluster_env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STRICT="${STRICT:-0}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "❌ Required command not found: $1"
    exit 1
  fi
}

require_cmd "$PYTHON_BIN"

PY_VERSION="$($PYTHON_BIN - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
PY_MAJOR="${PY_VERSION%%.*}"
PY_MINOR="${PY_VERSION##*.}"
if [[ "$PY_MAJOR" -lt 3 || "$PY_MINOR" -lt 8 ]]; then
  echo "❌ Python 3.8+ is required. Found: $PY_VERSION"
  exit 1
fi

VENV_DIR="$PROJECT_ROOT/$ENV_NAME"
if [[ -d "$VENV_DIR" ]]; then
  echo "ℹ️ Existing virtual environment detected: $VENV_DIR"
else
  echo "🚀 Creating Python virtual environment: $ENV_NAME"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "✅ Activating environment"
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # Unix-like
  source "$VENV_DIR/bin/activate"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
  # Git Bash / MSYS
  source "$VENV_DIR/Scripts/activate"
else
  echo "❌ Could not find venv activate script in $VENV_DIR"
  exit 1
fi

pip_install() {
  python -m pip install "$@"
}

pip_install_optional() {
  if pip_install "$@"; then
    return 0
  fi
  echo "⚠️ Optional install failed: $*"
  if [[ "$STRICT" == "1" ]]; then
    exit 1
  fi
  return 0
}

echo "⬆️ Upgrading pip / setuptools / wheel"
pip_install --upgrade pip setuptools wheel

echo "📦 Installing core scientific stack"
pip_install \
  numpy \
  scipy \
  einops \
  psutil \
  tqdm

pip_install nicegui
echo "🔥 Installing PyTorch"
OS="$(uname -s)"
ARCH="$(uname -m)"
if [[ -n "${TORCH_PIP_SPEC:-}" ]]; then
  # Example: TORCH_PIP_SPEC="torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
  pip_install ${TORCH_PIP_SPEC}
else
  case "${OS}-${ARCH}" in
    Linux-x86_64|Linux-amd64)
      pip_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
      ;;
    Darwin-*)
      pip_install torch torchvision torchaudio
      ;;
    *)
      echo "⚠️ Unknown platform (${OS}-${ARCH}). Installing torch from PyPI."
      echo "   Override with TORCH_PIP_SPEC if needed."
      pip_install torch torchvision torchaudio
      ;;
  esac
fi

echo "⚡ Installing distributed / system deps"
pip_install \
  pyzmq \
  msgpack \
  cloudpickle

echo "🧠 Installing Mamba / Triton dependencies"
pip_install \
  packaging \
  ninja

INSTALL_TRITON="${INSTALL_TRITON:-auto}"
if [[ "$INSTALL_TRITON" == "1" ]]; then
  pip_install triton
elif [[ "$INSTALL_TRITON" == "auto" ]]; then
  if [[ "$OS" == "Linux" && "$ARCH" == "x86_64" ]]; then
    pip_install_optional triton
  else
    echo "ℹ️ Skipping triton on ${OS}-${ARCH}. Set INSTALL_TRITON=1 to force."
  fi
else
  echo "ℹ️ Triton install disabled (INSTALL_TRITON=$INSTALL_TRITON)."
fi

echo "📂 Installing HuggingFace tooling"
pip_install \
  huggingface_hub \
  safetensors \
  tokenizers \
  transformers

echo "🧪 Installing dev / debugging tools"
pip_install \
  ipython \
  rich \
  pytest

INSTALL_RAY="${INSTALL_RAY:-auto}"
if [[ "$INSTALL_RAY" == "1" ]]; then
  pip_install ray
elif [[ "$INSTALL_RAY" == "auto" ]]; then
  pip_install_optional ray
else
  echo "ℹ️ Ray install disabled (INSTALL_RAY=$INSTALL_RAY)."
fi

echo "✅ Verifying critical imports"
python - << 'EOF'
import torch
import zmq
import einops
print("✔ torch:", torch.__version__)
print("✔ zmq:", zmq.__version__)
print("✔ einops OK")
try:
    import triton  # optional
    print("✔ triton:", triton.__version__)
except Exception as exc:
    print("ℹ️ triton not available:", exc)
EOF

echo "🎉 Python environment setup complete!"
echo "To activate later: source $ENV_NAME/bin/activate"

# -------------------------------
# Return to project root
# -------------------------------
cd "$PROJECT_ROOT"
echo "📂 Returned to project root: $PROJECT_ROOT"
