#!/usr/bin/env bash
set -e

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
ENV_NAME="open_cluster_env"
PYTHON_BIN=python3

echo "🚀 Creating Python virtual environment: $ENV_NAME"
$PYTHON_BIN -m venv $ENV_NAME

echo "✅ Activating environment"
source $ENV_NAME/bin/activate

echo "⬆️ Upgrading pip / setuptools / wheel"
pip install --upgrade pip setuptools wheel

echo "📦 Installing core scientific stack"
pip install \
  numpy \
  scipy \
  einops \
  psutil \
  tqdm

echo "🔥 Installing PyTorch"
# ---- CPU ONLY ----
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# ---- OR CUDA (uncomment ONE if needed) ----
# CUDA 12.1
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "⚡ Installing distributed / system deps"
pip install \
  pyzmq \
  msgpack \
  cloudpickle \
  ray \
  zmq

echo "🧠 Installing Mamba / Triton dependencies"
pip install \
  triton \
  packaging \
  ninja

echo "📂 Installing HuggingFace tooling"
pip install \
  huggingface_hub \
  safetensors \
  tokenizers \
  transformers

echo "🧪 Installing dev / debugging tools"
pip install \
  ipython \
  rich \
  pytest

echo "✅ Verifying critical imports"
python - << 'EOF'
import torch
import zmq
import einops
import triton
print("✔ torch:", torch.__version__)
print("✔ zmq:", zmq.__version__)
print("✔ einops OK")
print("✔ triton OK")
EOF

echo "🎉 Python environment setup complete!"
echo "To activate later: source $ENV_NAME/bin/activate"

# -------------------------------
# Return to project root
# -------------------------------
cd "$PROJECT_ROOT"
echo "📂 Returned to project root: $PROJECT_ROOT"
