#!/usr/bin/env bash
set -e

# ================================
# Main Install Script
# ================================

echo "==========================================="
echo "      Open Cluster AI Station Installer"
echo "==========================================="
echo

# Prompt user for installation type
echo "Select installation type:"
echo "  1) Full install (Python environment + ggml backend)"
echo "  2) Worker node only install (ggml backend only)"
read -p "Enter choice [1 or 2]: " INSTALL_TYPE

# Path to install scripts
INSTALL_DIR="$(dirname "$0")/install"

PROJECT_ROOT="$(pwd)"

PY_ENV_SCRIPT="$INSTALL_DIR/install_py_env.sh"
GGML_BACKEND_SCRIPT="$INSTALL_DIR/build_ggml_backend.sh"
LIBTORCH_SCRIPT="$INSTALL_DIR/install_libtorch.sh"

# Validate scripts exist
if [[ ! -f "$GGML_BACKEND_SCRIPT" ]]; then
    echo "❌ ggml backend build script not found: $GGML_BACKEND_SCRIPT"
    exit 1
fi
if [[ "$INSTALL_TYPE" == "1" && ! -f "$PY_ENV_SCRIPT" ]]; then
    echo "❌ Python environment install script not found: $PY_ENV_SCRIPT"
    exit 1
fi

# Execute based on choice
case "$INSTALL_TYPE" in
    1)
        echo "🚀 Running full install..."
        echo "🟢 Installing Python environment..."
        bash "$PY_ENV_SCRIPT"

        echo "🟢 Building ggml backend..."
        bash "$GGML_BACKEND_SCRIPT"
        ;;
    2)
        echo "🚀 Running worker node only install..."
        echo "🟢 Building ggml backend..."
        bash "$GGML_BACKEND_SCRIPT"
        ;;
    *)
        echo "❌ Invalid selection. Exiting."
        exit 1
        ;;
esac

echo "✅ Installation complete!"
