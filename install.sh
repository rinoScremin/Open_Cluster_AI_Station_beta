#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
trap 'echo "❌ Install failed at line $LINENO. See logs above." >&2' ERR

# ================================
# Main Install Script
# ================================

echo "==========================================="
echo "      Open Cluster AI Station Installer"
echo "==========================================="
echo

usage() {
    cat <<'EOF'
Usage: ./install.sh [--full|--worker] [--non-interactive]

Options:
  --full, -f            Full install (Python environment + ggml backend)
  --worker, -w          Worker node only install (ggml backend only)
  --non-interactive, -n Do not prompt; uses DEFAULT_INSTALL_TYPE if set (default: 1)
  --help, -h            Show this help
EOF
}

NON_INTERACTIVE=0
INSTALL_TYPE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full|-f) INSTALL_TYPE="1"; shift ;;
        --worker|-w) INSTALL_TYPE="2"; shift ;;
        --non-interactive|-n) NON_INTERACTIVE=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "❌ Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Resolve paths from script location
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$PROJECT_ROOT/install"
PY_ENV_SCRIPT="$INSTALL_DIR/install_py_env.sh"
GGML_BACKEND_SCRIPT="$INSTALL_DIR/build_ggml_backend.sh"

cd "$PROJECT_ROOT"

# Prompt user for installation type if needed
if [[ -z "$INSTALL_TYPE" ]]; then
    if [[ "$NON_INTERACTIVE" -eq 1 ]]; then
        INSTALL_TYPE="${DEFAULT_INSTALL_TYPE:-1}"
        echo "ℹ️ Non-interactive mode. Using install type: $INSTALL_TYPE"
    else
        echo "Select installation type:"
        echo "  1) Full install (Python environment + ggml backend)"
        echo "  2) Worker node only install (ggml backend only)"
        read -r -p "Enter choice [1 or 2] (default: 1): " INSTALL_TYPE
        INSTALL_TYPE="${INSTALL_TYPE:-1}"
    fi
fi

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
