#!/bin/bash
# Download and extract libtorch CPU-only version

LIBTORCH_VERSION="2.2.2"
LIBTORCH_BUILD="2.2.2+cpu"
LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_BUILD}.zip"

mkdir -p cluster_matrix/ggml/libtorch
cd cluster_matrix/ggml || exit

echo "Downloading libtorch ${LIBTORCH_BUILD}..."
curl -L -o libtorch.zip $LIBTORCH_URL

echo "Extracting libtorch..."
unzip -q libtorch.zip -d .
rm libtorch.zip

echo "libtorch installed to cluster_matrix/ggml/libtorch"
