#!/usr/bin/env bash
#set -euo pipefail

sudo ifconfig eno1 192.168.2.100 netmask 255.255.255.0

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ray-conda-env 2>/dev/null || echo "Conda env missing on $(hostname)"

