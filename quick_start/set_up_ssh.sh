#!/usr/bin/env bash
set -e

# -------------------------------
# Make sure land_nodes_IP.txt exists
# -------------------------------
if [ ! -f land_nodes_IP.txt ]; then
    echo "❌ land_nodes_IP.txt not found. Run get_land_ips.sh first."
    exit 1
fi

# -------------------------------
# Ensure SSH key exists
# -------------------------------
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "🔑 Generating new SSH key..."
    ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
else
    echo "✅ SSH key already exists"
fi

# -------------------------------
# Copy SSH key to all nodes
# -------------------------------
while IFS= read -r node; do
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" "echo 2>&1" >/dev/null; then
        echo "✅ SSH already working to $node"
    else
        echo "📡 Copying SSH key to $node..."
        ssh-copy-id "$node"
    fi
done < land_nodes_IP.txt

echo "🎉 SSH setup complete for all nodes!"
