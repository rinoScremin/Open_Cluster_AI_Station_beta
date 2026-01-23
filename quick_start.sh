#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess

# -------------------------------
# Resolve project root
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
LLM_DIR = PROJECT_ROOT / "llm_models"
QUICK_START_DIR = PROJECT_ROOT / "quick_start"

# -------------------------------
# List available models
# -------------------------------
available_models = [p.name for p in LLM_DIR.iterdir() if p.is_dir()]
if not available_models:
    print("❌ No models found in llm_models")
    exit(1)

print("📂 Available models:")
for idx, m in enumerate(available_models):
    print(f"{idx+1}: {m}")
print("0: Skip loading any model")

# -------------------------------
# Prompt user for model selection
# -------------------------------
selection = input("Select model number to run: ").strip()
try:
    selection = int(selection)
except ValueError:
    print("❌ Invalid input")
    exit(1)

if selection == 0:
    print("Skipping model loading")
    exit(0)
elif 1 <= selection <= len(available_models):
    model_name = available_models[selection - 1]
    model_path = LLM_DIR / model_name
    print(f"✅ Selected model: {model_name}")
else:
    print("❌ Invalid selection")
    exit(1)

# -------------------------------
# Determine script to run
# -------------------------------
if model_name.startswith("Meta"):
    launcher_script = PROJECT_ROOT / "GQA_cluster_chat_v1.py"
elif model_name.startswith("Mamba"):
    launcher_script = PROJECT_ROOT / "mamba_cluster_chat_v1.py"
else:
    print("❌ Unknown model type")
    exit(1)

# -------------------------------
# Prompt for single-node or multi-node test
# -------------------------------
print("\nSelect test mode:")
print("1: Single node test")
print("2: Multiple node test")
test_mode = input("Enter 1 or 2: ").strip()

if test_mode == "1":
    # -------------------------------
    # Single node cluster config
    # -------------------------------
    node_ips = input("Enter single node IP (example: 192.168.2.100): ").strip()
    backend_select = input("Enter backend (example: llama): ").strip()
    cpu_gpu_select = input("Enter CPU/GPU flag (1 for GPU, 0 for CPU) (example: 1): ").strip()
    node_percentages = input("Enter node percentage (example: 1): ").strip()

elif test_mode == "2":
    # -------------------------------
    # Multi-node setup
    # -------------------------------
    print("\n📡 Detecting nodes on LAN network...")
    # Run get_land_ips.sh
    subprocess.run([QUICK_START_DIR / "get_land_ips.sh", "192.168.2.100"])  # Replace with your local IP if needed

    print("\n🔑 Setting up SSH keys for all nodes...")
    subprocess.run([QUICK_START_DIR / "set_up_ssh.sh"])  # You need a script to copy keys

    # Kill any existing matrix_zmq_server processes and start the backend on all nodes
    with open("land_nodes_IP.txt", "r") as f:
        node_list = [line.strip() for line in f.readlines()]

    for node in node_list:
        print(f"🚀 Setting up backend on {node}...")
        ssh_cmd = f"""
            ssh {node} 'pkill -f ./build/cluster_backend/cluster_backend_main/matrix_zmq_server || true && \
            {PROJECT_ROOT}/cluster_matrix/ggml/build/cluster_backend/cluster_backend_main/matrix_zmq_server &'
        """
        subprocess.run(ssh_cmd, shell=True)

    # -------------------------------
    # Prompt user to enter cluster parameters
    # -------------------------------
    print("\nEnter cluster parameters for multi-node test:")
    print("Example formats:")
    print("node_ips = 192.168.2.100,192.168.2.100,192.168.2.101")
    print("backend_select = llama,llama,llama")
    print("cpu_gpu_select = 1,1,1")
    print("node_percentages = 0.4,0.3,0.3\n")

    node_ips = input("Enter node_ips: ").strip()
    backend_select = input("Enter backend_select list: ").strip()
    cpu_gpu_select = input("Enter cpu_gpu_select list: ").strip()
    node_percentages = input("Enter node_percentages list: ").strip()

else:
    print("❌ Invalid selection")
    exit(1)

# -------------------------------
# Precache weights first
# -------------------------------
precache_cmd = f"""
python {launcher_script} \
  --model-dir {model_path} \
  --node-ips {node_ips} \
  --backend-select-list {backend_select} \
  --cpu-gpu-select-list {cpu_gpu_select} \
  --node-percentages {node_percentages} \
  --weight-cache-mode save \
  --precache \
  --precache-only
"""
print(f"📦 Pre-caching weights for {model_name}...")
os.system(precache_cmd)

# -------------------------------
# Run model
# -------------------------------
run_cmd = f"""
python {launcher_script} \
  --model-dir {model_path} \
  --node-ips {node_ips} \
  --backend-select-list {backend_select} \
  --cpu-gpu-select-list {cpu_gpu_select} \
  --node-percentages {node_percentages} \
  --weight-cache-mode load \
  --max-new-tokens 64 \
  --micro-batch-size 8
"""
print(f"🚀 Running model {model_name}...")
os.system(run_cmd)
