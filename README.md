---

# 🚀 **Open Cluster AI Station – Mamba / GQA Multi-Node Setup**

This README explains the commands to **download models, cache weights, and run distributed inference** across multiple nodes in your local network.

---

## 🔹 1️⃣ Downloading Mamba2 & GQA Models

Use the Hugging Face CLI to download models locally.

### **Mamba-Codestral (SSM / mamba2)**

```bash
python -m huggingface_hub.cli download mistralai/Mamba-Codestral-1B-v0.1 \
    --local-dir /path/to/mamba_ssm/Mamba-Codestral-1B-v0.1 \
    --local-dir-use-symlinks False

python -m huggingface_hub.cli download mistralai/Mamba-Codestral-3B-v0.1 \
    --local-dir /path/to/mamba_ssm/Mamba-Codestral-3B-v0.1 \
    --local-dir-use-symlinks False

python -m huggingface_hub.cli download mistralai/Mamba-Codestral-7B-v0.1 \
    --local-dir /path/to/mamba_ssm/Mamba-Codestral-7B-v0.1 \
    --local-dir-use-symlinks False

python -m huggingface_hub.cli download mistralai/Mamba-Codestral-70B-v0.1 \
    --local-dir /path/to/mamba_ssm/Mamba-Codestral-70B-v0.1 \
    --local-dir-use-symlinks False
```

### **GQA / Meta-LLaMA Models**

```bash
python -m huggingface_hub.cli download mlabonne/Meta-Llama-3.1-1B-Instruct-abliterated \
    --local-dir /path/to/exo_models/Meta-Llama-3.1-1B-Instruct-abliterated \
    --local-dir-use-symlinks False

python -m huggingface_hub.cli download mlabonne/Meta-Llama-3.1-3B-Instruct-abliterated \
    --local-dir /path/to/exo_models/Meta-Llama-3.1-3B-Instruct-abliterated \
    --local-dir-use-symlinks False

python -m huggingface_hub.cli download mlabonne/Meta-Llama-3.1-7B-Instruct-abliterated \
    --local-dir /path/to/exo_models/Meta-Llama-3.1-7B-Instruct-abliterated \
    --local-dir-use-symlinks False

python -m huggingface_hub.cli download mlabonne/Meta-Llama-3.1-70B-Instruct-abliterated \
    --local-dir /path/to/exo_models/Meta-Llama-3.1-70B-Instruct-abliterated \
    --local-dir-use-symlinks False
```

> ⚡ **Tip:** Replace `/path/to/…` with your actual project path. Avoid symlinks to ensure compatibility with cluster caching.

---

## 🔹 2️⃣ Precache & Load Weights

Before running inference, **cache the model weights across your cluster** for faster startup:

```bash
python ../mamba_cluster_transformer_v1.py \
    --node-ips 192.168.2.100,192.168.2.100,192.168.2.101,192.168.2.104 \
    --backend-select-list llama,llama,llama,llama \
    --cpu-gpu-select-list 1,1,1,1 \
    --node-percentages 0.35,0.35,0.15,0.15 \
    --weight-cache-mode save \
    --precache \
    --precache-only
```

Later, **load the cached weights** for actual computation:

```bash
python ../mamba_cluster_transformer_v1.py \
    --node-ips 192.168.2.100,192.168.2.100,192.168.2.101,192.168.2.104 \
    --backend-select-list llama,llama,llama,llama \
    --cpu-gpu-select-list 1,1,1,1 \
    --node-percentages 0.35,0.35,0.15,0.15 \
    --weight-cache-mode load
```

> 🔹 You can scale node counts by duplicating IPs for multiple shards per machine.

---

## 🔹 3️⃣ Multi-Node Sharding Examples

### **6-node Example**

```bash
python ../mamba_cluster_transformer_v1.py \
    --node-ips 192.168.2.100,192.168.2.100,192.168.2.100,192.168.2.101,192.168.2.101,192.168.2.104 \
    --backend-select-list llama,llama,llama,llama,llama,llama \
    --cpu-gpu-select-list 1,1,1,1,1,1 \
    --node-percentages 0.30,0.30,0.10,0.10,0.05,0.15 \
    --weight-cache-mode save --precache --precache-only
```

* Duplicated IPs allow multiple shards per node
* Percentages control **matrix shard sizes** per node

---

### **Loading weights for 6-node setup**

```bash
python ../mamba_cluster_transformer_v1.py \
    --node-ips 192.168.2.100,192.168.2.100,192.168.2.100,192.168.2.101,192.168.2.101,192.168.2.104 \
    --backend-select-list llama,llama,llama,llama,llama,llama \
    --cpu-gpu-select-list 1,1,1,1,1,1 \
    --node-percentages 0.30,0.30,0.10,0.10,0.05,0.15 \
    --weight-cache-mode load
```

---

## 🔹 4️⃣ Running GQA / Meta-LLaMA Chat

Precache weights:

```bash
python ../GQA_cluster_chat_v1.py \
    --model-dir /path/to/mlabonne--Meta-Llama-3.1-8B-Instruct-abliterated \
    --node-ips 192.168.2.100,192.168.2.100,192.168.2.100,192.168.2.101,192.168.2.101,192.168.2.104 \
    --backend-select-list llama,llama,llama,llama,llama,llama \
    --cpu-gpu-select-list 1,1,1,1,1,1 \
    --node-percentages 0.30,0.30,0.10,0.10,0.05,0.15 \
    --weight-cache-mode save \
    --precache \
    --precache-only
```

Run inference:

```bash
python ../GQA_cluster_chat_v1.py \
    --model-dir /path/to/mlabonne--Meta-Llama-3.1-8B-Instruct-abliterated \
    --node-ips 192.168.2.100,192.168.2.100,192.168.2.100,192.168.2.101,192.168.2.101,192.168.2.104 \
    --backend-select-list llama,llama,llama,llama,llama,llama \
    --cpu-gpu-select-list 1,1,1,1,1,1 \
    --node-percentages 0.30,0.30,0.10,0.10,0.05,0.15 \
    --weight-cache-mode load \
    --max-new-tokens 64 \
    --micro-batch-size 8
```

> 🔹 Adjust `max-new-tokens` and `micro-batch-size` to control **memory usage** and **throughput**.

---

## 🔹 5️⃣ Key Notes

1. **Node duplication** allows multiple shards per physical machine.
2. **Node percentages** define matrix distribution sizes.
3. **Cache modes**:

| Mode | Description                   |
| ---- | ----------------------------- |
| save | Save model weights to cluster |
| load | Load cached weights           |

4. **Backend types**: `"llama"`, `"torch"`, `"opencl"`
5. **CPU/GPU selection**: `1` → use acceleration, `0` → CPU only

---

### ⚡ Quick Tips

* Always precache weights before running inference to reduce startup times.
* Use multiple shards per machine to fully utilize GPUs.
* Start with small `max-new-tokens` for large models to avoid OOM errors.
* Adjust `node-percentages` to match hardware memory limits.

---

