# 🚀 **The Vision: Democratizing AI Compute**

**We built this because we were tired of seeing perfectly good hardware collecting dust.** The AI revolution shouldn't require $10,000 GPUs and HPC expertise. **This system proves you can run massive models on whatever hardware you already have** - gaming PCs, old laptops, office workstations, even integrated graphics.

## 🎯 **What This Means For You:**

**If you have:**
- An old gaming PC with an AMD/NVIDIA GPU
- A MacBook with Apple Silicon  
- A Raspberry Pi cluster
- Office computers after hours
- Friends with PCs who want to contribute

**You now have:**
- A distributed supercomputer
- The ability to run models that previously needed A100s
- A system that scales by just adding more devices
- **Zero configuration complexity** - it just works

## 🔓 **Why We're Open Sourcing This:**

1. **Break the hardware monopoly** - AI shouldn't require specific NVIDIA GPUs
2. **Reduce e-waste** - Old hardware has value when it computes together
3. **Lower barriers** - Students, researchers, startups can afford distributed AI
4. **Community innovation** - Let's build the future of distributed computing together

## 🌱 **This is Version 1.0 - The Foundation**

The architecture is solid, it works, and it's **already useful**. But this is just the beginning. With community contributions, we can:
- Add more backends (ROCm, SYCL, WebGPU)
- Improve scheduling algorithms  
- Add containerization/K8s support
- Create web dashboards
- Build transformer-specific optimizations

## 🤝 **Join The Movement**

**If you believe:**
- AI should be accessible to everyone, not just big tech
- Old hardware deserves a second life
- Distributed systems should be simple to use
- Community-driven software beats closed ecosystems

**Then this project is for you.** Let's build the distributed computing platform that actually works for real people with real hardware. No PhD required, no $10k GPU needed - just the computers you already have, working together.

**This isn't just code. It's a statement: Your hardware is enough.** 🖥️💻🖥️💻=🚀


# 🔥 Cluster Matrix - Distributed Computation System

## 🎯 **What This Is**
A **hybrid distributed computing system** that turns ANY group of computers into a supercomputer. Mix CPUs, GPUs (NVIDIA/AMD/Intel), Apple Silicon - **all working together** on massive matrix operations.

---

# USING `cluster_matrix_v1 system #1`

This document demonstrates how to use `cluster_matrix_v1` system #1 for **distributed matrix operations** across multiple machines, GPUs, and backends, including **LLM inference workloads** (e.g. attention + MLP layers).

---

## 🧩 System 1 — Linear Shard Split & Join

## 🔹 End-to-End Example

### Input

```python
A = (1000, 512)
B = (1000, 512)
Split dimension = 0  (rows of B)
```

### Split

```
B0 = (250, 512)
B1 = (250, 512)
B2 = (250, 512)
B3 = (250, 512)
```

### Compute

```
C0 = A @ B0ᵀ → (1000, 250)
C1 = A @ B1ᵀ → (1000, 250)
C2 = A @ B2ᵀ → (1000, 250)
C3 = A @ B3ᵀ → (1000, 250)

```

### Join

```
C = cat([C0, C1, C2, C3], dim=1)
C shape → (1000, 1000)

```

---

## 🔹 Key Properties

✔ Deterministic ordering
✔ Zero numerical error (no accumulation)
✔ Fast reconstruction (single `torch::cat`)
✔ Ideal for:

* embedding transforms
* linear layers
* inference sharding
* batch-parallel workloads

---

## 📡 Cluster Configuration

### Node IP List

```python
IP_list = [
    "192.168.2.100",
    "192.168.2.100",
    "192.168.2.101",
    "192.168.2.104",
]

cluster_zmq_obj = cluster_zmq(IP_list) # init the cluster paths and networks connections 
```

#### How IP duplication works

If the **same IP is listed multiple times**, the C++ backend will attempt to bind each entry to **separate hardware** on that machine.

Example:

```
192.168.2.100
 ├─ GPU 0 → shard 1
 ├─ GPU 1 → shard 2
 └─ CPU BLAS → shard 3 (fallback if no GPU available)
```

Notes:

* If no additional GPU is available, the shard **falls back to CPU BLAS**
* CPU BLAS **may not support ADD operations correctly** on some systems
* If you encounter incorrect ADD results, disable CPU BLAS for those nodes

Hardware examples:

* `192.168.2.101` → Laptop with integrated GPU / APU

  * First shard → GPU
  * Additional shards → CPU BLAS

* `192.168.2.104` → Intel i5-6500

  * No GPU
  * Always uses CPU BLAS

---

### Matrix Split Percentages

```python
percentages = [0.35, 0.35, 0.15, 0.15]
```

Defines how **Matrix B** is distributed:

| Node   | Percentage |
| ------ | ---------- |
| Node 1 | 35%        |
| Node 2 | 35%        |
| Node 3 | 15%        |
| Node 4 | 15%        |

---

### Backend Acceleration Selection

```python
CPU_GPU_select_list = [True, True, True, True]
```

* `True` → use compiled backend acceleration
* `False` → CPU-only (no BLAS / GPU acceleration)

---

### Backend Type Selection

```python
backend_select_list = ["llama", "llama", "llama", "llama"]
```

Available backends:

| Backend  | Description                        |
| -------- | ---------------------------------- |
| `llama`  | GGML backend (CPU/GPU accelerated) |
| `torch`  | PyTorch backend                    |
| `opencl` | Custom OpenCL backend              |

You can **mix backends per node**:

```python
backend_select_list = ["llama", "torch", "opencl", "llama"]
```

This allows support for **custom or experimental hardware**, including OpenCL-based accelerators.

---

## 🧠 RMSNorm (Local Preprocessing)

```python
post_attn_ln_w = torch.load(post_attn_ln_path, map_location="cpu")

if post_attn_ln_w.ndim != 1:
    raise ValueError("LayerNorm weight must be 1D")

if post_attn_ln_w.shape[0] != residual.shape[1]:
    raise ValueError("Hidden size mismatch")

mlp_in = self.rms_norm(residual, post_attn_ln_w)
mlp_in_col = mlp_in.t().contiguous()
```

⚠️ **IMPORTANT**

`cluster_matrix_v1` does **not** perform tensor reshaping.

All operations like:

* `.contiguous()`
* `.transpose()`
* `.reshape()`

**must be done in PyTorch before sending the tensor to the cluster.**

---

## 📦 Creating Cluster Matrices

### Matrix A (Full / Not Sharded)

```python
mlp_in_cluster = cluster_matrix(
    matrix_file_path=mlp_in_col,
    cluster_zmq_object=cluster_zmq_obj,
    CPU_GPU_select_list=CPU_GPU_select_list,
    node_percentages=percentages,
    back_end_select_list=backend_select_list,
    split_matrix=False,
    dim=1,
    auto_set_up=[1, "save"],
    matrix_name="layer0_mlp_in",
)
```

---

### Matrix B (Sharded Weights)

```python
mlp_gate_cluster = cluster_matrix(
    matrix_file_path=mlp_gate_path,
    cluster_zmq_object=cluster_zmq_obj,
    CPU_GPU_select_list=CPU_GPU_select_list,
    node_percentages=percentages,
    back_end_select_list=backend_select_list,
    split_matrix=True,
    dim=1,
    auto_set_up=[1, "load"],
)
```

💡 **Recommendation**

* Cache all large weight tensors once using `"save"`
* Reuse them later using `"load"`
* Some tensors (e.g. token embeddings) **cannot be cached** and must always use `"save"`

---

## ⚙️ Distributed Attention Example

```python
x = self.rms_norm(input_token_embeddings, input_layernorm_weight)
x = x.unsqueeze(1)

x = cluster_matrix(
    matrix_file_path=x,
    cluster_zmq_object=cluster_zmq_obj,
    CPU_GPU_select_list=CPU_GPU_select_list,
    node_percentages=percentages,
    back_end_select_list=backend_select_list,
    split_matrix=False,
    dim=1,
    auto_set_up=[1, "save"],
    matrix_name="input_token_embeddings",
)
```

```python
q_flat = x.cluster_shard_operation(q, True, False, True)
k_flat = x.cluster_shard_operation(k, True, False, True)
v_flat = x.cluster_shard_operation(v, True, False, True)
```

By default:

* Results are **sent back**
* Returned as **PyTorch tensors**
* Ready for further local processing

---

## ➕ Distributed Matrix Addition Example NOTE: CLUSTER ADD ONLY USED WITH 'SYSTEM 1' DO NOT USE WITH 'SYSTEM 2'

### Cluster ADD (Sharded)

```python
big_new_matrixC = big_new_matrixA.cluster_shard_operation(
    big_new_matrixB,
    False,
    True,
    True,
    "add",
)
```

✔️ For ADD:

* **Both matrices must be split**
* Operation is performed shard-wise:

  ```
  C_i = A_i + B_i
  ```

---

## 🖥️ Single-PC / Single-GPU Mode

You can also use `cluster_matrix` **without a cluster**.

Useful if:

* You do not have CUDA
* You only have one GPU
* GPU supports Vulkan / Metal / OpenCL (via GGML)

```python
cluster_zmq([192.168.2.100])
big_new_matrixA_single_node = cluster_matrix(
    matrix_file_path=big_test_matrix_pathA_T,
    cluster_zmq_object=cluster_zmq,
    CPU_GPU_select_list=[True],
    node_percentages=[1],
    back_end_select_list=["llama"],
    split_matrix=False,
    dim=0,
    auto_set_up=[1, "save"],
)

big_new_matrixB_single_node = cluster_matrix(
    matrix_file_path=big_test_matrix_pathA_T,
    cluster_zmq_object=cluster_zmq,
    CPU_GPU_select_list=[True],
    node_percentages=[1],
    back_end_select_list=["llama"],
    split_matrix=False,
    dim=0,
    auto_set_up=[1, "save"],
)

big_new_matrixC = big_new_matrixA_single_node.cluster_shard_operation(
    big_new_matrixB_single_node,
    False,
    True,
    False ## IF YOU ARE USING 1 NODE DO NOT USE SEND BACK NOT NEEDED IT MUST BE SET TO FALSE FOR SINGLE NODE MATRIX OPERATION'S 
)
#small_c_ref = A3 @ B3.T
#torch.save(small_c_ref, 'model_model_matrices/small_c_ref.pt')
#check_combined_result_values('model_model_matrices/small_c_ref.pt',small_new_matrixC) # use the 'check_combined_result_values' function to make ssure
#values are correct 
```

---
USING 'cluster_matrix_v1 system #1' DEFAULTS
---
```python
import torch
import numpy as np

A3 = torch.from_numpy(np.random.rand(1500, 4500).astype(np.float16))
B3 = torch.from_numpy(np.random.rand(1000, 4500).astype(np.float16))

torch.save(A3, 'model_model_matrices/small_matrixA.pt')
torch.save(B3, 'model_model_matrices/small_matrixB.pt')

small_test_matrix_pathA = 'model_model_matrices/small_matrixA.pt'
small_test_matrix_pathB = 'model_model_matrices/small_matrixB.pt'

################################ BELOW IS 6 NODE DEFAULT SETUP ################################

IP_list = [
    '192.168.2.100',
    '192.168.2.100',
    '192.168.2.100',
    '192.168.2.101',
    '192.168.2.101',
    '192.168.2.104'
]

cluster_zmq_obj = cluster_zmq(IP_list)

small_big_new_matrixA = cluster_matrix(
    small_test_matrix_pathA,
    cluster_zmq_object=cluster_zmq_obj,
    split_matrix=False,
    dim=0,
    auto_set_up=[1, "save"]
)

## BELOW IS HOW THE DEFAULT VALUES WILL BE SET
# CPU_GPU_select_list = [True, True, True, True, True, True]
# backend_select_list = ['llama', 'llama', 'llama', 'llama', 'llama', 'llama']
# percentages = [0.17, 0.17, 0.17, 0.16, 0.16]

small_big_new_matrixB = cluster_matrix(
    small_test_matrix_pathB,
    cluster_zmq_object=cluster_zmq_obj,
    split_matrix=True,
    dim=0,
    auto_set_up=[1, "save"]
)

small_new_matrixC = small_big_new_matrixA.cluster_shard_operation(
    small_big_new_matrixB,
    False,
    True,
    True
)

## BELOW IS HOW THE DEFAULT VALUES WILL BE SET
# CPU_GPU_select_list = [True, True, True, True, True, True]
# backend_select_list = ['llama', 'llama', 'llama', 'llama', 'llama', 'llama']
# percentages = [0.17, 0.17, 0.17, 0.16, 0.16]


################################ BELOW IS 2 NODE DEFAULT SETUP ################################

IP_list = ['192.168.2.100', '192.168.2.100']  # 50/50 split default

cluster_zmq_obj = cluster_zmq(IP_list)

small_big_new_matrixA = cluster_matrix(
    small_test_matrix_pathA,
    cluster_zmq_object=cluster_zmq_obj,
    split_matrix=False,
    auto_set_up=[1, "save"]
)

## BELOW IS HOW THE DEFAULT VALUES WILL BE SET
# # dim = 0
# CPU_GPU_select_list = [True, True]
# backend_select_list = ['llama', 'llama']
# percentages = [0.5, 0.5]

small_big_new_matrixB = cluster_matrix(
    small_test_matrix_pathB,
    cluster_zmq_object=cluster_zmq_obj,
    split_matrix=True,
    auto_set_up=[1, "save"]
)

## BELOW IS HOW THE DEFAULT VALUES WILL BE SET
# dim = 0
# CPU_GPU_select_list = [True, True]
# backend_select_list = ['llama', 'llama']
# percentages = [0.5, 0.5]

small_new_matrixC = small_big_new_matrixA.cluster_shard_operation(
    small_big_new_matrixB,
    False,
    True,
    True
)
```
---

# USING `cluster_matrix_v1 system #2`

This document demonstrates how to use `cluster_matrix_v1` system #1 for **distributed matrix operations** across multiple machines, GPUs, and backends, including **LLM inference workloads** (e.g. attention + MLP layers).

---

```python
import torch
import numpy as np

A3 = torch.from_numpy(np.random.rand(1500, 4500).astype(np.float16))
B3 = torch.from_numpy(np.random.rand(1000, 4500).astype(np.float16))

torch.save(A3, 'model_model_matrices/small_matrixA.pt')
torch.save(B3, 'model_model_matrices/small_matrixB.pt')

small_test_matrix_pathA = 'model_model_matrices/small_matrixA.pt'
small_test_matrix_pathB = 'model_model_matrices/small_matrixB.pt'

IP_list = [
    '192.168.2.100',
    '192.168.2.100',
    '192.168.2.100',
    '192.168.2.101',
    '192.168.2.101',
    '192.168.2.104'
]

CPU_GPU_select_list = [True, True, True, True, True, True]
backend_select_list = ['llama', 'llama', 'llama', 'llama', 'llama', 'llama']

cluster_zmq_obj = cluster_zmq(IP_list)

small_big_new_matrixA = cluster_matrix(
    small_test_matrix_pathA,
    cluster_zmq_object=cluster_zmq_obj,
    CPU_GPU_select_list=CPU_GPU_select_list,
    back_end_select_list=backend_select_list,
    split_matrix=True,
    dim=0,
    auto_set_up=[2, "save"],
    matrix_labeling='a'
)

small_new_matrixB = cluster_matrix(
    small_test_matrix_pathB,
    cluster_zmq_object=cluster_zmq_obj,
    CPU_GPU_select_list=CPU_GPU_select_list,
    back_end_select_list=backend_select_list,
    split_matrix=True,
    dim=0,
    auto_set_up=[2, "save"],
    matrix_labeling='b'
)

small_new_matrixC = small_big_new_matrixA.cluster_shard_operation(
    small_new_matrixB, False, True, True
)

small_big_new_matrixA = cluster_matrix(
    small_test_matrix_pathA,
    cluster_zmq_object=cluster_zmq_obj,
    CPU_GPU_select_list=CPU_GPU_select_list,
    back_end_select_list=backend_select_list,
    split_matrix=True,
    dim=0,
    auto_set_up=[2, "load"],
    matrix_labeling='a'
)

small_new_matrixB = cluster_matrix(
    small_test_matrix_pathB,
    cluster_zmq_object=cluster_zmq_obj,
    CPU_GPU_select_list=CPU_GPU_select_list,
    back_end_select_list=backend_select_list,
    split_matrix=True,
    dim=0,
    auto_set_up=[2, "load"],
    matrix_labeling='b'
)

small_new_matrixC = small_big_new_matrixA.cluster_shard_operation(
    small_new_matrixB, False, True, True
)

############################# TESTING CLUSTER MATRIX DEFAULT OPERATIONS SYSTEM 2 #############################

IP_list = [
    '192.168.2.100',
    '192.168.2.100',
    '192.168.2.101',
    '192.168.2.104'
]

cluster_zmq_obj = cluster_zmq(IP_list)

matrixA_float16 = cluster_matrix(
    matrix_pathA_float16,
    cluster_zmq_object=cluster_zmq_obj,
    split_matrix=True,
    dim=1,
    auto_set_up=[2, "save"],
    matrix_labeling='a'
)

matrixB_float16 = cluster_matrix(
    matrix_pathB_float16,
    cluster_zmq_object=cluster_zmq_obj,
    split_matrix=True,
    dim=1,
    auto_set_up=[2, "save"],
    matrix_labeling='b'
)

big_new_matrixC = matrixA_float16.cluster_shard_operation(
    matrixB_float16, False, True, True
)

check_combined_result_values(
    'model_matrices/c_ref_float16.pt',
    c_ref_float16
)

```

---

# System 2 — Matrix Split & Combine

System 2 uses a **grid-based splitting pattern** for distributed matrix operations.

* This produces a **fixed number of matrix operations** (ops) for configurations with fewer than the threshold nodes.
* As you add nodes, the number of ops **remains the same** until the number of nodes exceeds the current ops.
* Once this threshold is crossed, the number of matrix ops **increases to the next multiple** (e.g., 8 → 12 → 16, etc.).
* This pattern continues as nodes are added.

---

## Node Mapping Examples

### Example 1 — 6 nodes

```python
IP_list = [
    '192.168.2.100','192.168.2.100','192.168.2.100',
    '192.168.2.101','192.168.2.101',
    '192.168.2.104'
]
```

* **Number of matrix ops:** 8

```
'192.168.2.100' → matrix op 1 and 7
'192.168.2.100' → matrix op 2 and 8
'192.168.2.100' → matrix op 3
'192.168.2.101' → matrix op 4
'192.168.2.101' → matrix op 5
'192.168.2.104' → matrix op 6
```

---

### Example 2 — 8 nodes

```python
IP_list = [
    '192.168.2.100','192.168.2.100','192.168.2.100',
    '192.168.2.101','192.168.2.101','192.168.2.104',
    '192.168.2.103','192.168.2.102'
]
```

* **Number of matrix ops:** 8

```
'192.168.2.100' → matrix op 1
'192.168.2.100' → matrix op 2
'192.168.2.100' → matrix op 3
'192.168.2.101' → matrix op 4
'192.168.2.101' → matrix op 5
'192.168.2.104' → matrix op 6
'192.168.2.103' → matrix op 7
'192.168.2.102' → matrix op 8
```

---

# Splitting & Join Workflow

### Base Example

```python
A3 = np.random.rand(1500, 1500)
B3 = np.random.rand(1500, 1500)
```

* **Split A along columns (`dim=1`)**
* **Split B along rows (`dim=0`)**

**A shards:**

```
A1 = (1500, 750)
A2 = (1500, 750)
```

**B shards:**

```
B1 = (750, 1500)
B2 = (750, 1500)
```

**Further split B shards:**

```
B1_1 = (250, 1500)
B1_2 = (250, 1500)
B1_3 = (250, 1500)

B2_1 = (250, 1500)
B2_2 = (250, 1500)
B2_3 = (250, 1500)
```

**Compute operations:**

```
A1[:, 0:250]   @ B1_1 = C1
A1[:, 250:500] @ B1_2 = C2
A1[:, 500:750] @ B1_3 = C3

A2[:, 0:250]   @ B2_1 = C4
A2[:, 250:500] @ B2_2 = C5
A2[:, 500:750] @ B2_3 = C6
```

**Final result:**

```
C = C1 + C2 + C3 + C4 + C5 + C6
```

---

# ✅ Example 1 — 4 nodes

**Target:**

```
C = C1 + C2 + C3 + C4
```

**Matrices:**

```python
A = np.random.rand(500, 1500)
B = np.random.rand(1500, 1000)
```

**Split strategy:**

```
Split A along dim = 1
Split B along dim = 0
```

**A shards (2 shards):**

```
A1 = (500, 750)
A2 = (500, 750)
```

**B shards (2 shards):**

```
B1 = (750, 1000)
B2 = (750, 1000)
```

**Further split B shards (2-way for 4 nodes):**

```
B1_1 = (375, 1000)
B1_2 = (375, 1000)

B2_1 = (375, 1000)
B2_2 = (375, 1000)
```

**Compute:**

```
A1[:, 0:375]   @ B1_1 = C1
A1[:, 375:750] @ B1_2 = C2

A2[:, 0:375]   @ B2_1 = C3
A2[:, 375:750] @ B2_2 = C4
```

**Final result:**

```
C = C1 + C2 + C3 + C4  → shape: (500, 1000)
```

---

# ✅ Example 2 — 8 nodes

**Target:**

```
C = C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8
```

**Matrices:**

```python
A = np.random.rand(500, 1500)
B = np.random.rand(1500, 1000)
```

**Split strategy:**

```
Split A along dim = 1
Split B along dim = 0
```

**A shards (2 shards):**

```
A1 = (500, 750)
A2 = (500, 750)
```

**B shards (2 shards):**

```
B1 = (750, 1000)
B2 = (750, 1000)
```

**Further split B shards (4-way split per shard):**

```
B1_1 = (187, 1000)
B1_2 = (187, 1000)
B1_3 = (188, 1000)
B1_4 = (188, 1000)

B2_1 = (187, 1000)
B2_2 = (187, 1000)
B2_3 = (188, 1000)
B2_4 = (188, 1000)
```

**Compute:**

```
A1[:, 0:187]   @ B1_1 = C1
A1[:, 187:374] @ B1_2 = C2
A1[:, 374:562] @ B1_3 = C3
A1[:, 562:750] @ B1_4 = C4

A2[:, 0:187]   @ B2_1 = C5
A2[:, 187:374] @ B2_2 = C6
A2[:, 374:562] @ B2_3 = C7
A2[:, 562:750] @ B2_4 = C8
```

**Final result:**

```
C = C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8  → shape: (500, 1000)
```

---

## 🧠 Core Pattern

* **A** is always split **by columns**
* **B** is always split **by rows**
* **Inner dimension chunks must match**
* Each node computes a **full output matrix**
* **Partial outputs are summed** to produce the final result

---

---

## Installing and Using `cluster_matrix_v1`

> ⚠️ Install script coming soon

### Build the C++ ZMQ backend

From your project root, run the following commands:

```bash
cd /your_project_dir/Open_Cluster_AI_Station_beta/cluster_matrix/ggml
```

Configure the build with CMake:

```bash
cmake -B build \
      -DGGML_VULKAN=ON \
      -DGGML_CUDA=OFF \
      -DGGML_METAL=OFF \
      -DGGML_OPENCL=OFF \
      -DGGML_BLAS=ON \
      -DGGML_BLAS_VENDOR=OpenBLAS
```

Build the ZMQ backend server:

```bash
cmake --build build --target matrix_zmq_server -j$(nproc)
```

Start the backend server:

```bash
./build/cluster_backend/cluster_backend_main/matrix_zmq_server
```

---

### Run the Python test script

In a **new terminal window**, run:

```bash
python cluster_matrix_test.py
```

---

### Notes

* The backend server **must be running** before executing the Python script.

---



