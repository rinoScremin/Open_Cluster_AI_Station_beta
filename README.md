---

# GQA Cluster Transformer

Distributed **LLaMA-style Transformer** with **Grouped Query Attention (GQA)** and **KV-cache**, powered by a custom ZeroMQ-based compute cluster backend.

This project enables running transformer attention and MLP layers across multiple nodes (CPU/GPU) using a matrix-splitting cluster system.

---

## 🚀 Features

* ✅ LLaMA-style architecture
* ✅ Grouped Query Attention (GQA)
* ✅ Rotary Positional Embeddings (RoPE)
* ✅ Sliding-window KV cache
* ✅ RMSNorm (LLaMA style)
* ✅ Top-p (nucleus) sampling
* ✅ Repetition penalty
* ✅ Streaming token callback support
* ✅ Local HuggingFace or SentencePiece tokenizer support
* ✅ Distributed matrix compute via `cluster_matrix_v1`
* ✅ Deterministic cuDNN + fixed seeds

---

## 🧠 Architecture Overview

The core model class:

```
llama_cluster_transformer
```

It:

* Uses a model handler (`hugging_face_model_handler`)
* Distributes Q/K/V/O projections across cluster nodes
* Maintains KV cache per layer
* Applies RoPE during attention
* Runs attention + MLP blocks
* Produces logits using tied embeddings or LM head

---

## 📦 Requirements

### Python

* Python 3.9+
* PyTorch
* Transformers
* SentencePiece
* NumPy

### Internal Dependencies

This project requires:

* `cluster_matrix_v1`
* `cluster_zmq`
* `gguf_parser`
* `transformer_model_handler`

These must exist in your project structure.

---

## 📁 Project Structure (Expected)

```
project_root/
│
├── GQA_cluster_transformer.py
├── transformer_model_handler.py
├── gguf_parser.py
├── cluster_matrix/
│   └── cluster_matrix_v1.py
│
└── llm_models/
```

---

## 🔧 Environment Variables

You can control KV cache behavior:

```
OPEN_CLUSTER_KV_WINDOW=2048
OPEN_CLUSTER_KV_CACHE_DTYPE=fp16  # options: fp16, bf16, fp32
```

---

## 🧩 Tokenizer Support

The `Tokenizer` class supports:

### 1️⃣ SentencePiece (`.model`)

Uses:

* `SentencePieceProcessor`

### 2️⃣ HuggingFace (`tokenizer.json` or model directory)

Uses:

* `transformers.AutoTokenizer`

Special tokens are normalized automatically.

---

## 🏗 Model Initialization

Example:

```python
from GQA_cluster_transformer import llama_cluster_transformer, Tokenizer
from transformer_model_handler import hugging_face_model_handler

tokenizer = Tokenizer("path_to_tokenizer_or_model")

model_handler = hugging_face_model_handler(
    model_path="path_to_model",
    ...
)

model = llama_cluster_transformer(tokenizer, model_handler)
```

---

## 🧪 Text Generation

```python
output = model.generate(
    prompts="Explain GQA in simple terms.",
    max_gen_len=200,
    temperature=0.8,
    top_p=0.95
)

print(output[0])
```

---

## ⚡ Cluster Execution

The distributed path runs through:

```
run_QKV_mlp_cluster()
```

This method:

1. RMSNorm
2. Distributes Q/K/V projections across cluster
3. Applies RoPE
4. Updates KV cache
5. Runs attention
6. Applies post-attention RMSNorm
7. Runs SwiGLU MLP
8. Residual connections

Matrix splitting is controlled by:

* `CPU_GPU_select_list`
* `percentages`
* `backend_select_list`
* `split_dim`

All are supplied by the model handler.

---

## 🧠 KV Cache

* Sliding window implementation
* Configurable window size
* Supports fp16 / bf16 / fp32
* Wrap-around circular buffer logic

Reset automatically per `generate()` call.

Manual clear:

```python
model.clear_kv_cache()
```

---

## 🧮 Attention Details

* Supports `torch.nn.functional.scaled_dot_product_attention` if available
* Fallback manual masked softmax
* Uses LLaMA-style:

  * RoPE
  * RMSNorm (pre + post attention)
  * SwiGLU MLP

---

## 🛑 Stopping Conditions

Supports:

* `stop_ids`
* `stop_words`
* Token ID stop sequences
* Default EOS handling
* Special token filtering

---

## 📡 Streaming Tokens

You can stream generated tokens:

```python
def on_token(batch_idx, token_id, token_text):
    print(token_text, end="", flush=True)

model.generate(
    prompts="Tell me a story",
    max_gen_len=200,
    on_token=on_token
)
```

---

## 🔒 Determinism

The script enforces:

```python
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

For reproducible runs.

---

## ⚠ Current Limitations

* Batch size >1 not supported in `run_QKV_mlp`
* Requires proper cluster backend setup
* Assumes LLaMA-style weight layout
* Memory usage scales with KV window × layers

---

## 🎯 Intended Use

This is designed for:

* Experimental distributed transformer inference
* Custom LLM cluster systems
* Performance experimentation
* Matrix-splitting research
* GQA architecture testing

---

## 🛠 Debugging

Debug log path:

```
output_logs/transformer_debug.log
```

Cluster test matrices (optional):

```
cluster_matrix/test_model_matrices/
```
