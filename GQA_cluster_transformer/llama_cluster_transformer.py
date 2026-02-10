# GQA_cluster_transformer.py
from __future__ import annotations
import os
import math
import numpy as np
from typing import Optional, List, Union, Callable
import sys
from gguf_parser import GGUFParser

# Ensure local package imports work when running as a script
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from transformer_model_handler import hugging_face_model_handler
except ModuleNotFoundError:
    from GQA_cluster_transformer.transformer_model_handler import hugging_face_model_handler
import re
from collections import defaultdict
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import queue
from typing import Optional, Tuple, Type
from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import List
import os

# 1. Fix seeds
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# 2. Force deterministic cuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


logger = getLogger()
# Add your project root imports
CLUSTER_MATRIX_DIR = os.path.join(PROJECT_ROOT, "cluster_matrix")
if CLUSTER_MATRIX_DIR not in sys.path:
    sys.path.insert(0, CLUSTER_MATRIX_DIR)

from cluster_matrix_v1 import cluster_matrix
from cluster_matrix_v1 import cluster_zmq
from cluster_matrix_v1 import check_combined_result_values

# Your GQA_cluster_transformer code here...
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def _build_causal_mask(
    q_len: int,
    kv_len: int,
    offset: int = 0,
    device=None,
    dtype: torch.dtype = torch.float32,
    additive: bool = True,
) -> torch.Tensor:
    """
    Build a causal mask in [q_len, kv_len] layout.
    offset shifts the query positions for decode (past_len).
    If additive=True, returns large-negative bias in the given dtype.
    """
    if device is None:
        device = "cpu"
    row_idx = torch.arange(q_len, device=device).unsqueeze(1) + int(offset)
    col_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    mask = col_idx > row_idx
    if not additive:
        return mask
    if dtype.is_floating_point:
        neg = torch.finfo(dtype).min
    else:
        neg = -1e9
    return mask.to(dtype) * neg

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

class llama_cluster_transformer:

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.kv_cache = {}

        self.cluster_zmq_object = self.model.cluster_zmq_object
        self.IP_list = self.model.cluster_zmq_object.node_IP_list
        self.percentages = self.model.percentages
        self.CPU_GPU_select_list = self.model.CPU_GPU_select_list
        self.backend_select_list = self.model.backend_select_list
        self.split_system = self.model.split_system
        self.split_dim = self.model.split_dim

        self.H = self.model.hidden_size
        self.dim = self.H  # hidden_size becomes dim
        self.n_layers = self.model.num_hidden_layers
        self.n_heads = self.model.num_attention_heads
        self.n_kv_heads = self.model.num_key_value_heads or self.n_heads
        if self.dim % self.n_heads != 0:
            raise ValueError(
                f"Hidden size {self.dim} not divisible by num_attention_heads={self.n_heads}"
            )
        self.head_dim = self.dim // self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"Invalid head config: num_attention_heads={self.n_heads} "
                f"not divisible by num_key_value_heads={self.n_kv_heads}"
            )
        self.group_size = self.n_heads // self.n_kv_heads
        self.scale = 1.0 / math.sqrt(float(self.head_dim))
        self.vocab_size = self.model.vocab_size

        self.embedding_matrix = getattr(self.model, "embedding_matrix", None)
        self.lm_head = getattr(self.model, "lm_head", None)
        tie_embeddings = bool(getattr(self.model, "tie_word_embeddings", False))
        if self.embedding_matrix is not None:
            if self.lm_head is None or tie_embeddings:
                self.lm_head = self.embedding_matrix
        self.final_norm_weight = getattr(self.model, "final_norm_weight", None)
        if self.final_norm_weight is None:
            self.final_norm_weight = getattr(self.model, "norm_weight", None)

        self.multiple_of = 256  # make SwiGLU hidden layer size multiple of large power of 2
        self.norm_eps = 1e-5

        self.max_batch_size = 32
        self.max_seq_len = int(getattr(self.model, "max_position_embeddings", 1024) or 1024)

        # RoPE cache
        self._rope_theta = getattr(self.model, "rope_theta", 10000.0) or 10000.0
        self._rope_rotary_dim = int(self.head_dim)
        self._rope_inv_freq = None
        self._rope_cos = None
        self._rope_sin = None
        self._freqs_cis = None
        self.debug_log_path = os.path.join(PROJECT_ROOT, "output_logs", "transformer_debug.log")
                
        self.system = "You are a helpful assistant."
        self.next_decode_token = ''
        # Initialize KV cache
        self._init_kv_cache()
        self._debug_attn = False
        # Other initialization...
        print(f"✅ Transformer initialized with dim={self.dim}, layers={self.n_layers}, heads={self.n_heads}")

    def _init_kv_cache(self):
        """Initialize key-value cache for attention."""
        self.cache_k = [
            torch.zeros(
                (self.max_batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim)
            )
            for _ in range(self.n_layers)
        ]
        self.cache_v = [
            torch.zeros(
                (self.max_batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim)
            )
            for _ in range(self.n_layers)
        ]
    
    def update_kv_cache(self, layer_id, k, v, bsz, seqlen, start_pos):
        if layer_id < 0 or layer_id >= len(self.cache_k):
            raise ValueError(f"Invalid layer_id for KV cache: {layer_id}")

        self.cache_k[layer_id] = self.cache_k[layer_id].to(k)
        self.cache_v[layer_id] = self.cache_v[layer_id].to(v)

        # k, v are [B, T, n_kv, hd]
        self.cache_k[layer_id][:bsz, start_pos:start_pos + seqlen, :, :] = k
        self.cache_v[layer_id][:bsz, start_pos:start_pos + seqlen, :, :] = v

        keys = self.cache_k[layer_id][:bsz, :start_pos + seqlen, :, :]
        values = self.cache_v[layer_id][:bsz, :start_pos + seqlen, :, :]

        return keys, values

    def get_kv_cache(self, layer_id, seq_len, start_pos):
        """Retrieve KV cache for a specific layer."""
        if layer_id < 0 or layer_id >= len(self.cache_k):
            return None, None
        keys = self.cache_k[layer_id][:, :start_pos + seq_len, :, :]
        values = self.cache_v[layer_id][:, :start_pos + seq_len, :, :]
        return keys, values
    
    def clear_kv_cache(self):
        """Clear the KV cache."""
        for i in range(len(self.cache_k)):
            self.cache_k[i].zero_()
            self.cache_v[i].zero_()

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
        y = (x_fp32 * rms).to(x.dtype)
        return y * weight.to(dtype=x.dtype, device=x.device)

    def _get_norm_weight(self, layer_id: int, kind: str) -> Optional[torch.Tensor]:
        """
        Fetch per-layer norm weights from the model handler lists.
        kind: "input" or "post".
        """
        if kind == "input":
            norm_list = getattr(self.model, "input_layernorm_list", [])
        elif kind == "post":
            norm_list = getattr(self.model, "post_attention_layernorm_list", [])
        else:
            return None

        if not norm_list or layer_id >= len(norm_list):
            return None
        entry = norm_list[layer_id]
        if isinstance(entry, (list, tuple)) and len(entry) > 0 and torch.is_tensor(entry[0]):
            return entry[0]
        return None

    def _ensure_rope_cache(self, max_pos: int) -> None:
        max_pos = int(max_pos)
        if max_pos <= 0:
            return
        if (
            self._rope_cos is not None
            and self._rope_sin is not None
            and self._rope_cos.shape[0] >= max_pos
        ):
            return

        rotary_dim = int(self._rope_rotary_dim)
        if rotary_dim % 2 != 0:
            raise ValueError(f"RoPE rotary_dim must be even, got {rotary_dim}")

        if self._rope_inv_freq is None:
            self._rope_inv_freq = 1.0 / (
                self._rope_theta
                ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / float(rotary_dim))
            )

        positions = torch.arange(max_pos, dtype=torch.float32).unsqueeze(1)
        freqs = positions * self._rope_inv_freq.unsqueeze(0)  # [max_pos, rotary_dim/2]
        self._rope_cos = freqs.cos()
        self._rope_sin = freqs.sin()

    def _ensure_freqs_cis(self, end: int) -> torch.Tensor:
        end = int(end)
        if end <= 0:
            raise ValueError("RoPE end position must be > 0")
        if self._freqs_cis is None or self._freqs_cis.shape[0] < end:
            self._freqs_cis = precompute_freqs_cis(
                self.head_dim,
                end,
                theta=self._rope_theta,
            )
        return self._freqs_cis

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor, seq_len: int, offset: int = 0):
        if seq_len == 0:
            return q, k

        head_dim = q.size(-1)
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")

        rotary_dim = int(self._rope_rotary_dim)
        if rotary_dim != head_dim:
            raise ValueError(f"RoPE rotary_dim ({rotary_dim}) must equal head_dim ({head_dim}) for this path")

        # RoPE on [B, T, n_heads, head_dim]
        if self._rope_inv_freq is None:
            self._rope_inv_freq = 1.0 / (
                self._rope_theta
                ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim))
            )
        positions = torch.arange(offset, offset + seq_len, device=q.device, dtype=torch.float32)
        freqs = torch.outer(positions, self._rope_inv_freq.to(device=q.device))  # [T, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, head_dim]
        cos = emb.cos()[None, :, None, :].to(dtype=q.dtype, device=q.device)
        sin = emb.sin()[None, :, None, :].to(dtype=q.dtype, device=q.device)

        def _rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_out = (q * cos) + (_rotate_half(q) * sin)
        k_out = (k * cos) + (_rotate_half(k) * sin)
        return q_out, k_out

    def _write_debug(self, lines: List[str], log_path: Optional[str] = None) -> None:
        if not log_path:
            log_path = self.debug_log_path
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    def _build_causal_mask(
        self,
        q_len: int,
        kv_len: int,
        offset: int = 0,
        device=None,
        dtype: torch.dtype = torch.float32,
        additive: bool = True,
    ) -> torch.Tensor:
        return _build_causal_mask(
            q_len=q_len,
            kv_len=kv_len,
            offset=offset,
            device=device,
            dtype=dtype,
            additive=additive,
        )

    def _compute_logits(self, hidden_out: torch.Tensor, last_token_only: bool = True) -> torch.Tensor:
        if hidden_out.dim() == 1:
            hidden_out = hidden_out.unsqueeze(0)

        # LLaMA order: final norm first, then select last token.
        if self.final_norm_weight is not None:
            hidden_out = self._rms_norm(hidden_out, self.final_norm_weight, eps=1e-5)

        if hidden_out.dim() == 3 and last_token_only:
            hidden_out = hidden_out[:, -1, :]
        elif hidden_out.dim() == 3 and hidden_out.shape[1] == 1:
            hidden_out = hidden_out.squeeze(1)

        return hidden_out.float() @ self.lm_head.T.float()

    def _should_stop(self, tokens, prompt_tokens, stop_ids, stop_words):
        if stop_ids is not None:
            do_stop = [False for _ in range(len(tokens))]
            for i, (t, p) in enumerate(zip(tokens, prompt_tokens)):
                g = t[len(p):].tolist()
                for stop_id in stop_ids:
                    if stop_id in g:
                        do_stop[i] = True

            if all(do_stop):
                return True

        if stop_words is not None:
            do_stop = [False for _ in range(len(tokens))]
            for i, (t, p) in enumerate(zip(tokens, prompt_tokens)):
                t = t.clone()
                g = t[len(p):]
                g[g == self.tokenizer.pad_id] = self.tokenizer.eos_id
                g = g.tolist()
                d = self.tokenizer.decode(g)
                for stop_word in stop_words:
                    if stop_word in d:
                        do_stop[i] = True

            if all(do_stop):
                return True

        return False

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop_ids: List[int] = None,
        stop_words: List[str] = None,
        repetition_penalty: float = 1.0,
        print_reply: bool = False,
        on_token: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]

        if self.embedding_matrix is None:
            raise RuntimeError("Embedding matrix not loaded in model handler.")
        if self.lm_head is None:
            raise RuntimeError("LM head not available (lm_head missing and tie_word_embeddings is False).")

        # Reset cache once per generation call (prevents cross-prompt leakage).
        self.clear_kv_cache()

        def format_prompt(p: str) -> str:
            model_name = str(getattr(self.model, "model_name", "")).lower()
            model_path = str(getattr(self.model, "model_path", "")).lower()
            if "chat" in model_name or "chat" in model_path:
                eos = "</s>"
                return f"<|system|>\n{self.system}{eos}\n<|user|>\n{p}{eos}\n<|assistant|>"
            return p

        def format_messages(msgs: List[tuple]) -> str:
            eos = "</s>"
            parts: List[str] = []
            has_system = any(role == "system" for role, _ in msgs)
            if not has_system:
                parts.append(f"<|system|>\nYou are a helpful assistant.{eos}")
            for role, text in msgs:
                role = role.strip().lower()
                if role not in ("system", "user", "assistant"):
                    role = "user"
                parts.append(f"<|{role}|>\n{text}{eos}")
            parts.append("<|assistant|>")
            return "\n".join(parts)

        stop_ids = stop_ids or []
        stop_sequences: List[List[int]] = []
        if stop_words:
            for w in stop_words:
                try:
                    ids = self.tokenizer.encode(w, bos=False, eos=False)
                except Exception:
                    ids = []
                if ids:
                    stop_sequences.append(ids)

        # Support a single conversation passed as a list of (role, text)
        if (
            isinstance(prompts, list)
            and prompts
            and isinstance(prompts[0], (list, tuple))
            and len(prompts[0]) == 2
        ):
            prompt_texts = [format_messages(prompts)]  # type: ignore[arg-type]
        else:
            prompt_texts = [format_prompt(x) for x in prompts]  # type: ignore[list-item]

        bsz = len(prompt_texts)
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompt_texts]
        num_input_tokens = [len(t) for t in prompt_tokens]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id

        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            hidden_out = self.run_QKV_mlp_cluster(
                tokens[:, prev_pos:cur_pos],
                start_pos=prev_pos,
                prefill=(prev_pos == 0 and cur_pos > prev_pos),
                split_dim=self.split_dim,
                verbos=False,
            )
            logits = self._compute_logits(hidden_out, last_token_only=True)

            if repetition_penalty != 1.0:
                logits_new = logits.clone()
                batch_size = len(tokens)
                for i in range(batch_size):
                    for token in set(tokens[i].tolist()):
                        if logits[i, token] < 0:
                            logits_new[i, token] = logits[i, token] * repetition_penalty
                        else:
                            logits_new[i, token] = logits[i, token] / repetition_penalty
                logits = logits_new

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p) if top_p < 1.0 else torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1)

            next_token = next_token.reshape(-1)
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

            if on_token is not None:
                for i in range(bsz):
                    if not input_text_mask[i, cur_pos]:
                        tid = int(next_token[i].item())
                        try:
                            ttext = self.tokenizer.decode([tid])
                        except Exception:
                            ttext = ""
                        if ttext:
                            on_token(i, tid, ttext)

            if self._should_stop(tokens, prompt_tokens, stop_ids, stop_words):
                break

        tokens[tokens == self.tokenizer.pad_id] = self.tokenizer.eos_id
        outputs: List[str] = []
        for i, t in enumerate(tokens.tolist()):
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            outputs.append(self.tokenizer.decode(t))

        if print_reply:
            for i, prompt in enumerate(prompts):
                reply_tokens = outputs[i][len(prompt):]
                print("\n--- Model Reply ---", flush=True)
                print(reply_tokens, flush=True)
                print("-------------------\n", flush=True)

        return outputs

    def forward(self, tokens: torch.Tensor, start_pos: int = 0, prefill: Optional[bool] = None):
        """
        LLaMA-style forward: takes token IDs [B, T] and start_pos.
        Returns final hidden states [T, H] (batch=1 supported).
        """
        return self.run_QKV_mlp_cluster(
            tokens,
            start_pos=int(start_pos),
            prefill=prefill,
        )

    def run_QKV_mlp_torch(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
        prefill: Optional[bool] = None,
        split_dim: Optional[int] = None,
        verbos: bool = False,
    ) -> torch.Tensor:
        """
        PyTorch-only forward for layer 0 of the LLaMA-style model.
        Returns hidden states [B, T, H].
        """

        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.dtype != torch.long:
            raise TypeError(f"Expected token_ids to be torch.long, got {token_ids.dtype}")

        bsz, seqlen = token_ids.shape
        if bsz != 1:
            raise ValueError("run_QKV_mlp currently supports batch size 1 only.")

        if prefill is None:
            prefill = bool(start_pos == 0 and seqlen > 1)

        if self.embedding_matrix is None:
            raise RuntimeError("Embedding matrix not loaded in model handler.")

        # ---- EMBEDDING LOOKUP ----
        embeddings = self.embedding_matrix[token_ids].to(torch.float32)  # [B, T, H]
        T = seqlen
        x = embeddings  # [B, T, H]

        debug_dir = os.path.join(PROJECT_ROOT, "cluster_matrix", "test_model_matrices")

        # ---- LOAD WEIGHTS ----
        def _load_weight(fname: str, required: bool = True) -> Optional[torch.Tensor]:
            path = os.path.join(debug_dir, fname)
            if not os.path.exists(path):
                if required:
                    raise FileNotFoundError(f"Missing weight file: {path}")
                return None
            return torch.load(path).to(torch.float32)

        Wq = _load_weight("model.layers.0.self_attn.q_proj.weight")
        Wk = _load_weight("model.layers.0.self_attn.k_proj.weight")
        Wv = _load_weight("model.layers.0.self_attn.v_proj.weight")
        Wo = _load_weight("model.layers.0.self_attn.o_proj.weight")

        up_w = _load_weight("model.layers.0.mlp.up_proj.weight")
        down_w = _load_weight("model.layers.0.mlp.down_proj.weight")
        gate_w = _load_weight("model.layers.0.mlp.gate_proj.weight")

        ln_weight = self._get_norm_weight(layer_id=0, kind="input")
        post_ln_weight = self._get_norm_weight(layer_id=0, kind="post")
        if ln_weight is None:
            ln_weight = _load_weight("model.layers.0.input_layernorm.weight", required=False)
        if post_ln_weight is None:
            post_ln_weight = _load_weight("model.layers.0.post_attention_layernorm.weight", required=False)

        # ---- RMSNorm (pre-attn) ----
        eps = 1e-5
        if ln_weight is None:
            ln_weight = torch.ones(self.H, dtype=torch.float32)
        x_norm = self._rms_norm(x, ln_weight, eps=eps)

        # ---- Q/K/V projections ----
        q = x_norm @ Wq.T  # [B, T, H]
        k = x_norm @ Wk.T  # [B, T, n_kv_heads*head_dim]
        v = x_norm @ Wv.T  # [B, T, n_kv_heads*head_dim]

        kv_dim = k.size(-1)
        if kv_dim % self.head_dim != 0:
            raise ValueError(f"k_proj output dim {kv_dim} not divisible by head_dim {self.head_dim}")
        n_kv_heads = kv_dim // self.head_dim
        if self.n_heads % n_kv_heads != 0:
            raise ValueError(
                f"Invalid head config from weights: n_heads={self.n_heads} not divisible by n_kv_heads={n_kv_heads}"
            )
        group_size = self.n_heads // n_kv_heads

        q = q.view(bsz, T, self.n_heads, self.head_dim)  # [B, T, n_heads, hd]
        k = k.view(bsz, T, n_kv_heads, self.head_dim)    # [B, T, n_kv, hd]
        v = v.view(bsz, T, n_kv_heads, self.head_dim)    # [B, T, n_kv, hd]

        q, k = self._apply_rotary(q, k, seq_len=T, offset=start_pos)

        # ---- KV cache ----
        layer_id = 0
        k_cache, v_cache = self.update_kv_cache(layer_id, k, v, bsz, T, start_pos)

        # ---- Attention ----
        q = q.transpose(1, 2)  # [B, n_heads, T, hd]
        k_cache = k_cache.transpose(1, 2)  # [B, n_kv, T, hd]
        v_cache = v_cache.transpose(1, 2)  # [B, n_kv, T, hd]

        if n_kv_heads != self.n_heads:
            k_rep = k_cache.repeat_interleave(group_size, dim=1)
            v_rep = v_cache.repeat_interleave(group_size, dim=1)
        else:
            k_rep, v_rep = k_cache, v_cache

        q_len = q.size(2)
        kv_len = k_rep.size(2)
        use_sdpa = hasattr(F, "scaled_dot_product_attention")
        use_sdpa = hasattr(F, "scaled_dot_product_attention")

        if use_sdpa:
            # Match HF SDPA path when available.
            attn_out = F.scaled_dot_product_attention(
                q,
                k_rep,
                v_rep,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
        else:
            attn_scores = torch.matmul(q, k_rep.transpose(-2, -1)) * self.scale  # [B, n_heads, q_len, kv_len]
            mask = self._build_causal_mask(
                q_len=q_len,
                kv_len=kv_len,
                offset=start_pos,
                device=attn_scores.device,
                dtype=attn_scores.dtype,
                additive=True,
            )
            attn_scores = attn_scores + mask
            attn_probs = torch.softmax(attn_scores.float(), dim=-1).type_as(q)
            attn_out = torch.matmul(attn_probs, v_rep)  # [B, n_heads, q_len, hd]
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.H)
        attn_out = attn_out @ Wo.T  # [B, q_len, H]

        x = x + attn_out

        # ---- RMSNorm (post-attn) ----
        if post_ln_weight is None:
            post_ln_weight = torch.ones(self.H, dtype=torch.float32)
        x_norm_2 = self._rms_norm(x, post_ln_weight, eps=eps)

        # ---- MLP ----
        hidden_up = x_norm_2 @ up_w.T  # [B, T, intermediate]
        hidden_gate = x_norm_2 @ gate_w.T
        hidden_glu = F.silu(hidden_gate) * hidden_up
        hidden_down = hidden_glu @ down_w.T  # [B, T, H]

        # ---- Residual ----
        hidden_out = x + hidden_down  # [B, T, H]
        return hidden_out

    def run_QKV_mlp_cluster(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
        prefill: Optional[bool] = None,
        split_dim: Optional[int] = None,
        verbos: bool = False,
    ) -> torch.Tensor:
        """
        PyTorch-only forward for layer 0 of the LLaMA-style model.
        Returns hidden states [B, T, H].
        """

        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.dtype != torch.long:
            raise TypeError(f"Expected token_ids to be torch.long, got {token_ids.dtype}")

        bsz, seqlen = token_ids.shape
        if bsz != 1:
            raise ValueError("run_QKV_mlp currently supports batch size 1 only.")

        if prefill is None:
            prefill = bool(start_pos == 0 and seqlen > 1)

        if self.embedding_matrix is None:
            raise RuntimeError("Embedding matrix not loaded in model handler.")

        # ---- EMBEDDING LOOKUP ----
        embeddings = self.embedding_matrix[token_ids].to(torch.float32)  # [B, T, H]
        T = seqlen
        x = embeddings  # [B, T, H]

        for layer_index in range(self.n_layers):
            ln_weight = self._get_norm_weight(layer_id=layer_index, kind="input")
            post_ln_weight = self._get_norm_weight(layer_id=layer_index, kind="post")

            # ---- RMSNorm (pre-attn) ----
            eps = 1e-5
            if ln_weight is None:
                ln_weight = torch.ones(self.H, dtype=torch.float32)
            x_norm = self._rms_norm(x, ln_weight, eps=eps)

            x_norm_cluster = cluster_matrix(
                x_norm,
                cluster_zmq_object=self.model.cluster_zmq_object,
                CPU_GPU_select_list=self.model.CPU_GPU_select_list,
                node_percentages=self.model.percentages,
                back_end_select_list=self.model.backend_select_list,
                split_matrix=False,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name='x_norm'
            )

            _, Q_weight, _, _ = self.model.q_proj_list[layer_index]
            _, K_weight, _, _ = self.model.k_proj_list[layer_index]
            _, V_weight, _, _ = self.model.v_proj_list[layer_index]
            _, O_weight, _, _ = self.model.o_proj_list[layer_index]
            _, hidden_up_weight, _, _ = self.model.mlp_up_list[layer_index]
            _, hidden_down_weight, _, _ = self.model.mlp_down_list[layer_index]
            _, hidden_gate_weight, _, _ = self.model.mlp_gate_list[layer_index]

            q_cluster = x_norm_cluster.cluster_shard_operation(Q_weight, False, False, True)
            k_cluster = x_norm_cluster.cluster_shard_operation(K_weight, False, False, True)
            v_cluster = x_norm_cluster.cluster_shard_operation(V_weight, False, False, True)

            q = q_cluster.unsqueeze(0) if q_cluster.dim() == 2 else q_cluster
            k = k_cluster.unsqueeze(0) if k_cluster.dim() == 2 else k_cluster
            v = v_cluster.unsqueeze(0) if v_cluster.dim() == 2 else v_cluster

            kv_dim = k.size(-1)
            if kv_dim % self.head_dim != 0:
                raise ValueError(f"k_proj output dim {kv_dim} not divisible by head_dim {self.head_dim}")
            n_kv_heads = kv_dim // self.head_dim
            if self.n_heads % n_kv_heads != 0:
                raise ValueError(
                    f"Invalid head config from weights: n_heads={self.n_heads} not divisible by n_kv_heads={n_kv_heads}"
                )
            group_size = self.n_heads // n_kv_heads

            q = q.view(bsz, T, self.n_heads, self.head_dim)  # [B, T, n_heads, hd]
            k = k.view(bsz, T, n_kv_heads, self.head_dim)    # [B, T, n_kv, hd]
            v = v.view(bsz, T, n_kv_heads, self.head_dim)    # [B, T, n_kv, hd]

            q, k = self._apply_rotary(q, k, seq_len=T, offset=start_pos)

            k_cache, v_cache = self.update_kv_cache(layer_index, k, v, bsz, T, start_pos)
            # ---- Attention ----
            q = q.transpose(1, 2)  # [B, n_heads, T, hd]
            k_cache = k_cache.transpose(1, 2)  # [B, n_kv, T, hd]
            v_cache = v_cache.transpose(1, 2)  # [B, n_kv, T, hd]

            if n_kv_heads != self.n_heads:
                k_rep = k_cache.repeat_interleave(group_size, dim=1)
                v_rep = v_cache.repeat_interleave(group_size, dim=1)
            else:
                k_rep, v_rep = k_cache, v_cache

            q_len = q.size(2)
            kv_len = k_rep.size(2)

            Q = cluster_matrix(
                q,
                cluster_zmq_object=self.model.cluster_zmq_object,
                CPU_GPU_select_list=self.model.CPU_GPU_select_list,
                node_percentages=self.model.percentages,
                back_end_select_list=self.model.backend_select_list,
                split_matrix=True,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name='Q_cluster'
            )

            K = cluster_matrix(
                k_rep,
                cluster_zmq_object=self.model.cluster_zmq_object,
                CPU_GPU_select_list=self.model.CPU_GPU_select_list,
                node_percentages=self.model.percentages,
                back_end_select_list=self.model.backend_select_list,
                split_matrix=True,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name='K_cluster'
            )

            V = cluster_matrix(
                v_rep,
                cluster_zmq_object=self.model.cluster_zmq_object,
                CPU_GPU_select_list=self.model.CPU_GPU_select_list,
                node_percentages=self.model.percentages,
                back_end_select_list=self.model.backend_select_list,
                split_matrix=True,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name='V_cluster'
            )

            mask = self._build_causal_mask(
                q_len=q_len,
                kv_len=kv_len,
                offset=start_pos,
                device=q.device,
                dtype=q.dtype,
                additive=True,
            )

            attn_out_cluster_2d = Q.cluster_flash_attn(
                cluster_matrixK=K,
                cluster_matrixV=V,
                TransposeQ=False,
                TransposeK=False,
                TransposeV=False,
                send_back_result=True,
                operation="flash_attn_ext",
                extra_param_value=self.scale,
                mask=mask
            )

            attn_out_cluster_2d = attn_out_cluster_2d.contiguous()

            attn_out_cluster_4d = attn_out_cluster_2d.view(
                bsz, self.n_heads, q_len, self.head_dim
            )

            attn_out_cluster_3d = (
                attn_out_cluster_4d.transpose(1, 2)
                .contiguous()
                .view(bsz, q_len, self.H)
            )

            attn_out_cluster = cluster_matrix(
                attn_out_cluster_3d,
                cluster_zmq_object=self.model.cluster_zmq_object,
                CPU_GPU_select_list=self.model.CPU_GPU_select_list,
                node_percentages=self.model.percentages,
                back_end_select_list=self.model.backend_select_list,
                split_matrix=False,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name='attn_out_cluster'
            )

            attn_out_cluster = attn_out_cluster.cluster_shard_operation(O_weight, False, False, True)
            if attn_out_cluster.dim() == 2:
                attn_out_cluster = attn_out_cluster.unsqueeze(0)

            x = x + attn_out_cluster

            # ---- RMSNorm (post-attn) ----
            if post_ln_weight is None:
                post_ln_weight = torch.ones(self.H, dtype=torch.float32)
            x_norm_2 = self._rms_norm(x, post_ln_weight, eps=eps)
        
            x_cluster = cluster_matrix(
                x_norm_2.squeeze(0),
                cluster_zmq_object=self.model.cluster_zmq_object,
                CPU_GPU_select_list=self.model.CPU_GPU_select_list,
                node_percentages=self.model.percentages,
                back_end_select_list=self.model.backend_select_list,
                split_matrix=False,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name='X_cluster'
            )

            # ---- MLP ----
            hidden_up_cluster = x_cluster.cluster_shard_operation(hidden_up_weight, False, False, True)
            hidden_gate_cluster = x_cluster.cluster_shard_operation(hidden_gate_weight, False, False, True)

            hidden_glu = F.silu(hidden_gate_cluster) * hidden_up_cluster

            hidden_glu_cluster = cluster_matrix(
                hidden_glu,
                cluster_zmq_object=self.model.cluster_zmq_object,
                CPU_GPU_select_list=self.model.CPU_GPU_select_list,
                node_percentages=self.model.percentages,
                back_end_select_list=self.model.backend_select_list,
                split_matrix=False,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name='hidden_glu_cluster'
            )

            hidden_down_cluster = hidden_glu_cluster.cluster_shard_operation(hidden_down_weight, False, False, True)
            if hidden_down_cluster.dim() == 2:
                hidden_down_cluster = hidden_down_cluster.unsqueeze(0)

            # ---- Residual ----
            x = x + hidden_down_cluster  # [B, T, H]

        return x

        
def main_test():
    # ----------------- CLUSTER CONFIG -----------------
    IP_list = ['192.168.2.100','192.168.2.100','192.168.2.101']   
    #percentages = [0.5, 0.25, 0.25]
    #percentages = [0.5, 0.3125, 0.1875]
    #percentages = [0.5625, 0.3125, 0.125]
    #percentages = [0.625, 0.25, 0.125]
    percentages = [0.5, 0.45, 0.05]
    CPU_GPU_select_list = [True, True, True]  
    backend_select_list = ['llama', 'llama', 'llama'] 

    cluster_zmq_obj = cluster_zmq(IP_list)


   # ----------------- TESTING -----------------
    # Force FP32 weights for closer numerical parity with vanilla LLaMA test.
    os.environ["CLUSTER_FORCE_FP32"] = "1"

    model = hugging_face_model_handler(
        model_path="/home/rino/Desktop/Open_Cluster_AI_Station_beta/llm_models/TinyLlama-1.1B-Chat-v1.0",
        cluster_zmq_object=cluster_zmq_obj,
        percentages=percentages,
        CPU_GPU_select_list=CPU_GPU_select_list,
        backend_select_list=backend_select_list,
    ) 
    
    print('\n')
    print('test4: /home/rino/Desktop/Open_Cluster_AI_Station_beta/llm_models/TinyLlama-1.1B-Chat-v1.0')
    print('\n')
    #cache_mode = "save" if os.environ.get("CLUSTER_FORCE_FP32") in ("1", "true", "True") else "load"
    model.cache_model_tensors(saveOrload='load')

    Tokenizer_test = Tokenizer('/home/rino/Desktop/Open_Cluster_AI_Station_beta/llm_models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model')

    llama_test = llama_cluster_transformer(Tokenizer_test, model)
    
    # test1: single forward pass (prefill) using this file's forward()
    
    prompt = "hello! are you working"
    tokens = Tokenizer_test.encode(prompt, bos=True, eos=False)
    tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    with torch.inference_mode():
        print('tokens_tensor.shape: ', tokens_tensor.shape, flush=True)
        print('tokens_tensor.data: ', tokens_tensor, flush=True)
        out = llama_test.forward(tokens_tensor, start_pos=0, prefill=True)

        # LLaMA-style logits from last token
        logits = llama_test._compute_logits(out, last_token_only=True)

    print("test1 logits shape:", tuple(logits.shape))
    print("test1 logits sample:", logits[0, :10].detach().cpu())
    
def main():
    # ----------------- CLUSTER CONFIG -----------------
    IP_list = ['192.168.2.100','192.168.2.100','192.168.2.101']   
    #percentages = [0.5, 0.25, 0.25]
    #percentages = [0.5, 0.3125, 0.1875]
    #percentages = [0.5625, 0.3125, 0.125]
    #percentages = [0.625, 0.25, 0.125]
    percentages = [0.5, 0.45, 0.05]
    CPU_GPU_select_list = [True, True, True]  
    backend_select_list = ['llama', 'llama', 'llama'] 

    cluster_zmq_obj = cluster_zmq(IP_list)


   # ----------------- TESTING -----------------
    # Force FP32 weights for closer numerical parity with vanilla LLaMA test.
    os.environ["CLUSTER_FORCE_FP32"] = "1"

    model = hugging_face_model_handler(
        model_path="/home/rino/Desktop/Open_Cluster_AI_Station_beta/llm_models/TinyLlama-1.1B-Chat-v1.0",
        cluster_zmq_object=cluster_zmq_obj,
        percentages=percentages,
        CPU_GPU_select_list=CPU_GPU_select_list,
        backend_select_list=backend_select_list,
    ) 
    
    print('\n')
    print('test4: /home/rino/Desktop/Open_Cluster_AI_Station_beta/llm_models/TinyLlama-1.1B-Chat-v1.0')
    print('\n')
    #cache_mode = "save" if os.environ.get("CLUSTER_FORCE_FP32") in ("1", "true", "True") else "load"
    model.cache_model_tensors(saveOrload='load')

    Tokenizer_test = Tokenizer('/home/rino/Desktop/Open_Cluster_AI_Station_beta/llm_models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model')

    llama_test = llama_cluster_transformer(Tokenizer_test, model)
    
    def _clean_reply(text: str) -> str:
        for tok in ("<|assistant|>", "<|user|>", "<|system|>", "</s>"):
            text = text.replace(tok, "")
        return text.strip()

    print("Type 'exit' or 'quit' to stop.")
    history: List[tuple] = []
    max_history_turns = 6

    while True:
        prompt = input("you: ").strip()
        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q", "bye"):
            break

        history.append(("user", prompt))
        if len(history) > max_history_turns * 2:
            history = history[-max_history_turns * 2:]

        with torch.inference_mode():
            outputs = llama_test.generate(
                history,
                max_gen_len=128,
                temperature=0.8,
                top_p=0.95,
                print_reply=False,
            )

        reply = _clean_reply(outputs[0]) if outputs else ""
        history.append(("assistant", reply))
        print(f"assistant: {reply}")


#if __name__ == "__main__":
#    main()
