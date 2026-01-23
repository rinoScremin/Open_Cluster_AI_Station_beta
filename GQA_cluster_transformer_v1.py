from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import math
import os
import re
import sys
import time
from typing import Iterator, Optional

import torch
from transformers import AutoConfig, AutoTokenizer


def _repo_root() -> str:
    return os.path.abspath(os.path.dirname(__file__))


def _import_cluster_matrix_v1():
    # `cluster_matrix_v1.py` lives in `<repo_root>/cluster_matrix/` (not a Python package).
    cluster_dir = os.path.join(_repo_root(), "cluster_matrix")
    if cluster_dir not in sys.path:
        sys.path.insert(0, cluster_dir)
    import cluster_matrix_v1  # type: ignore

    return cluster_matrix_v1


_cm = _import_cluster_matrix_v1()
cluster_matrix = _cm.cluster_matrix
cluster_zmq = _cm.cluster_zmq


_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_name(s: str, *, max_len: int = 48) -> str:
    s = _SAFE_NAME_RE.sub("_", str(s)).strip("_")
    if not s:
        s = "model"
    if len(s) > int(max_len):
        s = s[: int(max_len)]
    return s

class cluster_llm_transformer:
    def __init__(
        self,
        model_path,
        IP_list,
        percentages,
        CPU_GPU_select_list,
        backend_select_list,
        *,
        model_matrices_dir: Optional[str] = None,
        local_files_only: Optional[bool] = None,
        weight_cache_mode: str = "load",
    ):
        # --------------------------------------------------
        # Paths
        # --------------------------------------------------
        self.local_project_dir = os.environ.get("LOCAL_PROJECT_DIR", os.path.join(_repo_root(), "cluster_matrix"))
        if self.local_project_dir and not str(self.local_project_dir).endswith(os.sep):
            self.local_project_dir = str(self.local_project_dir) + os.sep

        self.model_path = str(model_path)

        if model_matrices_dir is not None:
            matrices_dir = str(model_matrices_dir)
        elif os.path.isdir(self.model_path):
            matrices_dir = os.path.join(self.model_path, "model_matrices")
        else:
            # Backward-compatible default: allow running from any CWD with a local `model_matrices/` folder.
            matrices_dir = os.path.join(os.getcwd(), "model_matrices")

        if matrices_dir and not matrices_dir.endswith(os.sep):
            matrices_dir = matrices_dir + os.sep
        self.model_matrix_fold_dir = matrices_dir
        os.makedirs(self.model_matrix_fold_dir, exist_ok=True)

        # --------------------------------------------------
        # LOAD METADATA ONLY (NO MODEL WEIGHTS)
        # --------------------------------------------------
        if local_files_only is None:
            local_files_only = bool(os.path.isdir(self.model_path))
        self.config = AutoConfig.from_pretrained(self.model_path, local_files_only=bool(local_files_only))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=bool(local_files_only))
        self.num_layers = self.config.num_hidden_layers


        # --------------------------------------------------
        # ATTENTION / MODEL GEOMETRY
        # --------------------------------------------------
        self.hidden_size = self.config.hidden_size

        # You already have this logic
        self.attention_type, self.num_q_heads, self.num_kv_heads = self.detect_attention_type()

        self.head_dim = self.hidden_size // self.num_q_heads
        self.kv_dim = self.num_kv_heads * self.head_dim

        self.Model_Attention = self.attention_type
        self.attention_Heads = [self.num_q_heads, self.num_kv_heads]
        self.Hidden_size = self.hidden_size

        # Note: shard cache isolation is handled by the chat wrapper via per-model disk folders.
        # Keep this for debug/logging only (not used for shard filenames).
        tag_seed = (
            f"{os.path.abspath(self.model_path) if os.path.exists(self.model_path) else self.model_path}"
            f"|{getattr(self.config, 'model_type', '')}"
            f"|hs={self.hidden_size}|L={self.num_layers}|hq={self.num_q_heads}|hkv={self.num_kv_heads}"
        )
        self._model_tag = hashlib.sha1(tag_seed.encode("utf-8")).hexdigest()[:8]

        wmode = str(weight_cache_mode).strip().lower()
        if wmode not in ("save", "load"):
            raise ValueError(f"weight_cache_mode must be 'save' or 'load', got {weight_cache_mode!r}")
        self.weight_cache_mode = wmode

        # --------------------------------------------------
        # RUNTIME STATE
        # --------------------------------------------------
        self.tokens = None
        self.seq_len = 0

        # --------------------------------------------------
        # CLUSTER CONFIG
        # --------------------------------------------------
        self.IP_list = IP_list
        self.cluster_zmq_object = cluster_zmq(self.IP_list)
        self.percentages = percentages
        self.CPU_GPU_select_list = CPU_GPU_select_list
        self.backend_select_list = backend_select_list

        # --------------------------------------------------
        # INFERENCE OPTIMIZATION FLAGS / CACHES
        # --------------------------------------------------
        # NOTE: These optimizations are inference-only and preserve numerics/outputs.
        self.verbose = False  # Set True for per-layer debug prints.
        self._inv_sqrt_head_dim = 1.0 / math.sqrt(float(self.head_dim))
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"Invalid head config: num_q_heads={self.num_q_heads} not divisible by num_kv_heads={self.num_kv_heads}"
            )
        self._kv_group_size = self.num_q_heads // self.num_kv_heads

        # Cache: on-demand loaded small tensors (layernorms, etc.)
        self._tensor_cache: dict[str, torch.Tensor] = {}
        # Cache: precomputed o_proj matmul matrices (per layer).
        self._attn_o_proj_mat: list[torch.Tensor | None] = [None] * int(self.num_layers)

        # Cache: cluster_matrix objects for weight shards (per matrix name).
        self._cluster_weight_cache: dict[str, cluster_matrix] = {}

        # KV cache (compact): [layers, kv_heads, capacity, head_dim], float32 for stable numerics.
        self._kv_cache_k: torch.Tensor | None = None
        self._kv_cache_v: torch.Tensor | None = None
        self._kv_cache_capacity: int = 0

        # RoPE cache: precompute cos/sin for positions up to current generation max.
        self._rope_theta = float(getattr(self.config, "rope_theta", 10000.0))
        self._rope_rotary_dim = int(self.head_dim)
        self._rope_inv_freq: torch.Tensor | None = None
        self._rope_cos: torch.Tensor | None = None  # [max_pos, rotary_dim]
        self._rope_sin: torch.Tensor | None = None  # [max_pos, rotary_dim]

        # Precompute common per-layer file paths once (cuts Python string overhead in decode loop).
        self._layer_paths: list[dict[str, str]] = []
        for layer_idx in range(int(self.num_layers)):
            self._layer_paths.append(
                {
                    "attn_q": f"{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_q_proj_weight.pt",
                    "attn_k": f"{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_k_proj_weight.pt",
                    "attn_v": f"{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_v_proj_weight.pt",
                    "attn_o": f"{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_o_proj_weight.pt",
                    "ln_in": f"{self.model_matrix_fold_dir}layers_{layer_idx}_input_layernorm_weight.pt",
                    "ln_post": f"{self.model_matrix_fold_dir}layers_{layer_idx}_post_attention_layernorm_weight.pt",
                    "mlp_gate": f"{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_gate_proj_weight.pt",
                    "mlp_up": f"{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_up_proj_weight.pt",
                    "mlp_down": f"{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_down_proj_weight.pt",
                }
            )

        # --------------------------------------------------
        # PLACEHOLDERS (NO TENSORS LOADED HERE)
        # --------------------------------------------------
        self.token_embedding_matrix = None
        self.embed_tokens_weight = None
        self.lm_head_weight = None
        self.final_norm_weight = None
        self.token_embedding_matrix_path = ""
        self.cluster_token_embedding_matrix = None
        self.full_token_embedding_matrix = None
        self._cluster_anchor = None
        self._final_norm_weight = None
        self._lm_head_weight = None

        # --------------------------------------------------
        # LOG
        # --------------------------------------------------
        print(f"🔍 Model: {getattr(self.config, 'model_type', 'unknown')}")
        print(f"🔍 Attention: {self.attention_type}")
        print(f"🔍 Heads: Q={self.num_q_heads}, KV={self.num_kv_heads}")
        print(f"🔍 Hidden size: {self.hidden_size}")
        print(f"🔍 Head dimension: {self.head_dim}")
        print(f"🔍 KV dimension: {self.kv_dim}")

    def _cache_matrix_name(self, base_name: str) -> str:
        # `cluster_matrix_v1.py` derives shard filenames from the `.pt` basename in load mode,
        # so shard isolation must be done via per-model folders (not by prefixing matrix names).
        return str(base_name)

    def _as_batch_first_2d(self, t: torch.Tensor, batch: int, width: int, *, name: str) -> torch.Tensor:
        """
        Normalize cluster outputs into [B, W] for micro-batching.

        The cluster backend sometimes returns:
          - [B, W] (ideal)
          - [W, B] (needs transpose)
          - [1, B*W] (flattened)
          - [B*W, 1] (flattened column)
        """
        batch = int(batch)
        width = int(width)
        if batch <= 0 or width <= 0:
            raise ValueError(f"{name}: invalid target shape batch={batch} width={width}")

        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name}: expected torch.Tensor, got {type(t)}")

        if t.ndim == 1:
            if int(t.numel()) != batch * width:
                raise ValueError(f"{name}: got 1D numel={int(t.numel())}, expected {batch*width}")
            return t.reshape(batch, width).contiguous()

        if t.ndim != 2:
            raise ValueError(f"{name}: expected 1D/2D, got {tuple(t.shape)}")

        if tuple(t.shape) == (batch, width):
            return t.contiguous()
        if tuple(t.shape) == (width, batch):
            return t.t().contiguous()

        if t.shape[0] == 1 and int(t.shape[1]) == batch * width:
            return t.reshape(batch, width).contiguous()
        if t.shape[1] == 1 and int(t.shape[0]) == batch * width:
            return t.reshape(batch, width).contiguous()

        raise ValueError(f"{name}: got {tuple(t.shape)}, cannot coerce to ({batch}, {width})")

    def _get_cluster_weight_cached(self, matrix_file_path: str) -> cluster_matrix:
        """
        Cache `cluster_matrix` objects for *static* weight matrices (q/k/v/o, MLP weights).

        This removes repeated `cluster_matrix(...)` construction (and its Python/IO overhead)
        inside the token decode loop. It preserves numerics because the underlying shards
        and `cluster_shard_operation` calls are unchanged.
        """
        def _regenerate_weight_shards() -> None:
            # Fallback path: only regenerate if `auto_set_up=[1, "load"]` fails.
            # This avoids false-positive "missing shard" checks on the head node, because
            # shards are intentionally distributed across nodes (not centralized on the head's disk).
            matrix_name = self._cache_matrix_name(os.path.basename(matrix_file_path).split(".pt")[0])

            # Recreate shards with the same transpose policy used by `save_distribute_model_matrices`.
            w = torch.load(matrix_file_path, map_location="cpu")
            if not isinstance(w, torch.Tensor) or w.ndim != 2:
                raise ValueError(f"Expected 2D torch.Tensor at {matrix_file_path}, got {type(w)} {getattr(w, 'shape', None)}")

            hidden = int(self.hidden_size)
            if matrix_file_path.endswith("_mlp_gate_proj_weight.pt") or matrix_file_path.endswith("_mlp_up_proj_weight.pt"):
                if w.shape[0] == hidden:
                    w = w.contiguous()
                elif w.shape[1] == hidden:
                    w = w.t().contiguous()
                else:
                    raise ValueError(f"Unexpected MLP gate/up shape at {matrix_file_path}: {tuple(w.shape)} (hidden={hidden})")
            elif matrix_file_path.endswith("_mlp_down_proj_weight.pt"):
                if w.shape[1] == hidden:
                    w = w.contiguous()
                elif w.shape[0] == hidden:
                    w = w.t().contiguous()
                else:
                    raise ValueError(f"Unexpected MLP down shape at {matrix_file_path}: {tuple(w.shape)} (hidden={hidden})")
            else:
                w = w.t().contiguous()

            cluster_matrix(
                matrix_file_path=w,
                cluster_zmq_object=self.cluster_zmq_object,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name=matrix_name,
            )
            del w
            gc.collect()

        cm = self._cluster_weight_cache.get(matrix_file_path)
        if cm is None:
            base_name = os.path.basename(matrix_file_path).split(".pt")[0]
            matrix_name = self._cache_matrix_name(base_name)
            if self.weight_cache_mode == "save":
                # Explicit cache build mode: write shards once (tagged) and keep the object.
                cm = cluster_matrix(
                    matrix_file_path=matrix_file_path,
                    cluster_zmq_object=self.cluster_zmq_object,
                    CPU_GPU_select_list=self.CPU_GPU_select_list,
                    node_percentages=self.percentages,
                    back_end_select_list=self.backend_select_list,
                    split_matrix=True,
                    dim=1,
                    auto_set_up=[1, "save"],
                    matrix_name=matrix_name,
                )
            else:
                try:
                    cm = cluster_matrix(
                        matrix_file_path=matrix_file_path,
                        cluster_zmq_object=self.cluster_zmq_object,
                        CPU_GPU_select_list=self.CPU_GPU_select_list,
                        node_percentages=self.percentages,
                        back_end_select_list=self.backend_select_list,
                        split_matrix=True,
                        dim=1,
                        auto_set_up=[1, "load"],
                        matrix_name=matrix_name,
                    )
                except Exception as e:
                    # Backward-compatible: if shards were cached before model-tagging existed,
                    # try the legacy untagged matrix name before failing.
                    legacy_name = str(base_name)
                    try:
                        cm = cluster_matrix(
                            matrix_file_path=matrix_file_path,
                            cluster_zmq_object=self.cluster_zmq_object,
                            CPU_GPU_select_list=self.CPU_GPU_select_list,
                            node_percentages=self.percentages,
                            back_end_select_list=self.backend_select_list,
                            split_matrix=True,
                            dim=1,
                            auto_set_up=[1, "load"],
                            matrix_name=legacy_name,
                        )
                        print(
                            f"⚠️  Using legacy shard cache name for {os.path.basename(matrix_file_path)} "
                            f"({legacy_name}). Consider re-caching to avoid cross-model collisions."
                        )
                    except Exception as e2:
                        raise FileNotFoundError(
                            f"Missing cached shards for {os.path.basename(matrix_file_path)}.\n"
                            f"Tried tagged name: {matrix_name}\n"
                            f"Tried legacy name: {legacy_name}\n"
                            f"Original error: {e}\n"
                            f"Legacy error: {e2}\n"
                            "Fix: run with --weight-cache-mode save --precache (or let the chat wrapper do it)."
                        ) from e2
            self._cluster_weight_cache[matrix_file_path] = cm
        return cm

    def _get_attn_o_proj_mat(self, layer_idx: int) -> torch.Tensor:
        """
        Returns a cached matrix `M` such that `attn_output_flat @ M` matches the original
        o_proj application, regardless of whether the checkpoint weight is [out,in] or [in,out].
        """
        layer_idx = int(layer_idx)
        mat = self._attn_o_proj_mat[layer_idx]
        if mat is not None:
            return mat

        attn_o_proj_path = self._layer_paths[layer_idx]["attn_o"]
        w = torch.load(attn_o_proj_path, map_location="cpu")
        if not isinstance(w, torch.Tensor) or w.ndim != 2:
            raise ValueError(f"attn_o_proj must be 2D torch.Tensor, got {type(w)} {getattr(w, 'shape', None)}")

        hidden = int(self.hidden_size)
        if w.shape == (hidden, hidden):
            # Ambiguous: could be HF [out,in] (common) or already transposed [in,out].
            # We preserve the previous runtime behavior:
            #   if w.shape[1] == hidden (true) -> treat as [out,in] and use w.T
            mat = w.t().contiguous()
        elif w.shape[1] == hidden:
            # HF weight: [out, in] -> use w.T so x @ w.T
            mat = w.t().contiguous()
        elif w.shape[0] == hidden:
            # Already transposed: [in, out] -> use w directly so x @ w
            mat = w.contiguous()
        else:
            raise ValueError(f"attn_o_proj shape {tuple(w.shape)} incompatible with hidden={hidden}")

        self._attn_o_proj_mat[layer_idx] = mat
        return mat

    def transpose_save_matrix(self, layer_path: str) -> int:
        matrix = torch.load(layer_path, map_location="cpu")
        if not isinstance(matrix, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor in {layer_path}, got {type(matrix)}")
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D tensor in {layer_path}, got {tuple(matrix.shape)}")
        torch.save(matrix.t().contiguous(), layer_path)
        return 0

    def transpose_mlp_layers(self) -> int:
        """
        In-place transpose+contiguous of MLP weight .pt files so runtime doesn't need `.t().contiguous()`.

        After running, files will be:
          - up/gate: [hidden, intermediate]
          - down:    [intermediate, hidden]
        """
        for layer_index in range(self.num_layers):
            mlp_up_path = f"{self.model_matrix_fold_dir}layers_{layer_index}_mlp_up_proj_weight.pt"
            mlp_down_path = f"{self.model_matrix_fold_dir}layers_{layer_index}_mlp_down_proj_weight.pt"
            mlp_gate_path = f"{self.model_matrix_fold_dir}layers_{layer_index}_mlp_gate_proj_weight.pt"
            print(f"transposing mlp layers {layer_index}")
            for path in (mlp_up_path, mlp_down_path, mlp_gate_path):
                if not os.path.exists(path):
                    print(f"⚠️  Missing MLP weight (skip): {path}")
                    continue
                self.transpose_save_matrix(path)
        return 0

    def cache_mlp_weight_shards(self, start_layer: int = 0, end_layer: int | None = None) -> int:
        """
        One-time cache: distribute transposed MLP weights as cluster shards so `mlp_layer` can `load` instead of `save`.
        """
        if end_layer is None:
            end_layer = self.num_layers - 1
        for layer_idx in range(int(start_layer), int(end_layer) + 1):
            hidden = int(self.Hidden_size)
            mlp_up_path = f"{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_up_proj_weight.pt"
            mlp_down_path = f"{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_down_proj_weight.pt"
            mlp_gate_path = f"{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_gate_proj_weight.pt"

            mlp_gate_w = torch.load(mlp_gate_path, map_location="cpu")
            mlp_up_w = torch.load(mlp_up_path, map_location="cpu")
            mlp_down_w = torch.load(mlp_down_path, map_location="cpu")

            # Backward compatible: if user didn't run transpose_mlp_layers yet, fix orientation here.
            if mlp_gate_w.ndim != 2 or mlp_up_w.ndim != 2 or mlp_down_w.ndim != 2:
                raise ValueError(
                    f"Expected 2D MLP weights at layer {layer_idx}: "
                    f"gate={tuple(mlp_gate_w.shape)} up={tuple(mlp_up_w.shape)} down={tuple(mlp_down_w.shape)}"
                )
            if mlp_gate_w.shape[0] != hidden and mlp_gate_w.shape[1] == hidden:
                mlp_gate_w = mlp_gate_w.t().contiguous()
            else:
                mlp_gate_w = mlp_gate_w.contiguous()
            if mlp_up_w.shape[0] != hidden and mlp_up_w.shape[1] == hidden:
                mlp_up_w = mlp_up_w.t().contiguous()
            else:
                mlp_up_w = mlp_up_w.contiguous()
            if mlp_down_w.shape[1] != hidden and mlp_down_w.shape[0] == hidden:
                mlp_down_w = mlp_down_w.t().contiguous()
            else:
                mlp_down_w = mlp_down_w.contiguous()

            cluster_matrix(
                matrix_file_path=mlp_gate_w,
                cluster_zmq_object=self.cluster_zmq_object,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name=self._cache_matrix_name(f"layer{layer_idx}_mlp_gate_w"),
            )
            cluster_matrix(
                matrix_file_path=mlp_up_w,
                cluster_zmq_object=self.cluster_zmq_object,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name=self._cache_matrix_name(f"layer{layer_idx}_mlp_up_w"),
            )
            cluster_matrix(
                matrix_file_path=mlp_down_w,
                cluster_zmq_object=self.cluster_zmq_object,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name=self._cache_matrix_name(f"layer{layer_idx}_mlp_down_w"),
            )
        return 0

    def _has_weight_shards(self, matrix_name: str) -> bool:
        local_ram_folder = os.environ.get("LOCAL_RAM_FOLDER", "/dev/shm/matrix_shards/")
        local_disk_folder = os.environ.get("LOCAL_DISK_FOLDER", "matrix_shards/")
        local_project_dir = os.environ.get("LOCAL_PROJECT_DIR", self.local_project_dir)
        ram_path = os.path.join(local_ram_folder, f"{matrix_name}_shard_0.bin")
        disk_path = os.path.join(local_project_dir, local_disk_folder, f"{matrix_name}_shard_0.bin")
        return os.path.exists(ram_path) or os.path.exists(disk_path)

    def _get_final_norm_weight_path(self) -> str:
        candidates = (
            f"{self.model_matrix_fold_dir}model_norm_weight.pt",
            f"{self.model_matrix_fold_dir}norm_weight.pt",
        )
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Final norm weight not found. Tried: {candidates}")

    def _get_lm_head_weight_path(self) -> str:
        path = f"{self.model_matrix_fold_dir}lm_head_weight.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(f"LM head weight not found: {path}")
        return path

    def decode_next_token(
        self,
        hidden_out: torch.Tensor,
        *,
        temperature: float = 0.0,
        top_k: int = 0,
        use_cluster: bool = False,
    ) -> tuple[int, torch.Tensor]:
        """
        Convert the final hidden state for a single token into logits and pick the next token id.
        Uses final RMSNorm + LM head.

        Returns:
            (next_token_id, logits_1d[vocab])
        """
        if hidden_out.ndim != 1:
            raise ValueError(f"decode_next_token expects [hidden], got {tuple(hidden_out.shape)}")

        if self._final_norm_weight is None:
            self._final_norm_weight = torch.load(self._get_final_norm_weight_path(), map_location="cpu")
        norm_w = self._final_norm_weight
        if norm_w.ndim != 1 or norm_w.shape[0] != hidden_out.shape[0]:
            raise ValueError(f"final_norm_weight mismatch: weight={tuple(norm_w.shape)} hidden={tuple(hidden_out.shape)}")

        hidden_norm = self.rms_norm(hidden_out.unsqueeze(0), norm_w).squeeze(0)  # [hidden]

        if self._lm_head_weight is None:
            self._lm_head_weight = torch.load(self._get_lm_head_weight_path(), map_location="cpu")
        lm_head_w = self._lm_head_weight  # [vocab, hidden]
        if lm_head_w.ndim != 2 or lm_head_w.shape[1] != hidden_norm.shape[0]:
            raise ValueError(f"lm_head_weight mismatch: weight={tuple(lm_head_w.shape)} hidden={tuple(hidden_norm.shape)}")

        if use_cluster:
            # Cluster decode is optional; local decode is the default for correctness.
            hidden_cluster = cluster_matrix(
                matrix_file_path=hidden_norm.unsqueeze(1).contiguous(),  # [hidden, 1]
                cluster_zmq_object=self.cluster_zmq_object,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=False,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name=self._cache_matrix_name("decode_hidden"),
            )
            lm_head_w_t_cluster = cluster_matrix(
                matrix_file_path=lm_head_w.t().contiguous(),  # [hidden, vocab]
                cluster_zmq_object=self.cluster_zmq_object,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name=self._cache_matrix_name("lm_head_w_t"),
            )
            logits_2d = hidden_cluster.cluster_shard_operation(lm_head_w_t_cluster, True, False, True)  # [1, vocab]
            logits = logits_2d.squeeze(0)
        else:
            logits = (hidden_norm.unsqueeze(0) @ lm_head_w.t()).squeeze(0)  # [vocab]

        if temperature is None or temperature <= 0.0:
            next_id = int(torch.argmax(logits).item())
            return next_id, logits

        scaled = logits / float(temperature)
        if top_k and top_k > 0:
            k = min(int(top_k), scaled.numel())
            top_vals, top_idx = torch.topk(scaled, k)
            probs = torch.softmax(top_vals, dim=-1)
            next_local = int(torch.multinomial(probs, num_samples=1).item())
            next_id = int(top_idx[next_local].item())
            return next_id, logits

        probs = torch.softmax(scaled, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        return next_id, logits

    def detect_attention_type(self):
        """
        Detect attention type (MHA / GQA / MQA) using config only.
        NO model weights required.
        """

        config = self.config

        # Default assumptions
        num_q_heads = getattr(config, "num_attention_heads", None)
        num_kv_heads = getattr(config, "num_key_value_heads", None)

        if num_q_heads is None:
            raise ValueError("Config missing num_attention_heads")

        # If num_key_value_heads not present → standard MHA
        if num_kv_heads is None:
            num_kv_heads = num_q_heads
            attention_type = "MHA"
        else:
            if num_kv_heads == 1:
                attention_type = "MQA"
            elif num_kv_heads < num_q_heads:
                attention_type = "GQA"
            else:
                attention_type = "MHA"

        return attention_type, num_q_heads, num_kv_heads

    def list_llm_layer(self):
        for name, param in self.model.named_parameters():
            print("LLM layer --> ", name)

    def tokenize_text(self, text, use_chat_template=False):
        if use_chat_template and getattr(self.tokenizer, "chat_template", None) and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": text}]
            chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            self.tokens = self.tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False)
        else:
            self.tokens = self.tokenizer(text, return_tensors="pt")
        return self.tokens.input_ids

    def save_all_model_layers(
        self,
        start_layer: int = 0,
        end_layer: int | None = None,
        batch_size: int = 4,
        *,
        dtype: str = "float16",
        overwrite: bool = False,
        prefer_safetensors: bool = True,
        allow_full_model_load: bool = False,
    ) -> int:
        """
        Save the model weights needed by this project to `model_matrices/` without loading the full model in RAM.

        Notes:
        - Prefers streaming tensors from `.safetensors` shards (lowest memory).
        - Falls back to a full model load only if safetensors aren't found (may OOM on large models).
        - `batch_size` is kept for backward compatibility; streaming mode ignores it.
        """
        import gc
        import glob
        import re
        import shutil
        import time

        if end_layer is None:
            end_layer = getattr(self.config, "num_hidden_layers", 32) - 1

        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bf16": torch.bfloat16}
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype={dtype!r}. Use one of: {sorted(dtype_map)}")
        save_dtype = dtype_map[dtype]

        os.makedirs(self.model_matrix_fold_dir, exist_ok=True)

        print(f"💾 SAVING MODEL WEIGHTS SAFELY {start_layer} to {end_layer}")
        print(f"📦 Output dir: {self.model_matrix_fold_dir}")
        print(f"🔢 Save dtype: {dtype}")
        print(f"🧱 Batch size (compat): {batch_size}")
        print("=" * 60)

        allowed_layer_suffixes = {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        }

        def key_to_outfile(key: str) -> str | None:
            if key == "model.embed_tokens.weight":
                return "embed_tokens_weight.pt"
            if key == "lm_head.weight":
                return "lm_head_weight.pt"
            if key == "model.norm.weight":
                return "model_norm_weight.pt"

            m = re.match(r"^model\.layers\.(\d+)\.(.+)$", key)
            if not m:
                return None
            layer_idx = int(m.group(1))
            if layer_idx < int(start_layer) or layer_idx > int(end_layer):
                return None
            suffix = m.group(2)
            if suffix not in allowed_layer_suffixes:
                return None
            # e.g. model.layers.0.self_attn.q_proj.weight -> layers_0_self_attn_q_proj_weight.pt
            return f"layers_{layer_idx}_{suffix.replace('.', '_')}.pt"

        total_saved = 0

        # Prefer `model.safetensors.index.json` when present (most HF sharded checkpoints).
        # This is more robust than iterating `safe_open(...).keys()` on very large shards.
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        if prefer_safetensors and os.path.exists(index_path):
            try:
                from safetensors.torch import safe_open  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    f"Found safetensors index at {index_path} but failed to import safetensors: {e}. "
                    "Install `safetensors` in your env."
                )

            import json

            idx = json.loads(open(index_path, "r", encoding="utf-8").read())
            weight_map: dict[str, str] = dict(idx.get("weight_map", {}))
            if not weight_map:
                raise RuntimeError(f"safetensors index missing weight_map: {index_path}")

            # Group wanted keys by shard filename.
            keys_by_file: dict[str, list[str]] = {}
            for key, filename in weight_map.items():
                out_name = key_to_outfile(key)
                if out_name is None:
                    continue
                out_path = os.path.join(self.model_matrix_fold_dir, out_name)
                if (not overwrite) and os.path.exists(out_path):
                    continue
                keys_by_file.setdefault(filename, []).append(key)

            shard_files = sorted(keys_by_file.keys())
            print(f"✅ Using safetensors index streaming from {len(shard_files)} file(s)")

            for fname in shard_files:
                st_path = os.path.join(self.model_path, fname)
                if not os.path.exists(st_path):
                    raise FileNotFoundError(f"Index points to missing shard file: {st_path}")
                print(f"\n📦 Reading: {st_path}")
                with safe_open(st_path, framework="pt", device="cpu") as f:
                    for key in keys_by_file[fname]:
                        out_name = key_to_outfile(key)
                        if out_name is None:
                            continue
                        out_path = os.path.join(self.model_matrix_fold_dir, out_name)
                        if (not overwrite) and os.path.exists(out_path):
                            continue

                        tensor = f.get_tensor(key)
                        if tensor.dtype != save_dtype:
                            tensor = tensor.to(dtype=save_dtype)
                        tensor = tensor.contiguous()
                        torch.save(tensor, out_path)
                        total_saved += 1

                        del tensor
                        gc.collect()

                time.sleep(0.05)

            # Convenience: keep `norm_weight.pt` as a copy if needed by older code.
            model_norm_path = os.path.join(self.model_matrix_fold_dir, "model_norm_weight.pt")
            norm_path = os.path.join(self.model_matrix_fold_dir, "norm_weight.pt")
            if os.path.exists(model_norm_path) and (overwrite or not os.path.exists(norm_path)):
                shutil.copyfile(model_norm_path, norm_path)
                total_saved += 1

            print(f"\n🎉 SAFELY SAVED {total_saved} tensors via safetensors index")
            return total_saved

        safetensors_files: list[str] = []
        if prefer_safetensors:
            safetensors_files = sorted(glob.glob(os.path.join(self.model_path, "*.safetensors")))

        if safetensors_files:
            try:
                from safetensors.torch import safe_open  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    f"Found .safetensors weights but failed to import safetensors: {e}. "
                    "Install `safetensors` in your env."
                )

            print(f"✅ Using safetensors streaming from {len(safetensors_files)} file(s)")

            for st_path in safetensors_files:
                print(f"\n📦 Reading: {st_path}")
                with safe_open(st_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        out_name = key_to_outfile(key)
                        if out_name is None:
                            continue
                        out_path = os.path.join(self.model_matrix_fold_dir, out_name)
                        if (not overwrite) and os.path.exists(out_path):
                            continue

                        tensor = f.get_tensor(key)
                        if tensor.dtype != save_dtype:
                            tensor = tensor.to(dtype=save_dtype)
                        tensor = tensor.contiguous()
                        torch.save(tensor, out_path)
                        total_saved += 1

                        del tensor
                        gc.collect()

                time.sleep(0.05)

            # Convenience: keep `norm_weight.pt` as a copy if needed by older code.
            model_norm_path = os.path.join(self.model_matrix_fold_dir, "model_norm_weight.pt")
            norm_path = os.path.join(self.model_matrix_fold_dir, "norm_weight.pt")
            if os.path.exists(model_norm_path) and (overwrite or not os.path.exists(norm_path)):
                shutil.copyfile(model_norm_path, norm_path)
                total_saved += 1

            print(f"\n🎉 SAFELY SAVED {total_saved} tensors via safetensors")
            return total_saved

        if not allow_full_model_load:
            raise RuntimeError(
                "No `.safetensors` weight shards found in `model_path`, and `allow_full_model_load=False`.\n"
                "To avoid crashing the PC, this function defaults to safetensors streaming only.\n"
                "Fix: ensure your model has `*.safetensors` files, or call with `allow_full_model_load=True` "
                "(may use lots of RAM)."
            )

        # Fallback: full model load (can OOM on large models).
        print("⚠️  No .safetensors files found; doing full model load (may use a lot of RAM).")
        try:
            from transformers import AutoModelForCausalLM  # type: ignore
        except Exception as e:
            raise RuntimeError(f"transformers not available for fallback load: {e}")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=save_dtype,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        model.eval()

        wanted_files: dict[str, torch.Tensor] = {}
        state = model.state_dict()
        for key, tensor in state.items():
            out_name = key_to_outfile(key)
            if out_name is None:
                continue
            out_path = os.path.join(self.model_matrix_fold_dir, out_name)
            if (not overwrite) and os.path.exists(out_path):
                continue
            wanted_files[out_path] = tensor.detach().to(dtype=save_dtype).contiguous().cpu()

        for out_path, tensor in wanted_files.items():
            torch.save(tensor, out_path)
            total_saved += 1
            del tensor
            gc.collect()

        del state
        del model
        gc.collect()

        print(f"\n🎉 SAVED {total_saved} tensors via full-model fallback")
        return total_saved

    def get_token_embeddings(self, input_prompt='tell me a short joke', use_chat_template=False):
        """Get and distribute token embeddings"""
        # Tokenize the input prompt
        self.tokenize_text(input_prompt, use_chat_template=use_chat_template)
        
        if self.tokens is None:
            raise RuntimeError("Tokenization failed; self.tokens is None")
        
        print("🔍 Getting and distributing token embeddings...")
        
        # Load embedding matrix once (hot across multiple `generate_text` calls).
        embedding_path = self.model_matrix_fold_dir + 'embed_tokens_weight.pt'
        if self.embed_tokens_weight is None:
            if not os.path.exists(embedding_path):
                raise FileNotFoundError(
                    f"Embedding weights not found: {embedding_path}\n"
                    "Fix: extract weights to model_matrices first (or run the chat wrapper with --weight-cache-mode save --precache)."
                )
            self.embed_tokens_weight = torch.load(embedding_path, map_location="cpu")
        embedding_matrix = self.embed_tokens_weight
        print(f"📊 Embedding matrix shape: {embedding_matrix.shape}")
        
        # Get token IDs
        token_ids = self.tokens.input_ids[0]
        print(f"📊 Token IDs: {token_ids.tolist()}")
        
        # Vectorized embedding lookup: [seq] -> [seq, hidden]
        all_embeddings = embedding_matrix[token_ids]
        print(f"📦 Token embeddings shape: {all_embeddings.shape}")
        
        self.token_embedding_matrix = all_embeddings
        return self.token_embedding_matrix

    def save_distribute_model_matrices(
        self,
        start_layer = 0,
        end_layer = 0,
        include_embed_tokens: bool = False,
        include_lm_head: bool = False,
        include_final_norm: bool = False,
        transpose_for_runtime: bool = True,
        keep_ram_copies: bool = False,
        cleanup_sleep_s: float = 0.0,
        gc_collect: bool = True,
    ):
        def _zmq_send_command(worker_ip: str, command: str) -> bool:
            socket_pool = getattr(self.cluster_zmq_object, "llama_socket_pool", None)
            if socket_pool and worker_ip in socket_pool:
                try:
                    socket_pool[worker_ip].send(command.encode("utf-8"))
                    return True
                except Exception as e:
                    print(f"❌ Error sending command to {worker_ip}: {e}")
                    return False
            print(f"❌ No socket found for worker {worker_ip}")
            return False

        def _cleanup_ram_bins(prefix: str) -> None:
            prefix = str(prefix)
            if not prefix:
                return

            local_ram = getattr(self.cluster_zmq_object, "local_RAM_folder", "/dev/shm/matrix_shards/")
            remote_ram = getattr(self.cluster_zmq_object, "remote_RAM_folder", "/dev/shm/matrix_shards/")
            if not local_ram.endswith("/"):
                local_ram += "/"
            if not remote_ram.endswith("/"):
                remote_ram += "/"

            # Local cleanup (head node)
            for path in glob.glob(os.path.join(local_ram, f"{prefix}*.bin")):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"⚠️  Failed to remove local RAM file {path}: {e}")

            # Remote cleanup (worker nodes)
            # Note: no ACK is guaranteed for Linux commands; we best-effort free RAM.
            unique_ips = sorted(set(getattr(self.cluster_zmq_object, "node_IP_list", self.IP_list)))
            head_ip = getattr(self.cluster_zmq_object, "IP", None)
            for ip in unique_ips:
                if head_ip and ip == head_ip:
                    continue
                cmd = f"rm -f {remote_ram}{prefix}*.bin"
                _zmq_send_command(ip, cmd)

            if cleanup_sleep_s:
                time.sleep(float(cleanup_sleep_s))

            if gc_collect:
                gc.collect()

        num_layers = getattr(self.config, "num_hidden_layers", 32)
        print(f"📊 Total layers: {num_layers}")
        print("-" * 70)

        if end_layer > num_layers or start_layer > num_layers:
            print("incorrect start/end layers")
            return 0

        if end_layer == 0:
            end_layer = self.num_layers

        extra_paths: list[str] = []
        if include_embed_tokens:
            extra_paths.append(f"{self.model_matrix_fold_dir}embed_tokens_weight.pt")
        if include_lm_head:
            extra_paths.append(f"{self.model_matrix_fold_dir}lm_head_weight.pt")
        if include_final_norm:
            extra_paths.extend(
                [
                    f"{self.model_matrix_fold_dir}model_norm_weight.pt",
                    f"{self.model_matrix_fold_dir}norm_weight.pt",
                ]
            )

        for extra_path in extra_paths:
            if not os.path.exists(extra_path):
                print(f"⚠️  Missing extra weight (skip): {extra_path}")
                continue
            extra_name = os.path.basename(extra_path).split(".pt")[0]
            matrix_name = self._cache_matrix_name(extra_name)
            cluster_matrix(
                matrix_file_path=extra_path,
                cluster_zmq_object=self.cluster_zmq_object,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=True,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name=matrix_name,
            )
            if not keep_ram_copies:
                _cleanup_ram_bins(matrix_name)
         
        for layer_idx in range(start_layer, end_layer):
            print(f"SAVING LAYER: {layer_idx}")
            # ------------------------------------------------------------
            # Paths
            # ------------------------------------------------------------
            attn_q_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_q_proj_weight.pt'
            attn_k_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_k_proj_weight.pt'
            attn_v_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_v_proj_weight.pt'
            attn_o_proj_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_self_attn_o_proj_weight.pt'

            mlp_gate_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_gate_proj_weight.pt'
            mlp_up_path   = f'{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_up_proj_weight.pt'
            mlp_down_path = f'{self.model_matrix_fold_dir}layers_{layer_idx}_mlp_down_proj_weight.pt'

            # ============================================================
            # ATTENTION MATRIX SHARDS (Matrix B)
            # ============================================================

            weight_paths = (
                attn_q_proj_path,
                attn_k_proj_path,
                attn_v_proj_path,
                attn_o_proj_path,
                mlp_gate_path,
                mlp_up_path,
                mlp_down_path,
            )
            for weight_path in weight_paths:
                if not os.path.exists(weight_path):
                    print(f"⚠️  Missing weight (skip): {weight_path}")
                    continue
                matrix_name = self._cache_matrix_name(os.path.basename(weight_path).split(".pt")[0])

                weight = torch.load(weight_path, map_location="cpu")
                if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                    raise ValueError(f"Expected 2D torch.Tensor at {weight_path}, got {type(weight)} {getattr(weight, 'shape', None)}")

                if transpose_for_runtime:
                    hidden = int(self.hidden_size)
                    # MLP shapes are non-square; avoid double-transpose if already pre-transposed.
                    if weight_path.endswith("_mlp_gate_proj_weight.pt") or weight_path.endswith("_mlp_up_proj_weight.pt"):
                        # desired: [hidden, intermediate]
                        if weight.shape[0] == hidden:
                            weight = weight.contiguous()
                        elif weight.shape[1] == hidden:
                            weight = weight.t().contiguous()
                        else:
                            raise ValueError(f"Unexpected MLP gate/up shape at {weight_path}: {tuple(weight.shape)} (hidden={hidden})")
                    elif weight_path.endswith("_mlp_down_proj_weight.pt"):
                        # desired: [intermediate, hidden]
                        if weight.shape[1] == hidden:
                            weight = weight.contiguous()
                        elif weight.shape[0] == hidden:
                            weight = weight.t().contiguous()
                        else:
                            raise ValueError(f"Unexpected MLP down shape at {weight_path}: {tuple(weight.shape)} (hidden={hidden})")
                    else:
                        # Attention projections (square): store transposed to match runtime matmul flags.
                        weight = weight.t().contiguous()
                else:
                    weight = weight.contiguous()

                cluster_matrix(
                    matrix_file_path=weight,
                    cluster_zmq_object=self.cluster_zmq_object,
                    CPU_GPU_select_list=self.CPU_GPU_select_list,
                    node_percentages=self.percentages,
                    back_end_select_list=self.backend_select_list,
                    split_matrix=True,
                    dim=1,
                    auto_set_up=[1, "save"],
                    matrix_name=matrix_name,
                )
                # Release CPU memory pressure early (these weights are huge).
                del weight
                if not keep_ram_copies:
                    _cleanup_ram_bins(matrix_name)
                elif gc_collect:
                    gc.collect()

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        # x: [..., hidden]
        # weight: [hidden]
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
        y = (x_fp32 * rms).to(x.dtype)
        return y * weight.to(x.dtype)

    def _get_tensor_cached(self, path: str) -> torch.Tensor:
        """
        Small-tensor cache for weights that are reused every token (layernorms, etc.).
        Keeps numerics identical; avoids repeated disk I/O + Python overhead.
        """
        t = self._tensor_cache.get(path)
        if t is None:
            t = torch.load(path, map_location="cpu")
            self._tensor_cache[path] = t
        return t

    def _ensure_rope_cache(self, max_pos: int) -> None:
        """
        Precompute RoPE cos/sin tables for positions [0..max_pos).
        This removes per-token `torch.arange` + trig overhead and preserves numerics.
        """
        max_pos = int(max_pos)
        if max_pos <= 0:
            return

        if self._rope_cos is not None and self._rope_sin is not None and self._rope_cos.shape[0] >= max_pos:
            return

        rotary_dim = int(self._rope_rotary_dim)
        if self._rope_inv_freq is None:
            self._rope_inv_freq = 1.0 / (
                self._rope_theta
                ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / float(rotary_dim))
            )  # [rotary_dim/2], float32 on CPU

        positions = torch.arange(max_pos, dtype=torch.float32).unsqueeze(1)  # [max_pos, 1]
        freqs = positions * self._rope_inv_freq.unsqueeze(0)  # [max_pos, rotary_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_pos, rotary_dim]
        self._rope_cos = emb.cos()
        self._rope_sin = emb.sin()

    def _ensure_kv_cache(self, capacity: int) -> None:
        """
        Preallocate compact KV cache buffers and write by index (no torch.cat in decode loop).
        """
        capacity = int(capacity)
        if capacity <= self._kv_cache_capacity and self._kv_cache_k is not None and self._kv_cache_v is not None:
            return

        # (Re)allocate. Values beyond current sequence length are never read.
        self._kv_cache_capacity = max(capacity, 1)
        self._kv_cache_k = torch.empty(
            (int(self.num_layers), int(self.num_kv_heads), self._kv_cache_capacity, int(self.head_dim)),
            dtype=torch.float32,
        )
        self._kv_cache_v = torch.empty(
            (int(self.num_layers), int(self.num_kv_heads), self._kv_cache_capacity, int(self.head_dim)),
            dtype=torch.float32,
        )

    def rope_apply(self,
        q: torch.Tensor,
        k: torch.Tensor,
        position: int,
        rope_theta: float = 10000.0,
        rotary_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Llama-style RoPE to Q and K for a single position.

        q: [..., head_dim]
        k: [..., head_dim]
        position: int (token index)
        """
        if rotary_dim is None:
            rotary_dim = self.head_dim
        if rotary_dim <= 0:
            return q, k

        dtype = q.dtype
        rotary_dim = int(rotary_dim)
        self._rope_theta = float(rope_theta)
        self._rope_rotary_dim = rotary_dim
        self._ensure_rope_cache(int(position) + 1)
        assert self._rope_cos is not None and self._rope_sin is not None
        cos = self._rope_cos[int(position)].to(dtype=dtype)
        sin = self._rope_sin[int(position)].to(dtype=dtype)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)

        q_out = torch.cat((q_rot, q_pass), dim=-1)
        k_out = torch.cat((k_rot, k_pass), dim=-1)
        return q_out, k_out

    def rope_apply_batch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        rope_theta: float = 10000.0,
        rotary_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Llama-style RoPE to Q and K for multiple positions at once.

        q: [B, ..., head_dim]
        k: [B, ..., head_dim]
        positions: [B] absolute token indices (int64)
        """
        if rotary_dim is None:
            rotary_dim = self.head_dim
        rotary_dim = int(rotary_dim)
        if rotary_dim <= 0:
            return q, k

        if positions.ndim != 1:
            raise ValueError(f"positions must be 1D [B], got {tuple(positions.shape)}")
        positions = positions.to(dtype=torch.long)

        self._rope_theta = float(rope_theta)
        self._rope_rotary_dim = rotary_dim
        max_pos = int(positions.max().item()) + 1
        self._ensure_rope_cache(max_pos)
        assert self._rope_cos is not None and self._rope_sin is not None

        dtype = q.dtype
        cos = self._rope_cos.index_select(0, positions).to(dtype=dtype)  # [B, rotary_dim]
        sin = self._rope_sin.index_select(0, positions).to(dtype=dtype)  # [B, rotary_dim]

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        # Broadcast cos/sin across all non-batch dims (heads, etc.).
        view_shape = (cos.shape[0],) + (1,) * (q_rot.ndim - 2) + (cos.shape[1],)
        cos_view = cos.view(view_shape)
        sin_view = sin.view(view_shape)

        q_rot = (q_rot * cos_view) + (rotate_half(q_rot) * sin_view)
        k_rot = (k_rot * cos_view) + (rotate_half(k_rot) * sin_view)

        q_out = torch.cat((q_rot, q_pass), dim=-1)
        k_out = torch.cat((k_rot, k_pass), dim=-1)
        return q_out, k_out

    def expand_kv(self, k, v):
        """
        Expand KV heads for Grouped Query Attention (GQA).

        Args:
            k: Tensor (num_kv_heads, head_dim)
            v: Tensor (num_kv_heads, head_dim)

        Returns:
            k_expanded: (num_q_heads, head_dim)
            v_expanded: (num_q_heads, head_dim)
        """
        assert k.shape == (self.num_kv_heads, self.head_dim)
        assert v.shape == (self.num_kv_heads, self.head_dim)

        group_size = self.num_q_heads // self.num_kv_heads
        assert self.num_q_heads % self.num_kv_heads == 0, "Invalid GQA head configuration"

        k_expanded = k.repeat_interleave(group_size, dim=0)
        v_expanded = v.repeat_interleave(group_size, dim=0)

        return k_expanded, v_expanded

    def run_transformer(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 16,
        use_chat_template: bool = False,
        temperature: float = 0.0,
        top_k: int = 0,
        micro_batch_size: int = 16,
        print_output: bool = True,
    ) -> str:
        out = self.generate_text(
            prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            temperature=temperature,
            top_k=top_k,
            micro_batch_size=micro_batch_size,
        )
        if print_output:
            print("\n=== model output ===\n")
            print(out)
            print("\n====================\n")
        return out

    def generate_text(
        self,
        prompt: str = "tell me a short joke",
        *,
        max_new_tokens: int = 20,
        use_chat_template: bool = False,
        temperature: float = 0.0,
        top_k: int = 0,
        micro_batch_size: int = 16,
    ) -> str:
        with torch.inference_mode():
            prompt_embeddings = self.get_token_embeddings(prompt, use_chat_template=use_chat_template)
            if prompt_embeddings is None:
                raise RuntimeError("Failed to get prompt token embeddings")

            if self.tokens is None:
                raise RuntimeError("Tokenization failed; self.tokens is None")

            token_ids: list[int] = self.tokens.input_ids[0].tolist()

            # Preallocate compact KV cache once for the whole generation run (no torch.cat in decode loop).
            # Capacity is prompt_len + max_new_tokens (safe upper bound; values beyond seq_len are never read).
            capacity = len(token_ids) + int(max_new_tokens)
            self._ensure_kv_cache(capacity)

            last_hidden: torch.Tensor | None = None
            mb = max(int(micro_batch_size), 1)
            if mb == 1:
                for token_position, token_embedding in enumerate(prompt_embeddings):
                    last_hidden = self.run_transformer_layers(token_embedding, token_position)
            else:
                seq_len = int(prompt_embeddings.shape[0])
                print(f"[MICROBATCH] prompt prefill enabled: micro_batch_size={mb}, prompt_len={seq_len}")
                for start in range(0, seq_len, mb):
                    chunk = prompt_embeddings[start : start + mb]  # [B, hidden]
                    print(f"[MICROBATCH] prefill chunk: start={start}, batch={int(chunk.shape[0])}")
                    chunk_out = self.run_transformer_layers_microbatch(chunk, start)  # [B, hidden]
                    last_hidden = chunk_out[-1]

            if last_hidden is None:
                raise RuntimeError("Prompt was empty; no hidden state produced")

            eos_id = getattr(self.tokenizer, "eos_token_id", None)

            for _ in range(int(max_new_tokens)):
                next_id, _logits = self.decode_next_token(last_hidden, temperature=temperature, top_k=top_k)
                token_ids.append(next_id)

                if eos_id is not None and next_id == int(eos_id):
                    break

                next_embedding = self.embed_tokens_weight[next_id]
                last_hidden = self.run_transformer_layers(next_embedding, len(token_ids) - 1)

            return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def generate_text_stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 20,
        use_chat_template: bool = False,
        temperature: float = 0.0,
        top_k: int = 0,
        micro_batch_size: int = 16,
        skip_special_tokens: bool = True,
    ) -> Iterator[str]:
        """
        Stream decoded text increments as tokens are generated.

        This yields *text pieces* (not token ids) using full-sequence decode + diff, which preserves
        correct spacing/punctuation across common tokenizers (BPE/SentencePiece).
        """

        def _decode(ids: list[int]) -> str:
            return self.tokenizer.decode(
                ids, skip_special_tokens=bool(skip_special_tokens), clean_up_tokenization_spaces=False
            )

        def _diff(prev: str, cur: str) -> str:
            if cur.startswith(prev):
                return cur[len(prev) :]
            # Fallback: find common prefix length (rare tokenizer edge cases).
            n = 0
            limit = min(len(prev), len(cur))
            while n < limit and prev[n] == cur[n]:
                n += 1
            return cur[n:]

        with torch.inference_mode():
            prompt_embeddings = self.get_token_embeddings(prompt, use_chat_template=use_chat_template)
            if prompt_embeddings is None:
                raise RuntimeError("Failed to get prompt token embeddings")

            if self.tokens is None:
                raise RuntimeError("Tokenization failed; self.tokens is None")

            token_ids: list[int] = self.tokens.input_ids[0].tolist()

            capacity = len(token_ids) + int(max_new_tokens)
            self._ensure_kv_cache(capacity)

            last_hidden: torch.Tensor | None = None
            mb = max(int(micro_batch_size), 1)
            if mb == 1:
                for token_position, token_embedding in enumerate(prompt_embeddings):
                    last_hidden = self.run_transformer_layers(token_embedding, token_position)
            else:
                seq_len = int(prompt_embeddings.shape[0])
                print(f"[MICROBATCH] prompt prefill enabled: micro_batch_size={mb}, prompt_len={seq_len}")
                for start in range(0, seq_len, mb):
                    chunk = prompt_embeddings[start : start + mb]  # [B, hidden]
                    print(f"[MICROBATCH] prefill chunk: start={start}, batch={int(chunk.shape[0])}")
                    chunk_out = self.run_transformer_layers_microbatch(chunk, start)  # [B, hidden]
                    last_hidden = chunk_out[-1]

            if last_hidden is None:
                raise RuntimeError("Prompt was empty; no hidden state produced")

            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            prev_text = _decode(token_ids)

            for _ in range(int(max_new_tokens)):
                next_id, _logits = self.decode_next_token(last_hidden, temperature=temperature, top_k=top_k)
                token_ids.append(next_id)

                cur_text = _decode(token_ids)
                yield _diff(prev_text, cur_text)
                prev_text = cur_text

                if eos_id is not None and next_id == int(eos_id):
                    break

                next_embedding = self.embed_tokens_weight[next_id]
                last_hidden = self.run_transformer_layers(next_embedding, len(token_ids) - 1)

    def run_transformer_layers(self, input_token_embeddings, token_position: int):
        out = self.run_transformer_layers_microbatch(input_token_embeddings, int(token_position))
        if out.ndim != 2 or out.shape[0] != 1:
            raise RuntimeError(f"Expected microbatch output [1, hidden], got {tuple(out.shape)}")
        return out[0]

    def run_transformer_layers_microbatch(self, input_token_embeddings: torch.Tensor, token_position: int) -> torch.Tensor:
        """
        Micro-batched forward for a *consecutive* run of tokens.

        Args:
            input_token_embeddings: [B, hidden] or [hidden]
            token_position: absolute position of the first token in the micro-batch

        Returns:
            hidden_out: [B, hidden] final layer output for each token in the micro-batch
        """
        if self._kv_cache_k is None or self._kv_cache_v is None:
            raise RuntimeError("KV cache not initialized. Call run_transformer() first.")

        token_position = int(token_position)

        if input_token_embeddings.ndim == 1:
            input_token_embeddings = input_token_embeddings.unsqueeze(0)  # [1, hidden]
        if input_token_embeddings.ndim != 2 or input_token_embeddings.shape[1] != int(self.Hidden_size):
            raise ValueError(
                f"input_token_embeddings must be [B, hidden] (hidden={int(self.Hidden_size)}), got {tuple(input_token_embeddings.shape)}"
            )

        batch = int(input_token_embeddings.shape[0])
        end_pos_exclusive = token_position + batch
        if end_pos_exclusive > int(self._kv_cache_capacity):
            self._ensure_kv_cache(end_pos_exclusive)

        positions = torch.arange(token_position, end_pos_exclusive, dtype=torch.long)

        for layer_idx in range(int(self.num_layers)):
            paths = self._layer_paths[int(layer_idx)]

            # ------------------------------------------------------------
            # 1) Attention pre-norm + QKV projections (cluster offload)
            #    Shape bump: [hidden, B] instead of [hidden, 1]
            # ------------------------------------------------------------
            input_layernorm_weight = self._get_tensor_cached(paths["ln_in"])
            x_norm = self.rms_norm(input_token_embeddings, input_layernorm_weight)  # [B, hidden]
            x_col = x_norm.t().contiguous()  # [hidden, B]
            if batch > 1 and layer_idx == 0:
                print(f"[MICROBATCH] dispatch A: x_col shape={tuple(x_col.shape)} (hidden,B)")

            x_cluster = cluster_matrix(
                matrix_file_path=x_col,
                cluster_zmq_object=self.cluster_zmq_object,
                CPU_GPU_select_list=self.CPU_GPU_select_list,
                node_percentages=self.percentages,
                back_end_select_list=self.backend_select_list,
                split_matrix=False,
                dim=1,
                auto_set_up=[1, "save"],
                matrix_name=self._cache_matrix_name("input_token_embeddings"),
            )

            q_w = self._get_cluster_weight_cached(paths["attn_q"])
            k_w = self._get_cluster_weight_cached(paths["attn_k"])
            v_w = self._get_cluster_weight_cached(paths["attn_v"])

            q_flat = x_cluster.cluster_shard_operation(q_w, True, False, True)
            k_flat = x_cluster.cluster_shard_operation(k_w, True, False, True)
            v_flat = x_cluster.cluster_shard_operation(v_w, True, False, True)

            q_flat = self._as_batch_first_2d(q_flat, batch, int(self.hidden_size), name=f"layer{layer_idx}.q_flat")
            k_flat = self._as_batch_first_2d(k_flat, batch, int(self.kv_dim), name=f"layer{layer_idx}.k_flat")
            v_flat = self._as_batch_first_2d(v_flat, batch, int(self.kv_dim), name=f"layer{layer_idx}.v_flat")

            # [B, hidden] -> [B, heads, head_dim]
            q = q_flat.reshape(batch, int(self.num_q_heads), int(self.head_dim))
            k = k_flat.reshape(batch, int(self.num_kv_heads), int(self.head_dim))
            v = v_flat.reshape(batch, int(self.num_kv_heads), int(self.head_dim))

            # ------------------------------------------------------------
            # 2) RoPE (position-aware for each token in micro-batch)
            # ------------------------------------------------------------
            q, k = self.rope_apply_batch(
                q,
                k,
                positions=positions,
                rope_theta=self._rope_theta,
                rotary_dim=self.head_dim,
            )

            # ------------------------------------------------------------
            # 3) KV cache write (single slice write for the micro-batch)
            #    Cache layout: [layers, Hkv, T, D]
            # ------------------------------------------------------------
            self._kv_cache_k[layer_idx, :, token_position:end_pos_exclusive, :] = k.transpose(0, 1).to(dtype=torch.float32)
            self._kv_cache_v[layer_idx, :, token_position:end_pos_exclusive, :] = v.transpose(0, 1).to(dtype=torch.float32)

            # ------------------------------------------------------------
            # 4) Attention with causal mask inside micro-batch
            # ------------------------------------------------------------
            T_all = end_pos_exclusive
            k_cache = self._kv_cache_k[layer_idx, :, :T_all, :]  # [Hkv, T, D]
            v_cache = self._kv_cache_v[layer_idx, :, :T_all, :]  # [Hkv, T, D]

            q_group = q.reshape(batch, int(self.num_kv_heads), int(self._kv_group_size), int(self.head_dim))  # [B, Hkv, G, D]
            q_group = q_group.permute(1, 2, 0, 3)  # [Hkv, G, B, D]

            # KV caches are float32; ensure attention matmuls use a consistent dtype.
            q_group = q_group.to(dtype=torch.float32)

            scores = torch.matmul(
                q_group,  # [Hkv, G, B, D]
                k_cache.transpose(-1, -2).unsqueeze(1),  # [Hkv, 1, D, T]
            )  # -> [Hkv, G, B, T]
            scores = scores * self._inv_sqrt_head_dim

            # Causal mask for micro-batch: token at position p cannot attend to keys > p.
            key_idx = torch.arange(T_all, dtype=torch.long, device=scores.device)
            pos_for_mask = positions.to(device=scores.device)
            causal = key_idx.unsqueeze(0) <= pos_for_mask.unsqueeze(1)  # [B, T]
            scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)

            attn_weights = torch.softmax(scores, dim=-1)  # [Hkv, G, B, T]
            attn_out = torch.matmul(attn_weights, v_cache.unsqueeze(1))  # [Hkv, G, B, D]

            attn_out = attn_out.permute(2, 0, 1, 3)  # [B, Hkv, G, D]
            attn_output = attn_out.reshape(batch, int(self.num_q_heads), int(self.head_dim))  # [B, Hq, D]

            # o_proj + residual (identical math, just batched)
            attn_output = attn_output.to(dtype=input_token_embeddings.dtype)
            attn_output_flat = attn_output.reshape(batch, int(self.Hidden_size))  # [B, hidden]
            attn_hidden = attn_output_flat @ self._get_attn_o_proj_mat(layer_idx)  # [B, hidden]
            hidden_out = input_token_embeddings + attn_hidden  # [B, hidden]

            # ------------------------------------------------------------
            # 5) MLP block (post-attn norm + 2 matmuls) in batch
            # ------------------------------------------------------------
            input_token_embeddings = self.mlp_layer(layer_idx, hidden_out)

        return input_token_embeddings

    def mlp_layer(self,layer_idx, hidden_out):
        layer_idx = int(layer_idx)
        paths = self._layer_paths[layer_idx]
        mlp_up_path = paths["mlp_up"]
        mlp_down_path = paths["mlp_down"]
        mlp_gate_path = paths["mlp_gate"]
        post_attn_ln_path = paths["ln_post"]

        squeeze_out = (hidden_out.ndim == 1)
        if hidden_out.ndim == 1:
            residual = hidden_out.unsqueeze(0)  # [1, hidden]
        elif hidden_out.ndim == 2:
            residual = hidden_out  # [B, hidden]
        else:
            raise ValueError(f"mlp_layer expects [hidden] or [B, hidden], got {tuple(hidden_out.shape)}")

        # Cache post-attn RMSNorm weights (hot in decode loop).
        post_attn_ln_w = self._get_tensor_cached(post_attn_ln_path)
        if post_attn_ln_w.ndim != 1:
            raise ValueError(f"post_attention_layernorm_weight must be 1D, got {tuple(post_attn_ln_w.shape)}")
        if post_attn_ln_w.shape[0] != residual.shape[1]:
            raise ValueError(
                f"post_attention_layernorm_weight hidden mismatch: weight={post_attn_ln_w.shape[0]} hidden={residual.shape[1]}"
            )
        mlp_in = self.rms_norm(residual, post_attn_ln_w)  # [B, hidden]
        mlp_in_col = mlp_in.t().contiguous()  # [hidden, B]

        mlp_in_cluster = cluster_matrix(
            matrix_file_path=mlp_in_col,
            cluster_zmq_object=self.cluster_zmq_object,
            CPU_GPU_select_list=self.CPU_GPU_select_list,
            node_percentages=self.percentages,
            back_end_select_list=self.backend_select_list,
            split_matrix=False,
            dim=1,
            auto_set_up=[1, "save"],
            matrix_name=self._cache_matrix_name(f"layer{layer_idx}_mlp_in"),
        )

        # Cache static MLP weight shards (avoid per-token cluster_matrix construction).
        mlp_gate_cluster = self._get_cluster_weight_cached(mlp_gate_path)
        mlp_up_cluster = self._get_cluster_weight_cached(mlp_up_path)
        mlp_down_cluster = self._get_cluster_weight_cached(mlp_down_path)

        gate = mlp_in_cluster.cluster_shard_operation(mlp_gate_cluster, True, False, True)  # [B, intermediate]
        up = mlp_in_cluster.cluster_shard_operation(mlp_up_cluster, True, False, True)      # [B, intermediate]
        gate = self._as_batch_first_2d(gate, int(residual.shape[0]), int(gate.shape[-1]), name=f"layer{layer_idx}.mlp_gate")
        up = self._as_batch_first_2d(up, int(residual.shape[0]), int(up.shape[-1]), name=f"layer{layer_idx}.mlp_up")
        intermediate = torch.nn.functional.silu(gate) * up                                   # [B, intermediate]

        intermediate_cluster = cluster_matrix(
            matrix_file_path=intermediate.t().contiguous(),  # [intermediate, B]
            cluster_zmq_object=self.cluster_zmq_object,
            CPU_GPU_select_list=self.CPU_GPU_select_list,
            node_percentages=self.percentages,
            back_end_select_list=self.backend_select_list,
            split_matrix=False,
            dim=1,
            auto_set_up=[1, "save"],
            matrix_name=self._cache_matrix_name(f"layer{layer_idx}_mlp_intermediate"),
        )
        mlp_out = intermediate_cluster.cluster_shard_operation(mlp_down_cluster, True, False, True)  # [B, hidden]
        mlp_out = self._as_batch_first_2d(mlp_out, int(residual.shape[0]), int(residual.shape[1]), name=f"layer{layer_idx}.mlp_out")

        # Residual connection (post-attn residual + MLP output)
        layer_out = residual + mlp_out
        if squeeze_out:
            return layer_out.squeeze(0)
        return layer_out

if __name__ == "__main__":
    def _parse_csv_list(s: str) -> list[str]:
        parts: list[str] = []
        for raw in str(s).replace(" ", ",").split(","):
            v = raw.strip()
            if v:
                parts.append(v)
        return parts

    def _parse_bool_list(s: str) -> list[bool]:
        out: list[bool] = []
        for raw in _parse_csv_list(s):
            v = raw.strip().lower()
            if v in ("1", "true", "t", "yes", "y", "gpu"):
                out.append(True)
            elif v in ("0", "false", "f", "no", "n", "cpu"):
                out.append(False)
            else:
                raise ValueError(f"Invalid boolean value in list: {raw!r}")
        return out

    def _parse_float_list(s: str) -> list[float]:
        return [float(x) for x in _parse_csv_list(s)]

    parser = argparse.ArgumentParser(description="GQA cluster transformer runner (one-shot).")
    parser.add_argument("--model-dir", required=True, help="Path (or HF id) for the model.")
    parser.add_argument(
        "--model-matrices-dir",
        default=None,
        help="Optional override for extracted `.pt` weights directory (default: <model-dir>/model_matrices).",
    )
    parser.add_argument("--node-ips", required=True, help="Comma-separated node IPs (duplicates allowed).")
    parser.add_argument("--backend-select-list", required=True, help="Comma-separated backends per node (e.g. llama).")
    parser.add_argument("--cpu-gpu-select-list", required=True, help="Comma-separated 1/0 or true/false per node.")
    parser.add_argument("--node-percentages", required=True, help="Comma-separated shard percentages per node.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--prompt", default=None, help="If omitted, reads from stdin.")
    args = parser.parse_args()

    IP_list = _parse_csv_list(args.node_ips)
    backend_select_list = _parse_csv_list(args.backend_select_list)
    CPU_GPU_select_list = _parse_bool_list(args.cpu_gpu_select_list)
    percentages = _parse_float_list(args.node_percentages)

    n = len(IP_list)
    if not (len(backend_select_list) == len(CPU_GPU_select_list) == len(percentages) == n):
        raise ValueError(
            "List length mismatch: "
            f"node-ips={n} backend-select-list={len(backend_select_list)} "
            f"cpu-gpu-select-list={len(CPU_GPU_select_list)} node-percentages={len(percentages)}"
        )

    model: Optional[cluster_llm_transformer] = None
    try:
        model = cluster_llm_transformer(
            args.model_dir,
            IP_list,
            percentages,
            CPU_GPU_select_list,
            backend_select_list,
            model_matrices_dir=args.model_matrices_dir,
        )

        prompt = str(args.prompt) if args.prompt is not None else input("Enter prompt: ").strip()
        if not prompt:
            raise SystemExit(0)

        out = model.run_transformer(
            prompt,
            max_new_tokens=int(args.max_new_tokens),
            use_chat_template=bool(args.use_chat_template),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            micro_batch_size=int(args.micro_batch_size),
            print_output=True,
        )
        # Preserve backward-friendly echo.
        print(prompt, "\n")
        print(out, "\n")
    finally:
        try:
            if model is not None and hasattr(model.cluster_zmq_object, "cleanup"):
                model.cluster_zmq_object.cleanup()
        except Exception:
            pass
     
