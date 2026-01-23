from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def _repo_root() -> str:
    return os.path.abspath(os.path.dirname(__file__))


def _import_cluster_matrix_v1():
    cluster_dir = os.path.join(_repo_root(), "cluster_matrix")
    if cluster_dir not in sys.path:
        sys.path.insert(0, cluster_dir)
    import cluster_matrix_v1  # type: ignore

    return cluster_matrix_v1


_BLOCK_NORM_FILE_RE = re.compile(r"^layer_(\d+)_block_norm_(weight|bias)\.pt$")


def _try_extract_from_safetensors(model_dir: str, tensor_key: str) -> torch.Tensor:
    """
    Load a tensor directly from the model's safetensors shards (or consolidated file).
    This is used as a fallback when a required `.pt` file is missing from `model_matrices/`.
    """
    try:
        from safetensors.torch import safe_open  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("safetensors is required to auto-extract missing weights") from e

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    candidates: list[str] = []

    if os.path.exists(index_path):
        try:
            idx = json.load(open(index_path, "r"))
            shard = idx.get("weight_map", {}).get(tensor_key)
            if shard:
                candidates.append(os.path.join(model_dir, str(shard)))
        except Exception:
            # If index parsing fails, fall back to consolidated/model files.
            pass

    for fname in ("consolidated.safetensors", "model.safetensors"):
        p = os.path.join(model_dir, fname)
        if os.path.exists(p):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"Could not locate safetensors files under {model_dir}")

    last_err: Optional[Exception] = None
    for st_path in candidates:
        try:
            with safe_open(st_path, framework="pt", device="cpu") as f:
                if tensor_key not in f.keys():
                    continue
                return f.get_tensor(tensor_key)
        except Exception as e:
            last_err = e
            continue

    raise KeyError(f"Tensor key not found in safetensors: {tensor_key}") from last_err


def _as_batch_first_2d(t: torch.Tensor, batch: int, width: int, *, name: str) -> torch.Tensor:
    batch = int(batch)
    width = int(width)
    # cluster_matrix_v1 often converts matrices to a 4D layout (1, 1, rows, cols).
    # Coerce back to a 2D (rows, cols) view before the usual shape checks.
    if isinstance(t, torch.Tensor) and t.ndim == 4:
        if int(t.shape[0]) == 1 and int(t.shape[1]) == 1:
            t = t[0, 0]
        else:
            t = t.reshape(-1, int(t.shape[-1]))
    if isinstance(t, torch.Tensor) and t.ndim == 3:
        if int(t.shape[0]) == 1:
            t = t[0]
        else:
            t = t.reshape(-1, int(t.shape[-1]))
    if t.ndim == 2 and tuple(t.shape) == (batch, width):
        return t.contiguous()
    if t.ndim == 2 and tuple(t.shape) == (width, batch):
        return t.t().contiguous()
    if t.ndim == 1 and int(t.numel()) == batch * width:
        return t.reshape(batch, width).contiguous()
    if t.ndim == 2 and t.shape[0] == 1 and int(t.shape[1]) == batch * width:
        return t.reshape(batch, width).contiguous()
    if t.ndim == 2 and t.shape[1] == 1 and int(t.shape[0]) == batch * width:
        return t.reshape(batch, width).contiguous()
    raise ValueError(f"{name}: got {tuple(t.shape)}, expected ({batch}, {width})")


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + float(eps))
    y = x_float * rms
    return (y * weight.float()).to(dtype=x.dtype)


def _rms_norm_gated_grouped(
    x: torch.Tensor,
    *,
    weight: torch.Tensor,
    z: torch.Tensor,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
) -> torch.Tensor:
    b, d = x.shape
    group_size = int(group_size)
    if d % group_size != 0:
        raise ValueError(f"group_size={group_size} must divide d={d}")
    g = d // group_size

    x_float = x.float()
    z_float = z.float()
    if not norm_before_gate:
        x_float = x_float * F.silu(z_float)

    xg = x_float.reshape(b, g, group_size)
    rms = torch.rsqrt(xg.pow(2).mean(dim=-1, keepdim=True) + float(eps))
    yg = xg * rms
    y = yg.reshape(b, d) * weight.float()

    if norm_before_gate:
        y = y * F.silu(z_float)
    return y.to(dtype=x.dtype)


@dataclass(frozen=True)
class ClusterRuntime:
    cluster_zmq_object: object
    node_ips: list[str]
    cpu_gpu_select: list[bool]
    node_percentages: list[float]
    backend_select: list[str]
    weight_cache_mode: str = "save"  # "save" (distribute/overwrite shards) or "load" (use existing shards)


class ClusterMatmulCache:
    def __init__(self, rt: ClusterRuntime):
        cm = _import_cluster_matrix_v1()
        self._cluster_matrix = cm.cluster_matrix
        self._rt = rt
        self._weight_cache: dict[tuple[str, int], object] = {}

    def get_weight(self, pt_path: str, *, split_dim: int, matrix_name: Optional[str] = None):
        key = (pt_path, int(split_dim))
        cached = self._weight_cache.get(key)
        if cached is not None:
            return cached

        base_name = os.path.basename(pt_path)
        derived = base_name.split(".pt")[0]
        matrix_name = matrix_name or derived

        mode = str(self._rt.weight_cache_mode).lower().strip()
        if mode not in ("save", "load"):
            raise ValueError(f"weight_cache_mode must be 'save' or 'load', got {self._rt.weight_cache_mode!r}")

        def _build(auto_set_up: list[object]):
            return self._cluster_matrix(
                matrix_file_path=pt_path,
                cluster_zmq_object=self._rt.cluster_zmq_object,
                CPU_GPU_select_list=self._rt.cpu_gpu_select,
                node_percentages=self._rt.node_percentages,
                back_end_select_list=self._rt.backend_select,
                split_matrix=True,
                dim=int(split_dim),
                auto_set_up=auto_set_up,
                matrix_name=matrix_name,
            )

        if mode == "save":
            # Force regeneration/distribution of shards (fixes stale/corrupt shard files and ensures
            # worker nodes have the shards on disk).
            w = _build([1, "save"])
        else:
            # Use existing shards if already distributed to all nodes.
            try:
                w = _build([1, "load"])
            except Exception:
                w = _build([1, "save"])

        self._weight_cache[key] = w
        return w

    def matmul(
        self,
        x_2d: torch.Tensor,  # [B, in]
        *,
        weight_pt_path: str,  # [out, in] in .pt
        out_features: int,
        transpose_weight: bool,
        op_name: str,
    ) -> torch.Tensor:
        x_2d = x_2d.detach().to("cpu").contiguous()
        b, in_features = x_2d.shape
        x_cm = self._cluster_matrix(
            matrix_file_path=x_2d,
            cluster_zmq_object=self._rt.cluster_zmq_object,
            CPU_GPU_select_list=self._rt.cpu_gpu_select,
            node_percentages=self._rt.node_percentages,
            back_end_select_list=self._rt.backend_select,
            split_matrix=False,
            dim=1,
            auto_set_up=[1, "save"],
            # Reuse filenames to avoid unbounded growth in `matrix_shards/`.
            matrix_name=f"{op_name}_x",
        )
        w_cm = self.get_weight(weight_pt_path, split_dim=0)
        out = x_cm.cluster_shard_operation(w_cm, False, bool(transpose_weight), True, operation="mul")
        return _as_batch_first_2d(out, int(b), int(out_features), name=op_name)


class MambaCodestral7BCluster:
    def __init__(self, model_dir: str, rt: ClusterRuntime):
        self.model_dir = model_dir
        self.model_matrices = os.path.join(model_dir, "model_matrices")
        if not os.path.isdir(self.model_matrices):
            raise FileNotFoundError(f"Missing model_matrices folder at {self.model_matrices}")

        def _ensure_block_norm_weights(num_layers: int) -> None:
            """
            Ensure `layer_{i}_block_norm_weight.pt` exists for all layers by extracting missing
            tensors from safetensors shards in a single pass (fast path).
            """
            num_layers = int(num_layers)
            missing_layers = [
                i
                for i in range(num_layers)
                if not os.path.exists(os.path.join(self.model_matrices, f"layer_{i}_block_norm_weight.pt"))
            ]
            if not missing_layers:
                return

            try:
                from safetensors.torch import safe_open  # type: ignore
            except Exception:
                # Defer to per-file fallback in `_load_matrix`.
                return

            index_path = os.path.join(self.model_dir, "model.safetensors.index.json")
            if not os.path.exists(index_path):
                # Defer to per-file fallback in `_load_matrix` (it can try consolidated/model).
                return

            try:
                idx = json.load(open(index_path, "r"))
                weight_map = idx.get("weight_map", {})
            except Exception:
                return

            by_shard: dict[str, list[int]] = {}
            for layer_idx in missing_layers:
                key = f"backbone.layers.{layer_idx}.norm.weight"
                shard = weight_map.get(key)
                if isinstance(shard, str) and shard:
                    by_shard.setdefault(shard, []).append(layer_idx)

            for shard, layers in by_shard.items():
                st_path = os.path.join(self.model_dir, shard)
                if not os.path.exists(st_path):
                    continue
                with safe_open(st_path, framework="pt", device="cpu") as f:
                    for layer_idx in layers:
                        key = f"backbone.layers.{layer_idx}.norm.weight"
                        if key not in f.keys():
                            continue
                        out_path = os.path.join(self.model_matrices, f"layer_{layer_idx}_block_norm_weight.pt")
                        if os.path.exists(out_path):
                            continue
                        t = f.get_tensor(key).to(dtype=torch.float16).contiguous()
                        torch.save(t, out_path)

        def _load_matrix(filename: str) -> torch.Tensor:
            path = os.path.join(self.model_matrices, filename)
            try:
                return torch.load(path, map_location="cpu")
            except FileNotFoundError as e:
                # Auto-extract missing per-layer block RMSNorm weights (needed for correct math).
                m = _BLOCK_NORM_FILE_RE.match(filename)
                if m:
                    layer_idx = int(m.group(1))
                    param = str(m.group(2))
                    key = f"backbone.layers.{layer_idx}.norm.{param}"
                    try:
                        t = _try_extract_from_safetensors(self.model_dir, key).to(dtype=torch.float16).contiguous()
                        torch.save(t, path)
                        return t
                    except Exception:
                        # Fall through to the structured error below.
                        pass
                raise FileNotFoundError(
                    f"Missing extracted weight '{filename}' in {self.model_matrices}. "
                    "Re-run the extractor in `mamba_ssm/utils/MambaWeightExtractor.py` to (re)generate model_matrices."
                ) from e

        cfg_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing config.json at {cfg_path}")
        import json

        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        if cfg.get("model_type") != "mamba2":
            raise ValueError(f"Expected model_type=mamba2, got {cfg.get('model_type')}")

        self.hidden_size = int(cfg["hidden_size"])
        self.num_layers = int(cfg["num_hidden_layers"])
        self.expand = int(cfg["expand"])
        self.d_inner = self.hidden_size * self.expand
        self.d_state = int(cfg["state_size"])
        self.d_conv = int(cfg["conv_kernel"])
        self.ngroups = int(cfg["n_groups"])
        self.nheads = int(cfg["num_heads"])
        self.headdim = int(cfg["head_dim"])
        self.eps = float(cfg.get("layer_norm_epsilon", 1e-5))
        self.norm_before_gate = bool(cfg.get("norm_before_gate", True))
        self.residual_in_fp32 = bool(cfg.get("residual_in_fp32", False))
        if self.d_inner != self.nheads * self.headdim:
            raise ValueError(f"hidden_size*expand={self.d_inner} must equal num_heads*head_dim={self.nheads*self.headdim}")
        self.group_size = self.d_inner // self.ngroups

        _ensure_block_norm_weights(self.num_layers)

        self._matmul = ClusterMatmulCache(rt)

        self.embed_tokens = _load_matrix("embed_tokens_weight.pt")
        self.model_norm_weight = _load_matrix("model_norm_weight.pt")

        self._layer_small: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            self._layer_small.append(
                {
                    # Block RMSNorm sits outside the mixer; required for correct forward math.
                    "block_norm_w": _load_matrix(f"layer_{layer_idx}_block_norm_weight.pt").contiguous(),
                    "conv_w": _load_matrix(f"layer_{layer_idx}_conv1d_weight.pt")
                    .reshape(-1, self.d_conv)
                    .contiguous(),
                    "conv_b": _load_matrix(f"layer_{layer_idx}_conv1d_bias.pt").contiguous(),
                    "dt_bias": _load_matrix(f"layer_{layer_idx}_dt_bias.pt").contiguous(),
                    "A_log": _load_matrix(f"layer_{layer_idx}_A_log.pt").contiguous(),
                    "D": _load_matrix(f"layer_{layer_idx}_D.pt").contiguous(),
                    "norm_w": _load_matrix(f"layer_{layer_idx}_norm_weight.pt").contiguous(),
                }
            )

        self._in_proj_paths = [os.path.join(self.model_matrices, f"layer_{i}_in_proj_weight.pt") for i in range(self.num_layers)]
        self._out_proj_paths = [os.path.join(self.model_matrices, f"layer_{i}_out_proj_weight.pt") for i in range(self.num_layers)]
        self._lm_head_path = os.path.join(self.model_matrices, "lm_head_weight.pt")

    def allocate_cache(self, batch_size: int):
        batch_size = int(batch_size)
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state  # 8192 + 2048 = 10240
        conv_states = [torch.zeros(batch_size, conv_dim, self.d_conv, dtype=torch.float16) for _ in range(self.num_layers)]
        ssm_states = [torch.zeros(batch_size, self.nheads, self.headdim, self.d_state, dtype=torch.float16) for _ in range(self.num_layers)]
        return conv_states, ssm_states

    def _mixer_step(self, hidden: torch.Tensor, layer_idx: int, conv_states, ssm_states) -> torch.Tensor:
        """One Mamba2 mixer step for a single token (B, D) -> (B, D)."""
        b = int(hidden.shape[0])

        # in_proj via cluster: (B, hidden) @ (hidden, d_in_proj)
        in_proj = self._matmul.matmul(
            hidden,
            weight_pt_path=self._in_proj_paths[layer_idx],
            out_features=(2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads),
            transpose_weight=True,
            op_name=f"layer{layer_idx}_in_proj",
        )

        z = in_proj[:, : self.d_inner]
        xBC = in_proj[:, self.d_inner : self.d_inner + (self.d_inner + 2 * self.ngroups * self.d_state)]
        dt = in_proj[:, -self.nheads :]

        # Causal depthwise conv update (local CPU torch ops)
        conv_state = conv_states[layer_idx]
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = xBC
        conv_states[layer_idx] = conv_state

        s = self._layer_small[layer_idx]
        xBC = (conv_state * s["conv_w"].unsqueeze(0)).sum(dim=-1)
        xBC = xBC + s["conv_b"].unsqueeze(0)
        xBC = F.silu(xBC)

        x = xBC[:, : self.d_inner].reshape(b, self.nheads, self.headdim)
        Bv = xBC[:, self.d_inner : self.d_inner + self.ngroups * self.d_state].reshape(b, self.ngroups, self.d_state)
        Cv = xBC[:, self.d_inner + self.ngroups * self.d_state :].reshape(b, self.ngroups, self.d_state)

        heads_per_group = self.nheads // self.ngroups
        B_h = Bv.repeat_interleave(heads_per_group, dim=1)  # (B, H, N)
        C_h = Cv.repeat_interleave(heads_per_group, dim=1)  # (B, H, N)

        # Match reference behavior: do softplus in fp32, then cast.
        dt = F.softplus(dt.float() + s["dt_bias"].float().unsqueeze(0)).to(dtype=dt.dtype)
        A = -torch.exp(s["A_log"].float())  # (H,)
        D = s["D"]

        # SSM update (local CPU torch ops)
        ssm_prev = ssm_states[layer_idx]
        dA = torch.exp(dt.float() * A.unsqueeze(0))  # (B, H) float32
        ssm = ssm_prev.float() * dA[:, :, None, None]
        ssm = ssm + (dt[:, :, None, None].float() * x.float()[:, :, :, None]) * B_h.float()[:, :, None, :]
        ssm_states[layer_idx] = ssm.to(dtype=ssm_prev.dtype)

        # Match reference behavior: compute y in the model hidden dtype.
        ssm_y = ssm_states[layer_idx].to(dtype=x.dtype)
        y = (ssm_y * C_h[:, :, None, :]).sum(dim=-1)
        y = y + D.to(dtype=x.dtype).unsqueeze(0).unsqueeze(-1) * x  # (B, H, P)
        y = y.reshape(b, self.d_inner)

        # Multiply "gate" branch and apply extra normalization layer inside the mixer
        y = _rms_norm_gated_grouped(
            y,
            weight=s["norm_w"],
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
        )

        # out_proj via cluster: (B, d_inner) @ (d_inner, hidden)
        out = self._matmul.matmul(
            y,
            weight_pt_path=self._out_proj_paths[layer_idx],
            out_features=self.hidden_size,
            transpose_weight=True,
            op_name=f"layer{layer_idx}_out_proj",
        )
        return out

    def step_token(self, token_ids: torch.Tensor, conv_states, ssm_states) -> torch.Tensor:
        if token_ids.ndim != 1:
            raise ValueError(f"token_ids must be 1D [B], got {tuple(token_ids.shape)}")
        b = int(token_ids.shape[0])

        hidden = self.embed_tokens[token_ids].to(dtype=torch.float16)  # (B, hidden)
        residual: Optional[torch.Tensor] = None

        # Mamba blocks use: Add -> RMSNorm -> Mixer (no post-add inside the block).
        for layer_idx in range(self.num_layers):
            residual = hidden if residual is None else (hidden + residual)
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

            s = self._layer_small[layer_idx]
            hidden_norm = _rms_norm(residual.to(dtype=s["block_norm_w"].dtype), s["block_norm_w"], self.eps)
            hidden = self._mixer_step(hidden_norm, layer_idx, conv_states, ssm_states)

        # Final add + norm
        residual = hidden if residual is None else (hidden + residual)
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden = _rms_norm(residual.to(dtype=self.model_norm_weight.dtype), self.model_norm_weight, self.eps)
        logits = self._matmul.matmul(
            hidden,
            weight_pt_path=self._lm_head_path,
            out_features=int(self.embed_tokens.shape[0]),
            transpose_weight=True,
            op_name="lm_head",
        )
        return logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=os.path.join(_repo_root(), "llm_models", "Mamba-Codestral-7B-v0.1"))
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--backend", default=os.environ.get("CLUSTER_BACKEND", "llama"), choices=["torch", "llama", "opencl"])
    p.add_argument("--use-gpu", action="store_true", default=False)
    p.add_argument("--node-ips", default=os.environ.get("CLUSTER_NODE_IPS", os.environ.get("HEAD_NODE_IP", "192.168.2.100")))
    p.add_argument(
        "--backend-select-list",
        default=os.environ.get("CLUSTER_BACKEND_SELECT_LIST", ""),
        help="Comma-separated backend per node slot (overrides --backend). Example: llama,llama,torch,torch",
    )
    p.add_argument(
        "--cpu-gpu-select-list",
        default=os.environ.get("CLUSTER_CPU_GPU_SELECT_LIST", ""),
        help="Comma-separated bool/int per node slot (overrides --use-gpu). Example: 1,1,0,0",
    )
    p.add_argument(
        "--node-percentages",
        default=os.environ.get("CLUSTER_NODE_PERCENTAGES", ""),
        help="Comma-separated percentages per node slot (must match --node-ips length). Default: uniform.",
    )
    p.add_argument(
        "--weight-cache-mode",
        default=os.environ.get("CLUSTER_WEIGHT_CACHE_MODE", "save"),
        choices=["save", "load"],
        help="Use 'save' once to (re)generate and distribute shard files to all nodes; 'load' assumes shards exist on all nodes.",
    )
    p.add_argument(
        "--precache",
        action="store_true",
        help="Pre-generate/distribute all layer weight shards before running the prompt (recommended for multi-node).",
    )
    p.add_argument("--precache-only", action="store_true", help="Exit after precache completes.")
    p.add_argument("--precache-start-layer", type=int, default=0)
    p.add_argument("--precache-end-layer", type=int, default=-1, help="Inclusive. -1 means last layer.")
    args = p.parse_args()

    # Make cluster_matrix_v1's on-disk shard paths consistent when running from repo root.
    cluster_dir = os.path.join(_repo_root(), "cluster_matrix")
    os.environ.setdefault("LOCAL_DISK_FOLDER", os.path.join(cluster_dir, "matrix_shards") + os.sep)
    os.environ.setdefault("REMOTE_DISK_FOLDER", "matrix_shards" + os.sep)

    def _parse_csv_list(s: str) -> list[str]:
        parts = []
        for raw in str(s).replace(" ", ",").split(","):
            v = raw.strip()
            if v:
                parts.append(v)
        return parts

    node_ips = _parse_csv_list(args.node_ips)
    if not node_ips:
        raise ValueError("--node-ips cannot be empty")
    nslots = len(node_ips)

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

    if str(args.cpu_gpu_select_list).strip():
        cpu_gpu_select = _parse_bool_list(args.cpu_gpu_select_list)
        if len(cpu_gpu_select) != nslots:
            raise ValueError(f"--cpu-gpu-select-list length {len(cpu_gpu_select)} must match --node-ips length {nslots}")
    else:
        cpu_gpu_select = [bool(args.use_gpu)] * nslots

    if str(args.backend_select_list).strip():
        backend_select = _parse_csv_list(args.backend_select_list)
        if len(backend_select) != nslots:
            raise ValueError(f"--backend-select-list length {len(backend_select)} must match --node-ips length {nslots}")
    else:
        backend_select = [str(args.backend)] * nslots

    if str(args.node_percentages).strip():
        node_percentages = [float(x) for x in _parse_csv_list(args.node_percentages)]
        if len(node_percentages) != nslots:
            raise ValueError(f"--node-percentages length {len(node_percentages)} must match --node-ips length {nslots}")
        s = float(sum(node_percentages))
        if s <= 0:
            raise ValueError("--node-percentages must sum to > 0")
        node_percentages = [float(x) / s for x in node_percentages]
    else:
        node_percentages = [1.0 / nslots] * nslots

    cm = _import_cluster_matrix_v1()
    cluster_zmq_object = cm.cluster_zmq(node_ips)
    rt = ClusterRuntime(
        cluster_zmq_object=cluster_zmq_object,
        node_ips=node_ips,
        cpu_gpu_select=cpu_gpu_select,
        node_percentages=node_percentages,
        backend_select=backend_select,
        weight_cache_mode=str(args.weight_cache_mode),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = MambaCodestral7BCluster(args.model_dir, rt)

    if bool(args.precache):
        start = int(args.precache_start_layer)
        end = int(args.precache_end_layer)
        if end < 0:
            end = int(model.num_layers) - 1
        if start < 0 or start >= model.num_layers:
            raise ValueError(f"--precache-start-layer out of range: {start}")
        if end < start or end >= model.num_layers:
            raise ValueError(f"--precache-end-layer out of range: {end}")
        if str(rt.weight_cache_mode).lower() != "save":
            raise ValueError("--precache requires --weight-cache-mode save")
        for layer_idx in range(start, end + 1):
            model._matmul.get_weight(model._in_proj_paths[layer_idx], split_dim=0)
            model._matmul.get_weight(model._out_proj_paths[layer_idx], split_dim=0)
        model._matmul.get_weight(model._lm_head_path, split_dim=0)
        if bool(args.precache_only):
            return

    prompt = input("Enter prompt: ")
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(torch.long)
    if input_ids.shape[0] != 1:
        raise ValueError("This runner expects batch_size=1 for now")
    batch = int(input_ids.shape[0])

    conv_states, ssm_states = model.allocate_cache(batch)

    # Prefill (updates states for all prompt tokens, keep only last logits)
    for t in range(int(input_ids.shape[1])):
        logits = model.step_token(input_ids[:, t], conv_states, ssm_states)

    generated = input_ids.clone()
    for _ in range(int(args.max_new_tokens)):
        next_id = int(torch.argmax(logits[0]).item())
        next_tok = torch.tensor([next_id], dtype=torch.long)
        generated = torch.cat([generated, next_tok.unsqueeze(0)], dim=1)
        logits = model.step_token(next_tok, conv_states, ssm_states)

    out_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(prompt, "\n")
    print(out_text, "\n")


if __name__ == "__main__":
    main()
