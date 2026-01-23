from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch


def _ensure_cluster_matrix_v1_on_path() -> None:
    """
    `cluster_matrix_v1.py` lives in `<repo_root>/cluster_matrix/` (not a Python package).
    Add that folder to `sys.path` so `import cluster_matrix_v1` works.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cluster_dir = os.path.join(repo_root, "cluster_matrix")
    if cluster_dir not in sys.path:
        sys.path.insert(0, cluster_dir)


_cluster_matrix_import_lock = threading.Lock()
_cluster_matrix_mod = None


def _import_cluster_matrix_v1():
    global _cluster_matrix_mod
    if _cluster_matrix_mod is not None:
        return _cluster_matrix_mod
    with _cluster_matrix_import_lock:
        if _cluster_matrix_mod is not None:
            return _cluster_matrix_mod
        _ensure_cluster_matrix_v1_on_path()
        import cluster_matrix_v1  # type: ignore

        _cluster_matrix_mod = cluster_matrix_v1
        return _cluster_matrix_mod


@dataclass(frozen=True)
class ClusterMatrixConfig:
    cluster_zmq_object: Any
    CPU_GPU_select_list: list[Any]
    node_percentages: list[float]
    back_end_select_list: list[str]
    # Use "save" by default so the first run reliably generates/distributes correct shard files.
    # Set to (1, "load") on subsequent runs if shards already exist on all nodes.
    weight_auto_setup: tuple[int, str] = (1, "save")  # (system_id, "load"|"save")
    input_auto_setup: tuple[int, str] = (1, "save")
    name_prefix: str = "mamba"


_GLOBAL_CLUSTER_CONFIG: Optional[ClusterMatrixConfig] = None


def set_global_cluster_config(cfg: Optional[ClusterMatrixConfig]) -> None:
    global _GLOBAL_CLUSTER_CONFIG
    _GLOBAL_CLUSTER_CONFIG = cfg


def get_global_cluster_config() -> Optional[ClusterMatrixConfig]:
    return _GLOBAL_CLUSTER_CONFIG


_name_counter_lock = threading.Lock()
_name_counter = 0


def _unique_matrix_name(prefix: str) -> str:
    global _name_counter
    with _name_counter_lock:
        _name_counter += 1
        n = _name_counter
    return f"{prefix}_{int(time.time_ns())}_{n}"


def _coerce_2d_shape(t: torch.Tensor, rows: int, cols: int, *, name: str) -> torch.Tensor:
    """
    Cluster backends sometimes return transposed/flattened 2D tensors. Coerce into [rows, cols].
    """
    rows = int(rows)
    cols = int(cols)
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name}: expected torch.Tensor, got {type(t)}")

    # cluster_matrix_v1 often returns 3D/4D tensors due to its internal binary layout.
    # Common layout: (1, 1, rows, cols). Normalize to 2D first.
    if t.ndim == 4:
        if int(t.shape[0]) == 1 and int(t.shape[1]) == 1:
            t = t[0, 0]
        else:
            t = t.reshape(-1, int(t.shape[-1]))
    if t.ndim == 3:
        if int(t.shape[0]) == 1:
            t = t[0]
        else:
            t = t.reshape(-1, int(t.shape[-1]))

    if t.ndim == 1:
        if int(t.numel()) != rows * cols:
            raise ValueError(f"{name}: got 1D numel={int(t.numel())}, expected {rows*cols}")
        return t.reshape(rows, cols).contiguous()

    if t.ndim != 2:
        raise ValueError(f"{name}: expected 1D/2D, got shape={tuple(t.shape)}")

    if tuple(t.shape) == (rows, cols):
        return t.contiguous()
    if tuple(t.shape) == (cols, rows):
        return t.t().contiguous()

    if t.shape[0] == 1 and int(t.shape[1]) == rows * cols:
        return t.reshape(rows, cols).contiguous()
    if t.shape[1] == 1 and int(t.shape[0]) == rows * cols:
        return t.reshape(rows, cols).contiguous()

    raise ValueError(f"{name}: got {tuple(t.shape)}, cannot coerce to ({rows}, {cols})")


_WEIGHT_CLUSTER_CACHE: dict[tuple[int, str], Any] = {}
_WEIGHT_CLUSTER_CACHE_LOCK = threading.Lock()


def get_or_create_weight_cluster(
    weight: torch.Tensor,
    *,
    cfg: ClusterMatrixConfig,
    matrix_name: str,
    split_dim: int,
) -> Any:
    """
    Create or reuse a `cluster_matrix` for a static weight tensor.

    We try `auto_set_up=cfg.weight_auto_setup` first. If that fails and the mode is "load",
    we fall back to "save" once (using the provided `weight` tensor).
    """
    mod = _import_cluster_matrix_v1()
    cluster_matrix = mod.cluster_matrix

    cache_key = (id(cfg.cluster_zmq_object), matrix_name)
    with _WEIGHT_CLUSTER_CACHE_LOCK:
        cached = _WEIGHT_CLUSTER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    system_id, mode = cfg.weight_auto_setup
    auto_setup = [int(system_id), str(mode)]

    def _build(auto_set_up):
        return cluster_matrix(
            matrix_file_path=weight.detach().to("cpu").contiguous(),
            cluster_zmq_object=cfg.cluster_zmq_object,
            CPU_GPU_select_list=cfg.CPU_GPU_select_list,
            node_percentages=cfg.node_percentages,
            back_end_select_list=cfg.back_end_select_list,
            split_matrix=True,
            dim=int(split_dim),
            auto_set_up=auto_set_up,
            matrix_name=matrix_name,
        )

    try:
        cm = _build(auto_setup)
    except Exception:
        # Common case: shards not present yet. Create them once.
        if str(mode) != "load":
            raise
        cm = _build([int(system_id), "save"])

    with _WEIGHT_CLUSTER_CACHE_LOCK:
        _WEIGHT_CLUSTER_CACHE[cache_key] = cm
    return cm


def cluster_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    cfg: ClusterMatrixConfig,
    weight_name: str,
) -> torch.Tensor:
    """
    Distributed replacement for `F.linear(x, weight, bias)`.

    - `x`: [..., in_features]
    - `weight`: [out_features, in_features]
    - Returns: [..., out_features]
    """
    if x.device.type != "cpu":
        raise ValueError("cluster_linear requires CPU tensors (cluster backend handles device placement).")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        x = x.float()
    x_cpu = x.detach().contiguous()
    weight_cpu = weight.detach().to("cpu").contiguous()

    in_features = int(weight_cpu.shape[1])
    out_features = int(weight_cpu.shape[0])
    if x_cpu.shape[-1] != in_features:
        raise ValueError(f"cluster_linear: x.shape[-1]={int(x_cpu.shape[-1])} != in_features={in_features}")

    leading_shape = tuple(x_cpu.shape[:-1])
    x_2d = x_cpu.reshape(-1, in_features).contiguous()

    mod = _import_cluster_matrix_v1()
    cluster_matrix = mod.cluster_matrix

    x_cm = cluster_matrix(
        matrix_file_path=x_2d,
        cluster_zmq_object=cfg.cluster_zmq_object,
        CPU_GPU_select_list=cfg.CPU_GPU_select_list,
        node_percentages=cfg.node_percentages,
        back_end_select_list=cfg.back_end_select_list,
        split_matrix=False,
        dim=1,  # output shards join on dim=1
        auto_set_up=[int(cfg.input_auto_setup[0]), str(cfg.input_auto_setup[1])],
        matrix_name=_unique_matrix_name(f"{cfg.name_prefix}_x"),
    )

    w_cm = get_or_create_weight_cluster(
        weight_cpu,
        cfg=cfg,
        matrix_name=weight_name,
        split_dim=0,  # split rows (out_features)
    )

    out = x_cm.cluster_shard_operation(w_cm, False, True, True)
    out_2d = _coerce_2d_shape(out, x_2d.shape[0], out_features, name=f"{weight_name}:matmul")
    if bias is not None:
        out_2d = out_2d + bias.detach().to("cpu").reshape(1, out_features)
    out_full = out_2d.reshape(*leading_shape, out_features)
    return out_full.to(dtype=x.dtype)
