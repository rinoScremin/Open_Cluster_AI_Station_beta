# Copyright (c) 2024, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # type: ignore
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined  # type: ignore
except Exception:
    mamba_chunk_scan_combined = None
    mamba_split_conv1d_scan_combined = None

from mamba_ssm.utils.cluster_matrix import ClusterMatrixConfig, cluster_linear, get_global_cluster_config


if RMSNormGated is None:
    RMSNormGated = None  # type: ignore[assignment]


class _RMSNormGatedRef(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, norm_before_gate: bool = True, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = float(eps)
        self.norm_before_gate = bool(norm_before_gate)

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_float = x.float()
        if z is not None and not self.norm_before_gate:
            x_float = x_float * F.silu(z.float())
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = x_float * rms * self.weight.float()
        if z is not None and self.norm_before_gate:
            y = y * F.silu(z.float())
        return y.to(dtype=x.dtype)


class Mamba2Simple(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        cluster_config: Optional[ClusterMatrixConfig] = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.cluster_config = cluster_config if cluster_config is not None else get_global_cluster_config()
        if self.cluster_config is not None:
            # Disable fused CUDA/Triton paths when using the cluster backend.
            self.use_mem_eff_path = False
        if mamba_split_conv1d_scan_combined is None:
            self.use_mem_eff_path = False

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        # Use a device-agnostic (non-Triton) implementation to support CPU/heterogeneous backends.
        self.norm = _RMSNormGatedRef(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def _cluster_weight_name(self, suffix: str) -> str:
        layer = -1 if self.layer_idx is None else int(self.layer_idx)
        prefix = self.cluster_config.name_prefix if self.cluster_config is not None else "mamba2"
        return f"{prefix}_layer{layer}_{suffix}"

    def _ssm_scan_naive(
        self,
        *,
        x: torch.Tensor,  # (B, L, H, P)
        dt: torch.Tensor,  # (B, L, H)
        A: torch.Tensor,  # (H,)
        B: torch.Tensor,  # (B, L, G, N)
        C: torch.Tensor,  # (B, L, G, N)
        D: torch.Tensor,  # (H,)
        initial_states: Optional[torch.Tensor],  # (B, H, P, N) or None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seqlen, nheads, headdim = x.shape
        d_state = int(B.shape[-1])

        if self.ngroups == 1:
            B_h = B.expand(batch, seqlen, nheads, d_state)
            C_h = C.expand(batch, seqlen, nheads, d_state)
        else:
            if nheads % self.ngroups != 0:
                raise ValueError(f"ngroups={self.ngroups} must divide nheads={nheads}")
            repeat_factor = nheads // self.ngroups
            B_h = B.repeat_interleave(repeat_factor, dim=2)
            C_h = C.repeat_interleave(repeat_factor, dim=2)

        state = (
            initial_states.to(dtype=x.dtype)
            if initial_states is not None
            else x.new_zeros((batch, nheads, headdim, d_state))
        )
        ys = []
        A = A.to(dtype=x.dtype)
        D = D.to(dtype=x.dtype)
        for t in range(seqlen):
            dt_t = dt[:, t, :]  # (B, H)
            x_t = x[:, t, :, :]  # (B, H, P)
            B_t = B_h[:, t, :, :]  # (B, H, N)
            C_t = C_h[:, t, :, :]  # (B, H, N)
            dA = torch.exp(dt_t * A.reshape(1, -1))  # (B, H)
            state = state * dA[:, :, None, None] + (dt_t[:, :, None, None] * x_t[:, :, :, None]) * B_t[:, :, None, :]
            y_t = (state * C_t[:, :, None, :]).sum(dim=-1) + D.reshape(1, -1, 1) * x_t
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, H, P)
        return y, state

    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        if self.cluster_config is not None:
            zxbcdt = cluster_linear(
                u,
                self.in_proj.weight,
                self.in_proj.bias,
                cfg=self.cluster_config,
                weight_name=self._cluster_weight_name("in_proj_weight"),
            )
        else:
            zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log.float())  # (nheads,)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path and mamba_split_conv1d_scan_combined is not None and u.is_cuda:
            # Fully fused path
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
        else:
            z, xBC, dt = torch.split(
                zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
            )
            dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
            assert self.activation in ["silu", "swish"]

            # 1D Convolution
            if self.cluster_config is not None or causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                xBC = xBC[:, :seqlen, :]
            else:
                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)

            # Split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            x_hp = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
            B_g = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
            C_g = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)
            if mamba_chunk_scan_combined is not None and u.is_cuda and self.cluster_config is None:
                y = mamba_chunk_scan_combined(
                    x_hp,
                    dt,
                    A,
                    B_g,
                    C_g,
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=None,
                    seq_idx=seq_idx,
                    initial_states=initial_states,
                    **dt_limit_kwargs,
                )
            else:
                y, _ = self._ssm_scan_naive(
                    x=x_hp,
                    dt=dt,
                    A=A,
                    B=B_g,
                    C=C_g,
                    D=self.D,
                    initial_states=initial_states,
                )
            y = rearrange(y, "b l h p -> b l (h p)")

            # Multiply "gate" branch and apply extra normalization layer
            y = self.norm(y, z)
            if self.cluster_config is not None:
                out = cluster_linear(
                    y,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    cfg=self.cluster_config,
                    weight_name=self._cluster_weight_name("out_proj_weight"),
                )
            else:
                out = self.out_proj(y)
        return out
