import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from cluster_matrix_v1 import cluster_matrix, cluster_zmq, check_combined_result_values
import time
import math


def test_flash_attn_2d_simple():
    print("\n================= 2D FLASH ATTENTION TEST =================")

    torch.manual_seed(0)

    # ---- VERY SMALL DIMENSIONS ----
    T = 512        # sequence length
    D = 2048       # head_dim
    scale = 1.0 / math.sqrt(D)

    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- CREATE Q K V (2D) ----
    Q = torch.randn(T, D, device=device, dtype=dtype)
    K = torch.randn(T, D, device=device, dtype=dtype)
    V = torch.randn(T, D, device=device, dtype=dtype)

    # ---- TORCH REFERENCE ----
    scores = (Q @ K.T) * scale
    attn = torch.softmax(scores, dim=-1)
    torch_out = attn @ V

    torch.save(torch_out, "torch_flash_2d_ref.pt")

    print("Torch output shape:", torch_out.shape)

    # ---- CLUSTER SETUP ----
    IP_list = ["192.168.2.100"]
    percentages = [1.0]
    CPU_GPU_select_list = [True]
    backend_select_list = ["llama"]

    cluster_zmq_obj = cluster_zmq(IP_list)

    cluster_Q = cluster_matrix(
        Q,
        cluster_zmq_object=cluster_zmq_obj,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,
        split_matrix=False,
        auto_set_up=[1, "save"],
        matrix_name="q_2d_test"
    )

    cluster_K = cluster_matrix(
        K,
        cluster_zmq_object=cluster_zmq_obj,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,
        split_matrix=False,
        auto_set_up=[1, "save"],
        matrix_name="k_2d_test"
    )

    cluster_V = cluster_matrix(
        V,
        cluster_zmq_object=cluster_zmq_obj,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,
        split_matrix=False,
        auto_set_up=[1, "save"],
        matrix_name="v_2d_test"
    )

    # ---- CLUSTER FLASH ATTENTION ----
    cluster_out = cluster_Q.cluster_flash_attn(
        cluster_matrixK=cluster_K,
        cluster_matrixV=cluster_V,
        TransposeQ=False,
        TransposeK=False,
        TransposeV=False,
        send_back_result=True,
        operation="flash_attn_ext",
        extra_param_value=scale
    )

    # ---- VERIFY ----
    print("\n🔎 Verifying 2D FlashAttention")
    check_combined_result_values(
        "torch_flash_2d_ref.pt",
        cluster_out
    )

    print("===========================================================\n")


def require_allclose(ref, combined, label, rtol=1e-2, atol=1e-2):
    if not torch.is_tensor(ref):
        ref = torch.as_tensor(ref)
    if not torch.is_tensor(combined):
        combined = torch.as_tensor(combined)
    if not torch.allclose(ref, combined, rtol=rtol, atol=atol):
        diff = (ref - combined).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        raise AssertionError(
            f"{label} allclose failed (rtol={rtol}, atol={atol}) "
            f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}"
        )


def main():
    # ---------------- CONFIG ----------------
    B = 1
    T = 256
    hidden_size = 512
    n_q_heads = 16
    n_kv_heads = 2
    head_dim = hidden_size // n_q_heads

    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- FAKE QKV ----------------
    torch.manual_seed(0)

    Q = torch.randn(B, n_q_heads, T, head_dim, device=device, dtype=dtype)
    K = torch.randn(B, n_kv_heads, T, head_dim, device=device, dtype=dtype)
    V = torch.randn(B, n_kv_heads, T, head_dim, device=device, dtype=dtype)

    # ---------------- GQA EXPANSION ----------------
    kv_repeat = n_q_heads // n_kv_heads
    K = K.repeat_interleave(kv_repeat, dim=1)
    V = V.repeat_interleave(kv_repeat, dim=1)

    # ---------------- TORCH FLASH ATTENTION ----------------
    torch_out_4d = F.scaled_dot_product_attention(
        Q,
        K,
        V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False  # ✅ MUST match backend
    )


    # ---- Torch → 2D (MATCH CLUSTER LAYOUT) ----
    torch_out_2d = (
        torch_out_4d.contiguous()
        .squeeze(0)                  # [32, T, 64]
        .reshape(n_q_heads * T, head_dim)
    )  # [32*T, 64]

    torch.save(torch_out_4d, "torch_out_4d.pt")
    torch.save(torch_out_2d, "torch_out_2d.pt")

    # ---------------- CLUSTER SETUP ----------------
    IP_list = ["192.168.2.100","192.168.2.100","192.168.2.101","192.168.2.104"]
    percentages = [0.4,0.4,0.1,0.1]
    CPU_GPU_select_list = [True,True,True,True]
    backend_select_list = ["llama","llama","llama","torch"]
    scale = 1.0 / math.sqrt(head_dim)

    cluster_zmq_obj = cluster_zmq(IP_list)


    # Keep torch layout [B, H, T, D]; binary writer encodes as (batch, depth, rows, cols)
    Qc = Q#.contiguous()
    Kc = K#.contiguous()
    Vc = V#.contiguous()
    cluster_Q = cluster_matrix(
        Qc,
        cluster_zmq_object=cluster_zmq_obj,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,
        split_matrix=True,
        dim=1,
        auto_set_up=[1, "save"],
        matrix_name="q_test"
    )

    cluster_K = cluster_matrix(
        Kc,
        cluster_zmq_object=cluster_zmq_obj,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,
        split_matrix=True,
        dim=1,
        auto_set_up=[1, "save"],
        matrix_name="k_test"
    )
    
    cluster_V = cluster_matrix(
        Vc,
        cluster_zmq_object=cluster_zmq_obj,
        CPU_GPU_select_list=CPU_GPU_select_list,
        node_percentages=percentages,
        back_end_select_list=backend_select_list,
        split_matrix=True,
        dim=1,
        auto_set_up=[1, "save"],
        matrix_name="v_test"
    )

    # ---------------- CLUSTER FLASH ATTENTION ----------------
    cluster_out_2d = cluster_Q.cluster_flash_attn(
        cluster_matrixK=cluster_K,
        cluster_matrixV=cluster_V,
        TransposeQ=False,
        TransposeK=False,
        TransposeV=False,
        send_back_result=True,
        operation="flash_attn_ext",
        extra_param_value=scale
    )
    # cluster_out_2d: [32*T, 64]

    torch.save(cluster_out_2d, "cluster_out_2d.pt")

    # ---------------- RESHAPE CLUSTER → 4D ----------------
    cluster_out_4d = (
        cluster_out_2d
        .view(n_q_heads, T, head_dim)
        .unsqueeze(0)
    )  # [1, 32, T, 64]

    torch.save(cluster_out_4d, "cluster_out_4d.pt")

    #test_flash_attn_2d_simple()


    # ---------------- VERIFY ----------------
    print("🔎 Verifying 2D layout...")
    check_combined_result_values("torch_out_2d.pt", cluster_out_2d)
    require_allclose(torch_out_2d, cluster_out_2d, "FlashAttn 2D")

    print("🔎 Verifying 4D layout...")
    check_combined_result_values("torch_out_4d.pt", cluster_out_4d)
    require_allclose(torch_out_4d, cluster_out_4d, "FlashAttn 4D")

    print("✅ FlashAttention correctness VERIFIED (2D + 4D)")

    # ---------------- CAUSAL MASK TEST ----------------
    print("\n================= CAUSAL MASK FLASH ATTENTION TEST =================")
    q_len = T
    kv_len = K.shape[2]
    row_idx = torch.arange(q_len, device=device).unsqueeze(1)
    col_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    causal_mask = (col_idx > row_idx).to(torch.float32) * -1e9

    torch_out_causal_4d = F.scaled_dot_product_attention(
        Q,
        K,
        V,
        attn_mask=causal_mask,
        dropout_p=0.0,
        is_causal=False
    )

    torch_out_causal_2d = (
        torch_out_causal_4d.contiguous()
        .squeeze(0)
        .reshape(n_q_heads * T, head_dim)
    )

    torch.save(torch_out_causal_4d, "torch_out_causal_4d.pt")
    torch.save(torch_out_causal_2d, "torch_out_causal_2d.pt")

    cluster_out_causal_2d = cluster_Q.cluster_flash_attn(
        cluster_matrixK=cluster_K,
        cluster_matrixV=cluster_V,
        TransposeQ=False,
        TransposeK=False,
        TransposeV=False,
        send_back_result=True,
        operation="flash_attn_ext",
        extra_param_value=scale,
        mask=causal_mask
    )

    cluster_out_causal_4d = (
        cluster_out_causal_2d
        .view(n_q_heads, T, head_dim)
        .unsqueeze(0)
    )

    print("🔎 Verifying causal 2D layout...")
    check_combined_result_values("torch_out_causal_2d.pt", cluster_out_causal_2d)
    require_allclose(torch_out_causal_2d, cluster_out_causal_2d, "Causal FlashAttn 2D")

    print("🔎 Verifying causal 4D layout...")
    check_combined_result_values("torch_out_causal_4d.pt", cluster_out_causal_4d)
    require_allclose(torch_out_causal_4d, cluster_out_causal_4d, "Causal FlashAttn 4D")

    print("✅ Causal FlashAttention correctness VERIFIED")


if __name__ == "__main__":
    main()
