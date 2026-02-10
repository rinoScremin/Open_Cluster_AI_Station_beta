import sys
import os
import time
import torch
import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from cluster_matrix_v1 import cluster_matrix
from cluster_matrix_v1 import cluster_zmq
from cluster_matrix_v1 import check_combined_result_values


######################################------ MAIN FUNCTION - BENCHMARKING -----######################################
if __name__ == "__main__":

    os.makedirs("model_matrices", exist_ok=True)

    # ----------------- MATRIX DEFINITIONS -----------------
    matrix_defs = {
        "small": {
            "A": (1500, 4500, np.float16),
            "B": (1000, 4500, np.float16),
        },
        "mid": {
            "A": (5000, 7000, np.float32),
            "B": (9000, 7000, np.float32),
        },
        "big": {
            "A": (10000, 20000, np.float32),
            "B": (15000, 20000, np.float32),
        },
    }

    paths = {}

    # ----------------- CREATE MATRICES (ONLY IF MISSING) -----------------
    for name, cfg in matrix_defs.items():
        pathA = f"model_matrices/{name}_matrixA.pt"
        pathB = f"model_matrices/{name}_matrixB.pt"
        paths[name] = (pathA, pathB)

        if not os.path.exists(pathA) or not os.path.exists(pathB):
            print(f"[INIT] Generating {name} matrices")

            A = np.random.rand(cfg["A"][0], cfg["A"][1]).astype(cfg["A"][2])
            B = np.random.rand(cfg["B"][0], cfg["B"][1]).astype(cfg["B"][2])

            torch.save(torch.from_numpy(A), pathA)
            torch.save(torch.from_numpy(B), pathB)
        else:
            print(f"[INIT] Using cached {name} matrices")

    # ----------------- PYTORCH REFERENCE -----------------
    pytorch_times = {}

    # Transpose flags must match the cluster operation call below.
    TRANSPOSE_A = False
    TRANSPOSE_B = True

    for name, (pathA, pathB) in paths.items():
        A = torch.load(pathA)
        B = torch.load(pathB)

        t0 = time.perf_counter()
        A_eff = A.T if TRANSPOSE_A else A
        B_eff = B.T if TRANSPOSE_B else B
        C = A_eff @ B_eff
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        pytorch_times[name] = time.perf_counter() - t0

        torch.save(C, f"model_matrices/{name}_c_ref.pt")

        print(f"[PyTorch] {name.upper():5s}: {pytorch_times[name]:.4f}s")

    # ----------------- CLUSTER TEST FUNCTION -----------------
    def run_cluster_config(IP_list, percentages, CPU_GPU_select_list, backend_select_list):
        print("\n======================================================")
        print(f"🔧 TESTING CONFIG ({len(IP_list)} NODES)")
        print("======================================================")

        cluster_zmq_obj = cluster_zmq(IP_list)
        results = {}

        def _safe_percentages_for_shards(node_percentages, total_rows):
            # Prevent generating 0-row shards (which produces a 20-byte header-only .bin and breaks streaming).
            n = len(node_percentages)
            if n == 0:
                return node_percentages
            # If any slot would get 0 rows under the current percentages, fall back to even split.
            try:
                if any(int(total_rows * float(p)) <= 0 for p in node_percentages):
                    return [1.0 / n] * n
            except Exception:
                return [1.0 / n] * n
            return node_percentages

        for name, (pathA, pathB) in paths.items():
            results[name] = {}

            for mode in ["save", "load"]:
                print(f"\n[CLUSTER] {name.upper()} | mode={mode}")

                # Make shard percentages safe for this specific B (depends on B.rows).
                B_local = torch.load(pathB, map_location="cpu")
                b_rows = int(B_local.shape[0])
                safe_percentages = _safe_percentages_for_shards(percentages, b_rows)

                A = cluster_matrix(
                    pathA,
                    cluster_zmq_object=cluster_zmq_obj,
                    CPU_GPU_select_list=CPU_GPU_select_list,
                    node_percentages=safe_percentages,
                    back_end_select_list=backend_select_list,
                    split_matrix=False,
                    dim=1,
                    auto_set_up=[1, mode]
                )

                B = cluster_matrix(
                    pathB,
                    cluster_zmq_object=cluster_zmq_obj,
                    CPU_GPU_select_list=CPU_GPU_select_list,
                    node_percentages=safe_percentages,
                    back_end_select_list=backend_select_list,
                    split_matrix=True,
                    dim=0,
                    auto_set_up=[1, mode]
                )

                t0 = time.perf_counter()
                C = A.cluster_shard_operation(B, TRANSPOSE_A, TRANSPOSE_B, True)
                elapsed = time.perf_counter() - t0

                check_combined_result_values(
                    f"model_matrices/{name}_c_ref.pt",
                    C
                )

                results[name][mode] = elapsed

        return results

    # ----------------- TEST CONFIGURATIONS -----------------

    configs = [
        # 1 NODE
        (
            ['192.168.2.100'],
            [1.0],
            [True],
            ['llama']
        ),
        # 2 NODES (imbalanced)
        (
            ['192.168.2.100', '192.168.2.100'],
            [0.55, 0.45],
            [True, True],
            ['llama', 'llama']
        ),

        # 3 NODES
        (
            ['192.168.2.100', '192.168.2.100', '192.168.2.100'],
            [0.45, 0.45, 0.10],
            [True, True, True],
            ['llama', 'llama', 'llama']
        ),

        # 4 NODES
        (
            ['192.168.2.100', '192.168.2.100', '192.168.2.101', '192.168.2.101'],
            [0.45, 0.4, 0.1, 0.05],
            [True, True, True, True],
            ['llama', 'llama', 'llama', 'llama']
        ),

        # 5 NODES
        (
            ['192.168.2.100', '192.168.2.100', '192.168.2.100',
             '192.168.2.101', '192.168.2.101'],
            # 5 slots => 5 percentages (must match IP_list length).
            [0.40, 0.40, 0.1, 0.05, 0.05],
            [True] * 5,
            ['llama'] * 5
        ),

    ]

    # ----------------- RUN ALL CONFIGS -----------------
    # Keep results per config entry (do not key only by node count, because configs can repeat counts).
    all_results = []

    for idx, cfg in enumerate(configs, start=1):
        IP_list, percentages, CPU_GPU_select_list, backend_select_list = cfg
        res = run_cluster_config(IP_list, percentages, CPU_GPU_select_list, backend_select_list)
        all_results.append(
            {
                "index": idx,
                "nodes": len(IP_list),
                "percentages": percentages,
                "results": res,
            }
        )

    # ----------------- SUMMARY -----------------
    print("\n================== BENCHMARK SUMMARY ==================\n")

    # Also persist the printed summary for later review.
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_output_logs"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cluster_matrix_time_bench_summary.txt")
    summary_lines = ["================= BENCHMARK SUMMARY ==================", ""]

    for entry in all_results:
        header = f"--- CONFIG #{entry['index']} ({entry['nodes']} SLOT) | percentages={entry['percentages']} ---"
        print(f"\n{header}")
        summary_lines.append(header)

        result = entry["results"]
        for name in result:
            line = (
                f"{name.upper():5s} | "
                f"PyTorch: {pytorch_times[name]:.4f}s | "
                f"Cluster(save): {result[name]['save']:.4f}s | "
                f"Cluster(load): {result[name]['load']:.4f}s"
            )
            print(line)
            summary_lines.append(line)

    summary_lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\n✅ Saved benchmark summary to: {out_path}\n")
