import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from cluster_matrix_v1 import cluster_matrix
from cluster_matrix_v1 import cluster_zmq
from cluster_matrix_v1 import check_combined_result_values
import torch
import numpy as np

#######################################------MAIN FUNCTION - TESTING-----######################################  
if __name__ == "__main__":
    
    # ----------------- CREATE MATRICES for dim = 0 split test -----------------

    torch.manual_seed(0)
    np.random.seed(0)

    A3 = torch.from_numpy(np.random.rand(1500, 4500).astype(np.float16))
    B3 = torch.from_numpy(np.random.rand(1000, 4500).astype(np.float16))

    ref = A3 @ B3.T
   
    #############################TESTING CLUSTER MATRIX OPERATIONS SYSTEM 1#############################
    
    # ----------------- CLUSTER TEST (BIG MATRIX) dim = 0 split/join test-----------------

    IP_list = ['192.168.2.100','192.168.2.100','192.168.2.101']   
    percentages = [0.5,0.45,0.05]  
    CPU_GPU_select_list = [ True, True, True ]  
    backend_select_list = ['llama','llama','llama'] 


    cluster_zmq_obj = cluster_zmq(IP_list)

    # ----------------- CLUSTER TEST (small MATRIX) -----------------
    
    small_big_new_matrixA = cluster_matrix(A3, 
                                    cluster_zmq_object=cluster_zmq_obj, 
                                    CPU_GPU_select_list=CPU_GPU_select_list, 
                                    node_percentages=percentages,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=False,
                                    dim=1,
                                    auto_set_up=[1, "save"]
                                    )

    small_new_matrixB = cluster_matrix(B3, 
                                    cluster_zmq_object=cluster_zmq_obj,  
                                    CPU_GPU_select_list=CPU_GPU_select_list, 
                                    node_percentages=percentages,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=True,
                                    dim=1,
                                    auto_set_up=[1, "save"]
                                    )

    small_new_matrixC = small_big_new_matrixA.cluster_shard_operation(
        small_new_matrixB, False, True, False
    )
    small_new_matrixC = small_new_matrixC.send_back()

    if not isinstance(small_new_matrixC, torch.Tensor):
        print("❌ send_back did not return a torch.Tensor")
        exit(1)

    # Compare against reference
    ref_f32 = ref.to(dtype=torch.float32)
    out_f32 = small_new_matrixC.to(dtype=torch.float32)
    if ref_f32.shape != out_f32.shape:
        print(f"❌ Shape mismatch! Reference: {ref_f32.shape}, Combined: {out_f32.shape}")
        exit(1)

    max_diff = (ref_f32 - out_f32).abs().max().item()
    print(f"Max diff: {max_diff}")
    if not torch.allclose(ref_f32, out_f32, rtol=1e-2, atol=1e-2):
        print("❌ Values do not match within tolerance")
        exit(1)
    print("✅ Values match within tolerance")
