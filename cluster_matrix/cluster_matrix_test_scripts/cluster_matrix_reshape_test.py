import sys  
import os  
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  
  
import torch  
import torch.nn.functional as F  
from cluster_matrix_v1 import cluster_matrix, cluster_zmq, check_combined_result_values  
import time  
import math  
  
# ------------------ SETUP ------------------  
os.makedirs("test_model_matrices", exist_ok=True)  
  
seq_len = 200  
d_k = 160  
torch.manual_seed(0)  
  
Q = torch.randn(seq_len, d_k)  
K = torch.randn(seq_len, d_k)  
V = torch.randn(seq_len, d_k)  
  
q_path = "test_model_matrices/q.pt"  
k_path = "test_model_matrices/k.pt"  
v_path = "test_model_matrices/v.pt"  
  
torch.save(Q, q_path)  
torch.save(K, k_path)  
torch.save(V, v_path)  
  
# ------------------ CLUSTER SETUP ------------------  
IP_list = ["192.168.2.100","192.168.2.100","192.168.2.101"]  
percentages = [0.50,0.45,0.05]  
CPU_GPU_select_list = [True,True,True]  
backend_select_list = ["llama","llama","llama"]  
  
cluster_zmq_obj = cluster_zmq(IP_list)  
  
cluster_obj = cluster_matrix(  
    q_path,  
    cluster_zmq_object=cluster_zmq_obj,  
    CPU_GPU_select_list=CPU_GPU_select_list,  
    node_percentages=percentages,  
    back_end_select_list=backend_select_list,  
    split_matrix=True,  
    auto_set_up=[1, "save"]  
)  
  
# ------------------ RESHAPE ------------------  
# Define target output dimensions for the COMPLETE matrix (torch order: batch, depth, rows, cols)  
# The function will automatically calculate per-shard dimensions  
target_output_dims = [10, 4, 100, 8]  # 10*4*100*8 = 32,000 elements  
  
print(f"Input matrix shape: [{seq_len}, {d_k}] = {seq_len * d_k} elements")  
print(f"Target output dimensions (complete matrix): {target_output_dims}")  
print(f"Target total elements: {target_output_dims[0] * target_output_dims[1] * target_output_dims[2] * target_output_dims[3]}")  
  
# Validate element count matches for complete matrix  
if seq_len * d_k != target_output_dims[0] * target_output_dims[1] * target_output_dims[2] * target_output_dims[3]:  
    print("❌ ERROR: Element count mismatch for complete matrix!")  
    exit(1)  
  
result = cluster_obj.cluster_reshape_operation(  
    input_shape=[seq_len, d_k],  
    Transpose=False,  
    send_back_result=True,  
    output_dims=target_output_dims  
)  
  
expected = Q.reshape(*target_output_dims)  
if result.shape != expected.shape:  
    print(f"❌ Shape mismatch: got {tuple(result.shape)}, expected {tuple(expected.shape)}")  
    exit(1)  
  
max_diff = (result - expected).abs().max().item()  
print(f"✅ Reshape operation completed successfully (max diff: {max_diff})")  
