import sys  
import os  
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  
  
import torch  
import torch.nn.functional as F  
from cluster_matrix_v1 import cluster_matrix, cluster_zmq, check_combined_result_values  
import time  
import math  
import numpy as np  
  
# ------------------ SETUP ------------------  
os.makedirs("test_model_matrices", exist_ok=True)  

test_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) 
  
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
  
# ------------------ EXECUTE REPEAT TEST ------------------  
print("🧪 Testing cluster repeat operation...")  
  
# Repeat 8 times along second dimension (columns)  
result = cluster_obj.cluster_repeat_operation(  
    repeat_dims=[1, 8, 1, 1],  # [cols, rows, depth, batch] in GGML order  
    Transpose=False,  
    send_back_result=True  # Get combined result for verification  
)  
  