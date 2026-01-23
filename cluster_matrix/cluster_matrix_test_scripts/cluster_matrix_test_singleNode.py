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
 
if __name__ == "__main__":
    
    A3_float16 = torch.from_numpy(np.random.rand(1500, 4500).astype(np.float16))
    B3_float16 = torch.from_numpy(np.random.rand(1000, 4500).astype(np.float16))
    torch.save(A3_float16, 'model_matrices/small_matrixA_float16.pt')
    torch.save(B3_float16, 'model_matrices/small_matrixB_float16.pt')
    print('A# DATA TYPE: ',A3_float16.dtype)
    print('B# DATA TYPE: ',B3_float16.dtype)

    c_ref_float16 = A3_float16 @ B3_float16.T
    torch.save(c_ref_float16, 'model_matrices/c_ref_float16.pt')
    matrix_pathA_float16 = 'model_matrices/small_matrixA_float16.pt'  
    matrix_pathB_float16 = 'model_matrices/small_matrixB_float16.pt'  

    #######################################------SINLGE NODE TEST-----###################################### 
    cluster_zmq_obj = cluster_zmq(['192.168.2.100'])
    matrixA_single_node = cluster_matrix(
        matrix_file_path=matrix_pathA_float16,
        cluster_zmq_object=cluster_zmq_obj,
        CPU_GPU_select_list=[True],
        node_percentages=[1],
        back_end_select_list=["llama"],
        split_matrix=False,
        dim=0,
        auto_set_up=[1, "save"],
    )
    matrixB_single_node = cluster_matrix(
        matrix_file_path=matrix_pathB_float16,
        cluster_zmq_object=cluster_zmq_obj,
        CPU_GPU_select_list=[True],
        node_percentages=[1],
        back_end_select_list=["llama"],
        split_matrix=False,
        dim=0,
        auto_set_up=[1, "save"],
    )
    matrixC_single_node = matrixA_single_node.cluster_shard_operation(
        matrixB_single_node,
        False,
        True,
        False ## IF YOU ARE USING 1 NODE DO NOT USE SEND BACK NOT NEEDED IT MUST BE SET TO FALSE FOR SINGLE NODE MATRIX OPERATION'S 
    )
    check_combined_result_values('model_matrices/c_ref_float16.pt',matrixC_single_node) # use the 'check_combined_result_values' function to make ssure

    #######################################------SINLGE NODE TEST DEFULTS-----###################################### 

    cluster_zmq_obj = cluster_zmq(['192.168.2.100'])
    matrixA_single_node = cluster_matrix(
        matrix_file_path=matrix_pathA_float16,
        cluster_zmq_object=cluster_zmq_obj,
        split_matrix=False,
        auto_set_up=[1, "save"],
    )

    matrixB_single_node = cluster_matrix(
        matrix_file_path=matrix_pathB_float16,
        cluster_zmq_object=cluster_zmq_obj,
        split_matrix=False,
        auto_set_up=[1, "save"],
    )

    matrixC_single_node = matrixA_single_node.cluster_shard_operation(
        matrixB_single_node,
        False,
        True,
        False ## IF YOU ARE USING 1 NODE DO NOT USE SEND BACK NOT NEEDED IT MUST BE SET TO FALSE FOR SINGLE NODE MATRIX OPERATION'S 
    )
    check_combined_result_values('model_matrices/c_ref_float16.pt',matrixC_single_node) # use the 'check_combined_result_values' function to make ssure
