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
    
    A3_float16 = torch.from_numpy(np.random.rand(1000, 4500).astype(np.float16))
    B3_float16 = torch.from_numpy(np.random.rand(1000, 4500).astype(np.float16))
    torch.save(A3_float16, 'model_matrices/small_matrixA_float16.pt')
    torch.save(B3_float16, 'model_matrices/small_matrixB_float16.pt')
    print('A# DATA TYPE: ',A3_float16.dtype)
    print('B# DATA TYPE: ',B3_float16.dtype)

    c_ref_float16 = A3_float16 + B3_float16
    torch.save(c_ref_float16, 'model_matrices/c_ref_float16.pt')

    matrix_pathA_float16 = 'model_matrices/small_matrixA_float16.pt'  
    matrix_pathB_float16 = 'model_matrices/small_matrixB_float16.pt'  

    #############################TESTING CLUSTER MATRIX OPERATIONS SYSTEM 1#############################
    
    IP_list = ['192.168.2.100','192.168.2.100','192.168.2.101','192.168.2.104']   
    percentages = [0.35,0.35,0.15,0.15]  
    backend_select_list = ['llama','llama','llama','llama']
    CPU_GPU_select_list = [ True, True, True, False ]   
    '''
    you normally dont need to do this but in my case the last node is only CPU
    blas only(no GPU at all) and will fail on CPU 'add' so i need to disable the back-end setting 
    the last node to flase 
    '''


    cluster_zmq_obj = cluster_zmq(IP_list)
    
    matrixA_float16 = cluster_matrix(matrix_pathA_float16, 
                                    cluster_zmq_object=cluster_zmq_obj, 
                                    CPU_GPU_select_list=CPU_GPU_select_list, 
                                    node_percentages=percentages,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=True,
                                    dim=0,
                                    auto_set_up=[1, "save"]
                                    )

    matrixB_float16 = cluster_matrix(matrix_pathB_float16, 
                                    cluster_zmq_object=cluster_zmq_obj,  
                                    CPU_GPU_select_list=CPU_GPU_select_list, 
                                    node_percentages=percentages,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=True,
                                    dim=0,
                                    auto_set_up=[1, "save"]
                                    )

    big_new_matrixC = matrixA_float16.cluster_shard_operation(matrixB_float16, False, True, True, 'add')  
    check_combined_result_values('model_matrices/c_ref_float16.pt',c_ref_float16)



