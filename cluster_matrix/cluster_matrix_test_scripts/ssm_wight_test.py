import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)


import os
from transformers import AutoTokenizer, AutoConfig
from cluster_matrix_v1 import cluster_matrix
from cluster_matrix_v1 import cluster_zmq
from cluster_matrix_v1 import check_combined_result_values
import torch
import time
import math
import gc
import glob




'''
ter_matrix$ ls model_matrices | grep layer_25
layer_25_A_log.pt
layer_25_conv1d_bias.pt
layer_25_conv1d_weight.pt
layer_25_D.pt
layer_25_dt_bias.pt
layer_25_in_proj_weight.pt
layer_25_norm_weight.pt
layer_25_out_proj_weight.pt
'''
if __name__ == "__main__":

    layer_25_conv1d_weight_path = 'model_matrices/layer_25_conv1d_weight.pt'  
    layer_25_conv1d_weight = torch.load(layer_25_conv1d_weight_path)
    print('layer_25_conv1d_weight.dtype: ', layer_25_conv1d_weight.dtype)
    print('layer_25_conv1d_weight.shape: ', layer_25_conv1d_weight.shape)


    layer_25_D_path = 'model_matrices/layer_25_D.pt'  
    layer_25_D = torch.load(layer_25_D_path)
    print('layer_25_D.dtype: ', layer_25_D.dtype)
    print('layer_25_D.shape: ', layer_25_D.shape)

    layer_25_dt_bias_path = 'model_matrices/layer_25_dt_bias.pt'
    layer_25_dt_bias = torch.load(layer_25_dt_bias_path)
    print('layer_25_dt_bias.dtype: ', layer_25_dt_bias.dtype)
    print('layer_25_dt_bias.shape: ', layer_25_dt_bias.shape)  
     
     
    layer_25_out_proj_weight_path = 'model_matrices/layer_25_out_proj_weight.pt'      
    layer_25_out_proj_weight = torch.load(layer_25_out_proj_weight_path)
    print('layer_25_out_proj_weight.dtype: ', layer_25_out_proj_weight.dtype)
    print('layer_25_out_proj_weight.shape: ', layer_25_out_proj_weight.shape)

    layer_25_norm_weight_path = 'model_matrices/layer_25_norm_weight.pt'    
    layer_25_norm_weight = torch.load(layer_25_norm_weight_path)
    print('layer_25_norm_weight.dtype: ', layer_25_norm_weight.dtype)
    print('layer_25_norm_weight.shape: ', layer_25_norm_weight.shape)

    layer_25_conv1d_bias_path = 'model_matrices/layer_25_conv1d_bias.pt'    
    layer_25_conv1d_bias = torch.load(layer_25_conv1d_bias_path)
    print('layer_25_A_log.dtype: ', layer_25_conv1d_bias.dtype)
    print('layer_25_A_log.shape: ', layer_25_conv1d_bias.shape)

    layer_25_A_log_path = 'model_matrices/layer_25_A_log.pt'
    layer_25_conv1d_weight = torch.load(layer_25_conv1d_weight_path)
    print('layer_25_conv1d_weight.dtype: ', layer_25_conv1d_weight.dtype)
    print('layer_25_conv1d_weight.shape: ', layer_25_conv1d_weight.shape)

    layer_25_in_proj_weight_path = 'model_matrices/layer_25_in_proj_weight.pt' 
    layer_25_in_proj_weight = torch.load(layer_25_in_proj_weight_path)
    print('layer_25_in_proj_weight.dtype: ', layer_25_in_proj_weight.dtype)
    print('layer_25_in_proj_weight.shape: ', layer_25_in_proj_weight.shape)

    #layer_25_conv1d_bias_path = layer_25_conv1d_bias_path.unsqueeze(0)


    #############################TESTING CLUSTER MATRIX OPERATIONS SYSTEM 1#############################
    
    IP_list = ['192.168.2.100','192.168.2.100','192.168.2.100','192.168.2.101','192.168.2.101','192.168.2.104']   
    percentages = [0.3,0.3,0.1,0.1,0.1,0.1]  
    CPU_GPU_select_list = [ True, True, True, True, True, True  ]  
    backend_select_list = ['llama','llama','llama','llama','llama','llama'] 
    cluster_zmq_obj = cluster_zmq(IP_list)

    matrixA_float16 = cluster_matrix(layer_25_in_proj_weight, 
                                    cluster_zmq_object=cluster_zmq_obj, 
                                    CPU_GPU_select_list=CPU_GPU_select_list, 
                                    node_percentages=percentages,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=False,
                                    dim=0,
                                    auto_set_up=[1, "save"],
                                    matrix_name='layer_25_in_proj_weight_A'
                                    )

    matrixB_float16 = cluster_matrix(layer_25_in_proj_weight, 
                                    cluster_zmq_object=cluster_zmq_obj,  
                                    CPU_GPU_select_list=CPU_GPU_select_list, 
                                    node_percentages=percentages,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=True,
                                    dim=0,
                                    auto_set_up=[1, "save"],
                                    matrix_name='layer_25_in_proj_weight_B'
                                    )

    test_wight_mul = matrixA_float16.cluster_shard_operation(matrixB_float16, False, False, True)  


    matrixA_float16 = cluster_matrix(layer_25_conv1d_weight, 
                                    cluster_zmq_object=cluster_zmq_obj, 
                                    CPU_GPU_select_list=CPU_GPU_select_list, 
                                    node_percentages=percentages,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=False,
                                    dim=0,
                                    auto_set_up=[1, "save"],
                                    matrix_name='layer_25_conv1d_weight_A'
                                    )

    matrixB_float16 = cluster_matrix(layer_25_conv1d_weight, 
                                    cluster_zmq_object=cluster_zmq_obj,  
                                    CPU_GPU_select_list=CPU_GPU_select_list, 
                                    node_percentages=percentages,
                                    back_end_select_list=backend_select_list,
                                    split_matrix=True,
                                    dim=0,
                                    auto_set_up=[1, "save"],
                                    matrix_name='layer_25_conv1d_weight_B'
                                    )

    test_wight_mul = matrixA_float16.cluster_shard_operation(matrixB_float16, False, False, True)  



    #check_combined_result_values(test_wight_mul,test_wight_mul)