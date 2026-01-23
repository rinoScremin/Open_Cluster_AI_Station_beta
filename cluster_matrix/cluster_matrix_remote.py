import ray
from cluster_matrix_v1 import cluster_matrix, cluster_zmq

@ray.remote
class cluster_matrix_rayRemote:
    def __init__(
        self,
        matrix_file_path,
        cluster_zmq_object,
        CPU_GPU_select_list,
        node_percentages,
        back_end_select_list,
        split_matrix=False,
        dim=0,
        auto_set_up=None,
        matrix_name=None,
    ):
        # Only pass matrix_name if provided
        kwargs = {}
        if matrix_name is not None:
            kwargs['matrix_name'] = matrix_name

        self.local_matrix = cluster_matrix(
            matrix_file_path=matrix_file_path,
            cluster_zmq_object=cluster_zmq_object,
            CPU_GPU_select_list=CPU_GPU_select_list,
            node_percentages=node_percentages,
            back_end_select_list=back_end_select_list,
            split_matrix=split_matrix,
            dim=dim,
            auto_set_up=auto_set_up,
            **kwargs
        )

    def cluster_shard_operation(
        self,
        other_matrix_actor=None,
        send_back_result=True,
        add_op=False,
        save_result=True,
        op_type="matmul"
    ):
        # If 'other_matrix_actor' is a Ray actor, fetch its local matrix tensor
        other_matrix = None
        if other_matrix_actor is not None:
            # Must call a method that returns a serializable object
            other_matrix = ray.get(other_matrix_actor.get_serializable_matrix.remote())

        # Perform the operation
        result = self.local_matrix.cluster_shard_operation(
            other_matrix,
            send_back_result=send_back_result,
            add_op=add_op,
            save_result=save_result,
            op_type=op_type
        )
        return result

    # Return only serializable data
    def get_serializable_matrix(self):
        """
        Returns a version of the local matrix that Ray can serialize.
        You may adjust this to return a tensor, list, or numpy array.
        """
        return self.local_matrix.matrix_file_path  # or convert to CPU tensor if needed

    def get_matrix_shape(self):
        """
        Returns shape info of the local matrix (safe for Ray)
        """
        if hasattr(self.local_matrix, 'OG_matrix_shape'):
            return self.local_matrix.OG_matrix_shape
        return None
