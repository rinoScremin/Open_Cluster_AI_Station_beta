import torch
import os
import time
import subprocess
import struct
import numpy as np
import zmq
import mmap
import math

def check_combined_result_values(c_ref_path, combined):
    c_ref = torch.load(c_ref_path)
    if c_ref.shape != combined.shape:
        print(f"❌ Shape mismatch! Reference: {c_ref.shape}, Combined: {combined.shape}")
        return
    print(f"✅ Shapes match: {c_ref.shape}")
    # Ensure tensors
    if not isinstance(c_ref, torch.Tensor):
        c_ref = torch.from_numpy(c_ref)
    if not isinstance(combined, torch.Tensor):
        combined = torch.from_numpy(combined)
    # -------------------------------
    # DTYPE & DEVICE HANDLING
    # -------------------------------
    print(f"Reference dtype: {c_ref.dtype}")
    print(f"Combined  dtype: {combined.dtype}")
    # Move to same device
    device = combined.device
    c_ref = c_ref.to(device=device)
    # Promote BOTH to a safe comparison dtype
    compare_dtype = torch.float32
    c_ref = c_ref.to(dtype=compare_dtype)
    combined = combined.to(dtype=compare_dtype)
    # -------------------------------
    # DIFFERENCE METRICS
    # -------------------------------
    diff = torch.abs(c_ref - combined)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"Max absolute difference:  {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    # -------------------------------
    # DTYPE-AWARE TOLERANCE
    # -------------------------------
    tolerance_map = {
        torch.float16: 1e-1,
        torch.bfloat16: 1e-1,
        torch.float32: 1e-3,
        torch.float64: 1e-6,
    }
    original_dtype = combined.dtype
    tolerance = tolerance_map.get(original_dtype, 1e-3)
    if torch.allclose(c_ref, combined, rtol=tolerance, atol=tolerance):
        print(f"✅ Results match within tolerance ({tolerance})")
    else:
        print(f"⚠️  Results differ beyond tolerance ({tolerance})")
    significant_diff = diff > tolerance
    num_different = significant_diff.sum().item()
    total_elements = c_ref.numel()
    print(
        f"Elements with > {tolerance} difference: "
        f"{num_different}/{total_elements} "
        f"({(num_different / total_elements * 100):.2f}%)"
    )


class cluster_zmq:
    def __init__(self, node_IP_list):
        # Keep the caller-provided slot list (may include duplicates).
        self.node_IP_list = list(node_IP_list)
        # =============== DEFAULT NODE PERCENTAGES SET UP ===============
        self.num_nodes = len(node_IP_list)
        base = 100 // self.num_nodes
        remainder = 100 % self.num_nodes
        # Start with an even split
        node_percentages = [base] * self.num_nodes
        # Distribute the remainder (+1) across the first nodes
        for i in range(remainder):
            node_percentages[i] += 1
        # Normalize to fractions (e.g., 17 -> 0.17)
        self.default_node_percentages = [p / 100 for p in node_percentages]
        self.default_back_end_select_list = ['llama'] * self.num_nodes
        self.default_CPU_GPU_select_list = [True] * self.num_nodes

        # =============== FOLDER PATH CONFIGURATION ===============
        print("\n📁 CONFIGURING STORAGE PATHS...")
        
        # Local paths (head node)
        self.local_matrix_results_RAM_folder = os.environ.get(
            'LOCAL_MATRIX_RESULTS_RAM_FOLDER', '/dev/shm/matrix_results/'
        )
        self.local_DISK_folder = os.environ.get('LOCAL_DISK_FOLDER', 'matrix_shards/')
        self.local_RAM_folder = os.environ.get('LOCAL_RAM_FOLDER', '/dev/shm/matrix_shards/')

        default_project_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep
        self.local_project_dir = os.environ.get('LOCAL_PROJECT_DIR', default_project_dir)
        if not self.local_project_dir.endswith(os.sep):
            self.local_project_dir += os.sep
        
        print(f"   Local Paths:")
        print(f"     - Disk Folder: {self.local_DISK_folder}")
        print(f"     - RAM Folder: {self.local_RAM_folder}")
        print(f"     - Project Dir: {self.local_project_dir}")
        
        # Remote paths (worker nodes)
        self.remote_matrix_results_RAM_folder = os.environ.get(
            'REMOTE_MATRIX_RESULTS_RAM_FOLDER', '/dev/shm/matrix_results/'
        )
        self.remote_DISK_folder = os.environ.get('REMOTE_DISK_FOLDER', 'matrix_shards/')
        self.remote_RAM_folder = os.environ.get('REMOTE_RAM_FOLDER', '/dev/shm/matrix_shards/')
        self.remote_project_dir = os.environ.get('REMOTE_PROJECT_DIR', default_project_dir)
        if not self.remote_project_dir.endswith(os.sep):
            self.remote_project_dir += os.sep
        
        print(f"   Remote Paths:")
        print(f"     - Disk Folder: {self.remote_DISK_folder}")
        print(f"     - RAM Folder: {self.remote_RAM_folder}")
        print(f"     - Project Dir: {self.remote_project_dir}")
        
        # Get Python executable path
        self.conda_env_dir = os.environ.get('CONDA_ENV_DIR', '/home/rino/anaconda3/envs/ray-conda-env')
        self.python_path = os.environ.get('OPEN_CLUSTER_SCRIPT_PATH', '/home/rino/anaconda3/envs/cluster-worker/bin/python')


        # =============== NETWORK AND PORT CONFIGURATION ===============
        print("\n🌐 CONFIGURING NETWORK SETTINGS...")
                
        # Get head node IP addresses from environment variables
        self.IP = os.environ.get('HEAD_NODE_IP', '192.168.2.100')
        self.wifi_IP = os.environ.get('HEAD_NODE_IP_WIFI', '192.168.50.113')

        # Optional WiFi IP mapping (best-effort)
        wifi_env = os.environ.get("WORKER_WIFI_IPS", "")
        if wifi_env:
            self.IP_list_wifi = [ip.strip() for ip in wifi_env.split(",") if ip.strip()]
        else:
            self.IP_list_wifi = ['192.168.3.13', '192.168.3.243', '192.168.3.165', '192.168.3.94']
        
        print(f"   Head Node Ethernet IP: {self.IP}")
        print(f"   Head Node WiFi IP: {self.wifi_IP}")
        
        # ZeroMQ ports for llama communication
        self.llama_head_node_PULL_port = os.environ.get("HEAD_NODE_PULL_PORT_C", "7779")
        self.llama_head_node_PUSH_port = os.environ.get("HEAD_NODE_PUSH_PORT_C", "7780")
        self.llama_worker_node_PULL_port = os.environ.get("WORKER_NODE_PULL_PORT_C", "5557")
        self.llama_worker_node_PUSH_port = os.environ.get("WORKER_NODE_PUSH_PORT_C", "5558")
        
        print(f"   Head Node Ports: PULL={self.llama_head_node_PULL_port}, PUSH={self.llama_head_node_PUSH_port}")
        print(f"   Worker Node Ports: PULL={self.llama_worker_node_PULL_port}, PUSH={self.llama_worker_node_PUSH_port}")
        
        # Python frontend ACK / cluster barrier port
        self.python_front_end_cluster_port = os.environ.get("PYTHON_FRONT_END_CLUSTER_PORT", "7790")
        print(f"   Cluster Barrier Port: {self.python_front_end_cluster_port}")

        # =============== ZEROMMQ SOCKET SETUP ===============
        print("\n🔌 SETTING UP ZEROMQ CONNECTIONS...")

        # Check if we're in worker mode (all IPs are 0.0.0.0)
        all_ips_are_zero = all(ip == '0.0.0.0' for ip in self.node_IP_list)

        if all_ips_are_zero:
            print("   ⚙️ Worker mode detected - skipping network connections")
            self.zmq_context = None
            self.llama_socket_pool = {}
            self.llama_socket_pool_wifi = {}
            self.ack_receiver_socket = None
        else:
            # Initialize ZeroMQ context
            self.zmq_context = zmq.Context()
            self.llama_socket_pool = {}  # For llama communication - ports 5557/5558
            self.llama_socket_pool_wifi = {}  # Placeholder to avoid cleanup errors
            self.timeout = 5000  # 5 second timeout
            
            # Create PUSH sockets for ALL remote nodes
            unique_IP_list = list(dict.fromkeys(self.node_IP_list))
            print(f"   Connecting to {len(unique_IP_list)} unique nodes...")
            
            for node_ip in unique_IP_list:
                if node_ip != self.IP:  # Remote nodes only
                    # Llama socket (port 5557 for computation)
                    try:
                        llama_socket = self.zmq_context.socket(zmq.PUSH)
                        llama_socket.connect(f"tcp://{node_ip}:{self.llama_worker_node_PULL_port}")
                        self.llama_socket_pool[node_ip] = llama_socket
                        print(f"   ✅ Connected to worker node {node_ip}:{self.llama_worker_node_PULL_port}")
                    except Exception as e:
                        print(f"   ❌ Failed to connect to {node_ip}: {e}")
            
            # Optional WiFi sockets (parallel transfer)
            for idx, node_ip in enumerate(unique_IP_list):
                if idx < len(self.IP_list_wifi):
                    wifi_ip = self.IP_list_wifi[idx]
                    try:
                        wifi_socket = self.zmq_context.socket(zmq.PUSH)
                        wifi_socket.connect(f"tcp://{wifi_ip}:{self.llama_worker_node_PULL_port}")
                        self.llama_socket_pool_wifi[wifi_ip] = wifi_socket
                        print(f"   ✅ Connected to worker WiFi {wifi_ip}:{self.llama_worker_node_PULL_port}")
                    except Exception as e:
                        print(f"   ❌ Failed to connect WiFi {wifi_ip}: {e}")

            # Connect to local head node as well
            try:
                llama_socket = self.zmq_context.socket(zmq.PUSH)
                llama_socket.connect(f"tcp://{self.IP}:{self.llama_head_node_PULL_port}")
                self.llama_socket_pool[self.IP] = llama_socket
                print(f"   ✅ Connected to head node (self) {self.IP}:{self.llama_head_node_PULL_port}")
            except Exception as e:
                print(f"   ❌ Failed to connect to head node: {e}")
            
            print(f"   Total sockets in pool: {len(self.llama_socket_pool)}")

            # =============== CLUSTER BARRIER/ACK RECEIVER SETUP ===============
            print("\n🔄 SETTING UP CLUSTER BARRIER/ACK RECEIVER...")
            
            # Initialize ack receiver socket (singleton pattern)
            if not hasattr(cluster_zmq, '_ack_receiver_socket'):
                try:
                    cluster_zmq._ack_receiver_socket = self.zmq_context.socket(zmq.PULL)
                    cluster_zmq._ack_receiver_socket.bind(f"tcp://0.0.0.0:{self.python_front_end_cluster_port}")
                    cluster_zmq._ack_owner_context = self.zmq_context
                    print(f"✅ Python frontend ACK receiver bound to port {self.python_front_end_cluster_port}")
                except Exception as e:
                    print(f"❌ Failed to bind ACK receiver: {e}")
                    raise
            else:
                print(f"✅ ACK receiver already exists on port {self.python_front_end_cluster_port}")
            
            # Reference it in the instance
            self.ack_receiver_socket = cluster_zmq._ack_receiver_socket
            # Cache for combined PT blobs streamed from C++ (PT_COMBINED=<name>)
            if not hasattr(cluster_zmq, "_combined_pt_payloads"):
                cluster_zmq._combined_pt_payloads = {}
            self.combined_pt_payloads = cluster_zmq._combined_pt_payloads

        # =============== CREATE LOCAL DIRECTORIES ===============
        print("\n📂 CREATING LOCAL DIRECTORIES...")
        directories_created = []
        if not os.path.exists(self.local_DISK_folder):
            os.makedirs(self.local_DISK_folder)
            directories_created.append(self.local_DISK_folder)
        if not os.path.exists(self.local_RAM_folder):
            os.makedirs(self.local_RAM_folder)
            directories_created.append(self.local_RAM_folder)
        if not os.path.exists(self.local_matrix_results_RAM_folder):
            os.makedirs(self.local_matrix_results_RAM_folder)
            directories_created.append(self.local_matrix_results_RAM_folder)
        
        if directories_created:
            print(f"✅ Created directories: {', '.join(directories_created)}")
        else:
            print("✅ All required directories already exist")
                
        # =============== CREATE REMOTE DIRECTORIES ===============
        print("\n📡 CREATING REMOTE DIRECTORIES ON WORKER NODES...")
            
        command = f'mkdir -p {self.remote_project_dir}{self.remote_DISK_folder} {self.remote_RAM_folder} {self.remote_matrix_results_RAM_folder}'
        print(f"   Sending command: {command}")
            
        for node_ip, socket in self.llama_socket_pool.items():
            try:
                socket.send_multipart([command.encode('utf-8')])
                print(f"   ✅ Directory creation command sent to {node_ip}")
            except Exception as e:
                print(f"   ❌ Failed to send command to {node_ip}: {e}")

    def send_ack_confirmation(self, ack_msg="ACK"):    
        """    
        Send ACK confirmation back to C++ backend    
        """    
        try:    
            # Create a separate socket for sending confirmations    
            if not hasattr(self, 'ack_confirmation_socket'):    
                self.ack_confirmation_socket = self.zmq_context.socket(zmq.PUSH)    
                # Use self.IP for the head node IP and define confirmation port  
                confirmation_port = os.environ.get("PYTHON_ACK_CONFIRMATION_PORT", "7791")  
                self.ack_confirmation_socket.connect(f"tcp://{self.IP}:{confirmation_port}")    
            
            # Send the confirmation message    
            self.ack_confirmation_socket.send_string(ack_msg)    
            print(f"✅ Sent confirmation: {ack_msg}")    
            
        except Exception as e:    
            print(f"❌ Failed to send confirmation: {e}")

    def wait_for_acks(self, expected_count, expected_msg="ACK", time_out=120):
        """
        Wait for ACKs from all expected nodes on the Python front end cluster port.
        
        Args:
            expected_count: Number of ACKs to wait for
            expected_msg: The expected message string (default: "ACK")
            time_out: Timeout in seconds (default: 120 seconds)
        
        Returns:
            Number of ACKs actually received (may be less than expected if timeout occurs)
        """
        ack_socket = getattr(self, "ack_receiver_socket", None) or getattr(cluster_zmq, "_ack_receiver_socket", None)
        if ack_socket is None:
            raise RuntimeError("ACK receiver socket is not available.")

        acks = 0
        start_time = time.time()
        
        while acks < expected_count:
            # Check if timeout has been reached
            if time.time() - start_time > time_out:
                print(f"⏰ TIMEOUT: Only received {acks}/{expected_count} ACKs after {time_out} seconds")
                return acks
            try:
                parts = ack_socket.recv_multipart(flags=zmq.NOBLOCK)
                if not parts:
                    continue

                # Single-part string ACK
                if len(parts) == 1:
                    msg = parts[0].decode("utf-8", errors="replace")
                    if msg == expected_msg:
                        acks += 1
                        print(f"✅ Received {expected_msg} {acks}/{expected_count}")
                    continue

                # Multipart payload: header + bytes
                header = parts[0].decode("utf-8", errors="replace")
                payload = parts[1]
                if header.startswith("BIN_COMBINED=") or header.startswith("PT_COMBINED="):
                    key = header.split("=", 1)[1]
                    store = getattr(self, "combined_pt_payloads", None) or getattr(cluster_zmq, "_combined_pt_payloads", None)
                    if store is not None:
                        store[key] = payload
            except zmq.Again:
                # No message yet, sleep briefly to avoid 100% CPU
                time.sleep(0.01)
        print("✅ All ACKs received!")
        return acks

    def wait_for_combined_pt(self, base_result_name, time_out=120, force_2d=True):
        ack_socket = getattr(self, "ack_receiver_socket", None) or getattr(cluster_zmq, "_ack_receiver_socket", None)
        if ack_socket is None:
            raise RuntimeError("ACK receiver socket is not available.")

        store = getattr(self, "combined_pt_payloads", None) or getattr(cluster_zmq, "_combined_pt_payloads", None)
        if store is None:
            store = {}

        def _bin_payload_to_tensor(payload: bytes):
            # v2: [dtype_tag(int32), batch, depth, rows, cols, data]
            mv = memoryview(payload)
            if len(mv) < 5 * 4:
                raise ValueError("Combined payload too small for v2 header")
            dtype_tag, b, d, r, c = struct.unpack_from("iiiii", mv, 0)
            offset = 5 * 4
            numel = int(b) * int(d) * int(r) * int(c)

            if dtype_tag == -1:
                np_dtype = np.float32
            elif dtype_tag == -2:
                np_dtype = np.float16
            elif dtype_tag == -3:
                np_dtype = np.uint16  # raw bf16 bits
            else:
                raise ValueError(f"Unsupported dtype_tag in combined payload: {dtype_tag}")

            need = numel * np.dtype(np_dtype).itemsize
            if len(mv) < offset + need:
                raise ValueError("Combined payload truncated")

            arr = np.frombuffer(mv, dtype=np_dtype, count=numel, offset=offset).reshape((b, d, r, c))
            if dtype_tag == -3:
                u16 = np.array(arr, copy=True, dtype=np.uint16)
                u32 = (u16.astype(np.uint32) << 16)
                f32 = u32.view(np.float32)
                t = torch.from_numpy(f32).to(dtype=torch.bfloat16)
            else:
                t = torch.from_numpy(np.array(arr, copy=True))

            if force_2d and t.dim() == 4:
                t = t.reshape(t.shape[0] * t.shape[1] * t.shape[2], t.shape[3])
            return t

        start_time = time.time()
        while True:
            if base_result_name in store:
                payload = store.pop(base_result_name)
                return _bin_payload_to_tensor(payload)

            if time.time() - start_time > time_out:
                raise TimeoutError(
                    f"Timed out waiting for combined PT payload for '{base_result_name}'"
                )

            try:
                parts = ack_socket.recv_multipart(flags=zmq.NOBLOCK)
                if not parts:
                    continue
                if len(parts) == 1:
                    # Ignore ACK strings here
                    continue
                header = parts[0].decode("utf-8", errors="replace")
                payload = parts[1]
                if header.startswith("BIN_COMBINED=") or header.startswith("PT_COMBINED="):
                    key = header.split("=", 1)[1]
                    store[key] = payload
            except zmq.Again:
                time.sleep(0.01)

    def zmq_send_command(self, worker_ip, command, timeout=5):
        """Send command using persistent connection pool"""
        socket_pool = getattr(self, "llama_socket_pool", None)
        if socket_pool and worker_ip in socket_pool:
            socket_eth = socket_pool[worker_ip]
            try:
                # MUST send bytes, NOT str.
                socket_eth.send(command.encode('utf-8'))
                return True
            except Exception as e:
                print(f"❌ Error sending command to {worker_ip}: {e}")
                return False
        else:
            print(f"❌ No socket found for worker {worker_ip}")
            return False
		 
    def zmq_send_file(self, worker_ip, local_file_path):
        socket_pool = getattr(self, "llama_socket_pool", None)
        if socket_pool and worker_ip in socket_pool:
            socket_eth = socket_pool[worker_ip]
            with open(local_file_path, 'rb') as f:
                file_data = f.read()
            
            # Use os.path.basename to get filename
            filename_only = os.path.basename(local_file_path)
            
            socket_eth.send_multipart([
                filename_only.encode(),
                file_data
            ])
            print(f"📤 Sent file {filename_only} to {worker_ip}")
	        
    def stream_matrix_binary(self, worker_ip, matrix, save_name):
        """
        Stream a matrix as binary data directly to a remote node without saving locally.
        Creates the binary file data in memory and sends it via ZeroMQ.
        
        Args:
            matrix: PyTorch tensor to stream
            worker_ip: IP address of the target worker node
            save_name: Filename to use for the streamed file
        """
        verbose = os.environ.get("STREAM_MATRIX_BINARY_VERBOSE", "1") == "1"
        if verbose:
            print(f"📤 Streaming matrix to {worker_ip} as {save_name}")
        
        socket_pool = getattr(self, "llama_socket_pool", None)

        if not socket_pool or worker_ip not in socket_pool:
            print(f"  ERROR: No socket connection to {worker_ip}")
            return
        
        socket_eth = socket_pool[worker_ip]

        # ===== CREATE BINARY FILE DATA IN MEMORY (FAST PATH) =====
        if verbose:
            print("  Creating binary file data in memory...")

        # Binary wire format (v2):
        # [dtype_tag(int32), batch(int32), depth(int32), rows(int32), cols(int32), data(bytes)]
        #
        # dtype_tag is NEGATIVE to stay backward compatible with legacy files where
        # the first int was `ndim` (typically 4).
        #   -1 = float32
        #   -2 = float16
        #   -3 = bfloat16 (payload is raw bf16 bits as int16/uint16)
        #
        # Legacy format (v1) is still accepted by readers:
        # [ndim(int32), dims..., data(float32)]

        if isinstance(matrix, torch.Tensor):
            t = matrix.detach()
            if t.device.type != "cpu":
                t = t.cpu()
            if t.ndim == 2:
                t = t.reshape(1, 1, t.shape[0], t.shape[1])
            elif t.ndim == 3:
                t = t.reshape(1, t.shape[0], t.shape[1], t.shape[2])
            elif t.ndim == 4:
                pass
            else:
                raise ValueError(f"Unsupported number of dimensions: {t.ndim}")
            if not t.is_contiguous():
                t = t.contiguous()

            if t.dtype == torch.float32:
                dtype_tag = -1
                matrix_np = t.numpy()  # zero-copy view on CPU
            elif t.dtype == torch.float16:
                dtype_tag = -2
                matrix_np = t.numpy()  # zero-copy view on CPU
            elif t.dtype == torch.bfloat16:
                dtype_tag = -3
                # numpy can't represent bf16 directly; send raw bf16 bits as int16.
                matrix_np = t.view(torch.int16).numpy()
            else:
                raise ValueError(f"Unsupported tensor dtype for binary stream: {t.dtype}")
        elif isinstance(matrix, np.ndarray):
            matrix_np = matrix
            if not matrix_np.flags.get("C_CONTIGUOUS", False):
                matrix_np = np.asarray(matrix_np, order="C")
            if matrix_np.ndim == 2:
                matrix_np = matrix_np.reshape(1, 1, matrix_np.shape[0], matrix_np.shape[1])
            elif matrix_np.ndim == 3:
                matrix_np = matrix_np.reshape(1, matrix_np.shape[0], matrix_np.shape[1], matrix_np.shape[2])
            elif matrix_np.ndim == 4:
                pass
            else:
                raise ValueError(f"Unsupported number of dimensions: {matrix_np.ndim}")

            if matrix_np.dtype == np.float32:
                dtype_tag = -1
            elif matrix_np.dtype == np.float16:
                dtype_tag = -2
            else:
                raise ValueError(f"Unsupported numpy dtype for binary stream: {matrix_np.dtype}")
        else:
            raise ValueError(f"Unsupported matrix type: {type(matrix)}")

        b, c, h, w = (int(x) for x in matrix_np.shape)
        header_size = 4 + 4 * 4  # ndim + 4 dims
        payload_size = header_size + matrix_np.nbytes
        payload = bytearray(payload_size)
        struct.pack_into("iiiii", payload, 0, dtype_tag, b, c, h, w)

        # Fill the payload data area without creating an intermediate `tobytes()` copy.
        out_view = np.frombuffer(
            payload,
            dtype=matrix_np.dtype,
            offset=header_size,
            count=matrix_np.size,
        ).reshape(matrix_np.shape)
        out_view[...] = matrix_np

        if verbose:
            print(f"  Dimensions: {b} × {c} × {h} × {w}")
            print(f"  Binary data size: {payload_size:,} bytes ({payload_size/(1024*1024):.2f} MB)")
        
        # ===== SEND VIA ZMQ =====
        # Use the exact same pattern as zmq_send_file
        try:
            # Zero-copy into ZMQ when possible (pyzmq will hold a reference to the bytearray buffer).
            socket_eth.send_multipart([save_name.encode(), payload], copy=False)
            if verbose:
                print(f"  ✓ Matrix streamed to {worker_ip}")
                print(f"  Sent: {save_name} ({payload_size:,} bytes)")
        except Exception as e:
            print(f"  ERROR streaming matrix to {worker_ip}: {e}")
            raise
        
        return payload_size

    def cleanup(self):
        if getattr(self, "_cleaned_up", False):
            return
        self._cleaned_up = True

        try:
            for socket in getattr(self, "llama_socket_pool", {}).values():
                try:
                    socket.close(linger=0)
                except Exception:
                    pass
            for socket in getattr(self, "llama_socket_pool_wifi", {}).values():
                try:
                    socket.close(linger=0)
                except Exception:
                    pass
            if hasattr(self, "ack_confirmation_socket"):
                try:
                    self.ack_confirmation_socket.close(linger=0)
                except Exception:
                    pass
        finally:
            # Do not close/term the ack receiver singleton; it may be shared across instances.
            # Also avoid terminating the context that owns the singleton ack receiver.
            try:
                ctx = getattr(self, "zmq_context", None)
                if ctx is not None and ctx is not getattr(cluster_zmq, "_ack_owner_context", None):
                    ctx.term()
            except Exception:
                pass

    def __del__(self):
        """Destructor as fallback cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Avoid exceptions in destructor
   

class cluster_matrix:
    def __init__(
        self,
        matrix_file_path,
        cluster_zmq_object=None,
        node_IP_list=None,
        CPU_GPU_select_list=None,
        node_percentages=None,
        back_end_select_list=None,
        split_matrix=False,
        dim=0,
        matrix_name='',
        matrix_labeling='',
        auto_set_up=None,
    ):
        
        print("=" * 70)
        print("🚀 INITIALIZING CLUSTER MATRIX DISTRIBUTION SYSTEM")
        print("=" * 70)

        # =============== NODE CONFIGURATION VALIDATION ===============
        # Check consistency of the node configuration
        if node_percentages is None:
            node_percentages = list(cluster_zmq_object.default_node_percentages)
        if back_end_select_list is None:
            back_end_select_list = list(cluster_zmq_object.default_back_end_select_list)
        if CPU_GPU_select_list is None:
            CPU_GPU_select_list = list(cluster_zmq_object.default_CPU_GPU_select_list)
        if auto_set_up is None:
            auto_set_up = []

        # Backwards-compatible positional signature support:
        #   cluster_matrix(matrix_path, node_IP_list, CPU_GPU_select_list, node_percentages,
        #                  back_end_select_list, split_matrix=False, dim=0, ...)
        #
        # After introducing `cluster_zmq_object` as the 2nd parameter, older code that passes
        # positional args will shift everything by one. Detect that pattern and re-map.
        if (
            cluster_zmq_object is not None
            and not isinstance(cluster_zmq_object, cluster_zmq)
            and isinstance(cluster_zmq_object, (list, tuple))
            and node_IP_list is not None
            and isinstance(node_IP_list, (list, tuple))
            and (len(node_IP_list) == 0 or isinstance(node_IP_list[0], bool))
            and isinstance(back_end_select_list, bool)
        ):
            old_node_IP_list = list(cluster_zmq_object)
            old_cpu_gpu = list(node_IP_list)
            old_percentages = CPU_GPU_select_list if CPU_GPU_select_list is not None else []
            old_backends = node_percentages if node_percentages is not None else []
            old_split_matrix = bool(back_end_select_list)
            old_dim = split_matrix if isinstance(split_matrix, int) else 0

            cluster_zmq_object = None
            node_IP_list = old_node_IP_list
            CPU_GPU_select_list = old_cpu_gpu
            node_percentages = old_percentages
            back_end_select_list = old_backends
            split_matrix = old_split_matrix
            dim = old_dim

        if CPU_GPU_select_list is None:
            raise ValueError("CPU_GPU_select_list is required")

        self._owns_cluster_zmq_object = False
        if cluster_zmq_object is None:
            if node_IP_list is None:
                raise ValueError("Provide either cluster_zmq_object or node_IP_list")
            cluster_zmq_object = cluster_zmq(node_IP_list)
            self._owns_cluster_zmq_object = True

        self.cluster_zmq_object = cluster_zmq_object
        node_IP_list = list(cluster_zmq_object.node_IP_list)

        if matrix_labeling != '' and len(node_percentages) == 0:
            node_percentages = [0] * len(node_IP_list)
                
        if not (len(node_IP_list) == len(CPU_GPU_select_list) == len(back_end_select_list) == len(node_percentages)):
            print("❌ NODE CONFIGURATION ERROR: Lengths do not match!")
            print(f"   - node_IP_list: {len(node_IP_list)} nodes")
            print(f"   - CPU_GPU_select_list: {len(CPU_GPU_select_list)} selections")
            print(f"   - back_end_select_list: {len(back_end_select_list)} backends")
            print(f"   - node_percentages: {len(node_percentages)} percentages")
            raise ValueError("Node configuration error: All lists must have the same length")
        
        # Get head node IP addresses from environment variables
        self.IP = cluster_zmq_object.IP
        self.wifi_IP = cluster_zmq_object.wifi_IP

        # ZMQ objects (shared via cluster_zmq_object)
        self.zmq_context = cluster_zmq_object.zmq_context
        self.llama_socket_pool = cluster_zmq_object.llama_socket_pool
        self.llama_socket_pool_wifi = cluster_zmq_object.llama_socket_pool_wifi
        self.ack_receiver_socket = cluster_zmq_object.ack_receiver_socket
        
        # =============== FOLDER PATH CONFIGURATION ===============
        print("\n📁 CONFIGURING STORAGE PATHS...")
        
        # Local paths (head node)
        self.local_DISK_folder = cluster_zmq_object.local_DISK_folder
        self.local_RAM_folder = cluster_zmq_object.local_RAM_folder
        self.local_project_dir = cluster_zmq_object.local_project_dir
        
        # Remote paths (worker nodes)
        self.remote_DISK_folder = cluster_zmq_object.remote_DISK_folder
        self.remote_RAM_folder = cluster_zmq_object.remote_RAM_folder
        self.remote_project_dir = cluster_zmq_object.remote_project_dir
                
        # Get Python executable path
        self.conda_env_dir = cluster_zmq_object.conda_env_dir
        self.python_path = cluster_zmq_object.python_path

        print(f"✅ Using Python path: {self.python_path}")

        # =============== INSTANCE VARIABLE INITIALIZATION ===============
        print("\n📊 INITIALIZING INSTANCE VARIABLES...")
        
        self.matrix_file_path = matrix_file_path
        self.node_IP_list = node_IP_list
        self.node_percentages = node_percentages
        self.dim = dim
        self.transpose = False
        self.CPU_GPU_select_list = CPU_GPU_select_list  # True for GPU, False for CPU
        self.back_end_select_list = back_end_select_list  # 'torch', 'llama', 'opencl'
        self.split_matrix = split_matrix
        self.OG_matrix_shape = []
        self.matrix_labeling= matrix_labeling

        # Extract matrix name from file path
        if torch.is_tensor(matrix_file_path):
            self.matrix_name = matrix_name
        else:
            matrix_file_path_split = matrix_file_path.split('/')
            self.matrix_name = matrix_file_path_split[len(matrix_file_path_split)-1].split('.pt')[0]
            print(f"   Matrix Name: {self.matrix_name}")
            print(f"   Split Matrix: {split_matrix}")
            print(f"   Dimension: {dim}")
        
        # If no backend specified, default to 'llama' for all nodes
        if self.back_end_select_list == []:
            print("   No backend specified, defaulting to 'llama' for all nodes")
            for CPU_GPU_select in self.CPU_GPU_select_list:
                self.back_end_select_list.append('llama')
        
        self.node_matrices = []
        self.matrix_file_paths_list = []  # List for storing matrix file paths
        
        # =============== MATRIX DISTRIBUTION LOGIC ===============
        # auto_set_up format: [system_id, "save"|"load"]

        if len(auto_set_up) == 2:
            if auto_set_up[0] == 1 and auto_set_up[1] == 'save' and self.split_matrix == False:
                self.save_distribute_full_matrix_bin()
            if auto_set_up[0] == 1 and auto_set_up[1] == 'load' and self.split_matrix == False:
                self.load_cluster_matrix()

            if auto_set_up[0] == 1 and auto_set_up[1] == 'save' and self.split_matrix:
                self.convert_to_cluster_matrix_shards()
                self.save_distribute_matrix_shards_bin()
            if auto_set_up[0] == 1 and auto_set_up[1] == 'load' and self.split_matrix:
                self.load_cluster_matrix_shards()    

            if auto_set_up[0] == 2 and auto_set_up[1] == 'save' and self.matrix_labeling == 'a':
                self.convert_to_cluster_matrix_grid()
                self.save_distribute_matrixA_grid_bin()
            if auto_set_up[0] == 2 and auto_set_up[1] == 'load' and self.matrix_labeling == 'a':
                self.load_cluster_matrixA_grid()

            if auto_set_up[0] == 2 and auto_set_up[1] == 'save' and self.matrix_labeling == 'b':
                self.convert_to_cluster_matrix_grid()
                self.save_distribute_matrix_shards_bin()
            if auto_set_up[0] == 2 and auto_set_up[1] == 'load' and self.matrix_labeling == 'b':
                self.load_cluster_matrixB_grid()   
                     
    def convert_to_cluster_matrix_grid(self):
        """
        System 2 (round-robin / block-tiling):

        - Matrix A (matrix_labeling='a'): split into 2 shards along `self.dim` (50/50).
          These two shards are broadcast by `save_distribute_matrixA_grid_bin` across the
          operation slots (first half gets shard_0, second half gets shard_1).

        - Matrix B (matrix_labeling='b'): split into `base_slots` shards along `self.dim`,
          then repeat that list twice so the shard list length equals `op_slots = base_slots * 2`.

        This enables `op_slots` independent GEMMs where:
          shard_i uses A_shard = (i < base_slots ? A0 : A1)
          shard_i uses B_shard = B_{i % base_slots}

        Slot sizing rule (requested):
        - Treat the passed-in lists as "compute slots" (may contain repeated IPs).
        - Keep `op_slots` at 8 (base_slots=4) until compute slots exceed 8.
        - When compute slots exceed 8, grow `op_slots` in steps of 4: 8, 12, 16, ...

        NOTE: In System 2 we ignore `node_percentages` for B chunk sizing and do an even split.
        We still expand `node_percentages` to match `op_slots` to keep list lengths consistent.
        """
        if torch.is_tensor(self.matrix_file_path):
            full_matrix = self.matrix_file_path
        else:
            full_matrix = torch.load(self.matrix_file_path, map_location="cpu")

        self.OG_matrix_shape = list(full_matrix.shape)
        split_dim = int(self.dim)

        # Preserve the original (unexpanded) slot lists so this function is idempotent.
        if not hasattr(self, "_sys2_original_slot_lists"):
            self._sys2_original_slot_lists = {
                "node_IP_list": list(self.node_IP_list),
                "CPU_GPU_select_list": list(self.CPU_GPU_select_list),
                "back_end_select_list": list(self.back_end_select_list),
                "node_percentages": list(self.node_percentages),
            }

        original_node_IP_list = self._sys2_original_slot_lists["node_IP_list"]
        original_cpu_gpu = self._sys2_original_slot_lists["CPU_GPU_select_list"]
        original_backend = self._sys2_original_slot_lists["back_end_select_list"]
        original_percentages = self._sys2_original_slot_lists["node_percentages"]

        compute_slots = len(original_node_IP_list)
        if compute_slots < 1:
            raise ValueError("System 2 requires at least 1 slot")

        # Determine the grid width (base_slots) from the compute slots. This keeps the math and
        # server-side behavior identical to the 4-slot case (8 ops) for up to 8 compute slots.
        base_slots = max(4, 2 * math.ceil(compute_slots / 4))
        op_slots = base_slots * 2

        def _cycle_to_length(items, n):
            if not items:
                raise ValueError("System 2 slot lists must be non-empty")
            return [items[i % len(items)] for i in range(n)]

        # Expand slot lists so `cluster_shard_operation` can dispatch exactly `op_slots` blocks.
        # This is round-robin "wrap-around" over the provided compute slots.
        self.node_IP_list = _cycle_to_length(original_node_IP_list, op_slots)
        self.CPU_GPU_select_list = _cycle_to_length(original_cpu_gpu, op_slots)
        self.back_end_select_list = _cycle_to_length(original_backend, op_slots)
        self.node_percentages = _cycle_to_length(original_percentages, op_slots) if original_percentages else [0.0] * op_slots
        self._sys2_round_robin_expanded = True

        self._sys2_compute_slots = compute_slots
        self._sys2_base_slots = base_slots
        self._sys2_op_slots = op_slots

        if self.matrix_labeling == 'a':
            # A: split into 2 shards along split_dim (views, no copy).
            dim_size = int(full_matrix.size(split_dim))
            s0 = dim_size // 2
            s1 = dim_size - s0
            self.node_matrices = [
                full_matrix.narrow(split_dim, 0, s0),
                full_matrix.narrow(split_dim, s0, s1),
            ]
            print(
                f"✅ Matrix A: {tuple(full_matrix.shape)} → "
                f"[{tuple(self.node_matrices[0].shape)}, {tuple(self.node_matrices[1].shape)}] "
                f"(split along dim={split_dim})"
            )
            print(
                f"   System 2 slots: compute_slots={compute_slots}, base_slots={base_slots}, op_slots={op_slots}"
            )
            return self.node_matrices

        if self.matrix_labeling == 'b':
            # B: split into `base_slots` shards along split_dim (even split), then repeat twice
            # to match `op_slots`.
            dim_size = int(full_matrix.size(split_dim))
            if dim_size < base_slots:
                raise ValueError(f"Cannot split size {dim_size} into {base_slots} non-empty shards")

            base = dim_size // base_slots
            rem = dim_size % base_slots
            sizes = [base + (1 if i < rem else 0) for i in range(base_slots)]

            chunks: list[torch.Tensor] = []
            start = 0
            for sz in sizes:
                chunks.append(full_matrix.narrow(split_dim, start, sz))
                start += sz

            # Order: B0..B{n-1}, then B0..B{n-1}
            self.node_matrices = chunks + chunks
            print(
                f"✅ Matrix B: {tuple(full_matrix.shape)} → {op_slots} shards "
                f"(split {base_slots} then repeat; split along dim={split_dim})"
            )
            print(
                f"   System 2 slots: compute_slots={compute_slots}, base_slots={base_slots}, op_slots={op_slots}"
            )
            return self.node_matrices

        raise ValueError(f"Unknown matrix_labeling={self.matrix_labeling!r} for System 2 grid")

    def convert_to_cluster_matrix_shards(self):
        if torch.is_tensor(self.matrix_file_path):
            full_matrix = self.matrix_file_path
        else:
        # Load full matrix
            full_matrix = torch.load(self.matrix_file_path)

        total_rows = full_matrix.size(self.dim)  # typically dim=0
        self.node_matrices = []

        # Convert percentages to row counts
        if hasattr(self, 'node_percentages') and self.node_percentages:
            total_percentage = sum(self.node_percentages)
            if abs(total_percentage - 1.0) > 1e-6:
                raise ValueError(f"Node percentages must sum to 1. Got {total_percentage}")
            rows_per_node = [int(total_rows * p) for p in self.node_percentages]
            # Adjust for rounding error
            diff = total_rows - sum(rows_per_node)
            if diff != 0:
                rows_per_node[-1] += diff
        else:
            # Default: even split among nodes
            num_nodes = len(self.node_IP_list)
            base_rows = total_rows // num_nodes
            rows_per_node = [base_rows] * num_nodes
            rows_per_node[-1] += total_rows - sum(rows_per_node)

        # Slice the full matrix into shards
        start_idx = 0
        for node_idx, row_count in enumerate(rows_per_node):
            end_idx = start_idx + row_count
            shard = full_matrix.narrow(self.dim, start_idx, row_count).clone()
            self.node_matrices.append(shard)
            start_idx = end_idx

        #print(f"✅ Created {len(self.node_matrices)} shards according to node percentages")
        #for i, shard in enumerate(self.node_matrices):
        #    print(f"  Node {i}: shard shape {shard.shape}")

        return self.node_matrices

    def merged_matrix(self, matrix_shards, start_index, end_index):
        """
        Merge row shards from start_index (inclusive) to end_index (exclusive).
        Returns the concatenated tensor along dim=0.
        """
        if start_index < 0 or end_index <= start_index:
            raise ValueError("Invalid start_index/end_index")
        end_index = min(end_index, len(matrix_shards))
        pieces = [matrix_shards[i] for i in range(start_index, end_index)]
        if not pieces:
            raise ValueError("No shards to merge")
        return torch.cat(pieces, dim=self.dim)

    def save_distribute_matrix_shards_bin(self):
        """Save matrix shards as binary files and distribute to appropriate nodes."""
        for shard_index, node_IP in enumerate(self.node_IP_list):
            print(f"Processing shard {shard_index} for node {node_IP}")
            # Create filename for this shard
            save_name = self.matrix_name.split('.pt')[0] + '_shard_' + str(shard_index)
            # Handle shard for HEAD NODE (local storage)
            if node_IP == self.IP:
                save_name += '.bin'
                save_file_path_DISK = os.path.join(self.local_DISK_folder, save_name)
                save_file_path_RAM = os.path.join(self.local_RAM_folder, save_name)
                
                print(f"  Head node: Saving to DISK={save_file_path_DISK}")
                print(f"  Head node: Saving to RAM={save_file_path_RAM}")
                
                # Save tensor to binary file in both locations
                self.save_matrix_binary(self.node_matrices[shard_index], save_file_path_DISK)
                self.save_matrix_binary(self.node_matrices[shard_index], save_file_path_RAM)
                
                # Store RAM path for later access
                self.matrix_file_paths_list.append(save_file_path_RAM)
                print(f"  Added RAM path to file list")
                    
            # Handle shard for REMOTE NODE
            elif node_IP != self.IP:
                save_name += '.bin'
                print(f"  Remote node {node_IP}: Beginning distribution")

                self.cluster_zmq_object.stream_matrix_binary(node_IP, self.node_matrices[shard_index], save_name)

                self.cluster_zmq_object.wait_for_acks(1, save_name)
                # Step 3: Tell remote node to copy from RAM to DISK
                remote_save_file_path_RAM = os.path.join(self.remote_RAM_folder, save_name)
                remote_disk_dir_full = os.path.join(self.remote_project_dir, self.remote_DISK_folder)
                remote_save_file_path_DISK = os.path.join(remote_disk_dir_full, save_name)
                copy_command = f'cp {remote_save_file_path_RAM} {remote_save_file_path_DISK}'
                print(f"  Step 3: Sending copy command to remote")
                self.cluster_zmq_object.zmq_send_command(node_IP, copy_command)
                # Step 4: Store remote RAM path (not local)
                self.matrix_file_paths_list.append(remote_save_file_path_RAM)
                print(f"  Added remote RAM path to file list: {remote_save_file_path_RAM}")
        return self.matrix_file_paths_list

    def save_distribute_full_matrix_bin(self):
        """
        Save a FULL matrix (no splitting) as binary and distribute to all nodes.
        """
        # Create filename: replace .pt with .bin
        save_name = self.matrix_name.split('.pt')[0] + '.bin'
        print(f"Preparing full matrix: {save_name}")
        
        # Define local file paths
        save_file_path_DISK = os.path.join(self.local_DISK_folder, save_name)
        local_save_file_path_RAM = os.path.join(self.local_RAM_folder, save_name)
        print(f"Local paths - DISK: {save_file_path_DISK}, RAM: {local_save_file_path_RAM}")
        
        # Load the full matrix from PyTorch file

        if torch.is_tensor(self.matrix_file_path):
            full_matrix = self.matrix_file_path
        else:
            print(f"Loading matrix from: {self.matrix_file_path}")
            full_matrix = torch.load(self.matrix_file_path)
            print(f"Matrix loaded - Shape: {full_matrix.shape}")
        
        # Save to binary format locally
        print("Saving to local storage...")
        self.save_matrix_binary(full_matrix, save_file_path_DISK)
        self.save_matrix_binary(full_matrix, local_save_file_path_RAM)

        # Define remote paths (absolute disk path)
        remote_disk_dir_full = os.path.join(self.remote_project_dir, self.remote_DISK_folder)
        remote_save_file_path_RAM = os.path.join(self.remote_RAM_folder, save_name)
        remote_save_file_path_DISK = os.path.join(remote_disk_dir_full, save_name)
        print(f"Remote paths - RAM: {remote_save_file_path_RAM}, DISK: {remote_save_file_path_DISK}")
        
        # Track file paths for each node
        for node_ip in self.node_IP_list:
            if node_ip == self.IP:
                # Head node uses local RAM path
                self.matrix_file_paths_list.append(local_save_file_path_RAM)
            else:
                # Remote nodes use remote RAM path
                self.matrix_file_paths_list.append(remote_save_file_path_RAM)
        
        # Get UNIQUE IPs (no duplicates)
        unique_node_IP_list = list(set(self.node_IP_list))
        unique_remote_count = len([ip for ip in unique_node_IP_list if ip != self.IP])
        
        print(f"Distributing to {unique_remote_count} remote node(s)...")
        
        # Send file to each unique remote node
        for node_ip in unique_node_IP_list:
            if node_ip != self.IP:  # Skip local node
                print(f"Sending to {node_ip}")

                # Step 1: Send the file to remote node's RAM
                self.cluster_zmq_object.zmq_send_file(node_ip, save_file_path_DISK)
                
                # Wait for acknowledgments from remote nodes
                self.cluster_zmq_object.wait_for_acks(1, save_name)

                # Step 2: Tell remote node to copy from RAM to DISK for persistence
                copy_command = f'cp {remote_save_file_path_RAM} {remote_save_file_path_DISK}'
                self.cluster_zmq_object.zmq_send_command(node_ip, copy_command)
        print(f"Full matrix distribution completed")
        return 0

    def save_distribute_matrixA_grid_bin(self):
        """
        Distribute broadcast shards for distributed GEMM.
        """

        # ---------------------------
        # MATRIX A — ROW SHARDS
        # ---------------------------
        if self.matrix_labeling == 'a':
            print("\n📤 Distributing Matrix A row shards")
            
            # ---------------------------
            # MATRIX A — SAVE LOCALLY (save both files to head node)
            # ---------------------------
            self.matrix_file_paths_list = []

            # Create disk folder path
            disk_folder_path = os.path.join(self.local_project_dir, self.local_DISK_folder)
            os.makedirs(disk_folder_path, exist_ok=True)

            # Create file paths
            matrixA1_file_path = os.path.join(self.local_RAM_folder, f'{self.matrix_name}_shard_0.bin')
            matrixA2_file_path = os.path.join(self.local_RAM_folder, f'{self.matrix_name}_shard_1.bin')

            # Save matrices locally to RAM
            self.save_matrix_binary(self.node_matrices[0], matrixA1_file_path)
            self.save_matrix_binary(self.node_matrices[1], matrixA2_file_path)

            # Copy shard 0 to both locations
            shard0_disk_path = os.path.join(disk_folder_path, f'{self.matrix_name}_shard_0.bin')
            subprocess.run(['cp', matrixA1_file_path, shard0_disk_path], check=True)
            print(f"  Copied shard 0 to: {self.local_project_dir}/{self.matrix_name}_shard_0.bin")
            print(f"  Copied shard 0 to: {shard0_disk_path}")
            
            # Copy shard 1 to both locations
            shard1_disk_path = os.path.join(disk_folder_path, f'{self.matrix_name}_shard_1.bin')
            subprocess.run(['cp', matrixA2_file_path, shard1_disk_path], check=True)
            print(f"  Copied shard 1 to: {self.local_project_dir}/{self.matrix_name}_shard_1.bin")
            print(f"  Copied shard 1 to: {shard1_disk_path}")

            # Determine how many nodes get each shard
            total_nodes = len(self.node_IP_list)
            half_nodes = total_nodes // 2  # Integer division
            
            # Track which IPs we've already sent files to
            shard0_sent_to_ips = set()
            shard1_sent_to_ips = set()
            
            # Temporary list to store [IP, file_path] pairs
            ip_shard_pairs = []
            
            for index, node_IP in enumerate(self.node_IP_list):
                # Determine which shard this node gets
                if index < half_nodes:
                    file_path = matrixA1_file_path
                    shard_type = 0
                else:
                    file_path = matrixA2_file_path
                    shard_type = 1
                
                # Store the IP and file path pair
                ip_shard_pairs.append([node_IP, file_path])
                
                # Send files to remote nodes (skip local IP)
                if node_IP != self.IP:
                    if shard_type == 0 and node_IP not in shard0_sent_to_ips:
                        # Send the file to remote RAM
                        self.cluster_zmq_object.zmq_send_file(node_IP, matrixA1_file_path)
                                                
                        # Send command to copy from RAM to disk
                        remote_filename = os.path.basename(matrixA1_file_path)
                        
                        remote_save_file_path_RAM = os.path.join(self.remote_RAM_folder, remote_filename)
                        remote_save_file_path_DISK = os.path.join(self.remote_project_dir, self.remote_DISK_folder, remote_filename)
                        copy_command = f'cp {remote_save_file_path_RAM} {remote_save_file_path_DISK}'
                        self.cluster_zmq_object.zmq_send_command(node_IP, copy_command)
                        shard0_sent_to_ips.add(node_IP)
                        print(f'Sent shard 0 to IP: {node_IP}')
                    
                    elif shard_type == 1 and node_IP not in shard1_sent_to_ips:
                        # Send the file to remote RAM
                        self.cluster_zmq_object.zmq_send_file(node_IP, matrixA2_file_path)
                        # Send command to copy from RAM to disk
                        remote_filename = os.path.basename(matrixA2_file_path)
                        self.cluster_zmq_object.wait_for_acks(1, remote_filename)
                        remote_save_file_path_RAM = os.path.join(self.remote_RAM_folder, remote_filename)
                        remote_save_file_path_DISK = os.path.join(self.remote_project_dir, self.remote_DISK_folder, remote_filename)
                        copy_command = f'cp {remote_save_file_path_RAM} {remote_save_file_path_DISK}'
                        self.cluster_zmq_object.zmq_send_command(node_IP, copy_command)
                        
                        shard1_sent_to_ips.add(node_IP)
                        print(f'Sent shard 1 to IP: {node_IP}')
            
            # Print the IP-shard assignments for debugging
            print("\n📋 Node shard assignments:")
            for ip, path in ip_shard_pairs:
                shard_name = "shard_0" if path == matrixA1_file_path else "shard_1"
                print(f"  {ip} -> {shard_name}")
            
            # Now extract just the file paths (remove IPs) and store in matrix_file_paths_list
            self.matrix_file_paths_list = [file_path for _, file_path in ip_shard_pairs]
            #print(f"\n✅ Final matrix_file_paths_list (paths only):")
            #for i, path in enumerate(self.matrix_file_paths_list):
            #    shard_name = "shard_0" if path == matrixA1_file_path else "shard_1"
            #    print(f"  Node {i}: {shard_name}")
        return self.matrix_file_paths_list

    def save_matrix_binary(self, matrix, filename):
        """
        Save a PyTorch tensor or numpy array as a binary file.
        
        Binary format (v2):
        [dtype_tag(int32), batch(int32), depth(int32), rows(int32), cols(int32), data(bytes)]

        dtype_tag is NEGATIVE to stay backward compatible with legacy files where
        the first int was `ndim` (typically 4).
          -1 = float32
          -2 = float16
          -3 = bfloat16 (payload is raw bf16 bits as int16/uint16)

        Legacy format (v1):
        [ndim(int32), dims..., data(float32)]

        Always saves as 4D format (batch, channel, height, width) for consistency.
        
        Args:
            matrix: PyTorch tensor or numpy array to save
            filename: Path where the binary file will be saved
        """
        verbose = os.environ.get("SAVE_MATRIX_BINARY_VERBOSE", "1") == "1"
        if verbose:
            print(f"Saving matrix to binary file: {filename}")

        parent = os.path.dirname(filename)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # ===== FAST CPU SAVE PATH =====
        # Avoid `.tobytes()` (copies the whole buffer). Use ndarray.tofile(...) instead.
        if isinstance(matrix, torch.Tensor):
            t = matrix.detach()
            if t.device.type != "cpu":
                t = t.cpu()
            if t.ndim == 2:
                t = t.reshape(1, 1, t.shape[0], t.shape[1])
            elif t.ndim == 3:
                t = t.reshape(1, t.shape[0], t.shape[1], t.shape[2])
            elif t.ndim == 4:
                pass
            else:
                raise ValueError(f"Invalid tensor dimensionality: {t.ndim}")

            if not t.is_contiguous():
                t = t.contiguous()

            if t.dtype == torch.float32:
                dtype_tag = -1
                payload_np = t.numpy()
            elif t.dtype == torch.float16:
                dtype_tag = -2
                payload_np = t.numpy()
            elif t.dtype == torch.bfloat16:
                dtype_tag = -3
                payload_np = t.view(torch.int16).numpy()
            else:
                raise ValueError(f"Unsupported tensor dtype for save_matrix_binary: {t.dtype}")

            shape = tuple(int(x) for x in t.shape)
            if verbose:
                print(f"Original shape: {tuple(matrix.shape)}")
                print(f"Converted to 4D: {shape}")
                print("Writing binary file...")

            with open(filename, "wb") as f:
                f.write(struct.pack("iiiii", dtype_tag, shape[0], shape[1], shape[2], shape[3]))
                payload_np.tofile(f)
            return

        if isinstance(matrix, np.ndarray):
            arr = np.asarray(matrix, order="C")
            if arr.dtype == np.float32:
                dtype_tag = -1
            elif arr.dtype == np.float16:
                dtype_tag = -2
            else:
                raise ValueError(f"Unsupported numpy dtype for save_matrix_binary: {arr.dtype}")

            if arr.ndim == 2:
                arr = arr.reshape(1, 1, arr.shape[0], arr.shape[1])
            elif arr.ndim == 3:
                arr = arr.reshape(1, arr.shape[0], arr.shape[1], arr.shape[2])
            elif arr.ndim == 4:
                pass
            else:
                raise ValueError(f"Invalid array dimensionality: {arr.ndim}")

            shape = tuple(int(x) for x in arr.shape)
            if verbose:
                print(f"Original shape: {tuple(matrix.shape)}")
                print(f"Converted to 4D: {shape}")
                print("Writing binary file...")

            with open(filename, "wb") as f:
                f.write(struct.pack("iiiii", dtype_tag, shape[0], shape[1], shape[2], shape[3]))
                arr.tofile(f)
            return

        raise ValueError("Unsupported input type for save_matrix_binary")

    def convert_bin_matrix_to_pt(self, filename, force_2d=True):  
        """  
        Load a binary matrix saved in the format:
        - v2: [dtype_tag(int32), batch(int32), depth(int32), rows(int32), cols(int32), data(bytes)]
        - v1: [ndim(int32), dims..., data(float32)]
        
        Args:  
            filename: path to binary file  
            force_2d: if True, flatten batch*depth*rows into 2D (rows x cols)  
            
        Returns:  
            PyTorch tensor  
        """  
        # Fast path: mmap + frombuffer avoids copying file payload into Python bytes.
        # We still detach into a normal (owned) tensor at the end so callers can delete the file safely.
        with open(filename, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            offset = 0
            if mm.size() < 4:
                raise ValueError("File too short to read ndim")
            tag_or_ndim = struct.unpack_from('i', mm, offset)[0]
            offset += 4

            # v2 header (dtype tag)
            if tag_or_ndim < 0:
                dtype_tag = tag_or_ndim
                ndim = 4

                header_bytes = 4 * 4
                if mm.size() < offset + header_bytes:
                    raise ValueError("File too short to read 4D dimensions")
                dims = list(struct.unpack_from('iiii', mm, offset))
                offset += header_bytes

                num_elements = int(np.prod(dims, dtype=np.int64))

                if dtype_tag == -1:
                    np_dtype = np.float32
                elif dtype_tag == -2:
                    np_dtype = np.float16
                elif dtype_tag == -3:
                    np_dtype = np.uint16  # raw bf16 bits
                else:
                    raise ValueError(f"Unsupported dtype_tag in binary file: {dtype_tag}")

                data_bytes_needed = num_elements * np.dtype(np_dtype).itemsize
                if mm.size() < offset + data_bytes_needed:
                    raise ValueError(f"File too short to read {num_elements} elements of {np_dtype}")

                data_np = np.frombuffer(mm, dtype=np_dtype, count=num_elements, offset=offset)
                tensor_np = data_np.reshape(dims)

            # v1 legacy header (ndim)
            else:
                ndim = tag_or_ndim
                dtype_tag = -1  # legacy files are float32

                header_bytes = ndim * 4
                if mm.size() < offset + header_bytes:
                    raise ValueError("File too short to read dimensions")
                dims = list(struct.unpack_from('i' * ndim, mm, offset))
                offset += header_bytes

                num_elements = int(np.prod(dims, dtype=np.int64))
                data_bytes_needed = num_elements * 4
                if mm.size() < offset + data_bytes_needed:
                    raise ValueError(f"File too short to read {num_elements} floats")

                data_np = np.frombuffer(mm, dtype=np.float32, count=num_elements, offset=offset)
                tensor_np = data_np.reshape(dims)
        finally:
            # `tensor_np` is a view over `mm`; we will detach below, so it's safe to close here.
            try:
                mm.close()
            except Exception:
                pass
    
        # Optionally flatten batch*depth*rows -> 2D for LLAMA-like 4D tensors  
        if force_2d and ndim == 4:  
            batch, depth, rows, cols = dims  
            tensor_np = tensor_np.reshape(batch * depth * rows, cols)  
    
        # Convert to PyTorch tensor (detach from mm-backed buffer)
        if dtype_tag == -3:
            # bf16 bits (uint16) -> float32 -> bfloat16
            u16 = np.array(tensor_np, copy=True, dtype=np.uint16)
            u32 = (u16.astype(np.uint32) << 16)
            f32 = u32.view(np.float32)
            tensor_pt = torch.from_numpy(f32).to(dtype=torch.bfloat16)
        else:
            tensor_pt = torch.from_numpy(np.array(tensor_np, copy=True))
    
        # Info  
        print(f"✅ Loaded {filename}")  
        print(f"  Original dims: {dims}")  
        print(f"  Result tensor shape: {tensor_pt.shape}, size: {tensor_pt.numel() * tensor_pt.element_size():,} bytes")
        stats_max_elems = int(os.environ.get("CONVERT_BIN_STATS_MAX_ELEMS", "2000000"))
        if tensor_pt.numel() <= stats_max_elems:
            print(f"  Data range: [{tensor_pt.min().item():.6f}, {tensor_pt.max().item():.6f}]")
        else:
            print(f"  Data range: [skipped; numel={tensor_pt.numel():,} > {stats_max_elems:,}]")
    
        return tensor_pt

    def load_cluster_matrix(self):
        """
        Load a full matrix (not split) from disk and distribute to all nodes.
        """
        try:
            # Create filename for the binary matrix
            base_name = self.matrix_name + '.bin'
            combined_name = self.matrix_name + '_combined.bin'
            print(f"Loading full matrix: {base_name}")
            
            base_disk_path = os.path.join(self.local_project_dir, self.local_DISK_folder, base_name)
            
            if not os.path.exists(base_disk_path):
                print(
                    "Error: Base matrix binary not found. Combined outputs are write-only "
                    "and cannot be reused as inputs. Regenerate shards or rerun the operation "
                    "with send_back=False to keep a distributed input."
                )
                return False

            source_path = base_disk_path
            source_filename = base_name
            print(f"Source file: {source_path}")
            # Copy to RAM for local access
            local_ram_path = os.path.join(self.local_RAM_folder, base_name)
            print(f"Copying to local RAM...")
            subprocess.run(f'cp {source_path} {self.local_RAM_folder}', shell=True, check=True)
            
            # Get unique nodes to avoid duplicate transfers
            unique_node_IP_list = list(set(self.node_IP_list))
            
            # Define remote paths (mirror the source filename)
            remote_disk_path = self.remote_DISK_folder + source_filename
            remote_RAM_path = self.remote_RAM_folder + source_filename
            
            # Track file paths for all nodes
            for node_ip in self.node_IP_list:
                if node_ip == self.IP:
                    # Head node uses local RAM path
                    self.matrix_file_paths_list.append(local_ram_path)
                else:
                    # Remote nodes use remote RAM path
                    self.matrix_file_paths_list.append(remote_RAM_path)
            
            # Distribute to remote nodes
            print(f"Distributing to remote nodes...")
            for node_ip in unique_node_IP_list:
                if node_ip != self.IP:
                    # Send file to remote node
                    self.cluster_zmq_object.zmq_send_file(node_ip, source_path)
                    
                    # Send command to copy from remote disk to remote RAM
                    copy_command = f'cp {self.remote_project_dir}{remote_disk_path} {self.remote_RAM_folder}'
                    self.cluster_zmq_object.zmq_send_command(node_ip, copy_command)
                    
        except Exception as e:
            print(f"Error loading matrix: {e}")
            return False
        
        print(f"Matrix loaded successfully")
        return True

    def load_cluster_matrix_shards(self):
        """
        Load distributed matrix shards from storage.
        
        This method checks if matrix shards are already in RAM, and if not,
        loads them from disk to RAM on the nodes that actually need them.
        
        Design (System 1):
        - Each "slot" in `self.node_IP_list` corresponds to a shard index.
        - Shard files are distributed across the cluster; the head node does NOT
          necessarily have all shard files on its own disk.
        - The head node only copies its *assigned* shards into its local RAM.
        - Each worker node only needs its assigned shard indices in its RAM.
        """
        
        # Initialize the file paths list
        self.matrix_file_paths_list = []
        
        print(f"Loading cluster matrix shards: {self.matrix_name}")
        print(f"Number of nodes/shard locations: {len(self.node_IP_list)}")

        # Map each IP to the shard indices it is responsible for.
        shard_indices_by_ip = {}
        for shard_index, node_ip in enumerate(self.node_IP_list):
            shard_indices_by_ip.setdefault(node_ip, []).append(shard_index)

        head_ip = self.IP
        head_shard_indices = shard_indices_by_ip.get(head_ip, [])

        # The controller tracks per-slot paths:
        # - head slots refer to head local RAM
        # - worker slots refer to worker RAM (same /dev/shm/... path on that machine)
        for shard_index, node_ip in enumerate(self.node_IP_list):
            shard_filename = f"{self.matrix_name}_shard_{shard_index}.bin"
            if node_ip == head_ip:
                self.matrix_file_paths_list.append(os.path.join(self.local_RAM_folder, shard_filename))
            else:
                self.matrix_file_paths_list.append(os.path.join(self.remote_RAM_folder, shard_filename))

        # ===== HEAD: ensure only its assigned shards exist in local RAM =====
        if head_shard_indices:
            print(f"Head node assigned shard indices: {head_shard_indices}")
            missing_local = []
            for shard_index in head_shard_indices:
                shard_filename = f"{self.matrix_name}_shard_{shard_index}.bin"
                local_ram_path = os.path.join(self.local_RAM_folder, shard_filename)
                if not os.path.exists(local_ram_path):
                    missing_local.append(shard_index)

            if not missing_local:
                print("Found required head-node shards in local RAM")
            else:
                print(f"Head-node shards missing in RAM: {missing_local} → loading from disk...")
                for shard_index in missing_local:
                    shard_filename = f"{self.matrix_name}_shard_{shard_index}.bin"
                    local_disk_source = os.path.join(self.local_project_dir, self.local_DISK_folder, shard_filename)
                    local_ram_dest = os.path.join(self.local_RAM_folder, shard_filename)
                    local_copy_command = f'cp "{local_disk_source}" "{local_ram_dest}"'
                    print(f"  Local copy command: {local_copy_command}")
                    subprocess.run(local_copy_command, shell=True, check=True)
        else:
            print("⚠️  Head node has no assigned shards in this slot list")

        # ===== WORKERS: best-effort copy only their assigned shards into remote RAM =====
        # This does not wait for ACKs because Linux commands are not ACKed, but it keeps
        # worker behavior aligned with the head's slot list.
        for node_ip, shard_indices in shard_indices_by_ip.items():
            if node_ip == head_ip:
                continue
            for shard_index in shard_indices:
                shard_filename = f"{self.matrix_name}_shard_{shard_index}.bin"
                remote_disk_path = os.path.join(self.remote_DISK_folder, shard_filename)
                remote_ram_path = os.path.join(self.remote_RAM_folder, shard_filename)
                remote_copy_command = f'cp "{self.remote_project_dir}{remote_disk_path}" "{remote_ram_path}"'
                print(f"  Remote node {node_ip}: {remote_copy_command}")
                self.cluster_zmq_object.zmq_send_command(node_ip, remote_copy_command)

        print(f"\nMatrix shard loading complete")
        return True
 
    def load_cluster_matrixA_grid(self):
        """
        Load Matrix A shards from disk to RAM for distributed GEMM.
        Simple version: just copy from local_DISK_folder to local_RAM_folder
        """
                
        print(f"\n📥 Loading Matrix A grid shards from disk to RAM")

        # System 2 sizing: expand the provided compute slots to the derived op_slots so that
        # `cluster_shard_operation` runs the correct number of grid operations (8 for up to 8
        # compute slots, then 12/16/...).
        if not hasattr(self, "_sys2_original_slot_lists"):
            self._sys2_original_slot_lists = {
                "node_IP_list": list(self.node_IP_list),
                "CPU_GPU_select_list": list(self.CPU_GPU_select_list),
                "back_end_select_list": list(self.back_end_select_list),
                "node_percentages": list(self.node_percentages),
            }

        original_node_IP_list = self._sys2_original_slot_lists["node_IP_list"]
        original_cpu_gpu = self._sys2_original_slot_lists["CPU_GPU_select_list"]
        original_backend = self._sys2_original_slot_lists["back_end_select_list"]
        original_percentages = self._sys2_original_slot_lists["node_percentages"]

        compute_slots = len(original_node_IP_list)
        if compute_slots < 1:
            raise ValueError("System 2 requires at least 1 slot")

        base_slots = max(4, 2 * math.ceil(compute_slots / 4))
        op_slots = base_slots * 2

        def _cycle_to_length(items, n):
            if not items:
                raise ValueError("System 2 slot lists must be non-empty")
            return [items[i % len(items)] for i in range(n)]

        self.node_IP_list = _cycle_to_length(original_node_IP_list, op_slots)
        self.CPU_GPU_select_list = _cycle_to_length(original_cpu_gpu, op_slots)
        self.back_end_select_list = _cycle_to_length(original_backend, op_slots)
        self.node_percentages = (
            _cycle_to_length(original_percentages, op_slots)
            if original_percentages
            else [0.0] * op_slots
        )
        self._sys2_round_robin_expanded = True
        self._sys2_compute_slots = compute_slots
        self._sys2_base_slots = base_slots
        self._sys2_op_slots = op_slots

        print(
            f"   System 2 slots: compute_slots={compute_slots}, "
            f"base_slots={base_slots}, op_slots={op_slots}"
        )
        
        # Initialize the file paths list
        self.matrix_file_paths_list = []
        
        # Determine how many nodes get each shard
        total_nodes = len(self.node_IP_list)
        half_nodes = total_nodes // 2  # Integer division
        
        # Create file names for Matrix A shards
        shard0_filename = f'{self.matrix_name}_shard_0.bin'
        shard1_filename = f'{self.matrix_name}_shard_1.bin'
        
        # Define disk paths - CORRECTED: should be in local_project_dir + local_DISK_folder
        local_shard0_disk_path = os.path.join(self.local_project_dir, self.local_DISK_folder, shard0_filename)
        local_shard1_disk_path = os.path.join(self.local_project_dir, self.local_DISK_folder, shard1_filename)
        
        # Define RAM paths
        local_shard0_ram_path = os.path.join(self.local_RAM_folder, shard0_filename)
        local_shard1_ram_path = os.path.join(self.local_RAM_folder, shard1_filename)
        
        # Check if shards exist in disk
        print(f"Looking for shards in: {os.path.join(self.local_project_dir, self.local_DISK_folder)}")
        print(f"  Shard 0 path: {local_shard0_disk_path}")
        print(f"  Shard 1 path: {local_shard1_disk_path}")
        
        if not os.path.exists(local_shard0_disk_path):
            print(f"❌ Error: shard_0 not found at: {local_shard0_disk_path}")
            return False
        
        if not os.path.exists(local_shard1_disk_path):
            print(f"❌ Error: shard_1 not found at: {local_shard1_disk_path}")
            return False
        
        # Copy shard 0 from disk to RAM
        print(f"\n📋 Copying shard_0 from disk to RAM...")
        shard0_copy_cmd = f'cp "{local_shard0_disk_path}" "{local_shard0_ram_path}"'
        print(f"  Command: {shard0_copy_cmd}")
        subprocess.run(shard0_copy_cmd, shell=True, check=True)
        print(f"  ✅ shard_0 copied to RAM")
        
        # Copy shard 1 from disk to RAM  
        print(f"\n📋 Copying shard_1 from disk to RAM...")
        shard1_copy_cmd = f'cp "{local_shard1_disk_path}" "{local_shard1_ram_path}"'
        print(f"  Command: {shard1_copy_cmd}")
        subprocess.run(shard1_copy_cmd, shell=True, check=True)
        print(f"  ✅ shard_1 copied to RAM")
        
        # Create the distribution pattern (same as save_distribute_matrixA_grid_bin)
        print(f"\n📋 Creating distribution pattern for {total_nodes} nodes:")
        
        # Track which IPs have been processed for remote commands
        shard0_processed_ips = set()
        shard1_processed_ips = set()
        
        for index, node_IP in enumerate(self.node_IP_list):
            if index < half_nodes:
                # First half gets shard_0
                self.matrix_file_paths_list.append(local_shard0_ram_path)
                print(f"  Node {index} ({node_IP}): assigned shard_0")
                
                # Send command to remote nodes to copy their shard from disk to RAM
                if node_IP != self.IP and node_IP not in shard0_processed_ips:
                    remote_disk_path = os.path.join(self.remote_DISK_folder, shard0_filename)
                    remote_ram_path = os.path.join(self.remote_RAM_folder, shard0_filename)
                    # CORRECTED: remote_disk_path should be prefixed with remote_project_dir
                    remote_copy_command = f'cp "{self.remote_project_dir}{remote_disk_path}" "{remote_ram_path}"'
                    
                    print(f"    Sending to remote {node_IP}: {remote_copy_command}")
                    self.cluster_zmq_object.zmq_send_command(node_IP, remote_copy_command)
                    shard0_processed_ips.add(node_IP)
                    
            else:
                # Second half gets shard_1
                self.matrix_file_paths_list.append(local_shard1_ram_path)
                print(f"  Node {index} ({node_IP}): assigned shard_1")
                
                # Send command to remote nodes to copy their shard from disk to RAM
                if node_IP != self.IP and node_IP not in shard1_processed_ips:
                    remote_disk_path = os.path.join(self.remote_DISK_folder, shard1_filename)
                    remote_ram_path = os.path.join(self.remote_RAM_folder, shard1_filename)
                    # CORRECTED: remote_disk_path should be prefixed with remote_project_dir
                    remote_copy_command = f'cp "{self.remote_project_dir}{remote_disk_path}" "{remote_ram_path}"'
                    
                    print(f"    Sending to remote {node_IP}: {remote_copy_command}")
                    self.cluster_zmq_object.zmq_send_command(node_IP, remote_copy_command)
                    shard1_processed_ips.add(node_IP)
        
        # ===== LOADING COMPLETE =====
        print(f"\n✅ Matrix A grid loading complete")
        print(f"   Total nodes: {total_nodes}")
        print(f"   First {half_nodes} nodes: shard_0")
        print(f"   Remaining {total_nodes - half_nodes} nodes: shard_1")
        print(f"   File paths tracked: {len(self.matrix_file_paths_list)}")
        
        return True

    def load_cluster_matrixB_grid(self):
        """
        System 2 (grid) Matrix B loader.

        Loads the required B shards from local disk into RAM and prepares
        `matrix_file_paths_list` for `cluster_shard_operation` using the same
        op-slot sizing rule as `convert_to_cluster_matrix_grid`.

        This is System 2 only and does not change System 1 behavior.
        """
        if self.matrix_labeling != 'b':
            raise ValueError(
                "load_cluster_matrixB_grid is only valid for System 2 Matrix B "
                "(matrix_labeling='b')."
            )

        print(f"\n📥 Loading Matrix B grid shards from disk to RAM")

        # Preserve the original (unexpanded) slot lists so sizing is stable across calls.
        if not hasattr(self, "_sys2_original_slot_lists"):
            self._sys2_original_slot_lists = {
                "node_IP_list": list(self.node_IP_list),
                "CPU_GPU_select_list": list(self.CPU_GPU_select_list),
                "back_end_select_list": list(self.back_end_select_list),
                "node_percentages": list(self.node_percentages),
            }

        original_node_IP_list = self._sys2_original_slot_lists["node_IP_list"]
        original_cpu_gpu = self._sys2_original_slot_lists["CPU_GPU_select_list"]
        original_backend = self._sys2_original_slot_lists["back_end_select_list"]
        original_percentages = self._sys2_original_slot_lists["node_percentages"]

        compute_slots = len(original_node_IP_list)
        if compute_slots < 1:
            raise ValueError("System 2 requires at least 1 slot")

        base_slots = max(4, 2 * math.ceil(compute_slots / 4))
        op_slots = base_slots * 2

        def _cycle_to_length(items, n):
            if not items:
                raise ValueError("System 2 slot lists must be non-empty")
            return [items[i % len(items)] for i in range(n)]

        # Expand compute slots to exactly op_slots (wrap-around).
        self.node_IP_list = _cycle_to_length(original_node_IP_list, op_slots)
        self.CPU_GPU_select_list = _cycle_to_length(original_cpu_gpu, op_slots)
        self.back_end_select_list = _cycle_to_length(original_backend, op_slots)
        self.node_percentages = (
            _cycle_to_length(original_percentages, op_slots)
            if original_percentages
            else [0.0] * op_slots
        )

        self._sys2_round_robin_expanded = True
        self._sys2_compute_slots = compute_slots
        self._sys2_base_slots = base_slots
        self._sys2_op_slots = op_slots

        print(
            f"   System 2 slots: compute_slots={compute_slots}, "
            f"base_slots={base_slots}, op_slots={op_slots}"
        )

        # Initialize the file paths list (must be length op_slots).
        self.matrix_file_paths_list = []

        # In System 2, Matrix B shard files are saved per *operation slot* index
        # (e.g. ..._shard_0.bin .. ..._shard_{op_slots-1}.bin) on the node that executed that
        # slot during the save path. During load we:
        # - copy local-slot shards from local disk -> local RAM
        # - instruct remote nodes to copy their shard from remote disk -> remote RAM
        processed_remote_pairs = set()  # (node_ip, op_index)
        ok = True

        for op_index, node_IP in enumerate(self.node_IP_list):
            shard_filename = f"{self.matrix_name}_shard_{op_index}.bin"

            local_disk_path = os.path.join(
                self.local_project_dir, self.local_DISK_folder, shard_filename
            )
            local_ram_path = os.path.join(self.local_RAM_folder, shard_filename)
            remote_disk_path = os.path.join(self.remote_DISK_folder, shard_filename)
            remote_ram_path = os.path.join(self.remote_RAM_folder, shard_filename)

            # Track per-slot RAM path (must align with the slot's node assignment).
            if node_IP == self.IP:
                self.matrix_file_paths_list.append(local_ram_path)
            else:
                self.matrix_file_paths_list.append(remote_ram_path)

            if node_IP == self.IP:
                if not os.path.exists(local_ram_path):
                    if not os.path.exists(local_disk_path):
                        print(f"❌ Error: local shard not found at: {local_disk_path}")
                        ok = False
                        continue
                    copy_cmd = f'cp "{local_disk_path}" "{local_ram_path}"'
                    subprocess.run(copy_cmd, shell=True, check=True)
            else:
                if (node_IP, op_index) in processed_remote_pairs:
                    continue
                remote_copy_command = (
                    f'cp "{self.remote_project_dir}{remote_disk_path}" '
                    f'"{remote_ram_path}"'
                )
                self.cluster_zmq_object.zmq_send_command(node_IP, remote_copy_command)
                processed_remote_pairs.add((node_IP, op_index))

        print(f"\n✅ Matrix B grid loading complete")
        print(f"   File paths tracked: {len(self.matrix_file_paths_list)}")
        if not ok:
            print("⚠️  One or more local shards were missing; regenerate with auto_set_up=[2,'save'].")
        return ok

    def cluster_shard_operation(self, cluster_matrixB, TransposeA=False, TransposeB=False, send_back_result=True, operation='mul'):
        """
        Perform a distributed matrix operation across the cluster.
        
        Args:
            cluster_matrixB: Another cluster_matrix instance for the second operand
            TransposeA: Whether to transpose matrix A
            TransposeB: Whether to transpose matrix B  
            send_back_result: Whether to combine results into single file (True) 
                            or keep distributed (False)
            operation: Operation to perform ('mul', 'add', 'sub')
        
        Returns:
            Base name of the result file(s)
        """
        
        print(f"\n{'='*60}")
        print(f"🚀 STARTING CLUSTER OPERATION")
        print(f"{'='*60}")
        print(f"Matrix A: {self.matrix_name}")
        print(f"Matrix B: {cluster_matrixB.matrix_name}")
        print(f"Operation: {operation}")
        print(f"Transpose A: {TransposeA}, Transpose B: {TransposeB}")
        node_IP_list_len = len(self.node_IP_list)

        # Single-node runs should not use send_back/combining logic.
        # The result will be written locally as a single shard (e.g. `<base>_shard_0.bin`)
        # and can be loaded directly via `convert_bin_matrix_to_pt(...)`.
        if node_IP_list_len == 1 and send_back_result:
            print("⚠️  Single-node mode detected: forcing send_back_result=False (no combine/send_back).")
            send_back_result = False

        print(f"Send back result: {send_back_result}")
        print(f"Number of shards: {node_IP_list_len}")
        
        # ===== TRACK GPU USAGE PER NODE =====
        # This ensures multiple GPUs on the same node get used properly
        node_gpu_counters = {}
        
        print(f"\n📤 DISTRIBUTING OPERATIONS TO NODES")
        
        # Send operation commands to each node for its assigned shard
        for shard_index, (node_IP, CPU_GPU_select, back_end_select, node_matrix) in enumerate(zip(
            self.node_IP_list,  
            self.CPU_GPU_select_list,
            self.back_end_select_list, 
            self.matrix_file_paths_list
        )):
            print(f"\nProcessing shard {shard_index}:")
            
            # Get GPU number for this node
            if node_IP not in node_gpu_counters:
                node_gpu_counters[node_IP] = 0
            
            current_gpu_number = node_gpu_counters[node_IP]
            
            # INCREMENT NOW for next operation on this node
            if CPU_GPU_select:
                node_gpu_counters[node_IP] += 1
            
            print(f"  Node: {node_IP}")
            print(f"  Backend: {back_end_select}")
            print(f"  Use GPU: {CPU_GPU_select} (GPU #{current_gpu_number})")
            print(f"  Next GPU for this node will be: #{node_gpu_counters[node_IP]}")
            
            # Get file paths for both matrices
            matrix_a = node_matrix  # Current matrix shard
            matrix_b = cluster_matrixB.matrix_file_paths_list[shard_index]  # Other matrix shard
            
            # Convert booleans to lowercase strings for command
            use_gpu_str = str(CPU_GPU_select).lower()  # "true" or "false"
            
            # ===== TRANSPOSE LOGIC HANDLING =====
            # Handle backend-specific transpose quirks.
            # GGML (llama) uses column-major; flip TransposeB and swap operand order
            # to mirror the previously working cross-backend behavior.
            local_TransposeA = TransposeA
            local_TransposeB = TransposeB
            if back_end_select == 'llama':
                local_TransposeB = not TransposeB

            TransposeA_str = str(local_TransposeA).lower()
            TransposeB_str = str(local_TransposeB).lower()
            print(f"  Final transpose flags - A: {TransposeA_str}, B: {TransposeB_str}")
            
            # ===== PREPARE SEND_BACK FLAG =====
            if not send_back_result:
                send_back = 0
                print("Send back result: No (keep distributed)")
            else:
                shard_count = node_IP_list_len
                # join_dim must ALWAYS be set
                join_dim = self.dim  # 0 or 1
                # Encode: join_dim * 10 + shard_count
                send_back = join_dim * 10 + shard_count
                # System 2 → negative
                if self.matrix_labeling in ('a', 'b'):
                    send_back = -send_back
                print(f"Send back result: Yes ({send_back} shards will be combined)")
                print(f"  → system={'2' if send_back < 0 else '1'}, join_dim={join_dim}, shards={shard_count}")

            # ===== BUILD COMMAND FOR SPECIFIC BACKEND =====
            command = (
                f"server_command={back_end_select} "
                f"{matrix_a} "          # GGML expects B first
                f"{TransposeA_str} "
                f"{matrix_b} "          # Then A
                f"{TransposeB_str} "
                f"{use_gpu_str} "
                f"{current_gpu_number} "
                f"{send_back} "
                f"{operation} "
                f"2 "
                f"{shard_index}"
            )
    
            # ===== SEND COMMAND TO NODE =====
            print(f"  Sending command to node...")
            socket_eth = self.llama_socket_pool[node_IP]
            socket_eth.send_multipart([command.encode()])
            print(f"  ✅ Command sent to node {node_IP}")
        
        # ===== WAIT FOR ACKS FROM ALL NODES =====
        expected_acks = len(self.node_IP_list)  # one ACK per shard/operation
        print(f"\n⏳ WAITING FOR ACKS FROM NODES ({expected_acks})")
        self.cluster_zmq_object.wait_for_acks(expected_acks, "ACK_matrixOp_complete")
        # ===== OPERATION COMPLETE =====
        print(f"✅ CLUSTER OPERATION COMPLETE")

        # When keep-distributed, return a cluster_matrix wired to the shard outputs
        # ===== SETUP RESULT FILENAMES =====
        # Result names match the operand order we send to the server:
        # self (matrix A) first, then cluster_matrixB (matrix B)
        base_result_name=''
        if back_end_select == 'torch':
            base_result_name = f"{self.matrix_name}x{cluster_matrixB.matrix_name}"
            print(f"\n📊 Result base: {base_result_name} (send_back={send_back_result})")
        if back_end_select == 'llama':
            base_result_name = f"{cluster_matrixB.matrix_name}x{self.matrix_name}"
            print(f"\n📊 Result base: {base_result_name} (send_back={send_back_result})")

        # CASE 1: Single node, keep distributed → just convert single shard
        if not send_back_result and node_IP_list_len == 1:
            path = self.local_RAM_folder + base_result_name + '_shard_0.bin'
            # Give the server a moment to finish writing the shard file (RAM-backed FS).
            if not os.path.exists(path):
                for _ in range(200):  # ~2s
                    time.sleep(0.01)
                    if os.path.exists(path):
                        break
            combined_matrix = self.convert_bin_matrix_to_pt(path)
            return combined_matrix

        # CASE 2: Multiple nodes, want combined result → wait for combined PT
        if send_back_result and node_IP_list_len > 1:  
            combined_matrix = self.cluster_zmq_object.wait_for_combined_pt(base_result_name)
            return combined_matrix

        # CASE 3: Multiple nodes, keep distributed → return cluster_matrix
        if not send_back_result and node_IP_list_len > 1: 
            result_cluster_matrix = cluster_matrix(  
                base_result_name,  
                self.node_IP_list,  
                self.CPU_GPU_select_list,  
                self.node_percentages,  
                self.back_end_select_list,  
                True  
            )  
            return result_cluster_matrix

        # fallback (should rarely happen)
        return False

