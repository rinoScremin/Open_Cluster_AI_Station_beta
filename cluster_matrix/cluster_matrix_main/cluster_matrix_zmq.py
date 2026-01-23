import torch
import os
import time
import struct
import numpy as np
import zmq

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

        import io

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
   