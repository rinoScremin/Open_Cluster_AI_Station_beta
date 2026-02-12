#include "ggml.h"
#include "ggml-backend.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdlib>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <csignal>
#include <zmq.hpp>
#include <filesystem>
#include <set>
#include <array>
#include <cstdio>
#include <sstream>
#include <map>
#include "matrix_backend.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>
#include <list>
#include <cstdint>
#include <algorithm>
#include <cctype>
#include <torch/torch.h>
#include <unordered_map>
#include <unordered_set>
#include <limits>


static inline float bf16_to_f32(uint16_t v) {
    uint32_t bits = uint32_t(v) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline float fp16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t(h & 0x8000u) << 16);
    uint32_t exp = (h & 0x7C00u) >> 10;
    uint32_t mant = (h & 0x03FFu);

    uint32_t f_bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            f_bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            const uint32_t f_exp = (exp + (127 - 15)) << 23;
            const uint32_t f_mant = mant << 13;
            f_bits = sign | f_exp | f_mant;
        }
    } else if (exp == 0x1F) {
        f_bits = sign | 0x7F800000u | (mant << 13);
    } else {
        const uint32_t f_exp = (exp + (127 - 15)) << 23;
        const uint32_t f_mant = mant << 13;
        f_bits = sign | f_exp | f_mant;
    }

    float out;
    std::memcpy(&out, &f_bits, sizeof(out));
    return out;
}

// Function to execute a shell command and capture its output
std::string exec_command(const char* cmd)
{
    // Buffer to store chunks of command output
    std::array<char, 128> buffer;
    // String to accumulate the full command output
    std::string result;
    
    // Lambda function to safely close the pipe
    // Acts as a custom deleter for the unique_ptr
    auto pipe_closer = [](FILE* pipe) 
    {
        if (pipe) pclose(pipe);
    };
    
    // Create a unique_ptr with custom deleter to ensure pipe cleanup
    // popen() opens a process by creating a pipe and forking/executing the command
    std::unique_ptr<FILE, decltype(pipe_closer)> pipe(popen(cmd, "r"), pipe_closer);
    
    // Check if pipe was successfully created
    if (!pipe) 
    {
        throw std::runtime_error("popen() failed!");
    }
    
    // Read command output chunk by chunk until EOF
    // fgets reads up to buffer.size()-1 characters or until newline/EOF
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) 
    {
        // Append each chunk to the result string
        result += buffer.data();
    }
    
    // Return the complete command output
    return result;
}

// Function to get the local IP address by executing a shell script
std::string get_local_ip() 
{
    try 
    {
        // Execute the shell script that retrieves the LAN interface IP address
        // The script is expected to return the IP address as a string
        std::string ip = exec_command("./get_land_interface.sh");
        
        // Remove trailing newline character if present
        // Shell commands typically output with a newline at the end
        if (!ip.empty() && ip[ip.length()-1] == '\n') 
        {
            ip.erase(ip.length()-1);
        }
        
        // Return the cleaned IP address string
        return ip;
    } 
    catch (const std::exception& e) 
    {
        // Log error if command execution fails
        std::cerr << "Error getting local IP: " << e.what() << std::endl;
        
        // Return localhost address as fallback in case of failure
        return "127.0.0.1";
    }
}

std::string get_env(const char* env_var, const char* default_val) 
{
    const char* env_value = std::getenv(env_var);
    return env_value ? std::string(env_value) : std::string(default_val);
}

static inline bool get_env_flag(const char* env_var, bool default_val = false) {
    const char* v = std::getenv(env_var);
    if (!v) return default_val;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return (s == "1" || s == "true" || s == "yes" || s == "on");
}

static inline int normalize_dtype_tag_from_bin_header_int(int tag_or_ndim) {
    // v2: dtype_tag (negative). v1: ndim (positive) => legacy float32.
    return (tag_or_ndim < 0) ? tag_or_ndim : -1;
}

static inline int merge_output_dtype_tag(int current, int incoming) {
    // "Don't promote to float32" policy:
    // - Once we know we want fp16/bf16 output, keep it even if some shards arrive as float32.
    // - Prefer bf16 over fp16 if both appear.
    if (incoming != -1 && incoming != -2 && incoming != -3) return current;
    if (current != -1 && current != -2 && current != -3) current = -1;

    if (current == -3 || incoming == -3) return -3;
    if (current == -2 || incoming == -2) return -2;
    return -1;
}

static inline int read_dtype_tag_from_bin_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return -1;
    }
    int tag_or_ndim = 0;
    file.read(reinterpret_cast<char*>(&tag_or_ndim), sizeof(int));
    if (!file) {
        return -1;
    }
    return normalize_dtype_tag_from_bin_header_int(tag_or_ndim);
}

class llama_zmq_server 
{       
    public:   
        std::string project_folder;
        std::string matrix_shard_folder;
        std::string matrix_results_folder;
        std::string head_node_ip_eth;
        std::string head_node_ip_wifi;
        std::string head_node_PULL_port;
        std::string head_node_PUSH_port;
        std::string worker_node_PULL_port;
        std::string worker_node_PUSH_port;
        
        std::string local_IP_eth;
        std::string local_IP_wifi;
        
        std::string eth_pull_port;
        std::string eth_push_port;
        std::string wifi_pull_port;
        std::string wifi_push_port;
        std::string worker_peer_port;
        
        zmq::context_t zmq_context;
        zmq::socket_t file_receiver_eth;
        zmq::socket_t file_sender_eth;
        zmq::socket_t file_receiver_wifi;
        zmq::socket_t file_sender_wifi;
        zmq::socket_t head_node_sender_eth;
        zmq::socket_t head_node_sender_wifi;
        zmq::socket_t ack_sender;  // For sending ACKs to Python front-end
        zmq::socket_t worker_peer_receiver; // Worker ↔ Worker peer communication

        // Unified reserved file structure to hold incoming files from any interface
        struct ReservedFiles {
            std::vector<std::string> save_parallel_file_name; // Filename(s) for parallel or single-file saves (use [0] for single)
            std::vector<uint8_t> received_data_eth_file;      // Data received via Ethernet interface
            std::vector<uint8_t> received_data_wifi_file;     // Data received via WiFi interface
            bool is_parallel = false;                         // True when this ReservedFiles holds ETH+WiFi halves
            bool processed = false;                           // Marked once processed by save_file_handler
        };

        // Central list that holds all incoming files (Ethernet, WiFi, parallel)
        std::vector<ReservedFiles> reserved_files_list;

        // Track combine-in-progress results by base matrix name (e.g. "AxB").
        std::unordered_map<std::string, combined_matrix_shards> combined_matrix_shards_map;
        std::mutex combined_matrix_shards_mutex;

        // In your class member variables:
        std::vector<std::string> matrix_file_paths;

        // In-memory matrix shard store (no /dev/shm files).
        std::list<matrix_shard_object> matrix_shard_object_list;
        std::mutex matrix_shard_object_mutex;

        std::vector<std::string> received_data_eth_linux_command;
        std::vector<std::string> received_data_wifi_linux_command;
        std::vector<std::string> received_data_eth_server_command;
        std::vector<std::string> received_data_wifi_server_command;
        
        // Thread-safe mutexes (ADD THESE)
        std::mutex linux_commands_mutex;
        std::mutex server_commands_mutex;
	    std::mutex file_data_mutex;
	    std::mutex wifi_commands_mutex;
	    std::mutex head_node_sender_mutex;
	    std::mutex ack_sender_mutex;
        
        std::atomic<bool> server_running;
        llama_matrix_backend matrix_backend_llama;

        int send_back_number_of_shards = 0;
        std::vector<std::string> worker_ip_list;
        std::vector<float> worker_percentages;


        std::map<std::string, std::vector<std::pair<int, std::vector<uint8_t>>>> pending_shards;  
        std::map<std::string, std::set<int>> received_shards;  
        std::mutex shared_memory_mutex;
	        // Fallback shard counters for outputs when inputs have no shard suffix
	        std::map<std::string, int> output_shard_counters;
	        std::mutex output_shard_mutex;

	        // Track matrices that require FlashAttention-specific combine behavior.
	        // Keyed by the base matrix name used by `handle_combine_matrix_shard_list` (no `_shard_N.bin`).
	        std::unordered_set<std::string> flash_atten_openartion_combine_list;
	        std::mutex flash_atten_openartion_combine_mutex;

	    private:
        static std::string normalize_matrix_key(const std::string& path_or_name) {
            return std::filesystem::path(path_or_name).filename().string();
        }

	        static bool decode_matrix_binary_payload(
	            const std::vector<uint8_t>& bytes,
	            matrix_shard_object& out,
	            const std::string& base_file_name
	        ) {
            if (bytes.size() < 5 * sizeof(int)) {
                std::cerr << "ERROR: Matrix payload too small for header: " << base_file_name << std::endl;
                return false;
            }

            int tag_or_ndim = 0;
            std::memcpy(&tag_or_ndim, bytes.data(), sizeof(int));

            int dtype_tag = -1; // legacy default float32
            int dims[4] = {1, 1, 1, 1}; // batch, depth, rows, cols
            int header_ints = 0;

            if (tag_or_ndim < 0) {
                dtype_tag = tag_or_ndim;
                header_ints = 5; // dtype_tag + 4 dims
                std::memcpy(&dims[0], bytes.data() + sizeof(int), 4 * sizeof(int));
            } else {
                const int ndim = tag_or_ndim;
                if (ndim != 2 && ndim != 3 && ndim != 4) {
                    std::cerr << "ERROR: Unsupported ndim=" << ndim << " for " << base_file_name << std::endl;
                    return false;
                }
                header_ints = 1 + ndim;
                std::vector<int> vdims((size_t)ndim);
                std::memcpy(vdims.data(), bytes.data() + sizeof(int), (size_t)ndim * sizeof(int));
                if (ndim == 2) {
                    dims[0] = 1; dims[1] = 1; dims[2] = vdims[0]; dims[3] = vdims[1];
                } else if (ndim == 3) {
                    dims[0] = vdims[0]; dims[1] = 1; dims[2] = vdims[1]; dims[3] = vdims[2];
                } else {
                    dims[0] = vdims[0]; dims[1] = vdims[1]; dims[2] = vdims[2]; dims[3] = vdims[3];
                }
            }

            if (dtype_tag != -1 && dtype_tag != -2 && dtype_tag != -3) {
                std::cerr << "ERROR: Unsupported dtype_tag=" << dtype_tag << " for " << base_file_name << std::endl;
                return false;
            }

            const int batch = dims[0];
            const int depth = dims[1];
            const int rows = dims[2];
            const int cols = dims[3];

            if (batch <= 0 || depth <= 0 || rows <= 0 || cols <= 0) {
                std::cerr << "ERROR: Invalid dims for " << base_file_name << ": "
                          << batch << "," << depth << "," << rows << "," << cols << std::endl;
                return false;
            }

            const size_t total_elements =
                static_cast<size_t>(batch) * static_cast<size_t>(depth) *
                static_cast<size_t>(rows) * static_cast<size_t>(cols);

            const size_t header_bytes = static_cast<size_t>(header_ints) * sizeof(int);
            const size_t elem_bytes = (dtype_tag == -1) ? sizeof(float) : sizeof(uint16_t);
            const size_t need_bytes = header_bytes + total_elements * elem_bytes;
            if (bytes.size() < need_bytes) {
                std::cerr << "ERROR: Truncated payload for " << base_file_name
                          << " need=" << need_bytes << " got=" << bytes.size() << std::endl;
                return false;
            }

            out.base_file_name = normalize_matrix_key(base_file_name);
            out.batchA = batch;
            out.depthA = depth;
	            out.rows_A = rows;
	            out.cols_A = cols;
	            out.output_dtype_tag = dtype_tag;
	            out.data = std::make_shared<std::vector<float>>(total_elements);

	            const uint8_t* payload = bytes.data() + header_bytes;
	            float* out_f32 = out.data->data();
	            if (dtype_tag == -1) {
	                std::memcpy(out_f32, payload, total_elements * sizeof(float));
	            } else {
	                const uint16_t* u16 = reinterpret_cast<const uint16_t*>(payload);
	                for (size_t i = 0; i < total_elements; ++i) {
	                    out_f32[i] = (dtype_tag == -2) ? fp16_to_f32(u16[i]) : bf16_to_f32(u16[i]);
	                }
	            }

	            return true;
	        }

	        static bool decode_matrix_binary_payload_to_f32(
	            const std::vector<uint8_t>& bytes,
	            std::unique_ptr<float[]>& out_data,
	            int& rows,
	            int& cols,
	            int& batch,
	            int& depth,
	            int& dtype_tag
	        ) {
	            if (bytes.size() < 5 * sizeof(int)) {
	                std::cerr << "ERROR: Matrix payload too small for header" << std::endl;
	                return false;
	            }

	            int tag_or_ndim = 0;
	            std::memcpy(&tag_or_ndim, bytes.data(), sizeof(int));

	            dtype_tag = -1; // legacy default float32
	            int dims[4] = {1, 1, 1, 1}; // batch, depth, rows, cols
	            int header_ints = 0;

	            if (tag_or_ndim < 0) {
	                dtype_tag = tag_or_ndim;
	                header_ints = 5; // dtype_tag + 4 dims
	                std::memcpy(&dims[0], bytes.data() + sizeof(int), 4 * sizeof(int));
	            } else {
	                const int ndim = tag_or_ndim;
	                if (ndim != 2 && ndim != 3 && ndim != 4) {
	                    std::cerr << "ERROR: Unsupported ndim=" << ndim << std::endl;
	                    return false;
	                }
	                header_ints = 1 + ndim;
	                std::vector<int> vdims((size_t)ndim);
	                std::memcpy(vdims.data(), bytes.data() + sizeof(int), (size_t)ndim * sizeof(int));
	                if (ndim == 2) {
	                    dims[0] = 1; dims[1] = 1; dims[2] = vdims[0]; dims[3] = vdims[1];
	                } else if (ndim == 3) {
	                    dims[0] = vdims[0]; dims[1] = 1; dims[2] = vdims[1]; dims[3] = vdims[2];
	                } else {
	                    dims[0] = vdims[0]; dims[1] = vdims[1]; dims[2] = vdims[2]; dims[3] = vdims[3];
	                }
	            }

	            if (dtype_tag != -1 && dtype_tag != -2 && dtype_tag != -3) {
	                std::cerr << "ERROR: Unsupported dtype_tag=" << dtype_tag << std::endl;
	                return false;
	            }

	            batch = dims[0];
	            depth = dims[1];
	            rows = dims[2];
	            cols = dims[3];

	            if (batch <= 0 || depth <= 0 || rows <= 0 || cols <= 0) {
	                std::cerr << "ERROR: Invalid dims in payload: "
	                          << batch << "," << depth << "," << rows << "," << cols << std::endl;
	                return false;
	            }

	            const size_t total_elements =
	                static_cast<size_t>(batch) * static_cast<size_t>(depth) *
	                static_cast<size_t>(rows) * static_cast<size_t>(cols);

	            const size_t header_bytes = static_cast<size_t>(header_ints) * sizeof(int);
	            const size_t elem_bytes = (dtype_tag == -1) ? sizeof(float) : sizeof(uint16_t);
	            const size_t need_bytes = header_bytes + total_elements * elem_bytes;
	            if (bytes.size() < need_bytes) {
	                std::cerr << "ERROR: Truncated payload"
	                          << " need=" << need_bytes << " got=" << bytes.size() << std::endl;
	                return false;
	            }

	            out_data = std::make_unique<float[]>(total_elements);

	            const uint8_t* payload = bytes.data() + header_bytes;
	            float* out_f32 = out_data.get();
	            if (dtype_tag == -1) {
	                std::memcpy(out_f32, payload, total_elements * sizeof(float));
	            } else {
	                const uint16_t* u16 = reinterpret_cast<const uint16_t*>(payload);
	                for (size_t i = 0; i < total_elements; ++i) {
	                    out_f32[i] = (dtype_tag == -2) ? fp16_to_f32(u16[i]) : bf16_to_f32(u16[i]);
	                }
	            }

	            return true;
	        }

	        void upsert_matrix_shard_object(matrix_shard_object obj) {
	            const std::string key = normalize_matrix_key(obj.base_file_name);
	            obj.base_file_name = key;

            std::lock_guard<std::mutex> lock(matrix_shard_object_mutex);
            for (auto& existing : matrix_shard_object_list) {
                if (existing.base_file_name == key) {
                    existing = std::move(obj);
                    return;
                }
            }
            matrix_shard_object_list.push_back(std::move(obj));
        }

        bool try_get_matrix_shard_object(const std::string& path_or_name, matrix_shard_object& out) {
            const std::string key = normalize_matrix_key(path_or_name);
            std::lock_guard<std::mutex> lock(matrix_shard_object_mutex);
            for (const auto& existing : matrix_shard_object_list) {
                if (existing.base_file_name == key) {
                    out = existing;
                    return true;
                }
            }
            return false;
        }

        void store_matrix_result_to_shard_list(
            const std::string& filename,
            const MatrixResult& result,
            int output_dtype_tag
        ) {
            matrix_shard_object obj;
            obj.base_file_name = normalize_matrix_key(filename);
            obj.batchA = result.dims[0];
            obj.depthA = result.dims[1];
            obj.rows_A = result.dims[2];
            obj.cols_A = result.dims[3];
            obj.output_dtype_tag = output_dtype_tag;

            size_t total_elements = 1;
	            for (int i = 0; i < 4; ++i) {
	                const int v = (result.dims[i] > 0) ? result.dims[i] : 1;
	                total_elements *= static_cast<size_t>(v);
	            }

	            obj.data = std::make_shared<std::vector<float>>(total_elements);
	            std::memcpy(obj.data->data(), result.data.get(), total_elements * sizeof(float));
	            upsert_matrix_shard_object(std::move(obj));
	        }

        bool load_matrix_from_shard_list(
            const std::string& path_or_name,
            std::unique_ptr<float[]>& out,
            int& rows,
            int& cols,
            int& batch,
            int& depth,
            int& dtype_tag
        ) {
            matrix_shard_object obj;
            if (!try_get_matrix_shard_object(path_or_name, obj)) {
                std::cerr << "❌ Matrix not in matrix_shard_object_list: " << path_or_name << std::endl;
                return false;
            }

            rows = obj.rows_A;
            cols = obj.cols_A;
            batch = obj.batchA;
            depth = obj.depthA;
            dtype_tag = obj.output_dtype_tag;

	            const size_t total_elements =
	                static_cast<size_t>(batch) * static_cast<size_t>(depth) *
	                static_cast<size_t>(rows) * static_cast<size_t>(cols);
	            out = std::make_unique<float[]>(total_elements);
	            if (!obj.data || obj.data->size() != total_elements) {
	                std::cerr << "❌ Matrix data missing or size mismatch for: " << path_or_name << std::endl;
	                return false;
	            }
	            std::memcpy(out.get(), obj.data->data(), total_elements * sizeof(float));
	            return true;
	        }

        torch::Tensor load_matrix_from_shard_list_as_torch(const std::string& path_or_name) {
            matrix_shard_object obj;
            if (!try_get_matrix_shard_object(path_or_name, obj)) {
                throw std::runtime_error("Matrix not in matrix_shard_object_list: " + path_or_name);
            }

            std::vector<int64_t> sizes;
            if (obj.batchA > 1 && obj.depthA > 1) {
                sizes = {obj.batchA, obj.depthA, obj.rows_A, obj.cols_A};
            } else if (obj.batchA > 1) {
                sizes = {obj.batchA, obj.rows_A, obj.cols_A};
            } else {
                sizes = {obj.rows_A, obj.cols_A};
	            }

	            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	            if (!obj.data) {
	                throw std::runtime_error("Matrix data missing in matrix_shard_object_list: " + path_or_name);
	            }
	            torch::Tensor t = torch::from_blob(obj.data->data(), sizes, options).clone();
	            return t;
	        }

        std::filesystem::path resolve_matrix_disk_path(const std::string& file_or_path) const {
            std::filesystem::path p(file_or_path);
            if (p.is_absolute() || p.has_parent_path()) {
                return p;
            }

            const std::filesystem::path shard_dir(matrix_shard_folder);
            const std::filesystem::path project_dir(project_folder);

            std::filesystem::path cand1 = shard_dir / p;
            if (std::filesystem::exists(cand1)) return cand1;

            std::filesystem::path cand2 = project_dir / shard_dir / p;
            if (std::filesystem::exists(cand2)) return cand2;

            std::filesystem::path cand3 = project_dir / p;
            if (std::filesystem::exists(cand3)) return cand3;

            return cand2;
        }

	        bool load_matrix_shard_object_list(const std::vector<std::string>& files_or_names) {
	            bool ok = true;
	            for (const auto& f : files_or_names) {
	                const std::string key = normalize_matrix_key(f);
	                matrix_shard_object already;
                if (try_get_matrix_shard_object(key, already)) {
                    continue;
                }

                const std::filesystem::path disk_path = resolve_matrix_disk_path(f);
                int rows = 0, cols = 0, depth = 1, batch = 1;
                std::unique_ptr<float[]> data = load_matrix_bin(disk_path.string().c_str(), rows, cols, depth, batch);
                if (!data) {
                    std::cerr << "❌ Failed to load matrix from disk: " << disk_path << std::endl;
                    ok = false;
                    continue;
                }

                const int dtype_tag = read_dtype_tag_from_bin_file(disk_path.string());
                const size_t total_elements =
                    static_cast<size_t>(batch) * static_cast<size_t>(depth) *
                    static_cast<size_t>(rows) * static_cast<size_t>(cols);

                matrix_shard_object obj;
                obj.base_file_name = key;
                obj.rows_A = rows;
                obj.cols_A = cols;
	                obj.batchA = batch;
	                obj.depthA = depth;
	                obj.output_dtype_tag = dtype_tag;
	                obj.data = std::make_shared<std::vector<float>>(total_elements);
	                std::memcpy(obj.data->data(), data.get(), total_elements * sizeof(float));
	                upsert_matrix_shard_object(std::move(obj));
	            }
	            return ok;
	        }

	        int default_vram_backend_index() const {
	            // Prefer the first non-CPU GGML backend (CPU backend is usually appended last).
	            for (size_t i = 0; i < matrix_backend_llama.ggml_backends.size(); ++i) {
	                ggml_backend_t backend = matrix_backend_llama.ggml_backends[i];
	                if (!backend) continue;
	                const char* n = ggml_backend_name(backend);
	                if (!n) continue;
	                std::string name(n);
	                std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return (char) std::tolower(c); });
	                if (name.find("cpu") == std::string::npos) {
	                    return (int) i;
	                }
	            }
	            return -1;
	        }

	        bool load_matrix_shard_object_list_VRAM(const std::vector<std::string>& files_or_names) {
	            const bool ok = load_matrix_shard_object_list(files_or_names);

	            const int backend_index = default_vram_backend_index();
	            if (backend_index < 0) {
	                return ok;
	            }
	            auto& vram = get_vram_cache_manager();
	            if (!vram.enabled(backend_index)) {
	                return ok;
	            }
	            const bool vram_only_mode = get_env_flag("OPEN_CLUSTER_VRAM_ONLY", false);

	            for (const auto& f : files_or_names) {
	                const std::string key = normalize_matrix_key(f);
	                matrix_shard_object obj;
	                if (!try_get_matrix_shard_object(key, obj) || !obj.data) {
	                    continue;
	                }
	                // GGML tensor dims are {cols, rows, depth, batch}
	                bool cached_to_vram = false;
	                if (vram.get_tensor(backend_index, key)) {
	                    cached_to_vram = true;
	                } else {
	                    cached_to_vram = vram.cache_tensor_f32_4d(
	                        backend_index,
	                        key,
	                        obj.data->data(),
	                        obj.cols_A,
	                        obj.rows_A,
	                        obj.depthA,
	                        obj.batchA
	                    );
	                }

	                if (vram_only_mode && cached_to_vram) {
	                    obj.vram_only = true;
	                    obj.data.reset();
	                    upsert_matrix_shard_object(std::move(obj));
	                }
	            }

	            return ok;
	        }


    public:            
        // Constructor - initializes ZMQ server with dual network interfaces
	        llama_zmq_server() : 
	            zmq_context(1),
            file_receiver_eth(zmq_context, zmq::socket_type::pull),
            file_sender_eth(zmq_context, zmq::socket_type::push),
            file_receiver_wifi(zmq_context, zmq::socket_type::pull),
            file_sender_wifi(zmq_context, zmq::socket_type::push),
            head_node_sender_eth(zmq_context, zmq::socket_type::push),
            head_node_sender_wifi(zmq_context, zmq::socket_type::push),
            worker_peer_receiver(zmq_context, zmq::socket_type::pull),
            server_running(true)
	        {

	            
	            // Load configuration from environment variables with defaults.
	            // Try to infer the `cluster_matrix/` root so disk paths match the Python frontend
	            // even when the server is launched from `cluster_matrix/ggml/`.
	            const auto infer_default_project_folder = []() -> std::string {
	                std::filesystem::path cwd;
	                try {
	                    cwd = std::filesystem::current_path();
	                } catch (...) {
	                    return std::string("./");
	                }

	                const auto normalize_dir = [](const std::filesystem::path& dir) -> std::string {
	                    std::string out = dir.string();
	                    if (!out.empty() && out.back() != '/') {
	                        out.push_back('/');
	                    }
	                    return out;
	                };

	                std::filesystem::path probe = cwd;
	                for (int i = 0; i < 8; ++i) {
	                    // If launched from within `cluster_matrix/`, detect it directly.
	                    if (std::filesystem::exists(probe / "cluster_matrix_v1.py")) {
	                        return normalize_dir(probe);
	                    }
	                    // If launched from repo root (or elsewhere), detect the `cluster_matrix/` folder.
	                    if (std::filesystem::exists(probe / "cluster_matrix" / "cluster_matrix_v1.py")) {
	                        return normalize_dir(probe / "cluster_matrix");
	                    }
	                    if (!probe.has_parent_path()) {
	                        break;
	                    }
	                    const std::filesystem::path parent = probe.parent_path();
	                    if (parent == probe) {
	                        break;
	                    }
	                    probe = parent;
	                }

	                return normalize_dir(cwd);
	            };

	            const std::string default_project_folder = infer_default_project_folder();
	            project_folder = get_env("OPEN_CLUSTER_PROJECT_DIRECTORY", default_project_folder.c_str());



            matrix_shard_folder = get_env("OPEN_CLUSTER_MATRIX_SHARD_DIRECTORY",
                                        "matrix_shards/");
            if (matrix_shard_folder.rfind("/dev/shm/", 0) == 0) {
                std::cerr << "WARNING: OPEN_CLUSTER_MATRIX_SHARD_DIRECTORY points to /dev/shm; "
                          << "overriding to 'matrix_shards/' to avoid RAM-backed FS." << std::endl;
                matrix_shard_folder = "matrix_shards/";
            }
            matrix_results_folder = get_env("OPEN_CLUSTER_MATRIX_RESULTS_DIRECTORY", 
                                        "/dev/shm/matrix_results/");
            
            head_node_ip_eth = get_env("HEAD_NODE_IP_ETH", "192.168.2.100");
            head_node_ip_wifi = get_env("HEAD_NODE_IP_WIFI", "192.168.50.113");
            head_node_PULL_port = get_env("HEAD_NODE_PULL_PORT_C", "7779");
            head_node_PUSH_port = get_env("HEAD_NODE_PUSH_PORT_C", "7780");
            worker_node_PULL_port = get_env("WORKER_NODE_PULL_PORT_C", "5557");
            worker_node_PUSH_port = get_env("WORKER_NODE_PUSH_PORT_C", "5558");
            
            // Initialize parallel file structures (now handled via reserved_files_list)
            
            // Get local network addresses
            local_IP_eth = get_local_ip();
            
            // Attempt to get WiFi IP address using system command
            try {
                local_IP_wifi = exec_command(
                    "ip -4 addr show $(ip -4 route ls | grep default | grep -o 'dev [^ ]*' "
                    "| awk '{print $2}') | grep inet | awk '{print $2}' | cut -d'/' -f1"
                );
                // Clean up newline from command output
                if (!local_IP_wifi.empty() && local_IP_wifi[local_IP_wifi.length()-1] == '\n') {
                    local_IP_wifi.erase(local_IP_wifi.length()-1);
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to get WiFi IP: " << e.what() << std::endl;
                local_IP_wifi = "127.0.0.1";
            }
            
            // Configure network ports based on whether this is head node or worker node
            if (local_IP_eth == head_node_ip_eth || local_IP_wifi == head_node_ip_wifi) {
                // Head node configuration
                eth_pull_port = "tcp://" + local_IP_eth + ":" + head_node_PULL_port;
                eth_push_port = "tcp://" + local_IP_eth + ":" + head_node_PUSH_port;
                wifi_pull_port = "tcp://" + local_IP_wifi + ":" + head_node_PULL_port;
                wifi_push_port = "tcp://" + local_IP_wifi + ":" + head_node_PUSH_port;
            } else {
                // Worker node configuration
                eth_pull_port = "tcp://" + local_IP_eth + ":" + worker_node_PULL_port;
                eth_push_port = "tcp://" + local_IP_eth + ":" + worker_node_PUSH_port;
                wifi_pull_port = "tcp://" + local_IP_wifi + ":" + worker_node_PULL_port;
                wifi_push_port = "tcp://" + local_IP_wifi + ":" + worker_node_PUSH_port;
            }
            
            // Bind file transfer sockets
            file_receiver_eth.bind(eth_pull_port);
            file_sender_eth.bind(eth_push_port);
            file_receiver_wifi.bind(wifi_pull_port);
            file_sender_wifi.bind(wifi_push_port);
            
	            // Connect to head node for coordination
	            head_node_sender_eth.connect("tcp://" + head_node_ip_eth + ":" + head_node_PULL_port);
	            head_node_sender_wifi.connect("tcp://" + head_node_ip_wifi + ":" + head_node_PULL_port);
	            // Prevent indefinite blocking on send if the head isn't reachable/reading.
	            head_node_sender_eth.set(zmq::sockopt::linger, 0);
	            head_node_sender_wifi.set(zmq::sockopt::linger, 0);
	            head_node_sender_eth.set(zmq::sockopt::sndtimeo, 10000);
	            head_node_sender_wifi.set(zmq::sockopt::sndtimeo, 10000);
            
            // Setup Python front-end ACK communication
            std::string python_frontend_ip = get_env("HEAD_NODE_IP", "192.168.2.100");
            std::string python_frontend_port = get_env("PYTHON_FRONT_END_CLUSTER_PORT", "7790");
            
	            ack_sender = zmq::socket_t(zmq_context, zmq::socket_type::push);
	            ack_sender.connect("tcp://" + python_frontend_ip + ":" + python_frontend_port);
	            ack_sender.set(zmq::sockopt::linger, 0);
	            ack_sender.set(zmq::sockopt::sndtimeo, 10000);
            
            // Worker-to-worker peer communication setup
            // TODO: Load worker IPs from environment or configuration file
            worker_ip_list = {
                "192.168.2.100",
                "192.168.2.100",  // Experimental: Multiple workers on same machine
                "192.168.2.100",  // Experimental: For load distribution testing
                "192.168.2.102",
                "192.168.2.103"
            };
            
            // Experimental feature - work distribution percentages for heterogeneous nodes
            // This enables adaptive load balancing based on worker capabilities
            worker_percentages = {0.45f, 0.35f, 0.10f, 0.05f, 0.05f};  // For experimental feature not yet implemented
            
            worker_peer_port = get_env("WORKER_PEER_PORT", "5560");
            worker_peer_receiver.bind("tcp://" + local_IP_eth + ":" + worker_peer_port);
            worker_peer_receiver.bind("tcp://" + local_IP_wifi + ":" + worker_peer_port);
            
            // Clean console output
            std::cout << "\n=== ZMQ Server Initialization ===" << std::endl;
            std::cout << "Network Configuration:" << std::endl;
            std::cout << "  Ethernet IP: " << local_IP_eth << std::endl;
            std::cout << "  WiFi IP: " << local_IP_wifi << std::endl;
            std::cout << "\nPort Bindings:" << std::endl;
            std::cout << "  Ethernet PULL: " << eth_pull_port << std::endl;
            std::cout << "  Ethernet PUSH: " << eth_push_port << std::endl;
            std::cout << "  WiFi PULL: " << wifi_pull_port << std::endl;
            std::cout << "  WiFi PUSH: " << wifi_push_port << std::endl;
            std::cout << "  Worker Peer: " << worker_peer_port << std::endl;
            std::cout << "  Worker IPs configured: " << worker_ip_list.size() << " nodes" << std::endl;
            
            // Initialize hardware backends
            #ifdef GGML_OPENCL
                std::cout << "\nInitializing OpenCL backends..." << std::endl;
                init_openCL_GPUS();
            #else
                std::cout << "\nOpenCL backend disabled at compile time" << std::endl;
	            #endif

	            // Initialize VRAM caching buffers (best-effort) and print GPU VRAM info.
	            set_VRAM_buffers(matrix_backend_llama);
	            inspect_GPU();
	            
	            std::cout << "\nServer initialization complete" << std::endl;
	            std::cout << "==============================\n" << std::endl;
	        }

	        void inspect_GPU()
	        {
	            std::cout << "\n=== GPU VRAM Inspection ===" << std::endl;
	            try {
	                std::vector<cl::Platform> platforms;
	                cl::Platform::get(&platforms);
	                size_t gpu_idx = 0;
	                for (const auto& plat : platforms) {
	                    std::vector<cl::Device> devices;
	                    plat.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	                    for (const auto& dev : devices) {
	                        std::string name = dev.getInfo<CL_DEVICE_NAME>();
	                        name.erase(std::remove(name.begin(), name.end(), '\n'), name.end());
	                        name.erase(std::remove(name.begin(), name.end(), '\r'), name.end());
	                        const cl_ulong total = dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
	                        std::cout << "OpenCL GPU " << gpu_idx++ << ": " << name
	                                  << " | Total VRAM: " << (double)total / (1024.0 * 1024.0 * 1024.0) << " GiB"
	                                  << std::endl;
	                    }
	                }
	                if (gpu_idx == 0) {
	                    std::cout << "OpenCL GPU: none detected (cannot query total VRAM via OpenCL)" << std::endl;
	                }
	            } catch (...) {
	                std::cout << "OpenCL GPU query failed (VRAM total unavailable)" << std::endl;
	            }

	            auto& mgr = get_vram_cache_manager();
	            std::cout << "GGML VRAM cache budget per backend:" << std::endl;
	            for (size_t i = 0; i < matrix_backend_llama.ggml_backends.size(); ++i) {
	                ggml_backend_t backend = matrix_backend_llama.ggml_backends[i];
	                const char* backend_name = backend ? ggml_backend_name(backend) : "(null)";
	                const size_t budget = mgr.budget_bytes((int)i);
	                const size_t used = mgr.used_bytes((int)i);
	                const size_t free_budget = mgr.free_budget_bytes((int)i);
	                std::cout << "  [" << i << "] " << backend_name
	                          << " | enabled=" << (mgr.enabled((int)i) ? "yes" : "no")
	                          << " | budget=" << (double)budget / (1024.0 * 1024.0) << " MiB"
	                          << " | used=" << (double)used / (1024.0 * 1024.0) << " MiB"
	                          << " | free_budget=" << (double)free_budget / (1024.0 * 1024.0) << " MiB"
	                          << std::endl;
	            }
	            std::cout << "===========================\n" << std::endl;
	        }

		void send_ack(std::string ack_msg = "ACK") 
		{
		    zmq::message_t ack(ack_msg.data(), ack_msg.size());
		    std::lock_guard<std::mutex> lock(ack_sender_mutex);
		    ack_sender.send(ack, zmq::send_flags::none);
		}

	    bool send_combined_bin_to_python(
	        const std::string& matrix_name,
	        const MatrixResult& full,
	        int output_dtype_tag
	    )
	    {
	            // v2 binary wire format:
	            // [dtype_tag(int32), batch(int32), depth(int32), rows(int32), cols(int32), data(bytes)]
	        const int dtype_tag = output_dtype_tag;
	        if (dtype_tag != -1 && dtype_tag != -2 && dtype_tag != -3) {
	            std::cerr << "ERROR: Unsupported output dtype_tag for combined stream: "
	                        << dtype_tag << std::endl;
	            return false;
	        }

	        const int ndim = 4;
	        size_t total_elements = 1;
	        for (int i = 0; i < ndim; ++i) {
	            const int v = (full.dims[i] > 0) ? full.dims[i] : 1;
	            total_elements *= static_cast<size_t>(v);
	        }

	        const size_t elem_bytes = (dtype_tag == -1) ? sizeof(float) : sizeof(uint16_t);
	        const size_t header_bytes = sizeof(int) * 5;
	        const size_t payload_bytes = header_bytes + total_elements * elem_bytes;

	        zmq::message_t payload_msg(payload_bytes);
	        auto* header = static_cast<int*>(payload_msg.data());
	        header[0] = dtype_tag;
	        for (int i = 0; i < ndim; ++i) {
	            header[i + 1] = (full.dims[i] > 0) ? full.dims[i] : 1;
	        }

	        uint8_t* data_ptr = static_cast<uint8_t*>(payload_msg.data()) + header_bytes;
	        if (dtype_tag == -1) {
	            std::memcpy(data_ptr, full.data.get(), total_elements * sizeof(float));
	        } else if (dtype_tag == -2) {
	            uint16_t* out = reinterpret_cast<uint16_t*>(data_ptr);
	            for (size_t i = 0; i < total_elements; ++i) {
	                out[i] = float_to_fp16_bits(full.data[i]);
	            }
	        } else {
	            uint16_t* out = reinterpret_cast<uint16_t*>(data_ptr);
	            for (size_t i = 0; i < total_elements; ++i) {
	                out[i] = float_to_bf16_bits(full.data[i]);
	            }
	        }

	        const std::string header_str = "BIN_COMBINED=" + matrix_name;
	        try {
	            zmq::message_t header_msg(header_str.data(), header_str.size());

	            std::lock_guard<std::mutex> lock(ack_sender_mutex);
	            ack_sender.send(header_msg, zmq::send_flags::sndmore);
	            ack_sender.send(payload_msg, zmq::send_flags::none);
	            std::cout << "📤 Streamed combined matrix to Python: "
	                        << matrix_name << " (" << payload_bytes << " bytes)" << std::endl;
	            return true;
	        } catch (const zmq::error_t& e) {
	            std::cerr << "ERROR: Failed to stream combined PT to Python: "
	                      << e.what() << std::endl;
	            return false;
	        }
	    }

        void run_server() 
        {
            std::cout << "🚀 C++ ZMQ Node Server starting..." << std::endl;
            
            // Start network listener threads for dual-interface operation
            std::thread eth_thread(&llama_zmq_server::listen_interface, this, "Ethernet");

            std::thread process_command_thread(&llama_zmq_server::process_command, this);
            
            // Detach threads to run as daemon processes (background services)
            eth_thread.detach();
            //wifi_thread.detach();
            process_command_thread.detach();
            
            std::cout << "✅ Network listeners started successfully" << std::endl;
            std::cout << "   • Ethernet interface: Active" << std::endl;
            std::cout << "   • WiFi interface: Active" << std::endl;
            std::cout << "   • Command processor: Active" << std::endl;
            std::cout << "\n📡 Server running. Press Ctrl+C to gracefully shutdown..." << std::endl;
            
            try 
            {
                // Register signal handler for graceful shutdown on Ctrl+C
                std::signal(SIGINT, [](int sig) { 
                    std::cout << "\n🛑 Received shutdown signal (Ctrl+C)" << std::endl;
                    std::cout << "   Shutting down ZMQ server..." << std::endl;
                    std::exit(0); 
                });
                
                // Main thread idle loop - keeps the server alive
                // This allows signal handling and keeps the process running
                while (server_running) 
                {
                    // Sleep to prevent CPU spinning while waiting for shutdown
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            } 
            catch (const std::exception& e) 
            {
                std::cerr << "\n❌ Critical server error: " << e.what() << std::endl;
                std::cerr << "   Server shutting down due to exception" << std::endl;
            }
            
            std::cout << "👋 Server shutdown complete" << std::endl;
        }

        void listen_interface(const std::string& interface_name)
        {
            // Determine which socket and which command containers/mutex to use
            zmq::socket_t* socket_ptr = nullptr;
            std::vector<std::string>* linux_cmd_ptr = nullptr;
            std::vector<std::string>* server_cmd_ptr = nullptr;
            std::mutex* linux_cmd_mutex = nullptr;
            std::mutex* server_cmd_mutex = nullptr;

            if (interface_name == "Ethernet")
            {
                socket_ptr = &file_receiver_eth;
                linux_cmd_ptr = &received_data_eth_linux_command;
                server_cmd_ptr = &received_data_eth_server_command;
                linux_cmd_mutex = &linux_commands_mutex;
                server_cmd_mutex = &server_commands_mutex;
            }
            else if (interface_name == "WiFi")
            {
                socket_ptr = &file_receiver_wifi;
                linux_cmd_ptr = &received_data_wifi_linux_command;
                server_cmd_ptr = &received_data_wifi_server_command;
                linux_cmd_mutex = &wifi_commands_mutex;
                server_cmd_mutex = &server_commands_mutex;
            }
            else
            {
                std::cerr << "❌ Unknown interface: " << interface_name << std::endl;
                return;
            }

            std::cout << "🔌 " << interface_name << " listener thread started" << std::endl;

            while (server_running)
            {
                try
                {
                    std::vector<zmq::message_t> parts;
                    bool more_parts = true;

                    // Receive multipart ZMQ message (could be 1 or 2 parts)
                    while (more_parts && server_running)
                    {
                        zmq::message_t message;
                        auto result = socket_ptr->recv(message, zmq::recv_flags::dontwait);

                        if (result)
                        {
                            more_parts = message.more();
                            parts.push_back(std::move(message));
                        }
                        else
                        {
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                            break;
                        }
                    }

                    if (parts.empty()) continue;

                    // Single-part: commands
                    if (parts.size() == 1)
                    {
                        std::string command = parts[0].to_string();
                        size_t server_cmd_pos = command.find("server_command=");

                        if (server_cmd_pos != std::string::npos)
                        {
                            std::string server_cmd = command.substr(server_cmd_pos + 15);
                            std::lock_guard<std::mutex> lock(*server_cmd_mutex);
                            server_cmd_ptr->push_back(server_cmd);
                            std::cout << "📋 " << interface_name << ": Received server command" << std::endl;
                        }
                        else
                        {
                            std::lock_guard<std::mutex> lock(*linux_cmd_mutex);
                            linux_cmd_ptr->push_back(command);
                            std::cout << "💻 " << interface_name << ": Received Linux command" << std::endl;
                        }
                    }
                    // Two-part: file transfer (either full file or parallel half)
                    else if (parts.size() == 2)
                    {
                        std::string filename_header = parts[0].to_string();
                        size_t parallel_send_pos = filename_header.find("P_SEND_");

                        const uint8_t* data_ptr = static_cast<const uint8_t*>(parts[1].data());
                        size_t data_size = parts[1].size();

                        if (parallel_send_pos != std::string::npos)
                        {
                            // Parallel half (ETH or WiFi)
                            std::string actual_filename = filename_header.substr(parallel_send_pos + 7);

                            std::lock_guard<std::mutex> lock(file_data_mutex);

                            bool found = false;
                            for (auto &rf : reserved_files_list)
                            {
                                if (!rf.save_parallel_file_name.empty() && rf.save_parallel_file_name[0] == actual_filename)
                                {
                                    if (interface_name == "Ethernet")
                                        rf.received_data_eth_file.assign(data_ptr, data_ptr + data_size);
                                    else
                                        rf.received_data_wifi_file.assign(data_ptr, data_ptr + data_size);

                                    rf.is_parallel = true;
                                    found = true;
                                    std::cout << "📂 " << interface_name << ": Added to parallel file '" << actual_filename << "'" << std::endl;
                                    break;
                                }
                            }

                            if (!found)
                            {
                                ReservedFiles rf;
                                rf.save_parallel_file_name.push_back(actual_filename);
                                if (interface_name == "Ethernet")
                                    rf.received_data_eth_file.assign(data_ptr, data_ptr + data_size);
                                else
                                    rf.received_data_wifi_file.assign(data_ptr, data_ptr + data_size);

                                rf.is_parallel = true;
                                reserved_files_list.push_back(std::move(rf));

                                std::cout << "📂 " << interface_name << ": Started parallel file '" << actual_filename << "' (" << interface_name << " half)" << std::endl;
                            }
                        }
                        else
                        {
                            // Full file transfer over single interface
                            std::string filename = filename_header;

                            std::vector<uint8_t> file_data;
                            file_data.assign(data_ptr, data_ptr + data_size);

                            {
                                std::lock_guard<std::mutex> lock(file_data_mutex);
                                bool found = false;
                                for (auto &rf : reserved_files_list)
                                {
                                    if (!rf.save_parallel_file_name.empty() && rf.save_parallel_file_name[0] == filename)
                                    {
                                        if (interface_name == "Ethernet") rf.received_data_eth_file = std::move(file_data);
                                        else rf.received_data_wifi_file = std::move(file_data);
                                        found = true;
                                        break;
                                    }
                                }

                                if (!found)
                                {
                                    ReservedFiles rf;
                                    rf.save_parallel_file_name.push_back(filename);
                                    if (interface_name == "Ethernet") rf.received_data_eth_file = std::move(file_data);
                                    else rf.received_data_wifi_file = std::move(file_data);
                                    reserved_files_list.push_back(std::move(rf));
                                }
                            }

                            std::cout << "📁 " << interface_name << ": Received file '" << filename << "' (" << data_size << " bytes)" << std::endl;
                        }

                        // Attempt to process saved files
                        save_file_handler();
                    }
                    else
                    {
                        std::cout << "⚠️ " << interface_name << ": Unexpected message format - " << parts.size() << " parts received" << std::endl;
                    }
                }
                catch (const std::exception& e)
                {
                    std::cerr << "❌ " << interface_name << " listener error: " << e.what() << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }

            std::cout << "🔌 " << interface_name << " listener thread stopping" << std::endl;
        }

        void process_command() 
        {
            std::cout << "⚙️ Command processor thread started" << std::endl;
            
            while (server_running) 
            {
                try 
                {
                    // --- Process Linux System Commands (Ethernet) ---
                    if (!received_data_eth_linux_command.empty()) 
                    {
                        std::vector<std::string> commands_to_process;
                        
                        // Safely copy commands from shared vector under lock
                        {
                            std::lock_guard<std::mutex> lock(linux_commands_mutex);
                            commands_to_process = received_data_eth_linux_command;
                            received_data_eth_linux_command.clear();  // Clear after copying
                        }
                        
                        std::cout << "\n🔧 Processing " << commands_to_process.size() 
                                << " Linux command(s)" << std::endl;
                        
                        for (const std::string &command : commands_to_process)
                        {
                            // Security note: system() calls should be validated in production
                            std::cout << "   • Executing: " << command << std::endl;
                            
                            int result = system(command.c_str());
                            if (result == 0) {
                                std::cout << "     ✅ Command completed successfully" << std::endl;
                            } else {
                                std::cout << "     ⚠️ Command returned exit code: " << result << std::endl;
                            }
                        }
                    }
                    
                    // --- Process Server Control Commands (Ethernet) ---
                    if (!received_data_eth_server_command.empty()) 
                    {
                        std::vector<std::string> server_commands_to_process;
                        
                        // Safely copy server commands from shared vector
                        {
                            std::lock_guard<std::mutex> lock(server_commands_mutex);
                            server_commands_to_process = received_data_eth_server_command;
                            received_data_eth_server_command.clear();  // Clear after copying
                        }
                        
                        std::cout << "\n🎮 Processing " << server_commands_to_process.size() 
                                << " server control command(s)" << std::endl;
                        
                        // Create threads for concurrent server command execution
                        std::vector<std::thread> command_threads;
                        
                        for (const std::string &command : server_commands_to_process)
                        {
                            std::cout << "   • Launching command: " 
                                    << (command.length() > 50 ? command.substr(0, 47) + "..." : command) 
                                    << std::endl;
                            
                            // Launch each server command in its own thread for parallel execution
                            command_threads.emplace_back([this, command]() {
                                try {
                                    run_server_command(command);
                                } catch (const std::exception& e) {
                                    std::cerr << "❌ Server command failed: " << e.what() 
                                            << " (Command: " << command << ")" << std::endl;
                                }
                            });
                        }
                        
                        // Detach threads to allow them to run independently
                        // Note: Using detach() means we don't wait for completion
                        // Use join() if synchronization is required
                        for (auto& thread : command_threads) {
                            thread.detach();
                        }
                        
                        std::cout << "     ✅ " << command_threads.size() 
                                << " command thread(s) launched" << std::endl;
                    }
                    
                    // Small delay to prevent CPU spinning when no commands are pending
                    // This also allows other threads to acquire locks
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    
                } 
                catch (const std::exception& e) 
                {
                    std::cerr << "❌ Command processor thread error: " << e.what() << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
            
            std::cout << "⚙️ Command processor thread stopping" << std::endl;
        }

        int run_server_command(const std::string& command) 
        {
            try 
            {
                std::cout << "🚀 Executing server command" << std::endl;
                
                // Tokenize command string into individual arguments
                std::vector<std::string> command_args;
                std::istringstream iss(command);
                std::string token;
                
                while (iss >> token) {
                    command_args.push_back(token);
                }

                // Validate minimum command structure
                if (command_args.empty()) {
                    std::cerr << "❌ Empty command received" << std::endl;
                    return -2;
                }

                const std::string& command_type = command_args[0];

                // ----------------------------
                // Matrix Load Operation (Disk -> In-Memory List)
                // ----------------------------
                if (command_type == "load_matrix_shard_object_list")
                {
                    if (command_args.size() < 2) {
                        std::cerr << "❌ load_matrix_shard_object_list requires at least 1 filename" << std::endl;
                        return -3;
                    }

                    std::vector<std::string> files(command_args.begin() + 1, command_args.end());
                    const bool ok = load_matrix_shard_object_list(files);
                    send_ack("ACK_load_matrix_shard_object_list_complete");
                    return ok ? 0 : -7;
                }

                if (command_type == "load_matrix_shard_object_list_VRAM")
                {
                    if (command_args.size() < 2) {
                        std::cerr << "❌ load_matrix_shard_object_list_VRAM requires at least 1 filename" << std::endl;
                        return -3;
                    }

                    std::vector<std::string> files(command_args.begin() + 1, command_args.end());
                    const bool ok = load_matrix_shard_object_list_VRAM(files);
                    // Keep ACK name stable for Python callers.
                    send_ack("ACK_load_matrix_shard_object_list_complete");
                    return ok ? 0 : -7;
                }

                // ----------------------------
                // Transformer Computation Operations
                // ----------------------------
                if (command_type == "transformerOp")
                {
                    if (command_args.size() < 9) {
                        std::cerr << "❌ Insufficient parameters for transformerOp "
                                << "(expected 9, got " << command_args.size() << ")"
                                << std::endl;
                        return -3;
                    }

                    // transformerOp <matrix>
                    //               <transposeA>
                    //               <use_gpu>
                    //               <gpu_id>
                    //               <send_back>
                    //               <operation_type>
                    //               <n_dims>
                    //               <shard_index_override>

                    const std::string& matrix_name = command_args[1];
                    bool transposeA = (command_args[2] == "true");
                    bool use_gpu    = (command_args[3] == "true");
                    int gpu_id      = std::stoi(command_args[4]);
                    int send_back   = std::stoi(command_args[5]);
                    std::string operation_type = command_args[6];
                    int n_dims      = std::stoi(command_args[7]);
                    int shard_index_override = std::stoi(command_args[8]);

                    std::cout << "**run_server_command** transformerOp\n"
                            << "  matrix: " << matrix_name << "\n"
                            << "  op: " << operation_type << "\n"
                            << "  gpu: " << use_gpu << " (id " << gpu_id << ")\n"
                            << "  send_back: " << send_back << "\n"
                            << "  shard_override: " << shard_index_override
                            << std::endl;

                    matrix_shard_object matrixA_obj;
                    if (!try_get_matrix_shard_object(matrix_name, matrixA_obj)) {
                        std::cerr << "❌ Missing input matrix in matrix_shard_object_list"
                                << std::endl;
                        std::cerr << "   Needed: '" << matrix_name << "'" << std::endl;
                        return -8;
                    }

                    const bool operation_success = transformer_operation(
                        "llama",               // ✅ backend selection
                        matrixA_obj,
                        transposeA,
                        use_gpu,
                        gpu_id,
                        send_back,
                        operation_type,
                        n_dims,
                        shard_index_override
                    );

                    if (operation_success) {
                        std::cout << "✅ Transformer operation completed successfully"
                                << std::endl;
                        return 0;
                    } else {
                        std::cerr << "❌ Transformer operation failed: "
                                << operation_type << std::endl;
                        return -7;
                    }
                }

                // ----------------------------
                // rope Computation Operations
                // ----------------------------
                if (command_type == "rope")
                {
                    // ----------------------------
                    // Parse command_args
                    // ----------------------------
                    const std::string& matrix_name = command_args[1];
                    bool transposeA = (command_args[2] == "true");
                    bool use_gpu    = (command_args[3] == "true");
                    int gpu_id      = std::stoi(command_args[4]);
                    int send_back   = std::stoi(command_args[5]);
                    std::string operation_type = command_args[6];
                    int rope_type   = std::stoi(command_args[7]);
                    int shard_index_override = std::stoi(command_args[8]);

                    // Parse extra parameters from Python side
                    std::vector<float> extra_params;
                    std::stringstream ss(command_args[9]);
                    std::string token;
                    while (std::getline(ss, token, ',')) {
                        extra_params.push_back(std::stof(token));
                    }

                    std::cout << "**run_server_command** rope\n"
                            << "  matrix: " << matrix_name << "\n"
                            << "  transpose: " << transposeA << "\n"
                            << "  gpu: " << use_gpu << " (id " << gpu_id << ")\n"
                            << "  send_back: " << send_back << "\n"
                            << "  op: " << operation_type << "\n"
                            << "  rope_type: " << rope_type << "\n"
                            << "  shard_override: " << shard_index_override
                            << std::endl;

                    // ----------------------------
                    // Fetch matrix shard object
                    // ----------------------------
                    matrix_shard_object matrixA_obj;
                    if (!try_get_matrix_shard_object(matrix_name, matrixA_obj)) {
                        std::cerr << "❌ Missing input matrix in matrix_shard_object_list\n"
                                << "   Needed: '" << matrix_name << "'"
                                << std::endl;
                        return -8;
                    }

                    // ----------------------------
                    // Construct RoPE structs
                    // ----------------------------
                    RoPEParams      base_params{};
                    RoPEExtParams   ext_params{};
                    RoPEMultiParams multi_params{};

                    RoPEParams*      base_ptr  = nullptr;
                    RoPEExtParams*   ext_ptr   = nullptr;
                    RoPEMultiParams* multi_ptr = nullptr;

                    if (rope_type == 0) {
                        base_params.pos_data = (extra_params.size() > 0)
                            ? reinterpret_cast<int32_t*>(static_cast<intptr_t>(extra_params[0]))
                            : nullptr;
                        for (int i = 0; i < 4 && extra_params.size() > i+1; ++i)
                            base_params.pos_dims[i] = static_cast<int>(extra_params[1 + i]);
                        base_params.n_dims = (extra_params.size() > 5) ? static_cast<int>(extra_params[5]) : 0;
                        base_params.mode   = (extra_params.size() > 6) ? static_cast<int>(extra_params[6]) : 0;
                        base_ptr = &base_params;
                    }
                    else if (rope_type == 1) {
                        ext_params.pos_data          = (extra_params.size() > 0) ? reinterpret_cast<int32_t*>(static_cast<intptr_t>(extra_params[0])) : nullptr;
                        ext_params.freq_factors_data = (extra_params.size() > 1) ? reinterpret_cast<float*>(static_cast<intptr_t>(extra_params[1])) : nullptr;
                        for (int i = 0; i < 4 && extra_params.size() > i+2; ++i) {
                            ext_params.pos_dims[i]  = static_cast<int>(extra_params[2 + i]);
                            ext_params.freq_dims[i] = (extra_params.size() > 6+i) ? static_cast<int>(extra_params[6 + i]) : 0;
                        }
                        ext_params.n_dims      = (extra_params.size() > 10) ? static_cast<int>(extra_params[10]) : 0;
                        ext_params.mode        = (extra_params.size() > 11) ? static_cast<int>(extra_params[11]) : 0;
                        ext_params.n_ctx_orig  = (extra_params.size() > 12) ? static_cast<int>(extra_params[12]) : 0;
                        ext_params.freq_base   = (extra_params.size() > 13) ? extra_params[13] : 0.0f;
                        ext_params.freq_scale  = (extra_params.size() > 14) ? extra_params[14] : 0.0f;
                        ext_params.ext_factor  = (extra_params.size() > 15) ? extra_params[15] : 0.0f;
                        ext_params.attn_factor = (extra_params.size() > 16) ? extra_params[16] : 0.0f;
                        ext_params.beta_fast   = (extra_params.size() > 17) ? extra_params[17] : 0.0f;
                        ext_params.beta_slow   = (extra_params.size() > 18) ? extra_params[18] : 0.0f;
                        ext_ptr = &ext_params;
                    }
                    else if (rope_type == 2) {
                        multi_params.pos_data          = (extra_params.size() > 0) ? reinterpret_cast<int32_t*>(static_cast<intptr_t>(extra_params[0])) : nullptr;
                        multi_params.freq_factors_data = (extra_params.size() > 1) ? reinterpret_cast<float*>(static_cast<intptr_t>(extra_params[1])) : nullptr;
                        for (int i = 0; i < 4; ++i) {
                            multi_params.pos_dims[i]  = (extra_params.size() > i+2) ? static_cast<int>(extra_params[2 + i]) : 0;
                            multi_params.freq_dims[i] = (extra_params.size() > i+6) ? static_cast<int>(extra_params[6 + i]) : 0;
                            multi_params.sections[i]  = (extra_params.size() > i+10) ? static_cast<int>(extra_params[10 + i]) : 0;
                        }
                        multi_params.n_dims      = (extra_params.size() > 14) ? static_cast<int>(extra_params[14]) : 0;
                        multi_params.mode        = (extra_params.size() > 15) ? static_cast<int>(extra_params[15]) : 0;
                        multi_params.n_ctx_orig  = (extra_params.size() > 16) ? static_cast<int>(extra_params[16]) : 0;
                        multi_params.freq_base   = (extra_params.size() > 17) ? extra_params[17] : 0.0f;
                        multi_params.freq_scale  = (extra_params.size() > 18) ? extra_params[18] : 0.0f;
                        multi_params.ext_factor  = (extra_params.size() > 19) ? extra_params[19] : 0.0f;
                        multi_params.attn_factor = (extra_params.size() > 20) ? extra_params[20] : 0.0f;
                        multi_params.beta_fast   = (extra_params.size() > 21) ? extra_params[21] : 0.0f;
                        multi_params.beta_slow   = (extra_params.size() > 22) ? extra_params[22] : 0.0f;
                        multi_ptr = &multi_params;
                    }
                    else {
                        std::cerr << "❌ Invalid rope_type: " << rope_type << std::endl;
                        return -3;
                    }

                    // ----------------------------
                    // Execute RoPE operation
                    // ----------------------------
                    bool success = rope_openartion(
                        "llama",
                        matrixA_obj,
                        transposeA,
                        use_gpu,
                        gpu_id,
                        send_back,
                        operation_type,
                        base_ptr,
                        ext_ptr,
                        multi_ptr,
                        shard_index_override
                    );

                    if (success) {
                        std::cout << "✅ RoPE operation completed successfully" << std::endl;
                        return 0;
                    } else {
                        std::cerr << "❌ RoPE operation failed" << std::endl;
                        return -7;
                    }
                }

                // ----------------------------
                // flash_attn Computation Operation
                // ----------------------------
                if (command_type == "flash_attn") 
                {
                    // Expected command structure:
                    // flash_attn <matrixQ> <TransposeQ> <matrixK> <TransposeK> <matrixV> <TransposeV> 
                    //            <use_gpu> <gpu_id> <send_back> <operation_type> <scale?> <max_bias?> <logit_softcap?> <backend=?> <mask=?> <shard_index?>

                    if (command_args.size() < 11) {
                        std::cerr << "❌ Insufficient parameters for flash_attn "
                                << "(expected at least 11, got " << command_args.size() << ")"
                                << std::endl;
                        return -3;
                    }

                    const std::string& matrixQ_name = command_args[1];
                    bool transposeQ = (command_args[2] == "true");
                    const std::string& matrixK_name = command_args[3];
                    bool transposeK = (command_args[4] == "true");
                    const std::string& matrixV_name = command_args[5];
                    bool transposeV = (command_args[6] == "true");

                    bool use_gpu = (command_args[7] == "true");
                    int gpu_id = std::stoi(command_args[8]);
                    int send_back = std::stoi(command_args[9]);
                    std::string operation_type = command_args[10];

                    // Optional extras: scale/max_bias/logit_softcap, optional mask=..., last token is shard_index (if present).
                    float scale = 0.0f;
                    float max_bias = 0.0f;
                    float logit_softcap = 0.0f;
                    int shard_index_override = -1;
                    std::string mask_name;
                    std::string backend_type = "llama";

                    auto is_number = [](const std::string& s) -> bool {
                        if (s.empty()) return false;
                        char* end = nullptr;
                        std::strtod(s.c_str(), &end);
                        return end != s.c_str() && *end == '\0';
                    };

                    if ((int)command_args.size() > 11) {
                        int last_idx = (int)command_args.size() - 1;
                        if (is_number(command_args[last_idx])) {
                            shard_index_override = std::stoi(command_args[last_idx]);
                            last_idx -= 1;
                        }

                        int numeric_seen = 0;
                        for (int i = 11; i <= last_idx; ++i) {
                            const std::string& tok = command_args[i];
                            if (tok.rfind("mask=", 0) == 0) {
                                mask_name = tok.substr(5);
                                continue;
                            }
                            if (tok.rfind("backend=", 0) == 0) {
                                backend_type = tok.substr(8);
                                continue;
                            }
                            if (is_number(tok)) {
                                if (numeric_seen == 0) scale = std::stof(tok);
                                else if (numeric_seen == 1) max_bias = std::stof(tok);
                                else if (numeric_seen == 2) logit_softcap = std::stof(tok);
                                numeric_seen++;
                                continue;
                            }
                            if (mask_name.empty()) {
                                mask_name = tok;
                            }
                        }
                    }

                    std::cout << "**run_server_command** flash_attn\n"
                            << "  Q: " << matrixQ_name << " transpose=" << transposeQ << "\n"
                            << "  K: " << matrixK_name << " transpose=" << transposeK << "\n"
                            << "  V: " << matrixV_name << " transpose=" << transposeV << "\n"
                            << "  GPU: " << (use_gpu ? "Yes (ID: " + std::to_string(gpu_id) + ")" : "No") << "\n"
                            << "  send_back: " << send_back << "\n"
                            << "  operation: " << operation_type << "\n"
                            << "  scale: " << scale << " max_bias: " << max_bias << " logit_softcap: " << logit_softcap << "\n"
                            << "  backend: " << backend_type << "\n"
                            << "  shard_override: " << shard_index_override << "\n"
                            << "  mask: " << (mask_name.empty() ? "none" : mask_name) << std::endl;

                    if (backend_type != "llama" && backend_type != "torch") {
                        std::cerr << "❌ Unsupported flash_attn backend: " << backend_type << std::endl;
                        return -3;
                    }

                    // Fetch matrix shard objects
                    matrix_shard_object matrixQ, matrixK, matrixV;
                    matrix_shard_object matrixMask;
                    if (!try_get_matrix_shard_object(matrixQ_name, matrixQ) ||
                        !try_get_matrix_shard_object(matrixK_name, matrixK) ||
                        !try_get_matrix_shard_object(matrixV_name, matrixV)) 
                    {
                        std::cerr << "❌ Missing one or more input matrices in matrix_shard_object_list" << std::endl;
                        return -8;
                    }
                    bool has_mask = false;
                    if (!mask_name.empty()) {
                        if (!try_get_matrix_shard_object(mask_name, matrixMask)) {
                            std::cerr << "❌ Mask matrix not found: " << mask_name << std::endl;
                            return -8;
                        }
                        has_mask = true;
                    }

                    // Call the flash_attn operation
                    bool success = flash_atten_openartion(
                        backend_type,     // backend_type
                        matrixQ, transposeQ,
                        matrixK, transposeK,
                        matrixV, transposeV,
                        has_mask ? &matrixMask : nullptr,
                        scale,
                        max_bias,
                        logit_softcap,
                        use_gpu,
                        gpu_id,
                        send_back,
                        operation_type,
                        shard_index_override
                    );

                    if (success) {
                        std::cout << "✅ flash_attn operation completed successfully" << std::endl;
                        return 0;
                    } else {
                        std::cerr << "❌ flash_attn operation failed" << std::endl;
                        return -7;
                    }
                }

                // ----------------------------  
                // Reshape Computation Operation  
                // ----------------------------  
                if (command_type == "reshape")  
                {  
                    // Expected command structure:  
                    // reshape <matrix> <transpose> <use_gpu> <gpu_id> <send_back> <output_dims> <shard_index>  
                
                    if (command_args.size() < 8) {  
                        std::cerr << "❌ Insufficient parameters for reshape "  
                                << "(expected at least 8, got " << command_args.size() << ")"  
                                << std::endl;  
                        return -3;  
                    }  
                
                    // Parse command arguments  
                    const std::string& matrix_name = command_args[1];  
                    bool transposeA = (command_args[2] == "true");  
                    bool use_gpu = (command_args[3] == "true");  
                    int gpu_id = std::stoi(command_args[4]);  
                    int send_back = std::stoi(command_args[5]);  
                    
                    // Parse output dimensions from comma-separated string  
                    std::vector<int> output_dims;  
                    std::stringstream ss_dims(command_args[6]);  
                    std::string dim_token;  
                    while (std::getline(ss_dims, dim_token, ',')) {  
                        output_dims.push_back(std::stoi(dim_token));  
                    }  
                    
                    if (output_dims.size() != 4) {  
                        std::cerr << "❌ Invalid output dimensions: expected 4 values, got " << output_dims.size() << std::endl;  
                        return -3;  
                    }  
                    
                    int shard_index_override = std::stoi(command_args[7]);  
                
                    std::cout << "**run_server_command** reshape\n"  
                            << "  matrix: " << matrix_name << "\n"  
                            << "  transpose: " << transposeA << "\n"  
                            << "  gpu: " << use_gpu << " (id " << gpu_id << ")\n"  
                            << "  send_back: " << send_back << "\n"  
                            << "  output_dims: [" << output_dims[0] << ", " << output_dims[1]   
                            << ", " << output_dims[2] << ", " << output_dims[3] << "]\n"  
                            << "  shard_override: " << shard_index_override << std::endl;  
                
                    // Fetch matrix shard object  
                    matrix_shard_object matrixA_obj;  
                    if (!try_get_matrix_shard_object(matrix_name, matrixA_obj)) {  
                        std::cerr << "❌ Missing input matrix in matrix_shard_object_list\n"  
                                << "   Needed: '" << matrix_name << "'" << std::endl;  
                        return -8;  
                    }  
                
                    // Execute reshape operation  
                    bool success = reshape_matrix(  
                        "llama",               // backend_type  
                        matrixA_obj,  
                        transposeA,  
                        use_gpu,  
                        gpu_id,  
                        send_back,  
                        output_dims.data(),    // output_dims[4]  
                        shard_index_override  
                    );  
                
                    if (success) {  
                        std::cout << "✅ Reshape operation completed successfully" << std::endl;  
                        return 0;  
                    } else {  
                        std::cerr << "❌ Reshape operation failed" << std::endl;  
                        return -7;  
                    }  
                }

                // ----------------------------  
                // Repeat Computation Operation  
                // ----------------------------  
                if (command_type == "repeat")  
                {  
                    // Expected command structure:  
                    // repeat <matrix> <transpose> <use_gpu> <gpu_id> <send_back> <repeat_dims> <shard_index>  
                
                    if (command_args.size() < 8) {  
                        std::cerr << "❌ Insufficient parameters for repeat "  
                                << "(expected at least 8, got " << command_args.size() << ")"  
                                << std::endl;  
                        return -3;  
                    }  
                
                    // Parse command arguments  
                    const std::string& matrix_name = command_args[1];  
                    bool transposeA = (command_args[2] == "true");  
                    bool use_gpu = (command_args[3] == "true");  
                    int gpu_id = std::stoi(command_args[4]);  
                    int send_back = std::stoi(command_args[5]);  
                    
                    // Parse repeat dimensions from comma-separated string  
                    std::vector<int> repeat_dims;  
                    std::stringstream ss_dims(command_args[6]);  
                    std::string dim_token;  
                    while (std::getline(ss_dims, dim_token, ',')) {  
                        repeat_dims.push_back(std::stoi(dim_token));  
                    }  
                    
                    if (repeat_dims.size() != 4) {  
                        std::cerr << "❌ Invalid repeat dimensions: expected 4 values, got " << repeat_dims.size() << std::endl;  
                        return -3;  
                    }  
                    
                    int shard_index_override = std::stoi(command_args[7]);  
                
                    std::cout << "**run_server_command** repeat\n"  
                            << "  matrix: " << matrix_name << "\n"  
                            << "  transpose: " << transposeA << "\n"  
                            << "  gpu: " << use_gpu << " (id " << gpu_id << ")\n"  
                            << "  send_back: " << send_back << "\n"  
                            << "  repeat_dims: [" << repeat_dims[0] << ", " << repeat_dims[1]  
                            << ", " << repeat_dims[2] << ", " << repeat_dims[3] << "]\n"  
                            << "  shard_override: " << shard_index_override << std::endl;  
                
                    // Fetch matrix shard object  
                    matrix_shard_object matrixA_obj;  
                    if (!try_get_matrix_shard_object(matrix_name, matrixA_obj)) {  
                        std::cerr << "❌ Missing input matrix in matrix_shard_object_list\n"  
                                << "   Needed: '" << matrix_name << "'" << std::endl;  
                        return -8;  
                    }  
                
                    // Execute repeat operation  
                    bool success = repeat_matrix(  
                        "llama",               // backend_type  
                        matrixA_obj,  
                        transposeA,  
                        use_gpu,  
                        gpu_id,  
                        send_back,  
                        repeat_dims.data(),    // repeat_dims[4]  
                        shard_index_override  
                    );  
                
                    if (success) {  
                        std::cout << "✅ Repeat operation completed successfully" << std::endl;  
                        return 0;  
                    } else {  
                        std::cerr << "❌ Repeat operation failed" << std::endl;  
                        return -7;  
                    }  
                }


                // ----------------------------
                // combine and send back shards
                // ----------------------------
                if (command_type == "send_back")
                {
                    // Expected command: send_back <matrix_name> <shard_index> <send_back>
                    if (command_args.size() < 4) {
                        std::cerr << "❌ Insufficient parameters for send_back "
                                << "(expected 4, got " << command_args.size() << ")" << std::endl;
                        try { send_ack("ACK_send_back_complete"); } catch (...) {}
                        return -3;
                    }

                    const std::string& matrix_name = command_args[1];
                    int shard_index = std::stoi(command_args[2]);
                    int send_back = std::stoi(command_args[3]);  // can be encoded join_dim/system

                    // Build shard filename if base name was provided
                    std::string shard_name = matrix_name;
                    if (shard_name.find("_shard_") == std::string::npos) {
                        shard_name += "_shard_" + std::to_string(shard_index) + ".bin";
                    }

                    std::cout << "**run_server_command** send_back\n"
                            << "  matrix: " << matrix_name << "\n"
                            << "  shard_index: " << shard_index << "\n"
                            << "  send_back: " << send_back << "\n"
                            << "  shard_name: " << shard_name << std::endl;

                    // Look up the shard
                    matrix_shard_object matrix_obj;
                    if (!try_get_matrix_shard_object(shard_name, matrix_obj)) {
                        std::cerr << "❌ Missing matrix in matrix_shard_object_list\n"
                                << "   Needed: '" << shard_name << "'" << std::endl;
                        try { send_ack("ACK_send_back_complete"); } catch (...) {}
                        return -8;
                    }

                    if (!matrix_obj.data) {
                        std::cerr << "❌ Matrix has no data: " << shard_name << std::endl;
                        try { send_ack("ACK_send_back_complete"); } catch (...) {}
                        return -9;
                    }

                    // Convert shard to MatrixResult
                    MatrixResult result;
                    result.dims[0] = std::max(1, matrix_obj.batchA);
                    result.dims[1] = std::max(1, matrix_obj.depthA);
                    result.dims[2] = std::max(1, matrix_obj.rows_A);
                    result.dims[3] = std::max(1, matrix_obj.cols_A);

                    const size_t total_elements =
                        static_cast<size_t>(result.dims[0]) *
                        static_cast<size_t>(result.dims[1]) *
                        static_cast<size_t>(result.dims[2]) *
                        static_cast<size_t>(result.dims[3]);

                    if (matrix_obj.data->size() != total_elements) {
                        std::cerr << "❌ Matrix size mismatch for " << shard_name
                                << " (expected " << total_elements
                                << ", got " << matrix_obj.data->size() << ")" << std::endl;
                        try { send_ack("ACK_send_back_complete"); } catch (...) {}
                        return -9;
                    }

                    result.data = std::make_unique<float[]>(total_elements);
                    std::memcpy(result.data.get(), matrix_obj.data->data(), total_elements * sizeof(float));

                    // Send back this shard to the head node
                    bool success = send_back_file(
                        shard_name,                 // local_file_path
                        shard_name,                 // filename
                        result,                     // MatrixResult
                        send_back,                  // encoded send_back flag
                        "llama",                    // backend (adjust if needed)
                        matrix_obj.output_dtype_tag // dtype tag
                    );

                    if (success) {
                        std::cout << "✅ Shard " << shard_index << " sent back successfully" << std::endl;
                    } else {
                        std::cerr << "❌ Failed to send back shard " << shard_index << std::endl;
                    }

                    try { send_ack("ACK_send_back_complete"); } catch (...) {}
                    return success ? 0 : -7;
                }
    

                // ----------------------------
                // Matrix Computation Operations
                // ----------------------------
                if (command_type == "llama" || command_type == "opencl" || command_type == "torch") 
                {
                    // Validate required parameters for matrix operations
                    if (command_args.size() < 10) {
                        std::cerr << "❌ Insufficient parameters for " << command_type 
                                << " operation (expected 10, got " << command_args.size() << ")" << std::endl;
                        return -3;
                    }

                    // Parse matrix operation parameters
                    bool transposeA = (command_args[2] == "true");
                    bool transposeB = (command_args[4] == "true");
                    bool use_gpu    = (command_args[5] == "true");
                    int gpu_id      = std::stoi(command_args[6]);
                    int send_back   = std::stoi(command_args[7]);  // Number of result shards to return
                    std::string operation_type = command_args[8];  // e.g., "matmul", "add", etc.
                    int n_dims      = std::stoi(command_args[9]);  // Matrix dimensions
                    int shard_index_override = -1;
                    if (command_args.size() > 10) {
                        shard_index_override = std::stoi(command_args[10]);
                    }
                    std::cout << "**run_server_command** shard_index_override: " << shard_index_override << std::endl;
                    std::cout << "**run_server_command** send_back: " << send_back << std::endl;
                    
                    bool operation_success = false;
                    std::string backend_name;

                    // Dispatch to unified matrix operation function
                    
                    if (command_type == "llama")
                    {
                        matrix_shard_object matrixA_obj;
                        matrix_shard_object matrixB_obj;
                        if (!try_get_matrix_shard_object(command_args[3], matrixA_obj) ||
                            !try_get_matrix_shard_object(command_args[1], matrixB_obj)) {
                            std::cerr << "❌ Missing input matrix in matrix_shard_object_list" << std::endl;
                            std::cerr << "   Needed: '" << command_args[3] << "' and '" << command_args[1] << "'" << std::endl;
                            return -8;
                        }
                        operation_success = matrix_operation(
                            command_type,
                            matrixA_obj,               // Matrix B (GGML order)
                            transposeB,
                            matrixB_obj,               // Matrix A (GGML order)
                            transposeA,
                            use_gpu,
                            gpu_id,
                            send_back,
                            operation_type,
                            n_dims,
                            shard_index_override
                        );
                    }
                    else
                    {
                        matrix_shard_object matrixA_obj;
                        matrix_shard_object matrixB_obj;
                        if (!try_get_matrix_shard_object(command_args[1], matrixA_obj) ||
                            !try_get_matrix_shard_object(command_args[3], matrixB_obj)) {
                            std::cerr << "❌ Missing input matrix in matrix_shard_object_list" << std::endl;
                            std::cerr << "   Needed: '" << command_args[1] << "' and '" << command_args[3] << "'" << std::endl;
                            return -8;
                        }

                        operation_success = matrix_operation(
                            command_type,
                            matrixA_obj,
                            transposeA,
                            matrixB_obj,
                            transposeB,
                            use_gpu,
                            gpu_id,
                            send_back,
                            operation_type,
                            n_dims,
                            shard_index_override
                        );
                    }
                    
                    if (command_type == "llama")
                        backend_name = "LLaMA/Vulkan";
                    else if (command_type == "torch")
                        backend_name = "PyTorch";
                    else if (command_type == "opencl")
                        backend_name = "OpenCL";

                    // Report operation outcome
                    if (operation_success) {
                        std::cout << "✅ " << backend_name << " operation completed successfully" << std::endl;
                        std::cout << "   • Operation: " << operation_type << std::endl;
                        std::cout << "   • GPU: " << (use_gpu ? "Yes (ID: " + std::to_string(gpu_id) + ")" : "No") << std::endl;
                        std::cout << "   • Result shards: " << send_back << std::endl;
                        return 0;
                    } else {
                        std::cerr << "❌ " << backend_name << " operation failed: " << operation_type << std::endl;
                        return -7;
                    }

                } 
                else 
                {
                    std::cerr << "❌ Unsupported server command type: '" << command_type << "'" << std::endl;
                    std::cerr << "   Supported commands: llama, opencl, torch" << std::endl;
                    return -6;
                }

            } 
            catch (const std::exception& e) 
            {
                std::cerr << "❌ Error executing server command: " << e.what() << std::endl;
                std::cerr << "   Command: " << command << std::endl;
                return -1;
            }
        }

        std::pair<std::string, int> get_matrix_name_and_shard_number(const std::string& shard_path) 
        {
            // Extract just the filename from the full path (remove directory portion)
            // Example: "/path/to/matrixA_shard_42.bin" -> "matrixA_shard_42.bin"
            std::string filename = shard_path.substr(shard_path.find_last_of("/") + 1);
            
            // Look for the shard pattern in the filename
            // Shard files follow the naming convention: <matrix_name>_shard_<number>.<extension>
            auto find_shard_pos = [](const std::string& name) -> std::pair<size_t, size_t> {
                size_t pos = name.find("_shard_");
                if (pos != std::string::npos) {
                    return {pos, 7};  // length of "_shard_"
                }
                // Accept legacy/pluralized variant to stay compatible with older runs
                pos = name.find("_shards_");
                if (pos != std::string::npos) {
                    return {pos, 8};  // length of "_shards_"
                }
                return {std::string::npos, 0};
            };

            auto [shard_pos, shard_token_len] = find_shard_pos(filename);
            
            if (shard_pos != std::string::npos) {
                // Extract the base matrix name (everything before the shard token)
                std::string matrix_name = filename.substr(0, shard_pos);
                
                // Extract the shard number portion (everything after the shard token)
                std::string shard_part = filename.substr(shard_pos + shard_token_len);
                
                // Remove file extension if present to isolate just the number
                // Example: "42.bin" -> "42"
                size_t dot_pos = shard_part.find_last_of(".");
                if (dot_pos != std::string::npos) {
                    shard_part = shard_part.substr(0, dot_pos);
                }
                
                try {
                    // Convert the shard number string to integer
                    int shard_number = std::stoi(shard_part);
                    
                    // Return both the matrix name and shard number
                    return {matrix_name, shard_number};
                } catch (const std::exception& e) {
                    // Handle parsing errors (e.g., if shard_part contains non-numeric characters)
                    // Return -1 as invalid shard number to indicate parsing failure
                    std::cerr << "⚠️ Failed to parse shard number from: '" << shard_part 
                            << "' in file: " << filename << std::endl;
                    return {matrix_name, -1};
                }
            }
            
            // If the filename doesn't match the shard pattern, return the entire filename
            // as the matrix name and -1 to indicate this is not a shard file
            // This handles cases like regular matrix files or incorrectly named files
            return {filename, -1};
        }

        std::string get_matrix_output_filename(
            const std::string& matrix_pathA,
            const std::string& matrix_pathB
        )
        {
            // Extract filenames only
            std::string a_filename =
                std::filesystem::path(matrix_pathA).filename().string();
            std::string b_filename =
                std::filesystem::path(matrix_pathB).filename().string();

            // -------------------------------
            // Remove ".bin" if present
            // -------------------------------
            auto strip_bin = [](std::string& name)
            {
                size_t pos = name.rfind(".bin");
                if (pos != std::string::npos)
                    name = name.substr(0, pos);
            };

            strip_bin(a_filename);
            strip_bin(b_filename);

            // -------------------------------
            // Extract shard numbers
            // -------------------------------
            auto [matrix_nameA, shard_numA] =
                get_matrix_name_and_shard_number(a_filename);
            auto [matrix_nameB, shard_numB] =
                get_matrix_name_and_shard_number(b_filename);

            // -------------------------------
            // Remove any lingering "_shard_X"
            // -------------------------------
            auto strip_shard = [](std::string& name)
            {
                size_t pos = name.find("_shard_");
                if (pos != std::string::npos)
                    name = name.substr(0, pos);
            };

            strip_shard(matrix_nameA);
            strip_shard(matrix_nameB);

            // -------------------------------
            // Determine shard number
            // -------------------------------
            int shard_num = -1;
            if (shard_numA != -1)
                shard_num = shard_numA;
            else if (shard_numB != -1)
                shard_num = shard_numB;

            // -------------------------------
            // Build output filename
            // -------------------------------
            std::string output_filename =
                matrix_nameA + "x" + matrix_nameB;

            // If no shard number could be inferred, assign a sequential shard index per output base
            if (shard_num == -1) {
                std::lock_guard<std::mutex> lock(output_shard_mutex);
                shard_num = output_shard_counters[output_filename]++;
            }

            if (shard_num != -1)
                output_filename += "_shard_" + std::to_string(shard_num);

            output_filename += ".bin";

            return output_filename;
        }

        void save_file_handler()
        {
            // Move reserved files to local copy under lock for processing
            std::vector<ReservedFiles> local_reserved_files;

            {
                std::lock_guard<std::mutex> lock(file_data_mutex);

                if (reserved_files_list.empty())
                {
                    std::cout << "No files to save" << std::endl;
                    return;
                }

                local_reserved_files = std::move(reserved_files_list);
                reserved_files_list.clear();

                std::cout << "Processing: " << local_reserved_files.size() << " reserved file(s)" << std::endl;
            }

			            auto process_payload = [&](const std::string& filename_in, const std::vector<uint8_t>& bytes) {
	                std::string filename = normalize_matrix_key(filename_in);
	                bool prefer_vram = false;
	                const bool vram_only_mode = get_env_flag("OPEN_CLUSTER_VRAM_ONLY", false);
	                const std::string vram_prefix = "VRAM|";
	                if (filename.rfind(vram_prefix, 0) == 0) {
	                    prefer_vram = true;
	                    filename = filename.substr(vram_prefix.size());
	                }
	                const bool is_sent_back = filename.rfind("sent_back=", 0) == 0 || filename.find("sent_back=") != std::string::npos;

	                if (is_sent_back)
	                {
	                    // Worker result shard streamed to head for combining.
	                    const bool is_head_node = (local_IP_eth == head_node_ip_eth || local_IP_wifi == head_node_ip_wifi);
	                    if (!is_head_node) {
	                        return;
	                    }

                    const size_t pos = filename.find("sent_back=");
                    std::string actual_filename = (pos == std::string::npos)
                        ? filename
                        : filename.substr(pos + 10);

                    // Optional: parse send_back from header (`sent_back=<send_back>|<filename>`).
                    // If absent (old workers), we fall back to `send_back=0` and will only
                    // be able to combine once a local/head shard supplies the metadata.
                    int send_back = 0;
                    const size_t sep = actual_filename.find('|');
                    if (sep != std::string::npos) {
                        const std::string send_back_str = actual_filename.substr(0, sep);
                        try {
                            send_back = std::stoi(send_back_str);
                        } catch (...) {
                            send_back = 0;
                        }
                        actual_filename = actual_filename.substr(sep + 1);
                    }

	                    std::unique_ptr<float[]> shard_data;
	                    int rows = 0;
	                    int cols = 0;
	                    int batch = 1;
	                    int depth = 1;
	                    int dtype_tag = -1;
	                    if (!decode_matrix_binary_payload_to_f32(bytes, shard_data, rows, cols, batch, depth, dtype_tag)) {
	                        std::cerr << "ERROR: Failed to decode sent_back payload: " << actual_filename << std::endl;
	                        return;
	                    }
		                    if (batch != 1 || depth != 1) {
		                        std::cout << "INFO: sent_back payload is 4D (batch=" << batch
		                                  << ", depth=" << depth << "); combine will flatten to 2D for "
		                                  << actual_filename << std::endl;
		                    }

		                    handle_combine_matrix_shard_list(
		                        actual_filename,
		                        std::move(shard_data),
		                        rows,
		                        cols,
		                        batch,
		                        depth,
		                        send_back,
		                        dtype_tag
		                    );

	                    return;
	                }

	                // Input matrices: keep fully in memory and ACK the sender.
	                matrix_shard_object obj;
	                if (!decode_matrix_binary_payload(bytes, obj, filename)) {
	                    std::cerr << "ERROR: Failed to decode matrix payload: " << filename << std::endl;
	                    return;
	                }

	                // If requested, try to cache this matrix into VRAM (best-effort).
	                bool cached_to_vram = false;
	                if (prefer_vram) {
	                    const int backend_index = default_vram_backend_index();
	                    auto& vram = get_vram_cache_manager();
	                    if (backend_index >= 0 && vram.enabled(backend_index) && obj.data) {
	                        if (vram.get_tensor(backend_index, obj.base_file_name)) {
	                            cached_to_vram = true;
	                        } else {
	                            cached_to_vram = vram.cache_tensor_f32_4d(
	                                backend_index,
	                                obj.base_file_name,
	                                obj.data->data(),
	                                obj.cols_A,
	                                obj.rows_A,
	                                obj.depthA,
	                                obj.batchA
	                            );
	                        }
	                    }
	                }

	                if (prefer_vram && cached_to_vram && vram_only_mode) {
	                    obj.vram_only = true;
	                    obj.data.reset();
	                }

	                upsert_matrix_shard_object(std::move(obj));

                // Persist the exact bytes to disk for later reload via `load_matrix_shard_object_list`.
                // (No /dev/shm usage; matrix_shard_folder defaults to `matrix_shards/`.)
                if (!(prefer_vram && cached_to_vram && vram_only_mode)) {
                    try {
                        std::filesystem::path shard_dir(matrix_shard_folder);
                        if (!shard_dir.is_absolute()) {
                            shard_dir = std::filesystem::path(project_folder) / shard_dir;
                        }
                        std::filesystem::create_directories(shard_dir);
                        std::filesystem::path out_path = shard_dir / filename;

                        std::ofstream file(out_path, std::ios::binary);
                        if (!file.is_open()) {
                            std::cerr << "ERROR: Failed to open for write: " << out_path << std::endl;
                        } else {
                            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                            file.close();
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "ERROR: Failed to persist matrix bytes for " << filename
                                  << ": " << e.what() << std::endl;
                    }
                }

                send_ack(filename);
            };

            for (auto &rf : local_reserved_files)
            {
                std::string filename = rf.save_parallel_file_name.empty() ? std::string("unknown") : rf.save_parallel_file_name[0];

                if ((rf.is_parallel) || (!rf.received_data_eth_file.empty() && !rf.received_data_wifi_file.empty()))
                {
                    std::vector<uint8_t> combined;
                    combined.reserve(rf.received_data_eth_file.size() + rf.received_data_wifi_file.size());
                    combined.insert(combined.end(), rf.received_data_eth_file.begin(), rf.received_data_eth_file.end());
                    combined.insert(combined.end(), rf.received_data_wifi_file.begin(), rf.received_data_wifi_file.end());
                    process_payload(filename, combined);
                }
                else if (!rf.received_data_eth_file.empty())
                {
                    process_payload(filename, rf.received_data_eth_file);
                }
                else if (!rf.received_data_wifi_file.empty())
                {
                    process_payload(filename, rf.received_data_wifi_file);
                }
                else
                {
                    std::cout << "Skipping empty ReservedFiles entry for: " << filename << std::endl;
                }
            }

            std::cout << "Save file handler completed" << std::endl;
        }

	    bool send_back_file(const std::string& local_file_path,
	                            const std::string& filename,
	                            MatrixResult& save_result,
	                            int total_shards,
	                            const std::string& selected_backend,
	                            int output_dtype_tag)
        {
            std::cout << "SENDING BACK FILE" << std::endl;
            bool is_head_node = (local_IP_eth == head_node_ip_eth || local_IP_wifi == head_node_ip_wifi);

            // ============================================================
            // WORKER NODE → STREAM RESULT BACK TO HEAD (NO DISK)
            // ============================================================
	            if (!is_head_node)
	            {
	                // Include the `send_back` encoding in the header so the head node can
	                // combine shards even when it doesn't compute a local shard.
	                // Backwards compatible format: `sent_back=<send_back>|<filename>`.
	                std::string send_back_filename =
	                    "sent_back=" + std::to_string(total_shards) + "|" + filename;
	                std::cout << "Worker streaming result back to head: "
	                        << send_back_filename << std::endl;

	                // Stream directly from memory using v2 binary format.
	                // IMPORTANT: results must be routed to the head node PULL socket (file_receiver_*),
	                // which is what `head_node_sender_*` is connected to.
	                bool ok = stream_matrix_binary(
	                    head_node_sender_eth,
	                    head_node_sender_mutex,
	                    head_node_ip_eth,     // destination label (for logging)
	                    save_result,          // MatrixResult (already in memory)
	                    send_back_filename,   // name expected by save_file_handler
	                    output_dtype_tag      // -1 f32, -2 fp16, -3 bf16
	                );

	                if (!ok)
	                {
	                    ok = stream_matrix_binary(
	                        head_node_sender_wifi,
	                        head_node_sender_mutex,
	                        head_node_ip_wifi,
	                        save_result,
	                        send_back_filename,
	                        output_dtype_tag
	                    );
	                }

	                if (!ok)
	                {
	                    std::cerr << "❌ Failed to stream matrix back to head: "
	                            << send_back_filename << std::endl;
                    return false;
                }

                return true;
            }

            // ============================================================
            // HEAD NODE → SAVE FILE + TRACK SHARDS
            // ============================================================
		            if (is_head_node)
		            {
		                // Extract shard dimensions
		                const int shard_batch = save_result.dims[0];
		                const int shard_depth = save_result.dims[1];
		                int shard_rows = save_result.dims[2];
		                int shard_cols = save_result.dims[3];
		                const size_t data_size = static_cast<size_t>(shard_rows) * static_cast<size_t>(shard_cols) * sizeof(float);

	                // Move shard buffer directly into combine path (avoid extra copy).
	                auto shard_data = std::move(save_result.data);
	                if (!shard_data) {
	                    std::cerr << "ERROR: Missing shard data for send_back: " << filename << std::endl;
	                    return false;
	                }


		                std::cout << "**send_back_file** total_shards: " << total_shards;
		                // Process shard through combination handler
		                bool result = handle_combine_matrix_shard_list(
	                    filename,
	                    std::move(shard_data),
	                    shard_rows,
	                    shard_cols,
	                    shard_batch,
	                    shard_depth,
	                    total_shards,
	                    output_dtype_tag
	                );

                std::cout << "Head node processed shard: " << filename 
                        << " (" << data_size << " bytes)" << std::endl;

                return result;
            }

            return false;
        }

	    bool stream_matrix_binary(
	        zmq::socket_t& out_socket,
	        std::mutex& out_socket_mutex,
	        const std::string& dest_label,
	        const MatrixResult& result,
	        const std::string& save_name,
	        int dtype_tag = -1
	    )
	    {
            // Validate dtype
            if (dtype_tag != -1 && dtype_tag != -2 && dtype_tag != -3) {
                std::cerr << "Unsupported dtype_tag for stream_matrix_binary: "
                        << dtype_tag << std::endl;
                return false;
            }

            // Expect fixed 4D layout (same as save_matrix_bin)
            const int ndim = 4;
            size_t total_elements = 1;
            for (int i = 0; i < ndim; i++) {
                total_elements *= result.dims[i];
            }

            const size_t elem_bytes = (dtype_tag == -1) ? 4 : 2;
            const size_t header_bytes = sizeof(int) * 5; // dtype + 4 dims
            const size_t payload_bytes = header_bytes + total_elements * elem_bytes;

	            // Allocate payload message (so we don't memcpy again into zmq::message_t)
	            zmq::message_t payload_msg(payload_bytes);

	            // ---- Write header ----
	            auto* header = static_cast<int*>(payload_msg.data());
	            header[0] = dtype_tag;
	            for (int i = 0; i < ndim; i++) {
	                header[i + 1] = result.dims[i];
	            }

	            // ---- Write payload ----
	            uint8_t* data_ptr = static_cast<uint8_t*>(payload_msg.data()) + header_bytes;

	            if (dtype_tag == -1) {
	                // float32 (direct copy)
	                std::memcpy(
	                    data_ptr,
                    result.data.get(),
                    total_elements * sizeof(float)
                );
            }
            else if (dtype_tag == -2) {
                // float16
                uint16_t* out = reinterpret_cast<uint16_t*>(data_ptr);
                for (size_t i = 0; i < total_elements; ++i) {
                    out[i] = float_to_fp16_bits(result.data[i]);
                }
            }
            else {
                // bfloat16
                uint16_t* out = reinterpret_cast<uint16_t*>(data_ptr);
                for (size_t i = 0; i < total_elements; ++i) {
                    out[i] = float_to_bf16_bits(result.data[i]);
                }
            }

	            // ---- Send via ZMQ (same pattern as zmq_send_file) ----
	            try {
	                zmq::message_t name_msg(save_name.data(), save_name.size());

	                std::lock_guard<std::mutex> lock(out_socket_mutex);
	                out_socket.send(name_msg, zmq::send_flags::sndmore);
	                out_socket.send(payload_msg, zmq::send_flags::none);

	                std::cout << "📤 Streamed matrix to " << dest_label
	                        << " as " << save_name
	                        << " (" << payload_bytes << " bytes)" << std::endl;
	            }
	            catch (const zmq::error_t& e) {
	                std::cerr << "ZMQ stream error: " << e.what() << std::endl;
                return false;
            }

            return true;
        }

	    bool handle_combine_matrix_shard_list(
	        const std::string& filename,
	        std::unique_ptr<float[]> data,
	        int shard_rows,
	        int shard_cols,
	        int shard_batch,
	        int shard_depth,
	        int total_shards,
	        int output_dtype_tag
	    )
	    {
            // Optimized combine path:
            // - store float shards directly (no shard_bytes / Torch cat)
            // - combine via memcpy into one output buffer when complete
            auto [matrix_name, shard_num] = get_matrix_name_and_shard_number(filename);
            if (shard_num < 0) {
                shard_num = 0;
            }

            // Parse send_back encoding when available:
            // - sign indicates System (System-2 uses negative)
            // - abs encodes join_dim (tens) + shards (ones) when >= 10
            int incoming_join_dim = 0;
            int incoming_shards_needed = 0;
            bool incoming_system2 = false;
            const bool has_send_back = (total_shards != 0);
            if (has_send_back) {
                const int abs_val = std::abs(total_shards);
                incoming_join_dim = 0;
                incoming_shards_needed = abs_val;
                if (abs_val >= 10) {
                    incoming_join_dim = abs_val / 10;
                    incoming_shards_needed = abs_val % 10;
                }
                incoming_system2 = (total_shards < 0);
            }

            bool completed = false;
            combined_matrix_shards done;
            int done_needed = 0;
            int done_output_dtype_tag = -1;

            {
                std::lock_guard<std::mutex> lock(combined_matrix_shards_mutex);

                auto [it, inserted] = combined_matrix_shards_map.try_emplace(matrix_name);
                combined_matrix_shards& combined = it->second;
                if (inserted || combined.file_name.empty()) {
                    combined.file_name = matrix_name;
                }

                // Persist shard-count/system/join-dim when we first learn it.
                if (combined.number_of_shards_needed == 0 && has_send_back) {
                    combined.join_dim = incoming_join_dim;
                    combined.number_of_shards_needed = incoming_shards_needed;
                    combined.is_system2 = incoming_system2;
                }

                combined.output_dtype_tag =
                    merge_output_dtype_tag(combined.output_dtype_tag, output_dtype_tag);

                // Dedupe retransmits
                if (combined.shards.find(shard_num) != combined.shards.end()) {
                    return true;
                }

	                combined_matrix_shards::ShardView view;
	                view.batch = shard_batch;
	                view.depth = shard_depth;
	                view.rows = shard_rows;
	                view.cols = shard_cols;
	                view.dtype_tag = output_dtype_tag;
	                view.data = std::move(data);
	                combined.shards.emplace(shard_num, std::move(view));
                combined.total_shards_reserved = static_cast<int>(combined.shards.size());

                if (!combined.have_index_range) {
                    combined.have_index_range = true;
                    combined.min_shard_index = shard_num;
                    combined.max_shard_index = shard_num;
                } else {
                    combined.min_shard_index = std::min(combined.min_shard_index, shard_num);
                    combined.max_shard_index = std::max(combined.max_shard_index, shard_num);
                }

                done_needed = combined.number_of_shards_needed;
                if (done_needed > 0 && combined.total_shards_reserved == done_needed) {
                    completed = true;
                    done_output_dtype_tag = combined.output_dtype_tag;
                    done = std::move(combined);
                    combined_matrix_shards_map.erase(it);
                }
            }

	            if (!completed) {
	                return true;
	            }

	            bool use_flash_combine = false;
	            {
	                std::lock_guard<std::mutex> lock(flash_atten_openartion_combine_mutex);
	                use_flash_combine = (flash_atten_openartion_combine_list.find(matrix_name) != flash_atten_openartion_combine_list.end());
	            }
	            if (!use_flash_combine) {
	                // Fallback: if the payload is 4D (batch/depth != 1), use the flattening combine.
	                for (const auto& kv : done.shards) {
	                    const auto& s = kv.second;
	                    if (s.batch != 1 || s.depth != 1) {
	                        use_flash_combine = true;
	                        break;
	                    }
	                }
	            }

	            MatrixResult full;
	            try {
	                if (done_needed == 1) {
	                    if (done.shards.size() != 1) {
	                        std::cerr << "ERROR: Expected 1 shard but got "
	                                  << done.shards.size() << " for " << matrix_name << std::endl;
	                        return true;
	                    }
	                    auto& only = done.shards.begin()->second;
	                    const int64_t rows2d = (int64_t) only.batch * (int64_t) only.depth * (int64_t) only.rows;
	                    if (rows2d <= 0 || rows2d > std::numeric_limits<int>::max()) {
	                        throw std::runtime_error("Invalid flattened rows for single-shard combine");
	                    }
	                    full.dims[0] = 1;
	                    full.dims[1] = 1;
	                    full.dims[2] = (int) rows2d;
	                    full.dims[3] = only.cols;
	                    full.data = std::move(only.data);
	                } else {
	                    if (use_flash_combine) {
	                        // For FlashAttention (and any 4D payloads), flatten (batch*depth*rows) to 2D before joining.
                            full = done.is_system2
	                            ? combine_flash_attn_shards_grid_2d(done)
	                            : matrix_backend_llama.combine_flash_attn_shards_ggml(done);

	                    } else {
	                        full = done.is_system2
	                            ? combine_matrix_shards_grid_2d(done)
	                            : combine_matrix_shards_2d(done);
	                    }
	                }
	            } catch (const std::exception& e) {
	                std::cerr << "ERROR: Combine failed for " << matrix_name << ": " << e.what() << std::endl;
	                // Avoid leaving stale flash-combine markers around.
	                if (use_flash_combine) {
	                    std::lock_guard<std::mutex> lock(flash_atten_openartion_combine_mutex);
	                    flash_atten_openartion_combine_list.erase(matrix_name);
	                }
	                return true;
	            }

	            if (full.data) {
	                const bool sent = send_combined_bin_to_python(
	                    matrix_name,
	                    full,
                    done_output_dtype_tag
                );
                if (!sent) {
                    std::cerr << "ERROR: Failed to stream combined PT for "
                              << matrix_name << std::endl;
	                }
	                send_ack("ACK_combined_matrix_saved");
	                if (use_flash_combine) {
	                    std::lock_guard<std::mutex> lock(flash_atten_openartion_combine_mutex);
	                    flash_atten_openartion_combine_list.erase(matrix_name);
	                }
	            } else {
	                std::cerr << "ERROR: Combine failed for " << matrix_name << std::endl;
	            }

	            return true;

#if 0
            // ============================================================
            // DEBUG: Print incoming parameters
            // ============================================================
            std::cout << "DEBUG: handle_combine_matrix_shard_list called" << std::endl;
            std::cout << "DEBUG: filename='" << filename << "'" << std::endl;
            std::cout << "DEBUG: shard_rows=" << shard_rows
                    << ", shard_cols=" << shard_cols << std::endl;
            std::cout << "DEBUG: total_shards=" << total_shards << std::endl;
            std::cout << "DEBUG: output_dtype_tag=" << output_dtype_tag << std::endl;

            // ============================================================
            // EXTRACT MATRIX NAME AND SHARD NUMBER
            // ============================================================
            auto [matrix_name, shard_num] = get_matrix_name_and_shard_number(filename);

            std::cout << "DEBUG: Extracted matrix_name='"
                    << matrix_name << "', shard_num=" << shard_num << std::endl;

            // ============================================================
            // BUILD SHARD BYTES WITH METADATA
            // ============================================================
            std::vector<uint8_t> shard_bytes;

            // v2 header: dtype_tag + fixed 4D dims (payload is float32 here)
            int dtype_tag = -1;
            shard_bytes.insert(
                shard_bytes.end(),
                reinterpret_cast<uint8_t*>(&dtype_tag),
                reinterpret_cast<uint8_t*>(&dtype_tag) + sizeof(int)
            );

            int dims[4] = {1, 1, shard_rows, shard_cols};
            for (int i = 0; i < 4; ++i)
            {
                shard_bytes.insert(
                    shard_bytes.end(),
                    reinterpret_cast<uint8_t*>(&dims[i]),
                    reinterpret_cast<uint8_t*>(&dims[i]) + sizeof(int)
                );
            }

            size_t data_size = static_cast<size_t>(shard_rows) * shard_cols * sizeof(float);
            shard_bytes.insert(
                shard_bytes.end(),
                reinterpret_cast<uint8_t*>(data.get()),
                reinterpret_cast<uint8_t*>(data.get()) + data_size
            );

            std::cout << "DEBUG: Created shard_bytes of size "
                    << shard_bytes.size() << std::endl;

            // ============================================================
            // TRACK EXISTING MATRIX
            // ============================================================
            std::cout << "DEBUG: Checking "
                    << combined_matrix_shards_list.size()
                    << " existing tracking entries" << std::endl;

            for (auto& combined : combined_matrix_shards_list)
            {
                auto [combined_name, _] = get_matrix_name_and_shard_number(combined.file_name);

                if (combined_name == matrix_name)
                {
                    std::cout << "DEBUG: FOUND MATCH for matrix '"
                            << matrix_name << "'" << std::endl;

                    // Dedupe retransmits (do not advance total_shards_reserved for duplicates)
                    if (combined.received_shard_numbers.find(shard_num) != combined.received_shard_numbers.end())
                    {
                        std::cout << "DEBUG: Duplicate shard " << shard_num
                                << " ignored for matrix '" << matrix_name << "'" << std::endl;
                        return true;
                    }

                    // Update shard count if first shard was placeholder
                    if (combined.number_of_shards_needed == 0 && total_shards != 0)
                    {
                        int abs_val = std::abs(total_shards);
                        int join_dim = 0;
                        int shards_needed = abs_val;

                        if (abs_val >= 10)
                        {
                            join_dim = abs_val / 10;
                            shards_needed = abs_val % 10;
                        }

                        combined.join_dim = join_dim;
                        combined.number_of_shards_needed = shards_needed;
                        combined.is_system2 = (total_shards < 0);

                        std::cout << "DEBUG: number_of_shards_needed UPDATED to "
                                << shards_needed
                                << ", join_dim=" << join_dim << std::endl;
                    }

                    combined.output_dtype_tag = merge_output_dtype_tag(combined.output_dtype_tag, output_dtype_tag);

                    combined.total_shards_reserved++;
                    combined.shard_numbers.push_back(shard_num);
                    combined.received_matrix_data.push_back(std::move(shard_bytes));
                    combined.dims_list.push_back({1, 1, shard_rows, shard_cols});
                    combined.received_shard_numbers.insert(shard_num);

                    std::cout << "DEBUG: Updated shard count to "
                            << combined.total_shards_reserved
                            << " of " << combined.number_of_shards_needed
                            << std::endl;

                    int needed = std::abs(combined.number_of_shards_needed);

                    if (needed > 0 && combined.total_shards_reserved == needed)
                    {
                        bool is_system2 = combined.is_system2;

                        std::cout << "DEBUG: ALL SHARDS RECEIVED!" << std::endl;

                        auto shard_bytes_to_result = [](const std::vector<uint8_t>& bytes) -> MatrixResult {
                            MatrixResult out;
                            if (bytes.size() < static_cast<size_t>(5 * sizeof(int))) {
                                return out;
                            }
                            int dtype_tag = 0;
                            int dims[4] = {0, 0, 0, 0};
                            std::memcpy(&dtype_tag, bytes.data(), sizeof(int));
                            std::memcpy(&dims[0], bytes.data() + sizeof(int), 4 * sizeof(int));
                            if (dtype_tag != -1) {
                                throw std::runtime_error("single-shard fast-path expects float32 v2 shard (dtype_tag=-1)");
                            }

                            const size_t numel =
                                static_cast<size_t>(dims[0]) * static_cast<size_t>(dims[1]) *
                                static_cast<size_t>(dims[2]) * static_cast<size_t>(dims[3]);
                            const size_t need_bytes = static_cast<size_t>(5 * sizeof(int)) + numel * sizeof(float);
                            if (bytes.size() < need_bytes) {
                                throw std::runtime_error("single-shard fast-path payload truncated");
                            }

                            out.dims[0] = dims[0];
                            out.dims[1] = dims[1];
                            out.dims[2] = dims[2];
                            out.dims[3] = dims[3];
                            out.data = std::make_unique<float[]>(numel);
                            std::memcpy(out.data.get(), bytes.data() + 5 * sizeof(int), numel * sizeof(float));
                            return out;
                        };

                        MatrixResult full;
                        if (needed == 1)
                        {
                            std::cout << "DEBUG: Single-shard result → stream directly (no combine)" << std::endl;
                            if (combined.received_matrix_data.empty()) {
                                std::cerr << "ERROR: No shard data for single-shard result: " << matrix_name << std::endl;
                            } else {
                                full = shard_bytes_to_result(combined.received_matrix_data.front());
                            }
                        }
                        else
                        {
                            std::cout << "DEBUG: Combining shards..." << std::endl;
                            full = is_system2
                                ? combine_matrix_shards_grid_2d(combined)
                                : combine_matrix_shards_2d(combined);

                            //matrix_backend_llama.combine_matrix_shards_2d_ggml(combined);
                            //matrix_backend_llama.combine_matrix_shards_grid_2d_ggml(combined);

                        }

	                        if (full.data)
	                        {
	                            const bool sent = send_combined_bin_to_python(
	                                matrix_name,
	                                full,
	                                combined.output_dtype_tag
	                            );
	                            if (!sent) {
	                                std::cerr << "ERROR: Failed to stream combined PT for "
	                                          << matrix_name << std::endl;
	                            }
	                            send_ack("ACK_combined_matrix_saved");
	                        }
                        else
                        {
                            std::cerr << "ERROR: Combine failed for "
                                    << matrix_name << std::endl;
                        }

                        // Remove completed matrix from tracking
                        combined_matrix_shards_list.erase(
                            std::remove_if(
                                combined_matrix_shards_list.begin(),
                                combined_matrix_shards_list.end(),
                                [&](const combined_matrix_shards& c)
                                {
                                    auto [n, __] =
                                        get_matrix_name_and_shard_number(c.file_name);
                                    return n == matrix_name;
                                }),
                            combined_matrix_shards_list.end()
                        );
                    }

                    return true;
                }
            }

            // ============================================================
            // FIRST SHARD → CREATE NEW TRACKING ENTRY
            // ============================================================
            combined_matrix_shards combined;
            combined.file_name = matrix_name;

            // ---- Parse send_back encoding
            int abs_val = std::abs(total_shards);
            int join_dim = 0;
            int shards_needed = abs_val;

            if (abs_val >= 10)
            {
                join_dim = abs_val / 10;
                shards_needed = abs_val % 10;
            }

            combined.join_dim = join_dim;
            combined.number_of_shards_needed = shards_needed;
            combined.total_shards_reserved = 1;
            combined.is_system2 = (total_shards < 0);
            combined.output_dtype_tag = merge_output_dtype_tag(combined.output_dtype_tag, output_dtype_tag);

            if (shards_needed == 1)
            {
                std::cout << "Single-shard result → stream directly (no combine/tracking)" << std::endl;

                auto shard_bytes_to_result = [](const std::vector<uint8_t>& bytes) -> MatrixResult {
                    MatrixResult out;
                    if (bytes.size() < static_cast<size_t>(5 * sizeof(int))) {
                        return out;
                    }
                    int dtype_tag = 0;
                    int dims[4] = {0, 0, 0, 0};
                    std::memcpy(&dtype_tag, bytes.data(), sizeof(int));
                    std::memcpy(&dims[0], bytes.data() + sizeof(int), 4 * sizeof(int));
                    if (dtype_tag != -1) {
                        throw std::runtime_error("single-shard fast-path expects float32 v2 shard (dtype_tag=-1)");
                    }

                    const size_t numel =
                        static_cast<size_t>(dims[0]) * static_cast<size_t>(dims[1]) *
                        static_cast<size_t>(dims[2]) * static_cast<size_t>(dims[3]);
                    const size_t need_bytes = static_cast<size_t>(5 * sizeof(int)) + numel * sizeof(float);
                    if (bytes.size() < need_bytes) {
                        throw std::runtime_error("single-shard fast-path payload truncated");
                    }

                    out.dims[0] = dims[0];
                    out.dims[1] = dims[1];
                    out.dims[2] = dims[2];
                    out.dims[3] = dims[3];
                    out.data = std::make_unique<float[]>(numel);
                    std::memcpy(out.data.get(), bytes.data() + 5 * sizeof(int), numel * sizeof(float));
                    return out;
                };

                MatrixResult full = shard_bytes_to_result(shard_bytes);
                if (full.data)
                {
                    const bool sent = send_combined_bin_to_python(
                        matrix_name,
                        full,
                        combined.output_dtype_tag
                    );
                    if (!sent) {
                        std::cerr << "ERROR: Failed to stream combined PT for " << matrix_name << std::endl;
                    }
                    send_ack("ACK_combined_matrix_saved");
                }
                else
                {
                    std::cerr << "ERROR: Single-shard stream failed for " << matrix_name << std::endl;
                }

                return true;
            }

            combined.shard_numbers.push_back(shard_num);
            combined.received_matrix_data.push_back(std::move(shard_bytes));
            combined.dims_list.push_back({1, 1, shard_rows, shard_cols});
            combined.received_shard_numbers.insert(shard_num);

            combined_matrix_shards_list.push_back(std::move(combined));

            std::cout << "Started tracking matrix: "
                    << matrix_name
                    << " | shards=" << shards_needed
                    << " | join_dim=" << join_dim
                    << " | system=" << (total_shards < 0 ? "2" : "1")
                    << std::endl;

            return true;
#endif
        }

	    MatrixResult combine_matrix_shards_2d(combined_matrix_shards& combined)
	    {
	        MatrixResult result;
            if (combined.shards.empty()) {
                return result;
            }

            const int expected = combined.number_of_shards_needed;
            if (expected <= 0) {
                throw std::runtime_error("combine_matrix_shards_2d called without known shard count");
            }
            if (static_cast<int>(combined.shards.size()) != expected) {
                throw std::runtime_error("combine_matrix_shards_2d missing shards");
            }

            std::vector<int> order;
            order.reserve((size_t)expected);
            if (combined.have_index_range && (combined.max_shard_index - combined.min_shard_index + 1 == expected)) {
                for (int i = combined.min_shard_index; i <= combined.max_shard_index; ++i) {
                    order.push_back(i);
                }
            } else {
                for (const auto& kv : combined.shards) {
                    order.push_back(kv.first);
                }
                std::sort(order.begin(), order.end());
            }

            const int join_dim = combined.join_dim;
            if (join_dim != 0 && join_dim != 1) {
                throw std::runtime_error("combine_matrix_shards_2d only supports join_dim 0 or 1");
            }

            int total_rows = 0;
            int total_cols = 0;
            int fixed_rows = -1;
            int fixed_cols = -1;

	            if (join_dim == 0) {
	                for (int idx : order) {
	                    const auto it = combined.shards.find(idx);
	                    if (it == combined.shards.end()) {
	                        throw std::runtime_error("Missing shard index during combine");
	                    }
	                    const auto& s = it->second;
	                    if (s.batch != 1 || s.depth != 1) {
	                        throw std::runtime_error("combine_matrix_shards_2d expects 2D shards (batch=1, depth=1)");
	                    }
	                    if (fixed_cols < 0) fixed_cols = s.cols;
	                    if (s.cols != fixed_cols) {
	                        throw std::runtime_error("Shard col mismatch for join_dim=0");
	                    }
	                    total_rows += s.rows;
	                }
                total_cols = fixed_cols;
	            } else {
	                for (int idx : order) {
	                    const auto it = combined.shards.find(idx);
	                    if (it == combined.shards.end()) {
	                        throw std::runtime_error("Missing shard index during combine");
	                    }
	                    const auto& s = it->second;
	                    if (s.batch != 1 || s.depth != 1) {
	                        throw std::runtime_error("combine_matrix_shards_2d expects 2D shards (batch=1, depth=1)");
	                    }
	                    if (fixed_rows < 0) fixed_rows = s.rows;
	                    if (s.rows != fixed_rows) {
	                        throw std::runtime_error("Shard row mismatch for join_dim=1");
	                    }
	                    total_cols += s.cols;
	                }
                total_rows = fixed_rows;
            }

            if (total_rows <= 0 || total_cols <= 0) {
                throw std::runtime_error("Invalid combined shape");
            }

            result.dims[0] = 1;
            result.dims[1] = 1;
            result.dims[2] = total_rows;
            result.dims[3] = total_cols;
            const size_t total = static_cast<size_t>(total_rows) * static_cast<size_t>(total_cols);
            result.data = std::make_unique<float[]>(total);

	            if (join_dim == 0) {
	                int row_off = 0;
	                for (int idx : order) {
	                    auto it = combined.shards.find(idx);
	                    auto& s = it->second;
	                    if (s.batch != 1 || s.depth != 1) {
	                        throw std::runtime_error("combine_matrix_shards_2d expects 2D shards (batch=1, depth=1)");
	                    }
	                    const size_t shard_elems = static_cast<size_t>(s.rows) * static_cast<size_t>(s.cols);
	                    if (!s.data || shard_elems == 0) {
	                        throw std::runtime_error("Missing shard payload during combine");
	                    }
                    std::memcpy(
                        result.data.get() + static_cast<size_t>(row_off) * static_cast<size_t>(total_cols),
                        s.data.get(),
                        shard_elems * sizeof(float)
                    );
                    row_off += s.rows;
                    s.data.reset();
                }
	            } else {
	                int col_off = 0;
	                for (int idx : order) {
	                    auto it = combined.shards.find(idx);
	                    auto& s = it->second;
	                    if (s.batch != 1 || s.depth != 1) {
	                        throw std::runtime_error("combine_matrix_shards_2d expects 2D shards (batch=1, depth=1)");
	                    }
	                    if (!s.data) {
	                        throw std::runtime_error("Missing shard payload during combine");
	                    }
                    for (int r = 0; r < total_rows; ++r) {
                        float* dst = result.data.get()
                            + static_cast<size_t>(r) * static_cast<size_t>(total_cols)
                            + static_cast<size_t>(col_off);
                        const float* src = s.data.get()
                            + static_cast<size_t>(r) * static_cast<size_t>(s.cols);
                        std::memcpy(dst, src, static_cast<size_t>(s.cols) * sizeof(float));
                    }
                    col_off += s.cols;
                    s.data.reset();
                }
            }

            return result;

#if 0
            if (combined.received_matrix_data.empty()) {
                return result;
            }

            // ============================================================
            // STEP 1: SORT SHARDS BY SHARD NUMBER
            // ============================================================
            struct ShardEntry {
                int shard_num;
                torch::Tensor tensor;
            };

            std::vector<ShardEntry> shards;

            auto shard_num_it = combined.shard_numbers.begin();
            auto data_it      = combined.received_matrix_data.begin();

	            for (; shard_num_it != combined.shard_numbers.end() &&
	                data_it      != combined.received_matrix_data.end();
	                ++shard_num_it, ++data_it)
	            {
	                const std::vector<uint8_t>& bytes = *data_it;
	                if (bytes.size() < static_cast<size_t>(5 * sizeof(int))) {
	                    throw std::runtime_error("Shard payload too small to contain v2 header");
	                }

	                int dtype_tag = 0;
	                int dims[4] = {0, 0, 0, 0};
	                std::memcpy(&dtype_tag, bytes.data(), sizeof(int));
	                std::memcpy(&dims[0], bytes.data() + sizeof(int), 4 * sizeof(int));

	                if (dtype_tag != -1) {
	                    throw std::runtime_error("combine_matrix_shards_2d expects float32 v2 shards (dtype_tag=-1)");
	                }

	                const int batch = dims[0];
	                const int depth = dims[1];
	                const int rows  = dims[2];
	                const int cols  = dims[3];

	                const size_t numel = static_cast<size_t>(batch) * depth * rows * cols;
	                const size_t need_bytes = static_cast<size_t>(5 * sizeof(int)) + numel * sizeof(float);
	                if (bytes.size() < need_bytes) {
	                    throw std::runtime_error("Shard payload truncated (not enough float32 data)");
	                }

	                const float* raw = reinterpret_cast<const float*>(bytes.data() + 5 * sizeof(int));
	                const auto options =
	                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

	                torch::Tensor t;
	                if (batch > 1 && depth > 1) {
	                    t = torch::from_blob((void*)raw, {batch, depth, rows, cols}, options).clone();
	                } else if (batch > 1) {
	                    t = torch::from_blob((void*)raw, {batch, rows, cols}, options).clone();
	                } else {
	                    t = torch::from_blob((void*)raw, {rows, cols}, options).clone();
	                }

	                shards.push_back({*shard_num_it, std::move(t)});
	            }

            std::sort(shards.begin(), shards.end(),
                    [](const ShardEntry& a, const ShardEntry& b) {
                        return a.shard_num < b.shard_num;
                    });

            // ============================================================
            // STEP 2: VALIDATE SHAPES
            // ============================================================
            const auto& ref = shards[0].tensor.sizes();
            int64_t batch = (ref.size() == 4) ? ref[0] : 1;
            int64_t depth = (ref.size() == 4) ? ref[1] : 1;

            for (const auto& s : shards) {
                const auto& sz = s.tensor.sizes();

                if (sz.size() != ref.size()) {
                    throw std::runtime_error("Shard rank mismatch");
                }

                if (sz.size() == 4) {
                    if (sz[0] != batch || sz[1] != depth) {
                        throw std::runtime_error("Batch/depth mismatch between shards");
                    }
                }
            }

            // ============================================================
            // STEP 3: TORCH CONCAT
            // ============================================================
            std::vector<torch::Tensor> tensors;
            for (auto& s : shards) {
                tensors.push_back(s.tensor);
            }

            int torch_join_dim;
            if (ref.size() == 2) {
                torch_join_dim = combined.join_dim;
            } else if (ref.size() == 3) {
                torch_join_dim = combined.join_dim + 1; // skip batch
            } else {
                torch_join_dim = combined.join_dim + 2; // skip batch + depth
            }

            torch::Tensor combined_tensor = torch::cat(tensors, torch_join_dim);

            // ============================================================
            // STEP 4: EXPORT BACK TO MatrixResult
            // ============================================================
            auto sizes = combined_tensor.sizes();

            result.dims[0] = (sizes.size() == 4) ? sizes[0] : 1;
            result.dims[1] = (sizes.size() == 4) ? sizes[1] : 1;
            result.dims[2] = sizes[sizes.size() - 2];
            result.dims[3] = sizes[sizes.size() - 1];

            size_t total = combined_tensor.numel();
            result.data = std::make_unique<float[]>(total);

            std::memcpy(
                result.data.get(),
                combined_tensor.contiguous().data_ptr<float>(),
                total * sizeof(float)
            );

            std::cout << "Torch combine complete → shape ("
                    << result.dims[2] << " x " << result.dims[3] << ")\n";

	            return result;
#endif
	    }

        // FlashAttention outputs can be 4D (batch/depth/rows/cols). For combine/send-back,
        // we flatten the leading dims into a 2D matrix of shape:
        //   rows2d = batch * depth * rows
        //   cols2d = cols
        MatrixResult combine_flash_attn_shards(combined_matrix_shards& combined)
        {
            MatrixResult result;

            if (combined.shards.empty()) {
                return result;
            }

            const int expected = combined.number_of_shards_needed;
            if (expected <= 0) {
                throw std::runtime_error("combine_flash_attn_shards called without known shard count");
            }
            if (static_cast<int>(combined.shards.size()) != expected) {
                throw std::runtime_error("combine_flash_attn_shards missing shards");
            }

            std::vector<int> order;
            order.reserve((size_t)expected);

            if (combined.have_index_range &&
                (combined.max_shard_index - combined.min_shard_index + 1 == expected)) {
                for (int i = combined.min_shard_index; i <= combined.max_shard_index; ++i) {
                    order.push_back(i);
                }
            } else {
                for (const auto& kv : combined.shards) {
                    order.push_back(kv.first);
                }
                std::sort(order.begin(), order.end());
            }

            const int join_dim = combined.join_dim;
            if (join_dim != 0 && join_dim != 1) {
                throw std::runtime_error("combine_flash_attn_shards only supports join_dim 0 or 1");
            }

            int64_t total_rows = 0;
            int total_cols = 0;
            int64_t fixed_rows = -1;
            int fixed_cols = -1;

            auto rows2d = [](const combined_matrix_shards::ShardView& s) -> int64_t {
                return (int64_t)s.batch * (int64_t)s.depth * (int64_t)s.rows;
            };

            if (join_dim == 0) {
                for (int idx : order) {
                    const auto it = combined.shards.find(idx);
                    if (it == combined.shards.end()) {
                        throw std::runtime_error("Missing shard index during flash combine");
                    }

                    const auto& s = it->second;
                    const int64_t r2 = rows2d(s);
                    if (r2 <= 0) {
                        throw std::runtime_error("Invalid shard rows during flash combine");
                    }

                    if (fixed_cols < 0) fixed_cols = s.cols;
                    if (s.cols != fixed_cols) {
                        throw std::runtime_error("Shard col mismatch for flash join_dim=0");
                    }

                    total_rows += r2;
                }
                total_cols = fixed_cols;
            } else {
                for (int idx : order) {
                    const auto it = combined.shards.find(idx);
                    if (it == combined.shards.end()) {
                        throw std::runtime_error("Missing shard index during flash combine");
                    }

                    const auto& s = it->second;
                    const int64_t r2 = rows2d(s);
                    if (r2 <= 0) {
                        throw std::runtime_error("Invalid shard rows during flash combine");
                    }

                    if (fixed_rows < 0) fixed_rows = r2;
                    if (r2 != fixed_rows) {
                        throw std::runtime_error("Shard row mismatch for flash join_dim=1");
                    }

                    total_cols += s.cols;
                }
                total_rows = fixed_rows;
            }

            if (total_rows <= 0 || total_cols <= 0) {
                throw std::runtime_error("Invalid flash combined shape");
            }
            if (total_rows > std::numeric_limits<int>::max()) {
                throw std::runtime_error("Flash combined rows exceed int range");
            }

            result.dims[0] = 1;
            result.dims[1] = 1;
            result.dims[2] = (int)total_rows;
            result.dims[3] = total_cols;

            const size_t total =
                static_cast<size_t>(total_rows) * static_cast<size_t>(total_cols);
            result.data = std::make_unique<float[]>(total);

            if (join_dim == 0) {
                int64_t row_off = 0;
                for (int idx : order) {
                    auto it = combined.shards.find(idx);
                    auto& s = it->second;

                    const int64_t r2 = rows2d(s);
                    const size_t shard_elems =
                        static_cast<size_t>(r2) * static_cast<size_t>(s.cols);

                    if (!s.data || shard_elems == 0) {
                        throw std::runtime_error("Missing shard payload during flash combine");
                    }

                    std::memcpy(
                        result.data.get()
                            + static_cast<size_t>(row_off) * static_cast<size_t>(total_cols),
                        s.data.get(),
                        shard_elems * sizeof(float)
                    );

                    row_off += r2;
                }
            } else {
                int col_off = 0;
                for (int idx : order) {
                    auto it = combined.shards.find(idx);
                    auto& s = it->second;

                    const int64_t r2 = rows2d(s);
                    if (!s.data) {
                        throw std::runtime_error("Missing shard payload during flash combine");
                    }

                    for (int64_t r = 0; r < r2; ++r) {
                        std::memcpy(
                            result.data.get()
                                + static_cast<size_t>(r) * static_cast<size_t>(total_cols)
                                + (size_t)col_off,
                            s.data.get()
                                + static_cast<size_t>(r) * static_cast<size_t>(s.cols),
                            static_cast<size_t>(s.cols) * sizeof(float)
                        );
                    }

                    col_off += s.cols;
                }
            }

            return result;
        }

        // System-2 grid variant for FlashAttention/4D payloads.
        // We flatten (batch*depth*rows) to a 2D height before tiling.
        MatrixResult combine_flash_attn_shards_grid_2d(combined_matrix_shards& combined)
        {
            MatrixResult result;

            if (combined.shards.empty()) {
                return result;
            }

            const int expected = combined.number_of_shards_needed;
            if (expected <= 0) {
                throw std::runtime_error(
                    "combine_flash_attn_shards_grid_2d called without known shard count"
                );
            }
            if (static_cast<int>(combined.shards.size()) != expected) {
                throw std::runtime_error(
                    "combine_flash_attn_shards_grid_2d missing shards"
                );
            }

            // System-2 dispatch is a 2 x N grid.
            constexpr int row_parts = 2;
            if (expected % row_parts != 0) {
                throw std::runtime_error(
                    "System-2 grid expects an even number of shards"
                );
            }
            const int col_parts = expected / row_parts;

            int min_i = 0;
            int max_i = -1;
            if (combined.have_index_range) {
                min_i = combined.min_shard_index;
                max_i = combined.max_shard_index;
            } else {
                min_i = std::numeric_limits<int>::max();
                max_i = std::numeric_limits<int>::min();
                for (const auto& kv : combined.shards) {
                    min_i = std::min(min_i, kv.first);
                    max_i = std::max(max_i, kv.first);
                }
            }

            if (max_i - min_i + 1 != expected) {
                throw std::runtime_error(
                    "System-2 combine expects contiguous shard indices; got gaps"
                );
            }

            std::vector<combined_matrix_shards::ShardView*> grid(
                (size_t)expected,
                nullptr
            );
            for (int i = min_i; i <= max_i; ++i) {
                auto it = combined.shards.find(i);
                if (it == combined.shards.end()) {
                    throw std::runtime_error(
                        "Missing System-2 shard index during combine"
                    );
                }
                grid[(size_t)(i - min_i)] = &it->second;
            }

            auto rows2d = [](const combined_matrix_shards::ShardView& s) -> int64_t {
                return (int64_t)s.batch * (int64_t)s.depth * (int64_t)s.rows;
            };

            std::vector<int> row_heights((size_t)row_parts, 0);
            std::vector<int> col_widths((size_t)col_parts, 0);

            for (int r = 0; r < row_parts; ++r) {
                auto* t0 = grid[(size_t)(r * col_parts)];
                if (!t0) {
                    throw std::runtime_error("Missing System-2 shard block");
                }

                const int64_t h = rows2d(*t0);
                if (h <= 0 || h > std::numeric_limits<int>::max()) {
                    throw std::runtime_error(
                        "Invalid System-2 shard height during flash combine"
                    );
                }

                row_heights[(size_t)r] = (int)h;

                for (int c = 0; c < col_parts; ++c) {
                    auto* blk = grid[(size_t)(r * col_parts + c)];
                    if (!blk) {
                        throw std::runtime_error("Missing System-2 shard block");
                    }

                    const int64_t bh = rows2d(*blk);
                    if (bh != row_heights[(size_t)r]) {
                        throw std::runtime_error(
                            "System-2 row-block height mismatch within same row"
                        );
                    }
                }
            }

            for (int c = 0; c < col_parts; ++c) {
                auto* t0 = grid[(size_t)c];
                if (!t0) {
                    throw std::runtime_error("Missing System-2 shard block");
                }

                col_widths[(size_t)c] = t0->cols;

                for (int r = 0; r < row_parts; ++r) {
                    auto* blk = grid[(size_t)(r * col_parts + c)];
                    if (!blk) {
                        throw std::runtime_error("Missing System-2 shard block");
                    }

                    if (blk->cols != col_widths[(size_t)c]) {
                        throw std::runtime_error(
                            "System-2 col-block width mismatch within same column"
                        );
                    }
                }
            }

            int total_rows = 0;
            for (int r = 0; r < row_parts; ++r) {
                total_rows += row_heights[(size_t)r];
            }

            int total_cols = 0;
            for (int c = 0; c < col_parts; ++c) {
                total_cols += col_widths[(size_t)c];
            }

            if (total_rows <= 0 || total_cols <= 0) {
                throw std::runtime_error("Invalid System-2 combined shape");
            }

            result.dims[0] = 1;
            result.dims[1] = 1;
            result.dims[2] = total_rows;
            result.dims[3] = total_cols;

            const size_t total =
                static_cast<size_t>(total_rows) * static_cast<size_t>(total_cols);
            result.data = std::make_unique<float[]>(total);

            int row_off = 0;
            for (int r = 0; r < row_parts; ++r) {
                int col_off = 0;
                for (int c = 0; c < col_parts; ++c) {
                    auto* blk = grid[(size_t)(r * col_parts + c)];
                    const int64_t h = rows2d(*blk);
                    const int w = blk->cols;

                    if (!blk->data) {
                        throw std::runtime_error(
                            "Missing shard payload during System-2 flash combine"
                        );
                    }

                    for (int rr = 0; rr < (int)h; ++rr) {
                        std::memcpy(
                            result.data.get()
                                + (size_t)(row_off + rr) * (size_t)total_cols
                                + (size_t)col_off,
                            blk->data.get()
                                + (size_t)rr * (size_t)w,
                            (size_t)w * sizeof(float)
                        );
                    }

                    col_off += w;
                }
                row_off += row_heights[(size_t)r];
            }

            return result;
        }

        MatrixResult combine_matrix_shards_grid_2d(combined_matrix_shards& combined)
        {
            MatrixResult result;

            if (combined.shards.empty()) {
                return result;
            }

            const int expected = combined.number_of_shards_needed;
            if (expected <= 0) {
                throw std::runtime_error("combine_matrix_shards_grid_2d called without known shard count");
            }
            if (static_cast<int>(combined.shards.size()) != expected) {
                throw std::runtime_error("combine_matrix_shards_grid_2d missing shards");
            }

            // System-2 dispatch is a 2 x N grid.
            constexpr int row_parts = 2;
            if (expected % row_parts != 0) {
                throw std::runtime_error("System-2 grid expects an even number of shards");
            }
            const int col_parts = expected / row_parts;

            int min_i = 0;
            int max_i = -1;
            if (combined.have_index_range) {
                min_i = combined.min_shard_index;
                max_i = combined.max_shard_index;
            } else {
                min_i = std::numeric_limits<int>::max();
                max_i = std::numeric_limits<int>::min();
                for (const auto& kv : combined.shards) {
                    min_i = std::min(min_i, kv.first);
                    max_i = std::max(max_i, kv.first);
                }
            }

            if (max_i - min_i + 1 != expected) {
                throw std::runtime_error("System-2 combine expects contiguous shard indices; got gaps");
            }

            std::vector<combined_matrix_shards::ShardView*> grid((size_t)expected, nullptr);
            for (int i = min_i; i <= max_i; ++i) {
                auto it = combined.shards.find(i);
                if (it == combined.shards.end()) {
                    throw std::runtime_error("Missing System-2 shard index during combine");
                }
                grid[(size_t)(i - min_i)] = &it->second;
            }

            // Row heights (2 parts) and column widths (N parts) may be uneven due to remainder.
            std::vector<int> row_heights((size_t)row_parts, 0);
            std::vector<int> col_widths((size_t)col_parts, 0);

	            for (int r = 0; r < row_parts; ++r) {
	                auto* t0 = grid[(size_t)(r * col_parts)];
	                if (!t0) throw std::runtime_error("Missing System-2 shard block");
	                if (t0->batch != 1 || t0->depth != 1) {
	                    throw std::runtime_error("System-2 combine expects 2D shards (batch=1, depth=1)");
	                }
	                row_heights[(size_t)r] = t0->rows;
	                for (int c = 0; c < col_parts; ++c) {
	                    auto* blk = grid[(size_t)(r * col_parts + c)];
	                    if (!blk) throw std::runtime_error("Missing System-2 shard block");
	                    if (blk->batch != 1 || blk->depth != 1) {
	                        throw std::runtime_error("System-2 combine expects 2D shards (batch=1, depth=1)");
	                    }
	                    if (blk->rows != row_heights[(size_t)r]) {
	                        throw std::runtime_error("System-2 row-block height mismatch within same row");
	                    }
	                }
	            }

	            for (int c = 0; c < col_parts; ++c) {
	                auto* t0 = grid[(size_t)c];
	                if (!t0) throw std::runtime_error("Missing System-2 shard block");
	                if (t0->batch != 1 || t0->depth != 1) {
	                    throw std::runtime_error("System-2 combine expects 2D shards (batch=1, depth=1)");
	                }
	                col_widths[(size_t)c] = t0->cols;
	                for (int r = 0; r < row_parts; ++r) {
	                    auto* blk = grid[(size_t)(r * col_parts + c)];
	                    if (!blk) throw std::runtime_error("Missing System-2 shard block");
	                    if (blk->batch != 1 || blk->depth != 1) {
	                        throw std::runtime_error("System-2 combine expects 2D shards (batch=1, depth=1)");
	                    }
	                    if (blk->cols != col_widths[(size_t)c]) {
	                        throw std::runtime_error("System-2 col-block width mismatch within same column");
	                    }
	                }
	            }

            int total_rows = 0;
            for (int r = 0; r < row_parts; ++r) total_rows += row_heights[(size_t)r];
            int total_cols = 0;
            for (int c = 0; c < col_parts; ++c) total_cols += col_widths[(size_t)c];

            if (total_rows <= 0 || total_cols <= 0) {
                throw std::runtime_error("Invalid System-2 combined shape");
            }

            result.dims[0] = 1;
            result.dims[1] = 1;
            result.dims[2] = total_rows;
            result.dims[3] = total_cols;
            const size_t total = static_cast<size_t>(total_rows) * static_cast<size_t>(total_cols);
            result.data = std::make_unique<float[]>(total);

            int row_off = 0;
            for (int r = 0; r < row_parts; ++r) {
                int col_off = 0;
                for (int c = 0; c < col_parts; ++c) {
                    auto* blk = grid[(size_t)(r * col_parts + c)];
                    if (!blk || !blk->data) {
                        throw std::runtime_error("Missing System-2 shard payload");
                    }
                    for (int rr = 0; rr < row_heights[(size_t)r]; ++rr) {
                        float* dst = result.data.get()
                            + static_cast<size_t>(row_off + rr) * static_cast<size_t>(total_cols)
                            + static_cast<size_t>(col_off);
                        const float* src = blk->data.get()
                            + static_cast<size_t>(rr) * static_cast<size_t>(blk->cols);
                        std::memcpy(dst, src, static_cast<size_t>(blk->cols) * sizeof(float));
                    }
                    col_off += col_widths[(size_t)c];
                    blk->data.reset();
                }
                row_off += row_heights[(size_t)r];
            }

            return result;

#if 0
            if (combined.received_matrix_data.empty()) {
                return result;
            }

            // ============================================================
            // STEP 1: LOAD SHARDS AS TORCH TENSORS (SORTED)
            // ============================================================
            struct Shard {
                int index;
                torch::Tensor t;
                int rows;
                int cols;
            };

            std::vector<Shard> shards;

            auto n_it = combined.shard_numbers.begin();
            auto d_it = combined.received_matrix_data.begin();
            auto s_it = combined.dims_list.begin();

            for (; n_it != combined.shard_numbers.end(); ++n_it, ++d_it, ++s_it) {

                const int shard_index = *n_it;
                const auto& dims = *s_it;   // [batch, depth, rows, cols]

                const int rows = dims[2];
                const int cols = dims[3];

                const uint8_t* p = d_it->data();
                p += sizeof(int) + 4 * sizeof(int); // skip ndim + dims

                const float* raw = reinterpret_cast<const float*>(p);

                auto t = torch::from_blob(
                    (void*)raw,
                    {rows, cols},
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
                ).clone(); // own memory

                shards.push_back({shard_index, t, rows, cols});
            }

            std::sort(shards.begin(), shards.end(),
                    [](const Shard& a, const Shard& b) {
                        return a.index < b.index;
                    });

            // ============================================================
            // STEP 2: SYSTEM-2 GRID SHAPE (2 x N)
            // ============================================================
            const int total_shards = (int) shards.size();
            if (total_shards == 0) {
                return result;
            }

            // System-2 dispatch in `cluster_matrix_v1.py` is a 2 x base_slots grid:
            // - first half of op slots use A_shard_0
            // - second half use A_shard_1
            // Each shard result is a unique (row, col) block that must be tiled into the
            // final matrix.
            constexpr int row_parts = 2;
            if (total_shards % row_parts != 0) {
                throw std::runtime_error("System-2 grid expects an even number of shards");
            }
            const int col_parts = total_shards / row_parts;

            // Normalize shard indices so we can validate a contiguous range and map
            // index -> (row, col).
            const int min_index = shards.front().index;
            const int max_index = shards.back().index;
            if (max_index - min_index + 1 != total_shards) {
                throw std::runtime_error(
                    "System-2 combine expects contiguous shard indices; got gaps (check shard_index_override)"
                );
            }

            std::vector<bool> seen((size_t) total_shards, false);
            std::vector<torch::Tensor> grid((size_t) total_shards);

            for (const auto& s : shards) {
                const int norm = s.index - min_index;
                if (norm < 0 || norm >= total_shards) {
                    throw std::runtime_error("System-2 shard index out of expected range");
                }
                if (seen[(size_t) norm]) {
                    throw std::runtime_error("Duplicate shard index received during System-2 combine");
                }
                seen[(size_t) norm] = true;
                grid[(size_t) norm] = s.t;
            }

            // Row heights (2 parts) and column widths (N parts) may be uneven due to remainder.
            std::vector<int64_t> row_heights((size_t) row_parts);
            std::vector<int64_t> col_widths((size_t) col_parts);

            for (int r = 0; r < row_parts; ++r) {
                const auto& t0 = grid[(size_t) (r * col_parts)];
                if (!t0.defined()) {
                    throw std::runtime_error("Missing System-2 shard block");
                }
                row_heights[(size_t) r] = t0.size(0);
                for (int c = 0; c < col_parts; ++c) {
                    const auto& t = grid[(size_t) (r * col_parts + c)];
                    if (!t.defined()) {
                        throw std::runtime_error("Missing System-2 shard block");
                    }
                    if (t.size(0) != row_heights[(size_t) r]) {
                        throw std::runtime_error("System-2 row-block height mismatch within same row");
                    }
                }
            }

            for (int c = 0; c < col_parts; ++c) {
                const auto& t0 = grid[(size_t) c];
                if (!t0.defined()) {
                    throw std::runtime_error("Missing System-2 shard block");
                }
                col_widths[(size_t) c] = t0.size(1);
                for (int r = 0; r < row_parts; ++r) {
                    const auto& t = grid[(size_t) (r * col_parts + c)];
                    if (!t.defined()) {
                        throw std::runtime_error("Missing System-2 shard block");
                    }
                    if (t.size(1) != col_widths[(size_t) c]) {
                        throw std::runtime_error("System-2 col-block width mismatch within same column");
                    }
                }
            }

            int64_t total_rows = 0;
            for (int r = 0; r < row_parts; ++r) {
                total_rows += row_heights[(size_t) r];
            }

            int64_t total_cols = 0;
            for (int c = 0; c < col_parts; ++c) {
                total_cols += col_widths[(size_t) c];
            }

            auto out = torch::zeros(
                {total_rows, total_cols},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
            );

            int64_t row_off = 0;
            for (int r = 0; r < row_parts; ++r) {
                int64_t col_off = 0;
                for (int c = 0; c < col_parts; ++c) {
                    const auto& block = grid[(size_t) (r * col_parts + c)];
                    out.narrow(0, row_off, row_heights[(size_t) r])
                        .narrow(1, col_off, col_widths[(size_t) c])
                        .copy_(block);
                    col_off += col_widths[(size_t) c];
                }
                row_off += row_heights[(size_t) r];
            }

            result.dims[0] = 1;
            result.dims[1] = 1;
            result.dims[2] = (int) total_rows;
            result.dims[3] = (int) total_cols;

            result.data = std::make_unique<float[]>((size_t) total_rows * (size_t) total_cols);
            std::memcpy(
                result.data.get(),
                out.contiguous().data_ptr<float>(),
                sizeof(float) * (size_t) total_rows * (size_t) total_cols
            );

            std::cout << "System-2 grid combine complete → shape ("
                    << result.dims[2] << " x " << result.dims[3] << ")\n";

            return result;
#endif
        }

        bool matrix_operation(
            const std::string& backend_type,
            const matrix_shard_object& matrixA,
            bool transposeA,
            const matrix_shard_object& matrixB,
            bool transposeB,
            bool use_gpu,
            int gpu_id,
            int send_back,
            const std::string& operation_type,
            int dim,
            int shard_index_override
        )
        {
            bool op_success = false;
            try {
                std::cout << "🚀 UNIFIED MATRIX OPERATION - Backend: " << backend_type << std::endl;

                // Common setup (all backends)
                std::string output_filename = get_matrix_output_filename(matrixA.base_file_name, matrixB.base_file_name);
                if (shard_index_override >= 0)
                {
                    // Force shard naming from caller
                    // Strip existing ".bin" and any trailing "_shard_<n>" before appending
                    size_t dot_pos = output_filename.rfind(".bin");
                    std::string base_name = (dot_pos != std::string::npos) ? output_filename.substr(0, dot_pos) : output_filename;
                    size_t shard_pos = base_name.rfind("_shard_");
                    if (shard_pos != std::string::npos)
                    {
                        base_name = base_name.substr(0, shard_pos);
                    }
                    output_filename = base_name + "_shard_" + std::to_string(shard_index_override) + ".bin";
                }

                // Determine output dtype tag from in-memory inputs.
                const int output_dtype_tag = merge_output_dtype_tag(matrixA.output_dtype_tag, matrixB.output_dtype_tag);

                // ============================================================
                // BACKEND: LLAMA / GGML / VULKAN
                // ============================================================
	            if (backend_type == "llama")
	            {
	                int rows_A = matrixA.rows_A;
	                int cols_A = matrixA.cols_A;
	                int batchA = matrixA.batchA;
	                int depthA = matrixA.depthA;
	                int rows_B = matrixB.rows_A;
	                int cols_B = matrixB.cols_A;
	                int batchB = matrixB.batchA;
	                int depthB = matrixB.depthA;

                    std::cout << "Matrix A shape:" << std::endl;
                    std::cout << "  rows_A: " << rows_A << std::endl;
                    std::cout << "  cols_A: " << cols_A << std::endl;
                    std::cout << "  batchA: " << batchA << std::endl;
                    std::cout << "  depthA: " << depthA << std::endl;

                    std::cout << "Matrix B shape:" << std::endl;
                    std::cout << "  rows_B: " << rows_B << std::endl;
                    std::cout << "  cols_B: " << cols_B << std::endl;
                    std::cout << "  batchB: " << batchB << std::endl;
                    std::cout << "  depthB: " << depthB << std::endl;


	                const bool has_a_data = (matrixA.data && !matrixA.data->empty());
	                const bool has_b_data = (matrixB.data && !matrixB.data->empty());

                    // Pre-check shapes to avoid ggml assertion failures for invalid ops.
                    {
                        int check_rows_A = rows_A;
                        int check_cols_A = cols_A;
                        int check_rows_B = rows_B;
                        int check_cols_B = cols_B;
                        if (transposeA) std::swap(check_rows_A, check_cols_A);
                        if (transposeB) std::swap(check_rows_B, check_cols_B);

                        const int dims_a[4] = { check_cols_A, check_rows_A, depthA, batchA };
                        const int dims_b[4] = { check_cols_B, check_rows_B, depthB, batchB };

                        auto invalid_dim = [](const int dims[4]) {
                            for (int i = 0; i < 4; ++i) {
                                if (dims[i] <= 0) return true;
                            }
                            return false;
                        };

                        auto print_dims = [](const char* name, const int dims[4]) {
                            std::cerr << "  " << name << " [cols,rows,depth,batch] = ["
                                      << dims[0] << "," << dims[1] << ","
                                      << dims[2] << "," << dims[3] << "]" << std::endl;
                        };

                        if (invalid_dim(dims_a) || invalid_dim(dims_b)) {
                            std::cerr << "❌ Invalid (non-positive) tensor dimensions for GGML op '"
                                      << operation_type << "' after transpose"
                                      << " (transposeA=" << (transposeA ? "true" : "false")
                                      << ", transposeB=" << (transposeB ? "true" : "false") << ")." << std::endl;
                            print_dims("A", dims_a);
                            print_dims("B", dims_b);
                            return false;
                        }

                        if (operation_type == "mul" || operation_type == "matmul") {
                            const bool can_mul =
                                (dims_a[0] == dims_b[0]) &&
                                (dims_b[2] % dims_a[2] == 0) &&
                                (dims_b[3] % dims_a[3] == 0);
                            if (!can_mul) {
                                std::cerr << "❌ GGML mul_mat shape mismatch after transpose"
                                          << " (transposeA=" << (transposeA ? "true" : "false")
                                          << ", transposeB=" << (transposeB ? "true" : "false") << ")." << std::endl;
                                print_dims("A", dims_a);
                                print_dims("B", dims_b);
                                std::cerr << "  Expected: A.cols == B.cols, "
                                          << "B.depth % A.depth == 0, B.batch % A.batch == 0." << std::endl;
                                return false;
                            }
                        } else if (operation_type == "add" || operation_type == "sub") {
                            const bool can_repeat =
                                (dims_a[0] % dims_b[0] == 0) &&
                                (dims_a[1] % dims_b[1] == 0) &&
                                (dims_a[2] % dims_b[2] == 0) &&
                                (dims_a[3] % dims_b[3] == 0);
                            if (!can_repeat) {
                                std::cerr << "❌ GGML " << operation_type << " shape mismatch after transpose"
                                          << " (transposeA=" << (transposeA ? "true" : "false")
                                          << ", transposeB=" << (transposeB ? "true" : "false") << ")." << std::endl;
                                print_dims("A", dims_a);
                                print_dims("B", dims_b);
                                std::cerr << "  Expected: A dims must be repeatable by B (A % B == 0 for each dim)." << std::endl;
                                return false;
                            }
                        }
                    }

	                    // Only lock long enough to read the backend vector; release before compute
	                ggml_backend_t backend;
	                {
	                    std::lock_guard<std::mutex> lock(matrix_backend_llama.backends_mutex);
	                    backend =
	                        (use_gpu && gpu_id >= 0 &&
	                        gpu_id < (int)matrix_backend_llama.ggml_backends.size())
	                        ? matrix_backend_llama.ggml_backends[gpu_id]
	                        : matrix_backend_llama.ggml_backends.back();
	                }

	                MatrixResult result;
	                result.dims[0] = result.dims[1] = result.dims[2] = result.dims[3] = 0;

	                // Fast path: use cached GGML tensors in VRAM when requested and possible.
	                const bool can_use_vram =
	                    use_gpu &&
	                    !transposeA &&
	                    !transposeB &&
	                    gpu_id >= 0 &&
	                    gpu_id < (int)matrix_backend_llama.ggml_backends.size();

	                if (can_use_vram) {
	                    auto& vram = get_vram_cache_manager();
	                    if (vram.enabled(gpu_id)) {
	                        ggml_tensor* a_t = vram.get_tensor(gpu_id, matrixA.base_file_name);
	                        if (!a_t && has_a_data) {
	                            vram.cache_tensor_f32_4d(
	                                gpu_id,
	                                matrixA.base_file_name,
	                                matrixA.data->data(),
	                                cols_A,
	                                rows_A,
	                                depthA,
	                                batchA
	                            );
	                            a_t = vram.get_tensor(gpu_id, matrixA.base_file_name);
	                        }

	                        ggml_tensor* b_t = vram.get_tensor(gpu_id, matrixB.base_file_name);
	                        if (!b_t && has_b_data) {
	                            vram.cache_tensor_f32_4d(
	                                gpu_id,
	                                matrixB.base_file_name,
	                                matrixB.data->data(),
	                                cols_B,
	                                rows_B,
	                                depthB,
	                                batchB
	                            );
	                            b_t = vram.get_tensor(gpu_id, matrixB.base_file_name);
	                        }

	                        if (a_t && b_t) {
	                            result = matrix_backend_llama.matrix_op_nd_tensors(a_t, b_t, backend, operation_type);
	                        }
	                    }
	                }

	                    // Fallback: host path (still runs on GPU if requested)
	                if (!result.data) {
	                    if (!has_a_data || !has_b_data) {
	                        throw std::runtime_error(
	                            "Missing matrix data for llama backend inputs (VRAM cache miss)"
	                        );
	                    }
	                    const float* a_ptr = matrixA.data->data();
	                    const float* b_ptr = matrixB.data->data();
	                    std::unique_ptr<float[]> a_tmp;
	                    std::unique_ptr<float[]> b_tmp;

	                    if (transposeA) {
	                        a_tmp = (depthA > 1 || batchA > 1)
	                            ? matrix_backend_llama.transpose_4d(const_cast<float*>(a_ptr), batchA, depthA, rows_A, cols_A)
	                            : matrix_backend_llama.transpose_2d(const_cast<float*>(a_ptr), rows_A, cols_A);
	                        a_ptr = a_tmp.get();
	                        std::swap(rows_A, cols_A);
	                    }

	                    if (transposeB) {
	                        b_tmp = (depthB > 1 || batchB > 1)
	                            ? matrix_backend_llama.transpose_4d(const_cast<float*>(b_ptr), batchB, depthB, rows_B, cols_B)
	                            : matrix_backend_llama.transpose_2d(const_cast<float*>(b_ptr), rows_B, cols_B);
	                        b_ptr = b_tmp.get();
	                        std::swap(rows_B, cols_B);
	                    }

	                    int dims_a[4] = { cols_A, rows_A, depthA, batchA };
	                    int dims_b[4] = { cols_B, rows_B, depthB, batchB };

		                    result = matrix_backend_llama.matrix_op_nd(
		                        const_cast<float*>(a_ptr), dims_a,
		                        const_cast<float*>(b_ptr), dims_b,
		                        backend, operation_type
		                    );

		                        // If GPU compute fails (often due to VRAM pressure), retry on CPU backend.
		                    if (!result.data && use_gpu) {
		                        ggml_backend_t cpu_backend;
		                        {
		                            std::lock_guard<std::mutex> lock(matrix_backend_llama.backends_mutex);
		                            cpu_backend = matrix_backend_llama.ggml_backends.back();
		                        }
		                        result = matrix_backend_llama.matrix_op_nd(
		                            const_cast<float*>(a_ptr), dims_a,
		                            const_cast<float*>(b_ptr), dims_b,
		                            cpu_backend, operation_type
		                        );
		                    }
		                }

		                if (!result.data) {
		                    throw std::runtime_error("GGML operation failed (no result data) for output: " + output_filename);
		                }

                    if (result.dims[0] == 0 && result.dims[1] == 0) {
                        result.dims[0] = 1;
                        result.dims[1] = 1;
                    }

	                    // SEND BACK LOGIC:
	                    // `send_back == 0`  => no combine, just save
	                    // `send_back != 0`  => combine on head (System-1: +, System-2: -)
	                    op_success = false;
                        store_matrix_result_to_shard_list(output_filename, result, output_dtype_tag);
	                    if (send_back != 0)
	                    {
	                        send_back_file(output_filename, output_filename, result, send_back, "llama", output_dtype_tag);
	                        op_success = true;  // assume success for now
	                    }
	                    else
	                    {
                            op_success = true;
	                    }

                }

                // ============================================================
                // BACKEND: PYTORCH / TORCH
                // ============================================================
                else if (backend_type == "torch")
                {
                    // GPU availability check
                    bool torch_gpu_available = false;
                    #ifdef USE_CUDA
                    torch_gpu_available = torch::cuda::is_available();
                    #endif
                    
                    if (use_gpu && !torch_gpu_available) {
                        std::cout << "⚠️  GPU requested but unavailable. Using CPU." << std::endl;
                    }

	                    auto to_torch = [](const matrix_shard_object& obj) -> torch::Tensor {
	                        if (!obj.data) {
	                            throw std::runtime_error("Missing matrix data for torch backend input: " + obj.base_file_name);
	                        }
	                        std::vector<int64_t> sizes;
	                        if (obj.batchA > 1 && obj.depthA > 1) {
	                            sizes = {obj.batchA, obj.depthA, obj.rows_A, obj.cols_A};
	                        } else if (obj.batchA > 1) {
	                            sizes = {obj.batchA, obj.rows_A, obj.cols_A};
	                        } else {
	                            sizes = {obj.rows_A, obj.cols_A};
	                        }
	                        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	                        return torch::from_blob((void*)obj.data->data(), sizes, options).clone();
	                    };

                    torch::Tensor A = to_torch(matrixA);
                    torch::Tensor B = to_torch(matrixB);

                    // Apply transposes (last 2 dims)
                    if (transposeA)
                        A = A.transpose(-2, -1).contiguous();
                    if (transposeB)
                        B = B.transpose(-2, -1).contiguous();

                    // Select device
                    torch::Device device = torch::kCPU;
                    if (use_gpu && torch_gpu_available) {
                        device = torch::Device(torch::kCUDA, gpu_id);
                    }
                    A = A.to(device);
                    B = B.to(device);

                    // Execute
                    torch::Tensor C;
                    if (operation_type == "mul") {
                        C = torch::matmul(A, B);
                    } else if (operation_type == "add") {
                        C = A + B;
                    } else if (operation_type == "sub") {
                        C = A - B;
                    } else {
                        std::cerr << "❌ Unknown op: " << operation_type << std::endl;
                        return false;
                    }

                    C = C.contiguous().to(torch::kCPU);

                    // Convert to MatrixResult
                    MatrixResult result;
                    auto sizes = C.sizes();
                    int ndim = sizes.size();
                    int batch = 1, depth = 1, rows = 1, cols = 1;
                    
                    if (ndim == 2) {
                        rows = sizes[0]; cols = sizes[1];
                    } else if (ndim == 3) {
                        batch = sizes[0]; rows = sizes[1]; cols = sizes[2];
                    } else if (ndim == 4) {
                        batch = sizes[0]; depth = sizes[1]; rows = sizes[2]; cols = sizes[3];
                    } else {
                        std::cerr << "❌ Unsupported rank: " << ndim << std::endl;
                        return false;
                    }

                    result.dims[0] = batch;
                    result.dims[1] = depth;
                    result.dims[2] = rows;
                    result.dims[3] = cols;

                    int64_t total = C.numel();
                    result.data = std::make_unique<float[]>(total);
                    memcpy(result.data.get(), C.data_ptr<float>(), total * sizeof(float));

	                    // SEND BACK LOGIC:
	                    // `send_back == 0`  => no combine, just save
	                    // `send_back != 0`  => combine on head (System-1: +, System-2: -)
	                    op_success = false;
                        store_matrix_result_to_shard_list(output_filename, result, output_dtype_tag);
	                    if (send_back != 0)
	                    {
	                        send_back_file(output_filename, output_filename, result, send_back, "torch", output_dtype_tag);
	                        op_success = true;  // assume success for now
	                    }
	                    else
	                    {
                            op_success = true;
	                    }
                }

                // ============================================================
                // BACKEND: OPENCL
                // ============================================================
                else if (backend_type == "opencl")
                {
                    if (gpu_id < 0 || gpu_id >= (int)openCL_GPU_select_list.size()) {
                        std::cerr << "❌ Invalid OpenCL GPU ID" << std::endl;
                        return false;
                    }

	                    auto to_torch = [](const matrix_shard_object& obj) -> torch::Tensor {
	                        if (!obj.data) {
	                            throw std::runtime_error("Missing matrix data for opencl backend input: " + obj.base_file_name);
	                        }
	                        std::vector<int64_t> sizes;
	                        if (obj.batchA > 1 && obj.depthA > 1) {
	                            sizes = {obj.batchA, obj.depthA, obj.rows_A, obj.cols_A};
	                        } else if (obj.batchA > 1) {
	                            sizes = {obj.batchA, obj.rows_A, obj.cols_A};
	                        } else {
	                            sizes = {obj.rows_A, obj.cols_A};
	                        }
	                        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	                        return torch::from_blob((void*)obj.data->data(), sizes, options).clone();
	                    };

                    torch::Tensor tensorA = to_torch(matrixA);
                    torch::Tensor tensorB = to_torch(matrixB);

                    // Apply transposes
                    if (transposeA)
                        tensorA = tensorA.transpose(-2, -1).contiguous();
                    if (transposeB)
                        tensorB = tensorB.transpose(-2, -1).contiguous();

                    float* A_ptr = tensorA.data_ptr<float>();
                    float* B_ptr = tensorB.data_ptr<float>();
                    int M = tensorA.size(-2);
                    int K = tensorA.size(-1);
                    int N = tensorB.size(-1);

                    if (K != tensorB.size(-2)) {
                        std::cerr << "❌ Dimension mismatch" << std::endl;
                        return false;
                    }

                    // OpenCL execution
                    cl::Device device = openCL_GPU_select_list[gpu_id];
                    cl::Context context(device);
                    cl::CommandQueue queue(context, device);
                    
                    cl::Buffer bufA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                   sizeof(float) * tensorA.numel(), A_ptr);
                    cl::Buffer bufB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                   sizeof(float) * tensorB.numel(), B_ptr);
                    cl::Buffer bufC(context, CL_MEM_WRITE_ONLY, sizeof(float) * M * N);
                    
                    cl::Program program(context, openCL_kernel_matmul);
                    program.build({device});
                    cl::Kernel kernel(program, "matmul");

                    kernel.setArg(0, bufA);
                    kernel.setArg(1, bufB);
                    kernel.setArg(2, bufC);
                    kernel.setArg(3, M);
                    kernel.setArg(4, N);
                    kernel.setArg(5, K);

                    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(M, N), cl::NDRange(16, 16));
                    queue.finish();

                    // Prepare result
                    MatrixResult result;
                    result.dims[0] = 1;
                    result.dims[1] = 1;
                    result.dims[2] = M;
                    result.dims[3] = N;
                    result.data = std::make_unique<float[]>(M * N);
                    queue.enqueueReadBuffer(bufC, CL_TRUE, 0, sizeof(float) * M * N, result.data.get());

	                    // SEND BACK LOGIC:
	                    // `send_back == 0`  => no combine, just save
	                    // `send_back != 0`  => combine on head (System-1: +, System-2: -)
	                    op_success = false;
                        store_matrix_result_to_shard_list(output_filename, result, output_dtype_tag);
	                    if (send_back != 0)
	                    {
	                        send_back_file(output_filename, output_filename, result, send_back, "opencl", output_dtype_tag);
	                        op_success = true;  // assume success for now
	                    }
	                    else
	                    {
                            op_success = true;
	                    }
                }

                else {
                    std::cerr << "❌ Unknown backend: " << backend_type << std::endl;
                    op_success = false;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "❌ Exception: " << e.what() << std::endl;
                op_success = false;
            }

            try { send_ack("ACK_matrixOp_complete"); } catch (...) {}
            return op_success;
        }

        bool transformer_operation(
            const std::string& transformer_op_select,
            const matrix_shard_object& matrixA,
            bool transposeA,
            bool use_gpu,
            int gpu_id,
            int send_back,
            const std::string& operation_type,
            int dim,
            int shard_index_override
        )
        {
            bool op_success = false;

            try {
                std::cout << "🚀 TRANSFORMER OP - Backend: " << transformer_op_select << std::endl;

                // ---------------- Output filename ----------------
                std::string output_filename = matrixA.base_file_name;

                if (shard_index_override >= 0) {
                    size_t dot = output_filename.rfind(".bin");
                    std::string base = (dot != std::string::npos)
                                    ? output_filename.substr(0, dot)
                                    : output_filename;
                    output_filename = base + "_shard_" + std::to_string(shard_index_override) + ".bin";
                }

                const int output_dtype_tag = matrixA.output_dtype_tag;

                // ============================================================
                // BACKEND: LLAMA / GGML
                // ============================================================
                if (transformer_op_select == "llama") {

                    int rows  = matrixA.rows_A;
                    int cols  = matrixA.cols_A;
                    int batch = matrixA.batchA;
                    int depth = matrixA.depthA;

                    const bool has_a_data = (matrixA.data && !matrixA.data->empty());

                    ggml_backend_t backend;
                    {
                        std::lock_guard<std::mutex> lock(matrix_backend_llama.backends_mutex);
                        backend = (use_gpu && gpu_id >= 0 &&
                                gpu_id < (int)matrix_backend_llama.ggml_backends.size())
                                ? matrix_backend_llama.ggml_backends[gpu_id]
                                : matrix_backend_llama.ggml_backends.back();
                    }

                    MatrixResult result;
                    result.dims[0] = result.dims[1] = result.dims[2] = result.dims[3] = 0;

                    // ---------------- VRAM fast path (NO transpose) ----------------
                    if (use_gpu && !transposeA &&
                        gpu_id >= 0 &&
                        gpu_id < (int)matrix_backend_llama.ggml_backends.size())
                    {
                        auto& vram = get_vram_cache_manager();

                        if (vram.enabled(gpu_id)) {
                            ggml_tensor* a_t = vram.get_tensor(gpu_id, matrixA.base_file_name);
                            if (!a_t && has_a_data) {
                                vram.cache_tensor_f32_4d(
                                    gpu_id,
                                    matrixA.base_file_name,
                                    matrixA.data->data(),
                                    cols, rows, depth, batch
                                );
                                a_t = vram.get_tensor(gpu_id, matrixA.base_file_name);
                            }

                            if (a_t) {
                                result = matrix_backend_llama.matrix_op_nd_tensors(
                                    a_t,
                                    nullptr,
                                    backend,
                                    operation_type
                                );
                            }
                        }
                    }

                    // ---------------- Host fallback ----------------
                    if (!result.data) {
                        if (!has_a_data) {
                            throw std::runtime_error(
                                "Missing matrix data for transformer operation (VRAM cache miss)"
                            );
                        }

                        const float* a_ptr = matrixA.data->data();
                        std::unique_ptr<float[]> a_tmp;

                        if (transposeA) {
                            a_tmp = (depth > 1 || batch > 1)
                                ? matrix_backend_llama.transpose_4d(
                                    const_cast<float*>(a_ptr),
                                    batch, depth, rows, cols)
                                : matrix_backend_llama.transpose_2d(
                                    const_cast<float*>(a_ptr),
                                    rows, cols);

                            a_ptr = a_tmp.get();
                            std::swap(rows, cols);
                        }

                        int dims_a[4] = { cols, rows, depth, batch };
                        int dims_b[4] = { 1, 1, 1, 1 }; // dummy

                        result = matrix_backend_llama.matrix_op_nd(
                            const_cast<float*>(a_ptr), dims_a,
                            nullptr, dims_b,
                            backend,
                            operation_type
                        );

                        // GPU fallback → CPU
                        if (!result.data && use_gpu) {
                            ggml_backend_t cpu_backend;
                            {
                                std::lock_guard<std::mutex> lock(matrix_backend_llama.backends_mutex);
                                cpu_backend = matrix_backend_llama.ggml_backends.back();
                            }

                            result = matrix_backend_llama.matrix_op_nd(
                                const_cast<float*>(a_ptr), dims_a,
                                nullptr, dims_b,
                                cpu_backend,
                                operation_type
                            );
                        }
                    }

                    if (!result.data) {
                        throw std::runtime_error("GGML transformer op failed: " + operation_type);
                    }

                    if (result.dims[0] == 0 && result.dims[1] == 0) {
                        result.dims[0] = 1;
                        result.dims[1] = 1;
                    }

                    // ---------------- Store / Send back ----------------
                    store_matrix_result_to_shard_list(output_filename, result, output_dtype_tag);

                    if (send_back != 0) {
                        int abs_send_back = std::abs(send_back);

                        if (abs_send_back == 1) {
                            // Single shard → immediately send to Python
                            const bool sent = send_combined_bin_to_python(
                                output_filename,
                                result,
                                output_dtype_tag
                            );
                            if (!sent) {
                                std::cerr << "ERROR: Failed to stream single-shard PT for "
                                        << output_filename << std::endl;
                            }
                            send_ack("ACK_combined_matrix_saved");
                        } else {
                            // Multi-shard → send_back_file() triggers combine on head
                            send_back_file(output_filename, output_filename, result,
                                        send_back, "llama", output_dtype_tag);
                        }
                    }

                    op_success = true;
                } else {
                    std::cerr << "❌ Unsupported backend for transformer_operation: "
                            << transformer_op_select << std::endl;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "❌ Exception: " << e.what() << std::endl;
                op_success = false;
            }

            try { send_ack("ACK_transformerOp_complete"); } catch (...) {}
            return op_success;
        }

        bool flash_atten_openartion(  
            const std::string& backend_type,  
            const matrix_shard_object& matrixQ,  
            bool transposeQ,  
            const matrix_shard_object& matrixK,  
            bool transposeK,  
            const matrix_shard_object& matrixV,  
            bool transposeV,  
            const matrix_shard_object* matrixMask,
            float scale,  
            float max_bias,  
            float logit_softcap,  
            bool use_gpu,  
            int gpu_id,  
            int send_back,  
            const std::string& operation_type,  
            int shard_index_override)  
        {  
            bool op_success = false;  
        
            try {  
                if (backend_type != "llama" && backend_type != "torch") {  
                    throw std::runtime_error("flash_attn only implemented for llama/torch backend");  
                }  
        
                const bool has_q_data = (matrixQ.data && !matrixQ.data->empty());
                const bool has_k_data = (matrixK.data && !matrixK.data->empty());
                const bool has_v_data = (matrixV.data && !matrixV.data->empty());
        
                // ---------------- Output filename ----------------  
                auto strip_bin = [](std::string& name) {  
                    const size_t pos = name.rfind(".bin");  
                    if (pos != std::string::npos) name = name.substr(0, pos);  
                };  
                auto strip_shard = [](std::string& name) {  
                    const size_t pos = name.find("_shard_");  
                    if (pos != std::string::npos) name = name.substr(0, pos);  
                };  
                auto base_name = [&](const std::string& file) {  
                    std::string out = std::filesystem::path(file).filename().string();  
                    strip_bin(out);  
                    strip_shard(out);  
                    return out;  
                };  
        
                const std::string q_name = base_name(matrixQ.base_file_name);  
                const std::string k_name = base_name(matrixK.base_file_name);  
                const std::string v_name = base_name(matrixV.base_file_name);  
                const std::string base_result = q_name + "x" + k_name + "x" + v_name;  
        
                // Mark this base as requiring FlashAttention-aware combine on the head node.  
                const bool is_head_node = (local_IP_eth == head_node_ip_eth || local_IP_wifi == head_node_ip_wifi);  
                if (is_head_node && send_back != 0) {  
                    std::lock_guard<std::mutex> lock(flash_atten_openartion_combine_mutex);  
                    flash_atten_openartion_combine_list.insert(base_result);  
                }  
        
                std::string output_filename = base_result;  
                int shard_num = shard_index_override;  
                if (shard_num < 0) {  
                    std::lock_guard<std::mutex> lock(output_shard_mutex);  
                    shard_num = output_shard_counters[base_result]++;  
                }  
                output_filename += "_shard_" + std::to_string(shard_num) + ".bin";  
        
                // dtype tag for downstream saves/streams  
                int output_dtype_tag = merge_output_dtype_tag(matrixQ.output_dtype_tag, matrixK.output_dtype_tag);  
                output_dtype_tag = merge_output_dtype_tag(output_dtype_tag, matrixV.output_dtype_tag);  

                const bool has_mask = (matrixMask && matrixMask->data);
                int colsM = 0, rowsM = 0, depthM = 0, batchM = 0;
                const float* mask_ptr = nullptr;
                if (has_mask) {
                    colsM = matrixMask->cols_A;
                    rowsM = matrixMask->rows_A;
                    depthM = matrixMask->depthA;
                    batchM = matrixMask->batchA;
                    if (colsM <= 0 || rowsM <= 0 || depthM <= 0 || batchM <= 0) {
                        throw std::runtime_error("Invalid mask dimensions for flash_attn");
                    }
                    mask_ptr = matrixMask->data->data();
                    if (!mask_ptr) {
                        throw std::runtime_error("Mask data pointer is null");
                    }
                }
        
                // After matrix validation, add:  
                printf("Q dims: [%d,%d,%d,%d] type:%d\n",   
                    matrixQ.cols_A, matrixQ.rows_A, matrixQ.depthA, matrixQ.batchA,   
                    matrixQ.output_dtype_tag);  
                printf("K dims: [%d,%d,%d,%d] type:%d\n",   
                    matrixK.cols_A, matrixK.rows_A, matrixK.depthA, matrixK.batchA,  
                    matrixK.output_dtype_tag);  
                printf("V dims: [%d,%d,%d,%d] type:%d\n",   
                    matrixV.cols_A, matrixV.rows_A, matrixV.depthA, matrixV.batchA,  
                    matrixV.output_dtype_tag);

                printf("Input dims - Q: [%d,%d,%d,%d] K: [%d,%d,%d,%d] V: [%d,%d,%d,%d]\n",  
                    matrixQ.cols_A, matrixQ.rows_A, matrixQ.depthA, matrixQ.batchA,  
                    matrixK.cols_A, matrixK.rows_A, matrixK.depthA, matrixK.batchA,  
                    matrixV.cols_A, matrixV.rows_A, matrixV.depthA, matrixV.batchA);
                MatrixResult result;  
                result.data = nullptr;  
                result.dims[0] = result.dims[1] = result.dims[2] = result.dims[3] = 0;  

                if (backend_type == "llama") {
                    auto dtype_tag_to_ggml = [](int tag) -> ggml_type {  
                        if (tag == -2) return GGML_TYPE_F16;  
                        if (tag == -3) return GGML_TYPE_BF16;  
                        return GGML_TYPE_F32;  
                    };  
                    const ggml_type data_type = dtype_tag_to_ggml(output_dtype_tag);
        
                    // ---------------- Select backend ----------------  
                    ggml_backend_t backend;  
                    {  
                        std::lock_guard<std::mutex> lock(matrix_backend_llama.backends_mutex);  
                        backend = (use_gpu && gpu_id >= 0 && gpu_id < (int) matrix_backend_llama.ggml_backends.size())  
                            ? matrix_backend_llama.ggml_backends[gpu_id]  
                            : matrix_backend_llama.ggml_backends.back();  
                    }  

                    // ---------------- VRAM fast-path ----------------  
                    const bool can_use_vram =  
                        use_gpu &&  
                        !transposeQ && !transposeK && !transposeV &&  
                        gpu_id >= 0 &&  
                        gpu_id < (int) matrix_backend_llama.ggml_backends.size() &&  
                        data_type == GGML_TYPE_F32 &&
                        !has_mask;  
        
                    if (can_use_vram) {  
                        auto& vram = get_vram_cache_manager();  
        
                        auto cache_tensor_if_needed = [&](const matrix_shard_object& mat) {  
                            if (!vram.enabled(gpu_id)) return;  
                            if (vram.get_tensor(gpu_id, mat.base_file_name)) return;  
                            if (!mat.data || mat.data->empty()) return;
        
                            vram.cache_tensor_f32_4d(  
                                gpu_id,  
                                mat.base_file_name,  
                                mat.data->data(),  
                                mat.cols_A,  
                                mat.rows_A,  
                                mat.depthA,  
                                mat.batchA  
                            );  
                        };  
        
                        cache_tensor_if_needed(matrixQ);  
                        cache_tensor_if_needed(matrixK);  
                        cache_tensor_if_needed(matrixV);  
        
                        ggml_tensor* q_t = vram.get_tensor(gpu_id, matrixQ.base_file_name);  
                        ggml_tensor* k_t = vram.get_tensor(gpu_id, matrixK.base_file_name);  
                        ggml_tensor* v_t = vram.get_tensor(gpu_id, matrixV.base_file_name);  
        
                        if (q_t && k_t && v_t) {  
                            float scale_use = scale;  
                            if (scale_use <= 0.0f) {  
                                scale_use = 1.0f / std::sqrt((float) std::max<int64_t>(1, q_t->ne[0]));  
                            }  
                            FlashAttnTensorParams p{v_t, nullptr, scale_use, max_bias, logit_softcap};  
                            result = matrix_backend_llama.matrix_op_nd_tensors(q_t, k_t, backend, operation_type, &p);  
                        }  
                    }  
        
                // ---------------- Host fallback ----------------  
                if (!result.data) {
                    if (!has_q_data || !has_k_data || !has_v_data) {
                        throw std::runtime_error("Missing matrix data for flash_attn inputs (VRAM cache miss)");
                    }
                    // Host fallback: force F32 tensors to avoid re-quantization/stride issues.
                    const ggml_type host_type = GGML_TYPE_F32;
                    int colsQ = matrixQ.cols_A, rowsQ = matrixQ.rows_A, depthQ = matrixQ.depthA, batchQ = matrixQ.batchA;
                    int colsK = matrixK.cols_A, rowsK = matrixK.rows_A, depthK = matrixK.depthA, batchK = matrixK.batchA;
                    int colsV = matrixV.cols_A, rowsV = matrixV.rows_A, depthV = matrixV.depthA, batchV = matrixV.batchA;
        
                    const float* q_ptr = matrixQ.data->data();
                    const float* k_ptr = matrixK.data->data();
                    const float* v_ptr = matrixV.data->data();
        
                    std::unique_ptr<float[]> q_tmp;  
                    std::unique_ptr<float[]> k_tmp;  
                    std::unique_ptr<float[]> v_tmp;  
        
                    if (transposeQ) {  
                        q_tmp = (depthQ > 1 || batchQ > 1)  
                            ? matrix_backend_llama.transpose_4d(const_cast<float*>(q_ptr), batchQ, depthQ, rowsQ, colsQ)  
                            : matrix_backend_llama.transpose_2d(const_cast<float*>(q_ptr), rowsQ, colsQ);  
                        q_ptr = q_tmp.get();  
                        std::swap(rowsQ, colsQ);  
                    }  
                    if (transposeK) {  
                        k_tmp = (depthK > 1 || batchK > 1)  
                            ? matrix_backend_llama.transpose_4d(const_cast<float*>(k_ptr), batchK, depthK, rowsK, colsK)  
                            : matrix_backend_llama.transpose_2d(const_cast<float*>(k_ptr), rowsQ, colsK);  
                        k_ptr = k_tmp.get();  
                        std::swap(rowsK, colsK);  
                    }  
                    if (transposeV) {  
                        v_tmp = (depthV > 1 || batchV > 1)  
                            ? matrix_backend_llama.transpose_4d(const_cast<float*>(v_ptr), batchV, depthV, rowsV, colsV)  
                            : matrix_backend_llama.transpose_2d(const_cast<float*>(v_ptr), rowsQ, colsV);  
                        v_ptr = v_tmp.get();  
                        std::swap(rowsV, colsV);  
                    }  
        
                    int dimsQ[4] = { colsQ, rowsQ, depthQ, batchQ };  
                    int dimsK[4] = { colsK, rowsK, depthK, batchK };  
                    std::cout << "Host dims after transpose Q/K/V: "
                              << "[" << dimsQ[0] << "," << dimsQ[1] << "," << dimsQ[2] << "," << dimsQ[3] << "] "
                              << "[" << dimsK[0] << "," << dimsK[1] << "," << dimsK[2] << "," << dimsK[3] << "] "
                              << "[" << colsV << "," << rowsV << "," << depthV << "," << batchV << "]"
                              << std::endl;
        
                    float scale_use = scale;  
                    if (scale_use <= 0.0f) {  
                        scale_use = 1.0f / std::sqrt((float) std::max(1, dimsQ[0]));  
                    }  
        
                    // Build/normalize mask in GGML layout (ne0 = n_kv, ne1 = n_q padded).
                    int mask_cols = 0;
                    int mask_rows = 0;
                    int mask_rows_pad = 0;
                    int mask_depth = 0;
                    int mask_batch = 0;
                    const float* mask_ptr_upload = nullptr;
                    std::vector<float> mask_f32_work;
                    if (has_mask) {
                        const int expected_cols = rowsK; // n_kv
                        const int expected_rows = rowsQ; // n_q
                        const bool dims_ok = (colsM == expected_cols && rowsM == expected_rows);
                        const bool dims_swapped = (colsM == expected_rows && rowsM == expected_cols);
                        if (!dims_ok && !dims_swapped) {
                            throw std::runtime_error(
                                "Mask dims do not match Q/K (expected [n_kv,n_q] or [n_q,n_kv])");
                        }
                        if (depthM <= 0 || batchM <= 0) {
                            throw std::runtime_error("Mask depth/batch invalid");
                        }
                        if (depthQ % depthM != 0 || batchQ % batchM != 0) {
                            throw std::runtime_error("Mask depth/batch not broadcastable to Q/K");
                        }

                        mask_cols = expected_cols;
                        mask_rows = expected_rows;
                        mask_rows_pad = std::max(mask_rows, (int)GGML_PAD(mask_rows, GGML_KQ_MASK_PAD));
                        mask_depth = depthM;
                        mask_batch = batchM;

                        const size_t out_elems =
                            (size_t)mask_cols * mask_rows_pad * mask_depth * mask_batch;
                        mask_f32_work.assign(out_elems, 0.0f);

                        for (int b = 0; b < mask_batch; ++b) {
                            for (int d = 0; d < mask_depth; ++d) {
                                for (int r = 0; r < mask_rows; ++r) {
                                    for (int c = 0; c < mask_cols; ++c) {
                                        size_t src_idx;
                                        if (dims_ok) {
                                            // mask stored as [rows= n_q, cols= n_kv]
                                            src_idx = (((size_t)b * depthM + d) * (size_t)rowsM + r) * (size_t)colsM + c;
                                        } else {
                                            // mask stored as [rows= n_kv, cols= n_q] -> transpose on read
                                            src_idx = (((size_t)b * depthM + d) * (size_t)rowsM + c) * (size_t)colsM + r;
                                        }
                                        const size_t dst_idx =
                                            (((size_t)b * mask_depth + d) * (size_t)mask_rows_pad + r) * (size_t)mask_cols + c;
                                        mask_f32_work[dst_idx] = mask_ptr[src_idx];
                                    }
                                }
                            }
                        }
                        mask_ptr_upload = mask_f32_work.data();
                    }

                    auto run_flash_attn_typed = [&](ggml_backend_t be) -> MatrixResult {
                        MatrixResult out;
                        out.data = nullptr;
                        out.dims[0] = out.dims[1] = out.dims[2] = out.dims[3] = 0;
        
                        ggml_context* ctx = matrix_backend_llama.get_thread_graph_ctx(  
                            0, /*bump_for_flash_attn=*/true);  
                        if (!ctx) return out;  
        
                        ggml_tensor* q_t = ggml_new_tensor_4d(ctx, host_type, dimsQ[0], dimsQ[1], dimsQ[2], dimsQ[3]);
                        ggml_tensor* k_t = ggml_new_tensor_4d(ctx, host_type, dimsK[0], dimsK[1], dimsK[2], dimsK[3]);
                        ggml_tensor* v_t = ggml_new_tensor_4d(ctx, host_type, colsV, rowsV, depthV, batchV);
                        ggml_tensor* mask_t = nullptr;
                        if (has_mask) {
                            mask_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, mask_cols, mask_rows_pad, mask_depth, mask_batch);
                        }
                        if (!q_t || !k_t || !v_t || (has_mask && !mask_t)) return out;

                        const size_t type_size = ggml_type_size(host_type);
                        q_t->nb[0] = type_size;
                        q_t->nb[1] = q_t->nb[0] * dimsQ[0];
                        q_t->nb[2] = q_t->nb[1] * dimsQ[1];
                        q_t->nb[3] = q_t->nb[2] * dimsQ[2];
        
                        k_t->nb[0] = type_size;  
                        k_t->nb[1] = k_t->nb[0] * dimsK[0];  
                        k_t->nb[2] = k_t->nb[1] * dimsK[1];  
                        k_t->nb[3] = k_t->nb[2] * dimsK[2];  
        
                        v_t->nb[0] = type_size;
                        v_t->nb[1] = v_t->nb[0] * colsV;
                        v_t->nb[2] = v_t->nb[1] * rowsV;
                        v_t->nb[3] = v_t->nb[2] * depthV;
                        if (mask_t) {
                            mask_t->nb[0] = sizeof(ggml_fp16_t);
                            mask_t->nb[1] = mask_t->nb[0] * mask_cols;
                            mask_t->nb[2] = mask_t->nb[1] * mask_rows_pad;
                            mask_t->nb[3] = mask_t->nb[2] * mask_depth;
                        }
        
                        ggml_tensor* result_t = ggml_flash_attn_ext(  
                            ctx, q_t, k_t, v_t, mask_t,  
                            scale_use, max_bias, logit_softcap);  
                        if (!result_t) return out;  
        
                        ggml_cgraph* gf = ggml_new_graph(ctx);  
                        ggml_build_forward_expand(gf, result_t);  
        
                        if (be && !ggml_backend_supports_op(be, result_t)) {  
                            return out;  
                        }  
        
                        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);  
                        if (!buf) return out;  
        
                        auto set_tensor_data = [&](ggml_tensor* t, const float* src, int64_t n) {
                            ggml_backend_tensor_set(
                                t, src, 0, (size_t)n * sizeof(float));
                        };
        
                        const int64_t n_q = (int64_t)dimsQ[0] * dimsQ[1] * dimsQ[2] * dimsQ[3];  
                        const int64_t n_k = (int64_t)dimsK[0] * dimsK[1] * dimsK[2] * dimsK[3];  
                        const int64_t n_v = (int64_t)colsV * rowsV * depthV * batchV;  
                        set_tensor_data(q_t, q_ptr, n_q);  
                        set_tensor_data(k_t, k_ptr, n_k);  
                        set_tensor_data(v_t, v_ptr, n_v);  
                        if (mask_t && mask_ptr_upload) {
                            const size_t mask_elems = (size_t)mask_cols * mask_rows_pad * mask_depth * mask_batch;
                            std::vector<ggml_fp16_t> mask_f16(mask_elems);
                            ggml_fp32_to_fp16_row(mask_ptr_upload, mask_f16.data(), (int64_t)mask_elems);
                            ggml_backend_tensor_set(mask_t, mask_f16.data(), 0, ggml_nbytes(mask_t));
                        }
        
                        ggml_backend_graph_compute(be, gf);  
        
                        const int64_t total = result_t->ne[3] * result_t->ne[2] * result_t->ne[1] * result_t->ne[0];
                        // ggml_flash_attn_ext returns ne=[D, H, T, B]. Convert to [B, H, T, D].
                        out.dims[0] = (int)result_t->ne[3];
                        out.dims[1] = (int)result_t->ne[1];
                        out.dims[2] = (int)result_t->ne[2];
                        out.dims[3] = (int)result_t->ne[0];

                        std::cout << "  flash_attn_ext result ne=["
                                  << result_t->ne[0] << "," << result_t->ne[1] << ","
                                  << result_t->ne[2] << "," << result_t->ne[3] << "]"
                                  << " -> out dims [B,H,T,D]=["
                                  << out.dims[0] << "," << out.dims[1] << ","
                                  << out.dims[2] << "," << out.dims[3] << "]"
                                  << std::endl;

                        std::vector<float> tmp_f32((size_t)total);
                        const ggml_type out_type = result_t->type;
                        if (out_type == GGML_TYPE_F32) {
                            ggml_backend_tensor_get(
                                result_t, tmp_f32.data(), 0,
                                sizeof(float) * (size_t)total);
                        } else if (out_type == GGML_TYPE_F16) {
                            std::vector<ggml_fp16_t> tmp((size_t)total);
                            ggml_backend_tensor_get(
                                result_t, tmp.data(), 0,
                                sizeof(ggml_fp16_t) * (size_t)total);
                            ggml_fp16_to_fp32_row(tmp.data(), tmp_f32.data(), total);
                        } else if (out_type == GGML_TYPE_BF16) {
                            std::vector<uint16_t> tmp((size_t)total);
                            ggml_backend_tensor_get(
                                result_t, tmp.data(), 0,
                                sizeof(uint16_t) * (size_t)total);
                            for (int64_t i = 0; i < total; ++i) {
                                tmp_f32[(size_t)i] = bf16_bits_to_float(tmp[(size_t)i]);
                            }
                        } else {
                            ggml_backend_tensor_get(
                                result_t, tmp_f32.data(), 0,
                                sizeof(float) * (size_t)total);
                        }

                        out.data = std::make_unique<float[]>(total);
                        const int64_t D = result_t->ne[0];
                        const int64_t H = result_t->ne[1];
                        const int64_t T = result_t->ne[2];
                        const int64_t B = result_t->ne[3];
                        for (int64_t b = 0; b < B; ++b) {
                            for (int64_t h = 0; h < H; ++h) {
                                for (int64_t t = 0; t < T; ++t) {
                                    const int64_t base_src = (b * T * H + t * H + h) * D;
                                    const int64_t base_dst = (b * H * T + h * T + t) * D;
                                    std::memcpy(
                                        out.data.get() + base_dst,
                                        tmp_f32.data() + base_src,
                                        (size_t)D * sizeof(float));
                                }
                            }
                        }
        
                        ggml_backend_buffer_free(buf);  
                        return out;  
                    };  
        
                    result = run_flash_attn_typed(backend);  
        
                    if (!result.data && use_gpu) {  
                        ggml_backend_t cpu_backend;  
                        {  
                            std::lock_guard<std::mutex> lock(matrix_backend_llama.backends_mutex);  
                            cpu_backend = matrix_backend_llama.ggml_backends.back();  
                        }  
                        result = run_flash_attn_typed(cpu_backend);  
                    }  
                }  
                } else if (backend_type == "torch") {
                    bool torch_gpu_available = false;
                    #ifdef USE_CUDA
                    torch_gpu_available = torch::cuda::is_available();
                    #endif

                    if (use_gpu && !torch_gpu_available) {
                        std::cout << "⚠️  GPU requested but unavailable. Using CPU." << std::endl;
                    }

                    auto to_torch_4d = [](const matrix_shard_object& obj) -> torch::Tensor {
                        if (!obj.data) {
                            throw std::runtime_error("Missing matrix data for torch flash_attn input: " + obj.base_file_name);
                        }
                        std::vector<int64_t> sizes = {obj.batchA, obj.depthA, obj.rows_A, obj.cols_A};
                        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
                        return torch::from_blob((void*)obj.data->data(), sizes, options).clone();
                    };

                    torch::Tensor Q = to_torch_4d(matrixQ);
                    torch::Tensor K = to_torch_4d(matrixK);
                    torch::Tensor V = to_torch_4d(matrixV);

                    if (transposeQ) Q = Q.transpose(-2, -1).contiguous();
                    if (transposeK) K = K.transpose(-2, -1).contiguous();
                    if (transposeV) V = V.transpose(-2, -1).contiguous();

                    const int64_t Bq = Q.size(0);
                    const int64_t Hq = Q.size(1);
                    const int64_t Tq = Q.size(2);
                    const int64_t Dq = Q.size(3);

                    const int64_t Tk = K.size(2);
                    const int64_t Dk = K.size(3);

                    const int64_t Tv = V.size(2);

                    if (Dq != Dk) {
                        throw std::runtime_error("flash_attn torch: Q/K head dim mismatch");
                    }
                    if (Tk != Tv) {
                        throw std::runtime_error("flash_attn torch: K/V sequence mismatch");
                    }

                    auto repeat_interleave_or_throw = [&](torch::Tensor t, int64_t target, int64_t dim, const char* label) -> torch::Tensor {
                        const int64_t current = t.size(dim);
                        if (current == target) return t;
                        if (current <= 0 || target % current != 0) {
                            throw std::runtime_error(std::string("flash_attn torch: cannot broadcast ") + label);
                        }
                        const int64_t rep = target / current;
                        return t.repeat_interleave(rep, dim);
                    };
                    auto repeat_tile_or_throw = [&](torch::Tensor t, int64_t target, int64_t dim, const char* label) -> torch::Tensor {
                        const int64_t current = t.size(dim);
                        if (current == target) return t;
                        if (current <= 0 || target % current != 0) {
                            throw std::runtime_error(std::string("flash_attn torch: cannot broadcast ") + label);
                        }
                        std::vector<int64_t> reps(t.dim(), 1);
                        reps[(size_t)dim] = target / current;
                        return t.repeat(reps);
                    };

                    // Match GGML broadcast semantics (grouped repeat).
                    K = repeat_interleave_or_throw(K, Bq, 0, "K batch");
                    K = repeat_interleave_or_throw(K, Hq, 1, "K heads");
                    V = repeat_interleave_or_throw(V, Bq, 0, "V batch");
                    V = repeat_interleave_or_throw(V, Hq, 1, "V heads");

                    torch::Tensor attn_mask;
                    if (has_mask) {
                        const int64_t expected_cols = Tk; // n_kv
                        const int64_t expected_rows = Tq; // n_q
                        const bool dims_ok = (colsM == expected_cols && rowsM == expected_rows);
                        const bool dims_swapped = (colsM == expected_rows && rowsM == expected_cols);
                        if (!dims_ok && !dims_swapped) {
                            throw std::runtime_error("Mask dims do not match Q/K (expected [n_kv,n_q] or [n_q,n_kv])");
                        }

                        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
                        torch::Tensor mask = torch::from_blob((void*)mask_ptr, {batchM, depthM, rowsM, colsM}, options).clone();
                        if (dims_swapped && !dims_ok) {
                            mask = mask.transpose(-2, -1).contiguous();
                        }

                        if (mask.size(2) != expected_rows || mask.size(3) != expected_cols) {
                            throw std::runtime_error("Mask dims after transpose do not match Q/K");
                        }

                        // Broadcast mask across depth/batch (tiled repeat).
                        mask = repeat_tile_or_throw(mask, Hq, 1, "mask depth");
                        mask = repeat_tile_or_throw(mask, Bq, 0, "mask batch");

                        if (max_bias > 0.0f) {
                            int64_t n_head = Hq;
                            int64_t n_head_log2 = 1;
                            while ((n_head_log2 << 1) <= n_head) {
                                n_head_log2 <<= 1;
                            }
                            const float m0 = std::pow(2.0f, -(max_bias) / (float)n_head_log2);
                            const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);
                            std::vector<float> slopes((size_t)n_head, 1.0f);
                            for (int64_t h = 0; h < n_head; ++h) {
                                if (h < n_head_log2) {
                                    slopes[(size_t)h] = std::pow(m0, (float)(h + 1));
                                } else {
                                    const float expn = (float)(2 * (h - n_head_log2) + 1);
                                    slopes[(size_t)h] = std::pow(m1, expn);
                                }
                            }
                            torch::Tensor slope_t = torch::from_blob(slopes.data(), {n_head}, options).clone();
                            slope_t = slope_t.view({1, n_head, 1, 1});
                            mask = mask * slope_t;
                        }

                        attn_mask = mask;
                    } else if (max_bias > 0.0f) {
                        throw std::runtime_error("flash_attn torch: max_bias requires mask");
                    }

                    torch::Device device = torch::kCPU;
                    if (use_gpu && torch_gpu_available) {
                        device = torch::Device(torch::kCUDA, gpu_id);
                    }

                    torch::ScalarType desired_dtype = torch::kFloat32;
                    if (output_dtype_tag == -2) {
                        desired_dtype = torch::kFloat16;
                    } else if (output_dtype_tag == -3) {
                        desired_dtype = torch::kBFloat16;
                    }
                    if (device.is_cpu()) {
                        // CPU kernels are most reliable in float32.
                        desired_dtype = torch::kFloat32;
                    }

                    Q = Q.to(device);
                    K = K.to(device);
                    V = V.to(device);
                    if (Q.scalar_type() != desired_dtype) Q = Q.to(desired_dtype);
                    if (K.scalar_type() != desired_dtype) K = K.to(desired_dtype);
                    if (V.scalar_type() != desired_dtype) V = V.to(desired_dtype);

                    if (attn_mask.defined()) {
                        attn_mask = attn_mask.to(device);
                    }

                    float scale_use = scale;
                    if (scale_use <= 0.0f) {
                        scale_use = 1.0f / std::sqrt((float)std::max<int64_t>(1, Dq));
                    }
                    if (logit_softcap != 0.0f) {
                        scale_use /= logit_softcap;
                    }

                    torch::Tensor out;
                    if (logit_softcap == 0.0f && max_bias == 0.0f && !attn_mask.defined()) {
                        const c10::optional<torch::Tensor> mask_opt =
                            attn_mask.defined() ? c10::optional<torch::Tensor>(attn_mask) : c10::nullopt;
                        out = at::scaled_dot_product_attention(
                            Q, K, V,
                            mask_opt,
                            0.0,
                            false,
                            scale_use
                        );
                    } else {
                        torch::Tensor Qm = Q;
                        torch::Tensor Km = K;
                        torch::Tensor Vm = V;
                        torch::Tensor Mm = attn_mask;
                        if (Q.scalar_type() != torch::kFloat32) Qm = Q.to(torch::kFloat32);
                        if (K.scalar_type() != torch::kFloat32) Km = K.to(torch::kFloat32);
                        if (V.scalar_type() != torch::kFloat32) Vm = V.to(torch::kFloat32);
                        if (Mm.defined() && Mm.scalar_type() != torch::kFloat32) Mm = Mm.to(torch::kFloat32);

                        torch::Tensor logits = torch::matmul(Qm, Km.transpose(-2, -1)) * scale_use;
                        if (logit_softcap != 0.0f) {
                            logits = logit_softcap * torch::tanh(logits);
                        }
                        if (Mm.defined()) {
                            logits = logits + Mm;
                        }

                        torch::Tensor probs = torch::softmax(logits, -1);
                        out = torch::matmul(probs, Vm);
                    }
                    out = out.contiguous().to(torch::kCPU);
                    if (out.scalar_type() != torch::kFloat32) {
                        out = out.to(torch::kFloat32);
                    }

                    const int64_t total = out.numel();
                    result.dims[0] = (int)out.size(0);
                    result.dims[1] = (int)out.size(1);
                    result.dims[2] = (int)out.size(2);
                    result.dims[3] = (int)out.size(3);
                    result.data = std::make_unique<float[]>((size_t)total);
                    std::memcpy(result.data.get(), out.data_ptr<float>(), (size_t)total * sizeof(float));
                }

                if (!result.data) {  
                    throw std::runtime_error("flash_attn_ext computation failed");  
                }  

                // Store result in shard list for future ops  
                store_matrix_result_to_shard_list(output_filename, result, output_dtype_tag);  
        
                if (send_back != 0) {  
                    send_back_file(output_filename, output_filename, result, send_back, backend_type, output_dtype_tag);  
                }  

                op_success = true;  
            }  
            catch (const std::exception& ex) {  
                std::cerr << "Error in flash_atten_openartion: " << ex.what() << std::endl;  
                op_success = false;  
            }  
        
            try { send_ack("ACK_matrixOp_complete"); } catch (...) {}
                    return op_success;

        }
        
        bool rope_openartion(
            const std::string& backend_type,
            const matrix_shard_object& matrixA,
            bool transposeA,
            bool use_gpu,
            int gpu_id,
            int send_back,
            const std::string& op_type,  // "rope", "rope_ext", "rope_multi"
            const RoPEParams* rope_base_params,
            const RoPEExtParams* rope_ext_params,
            const RoPEMultiParams* rope_multi_params,
            int shard_index_override
        )
        {
            bool op_success = false;

            try {
                if (backend_type != "llama") {
                    throw std::runtime_error("RoPE only implemented for llama backend");
                }

                if (!matrixA.data) {
                    throw std::runtime_error("Missing matrix data for RoPE operation");
                }

                // ---------------- Validate exactly ONE params ----------------
                const int param_count =
                    (rope_base_params  != nullptr) +
                    (rope_ext_params   != nullptr) +
                    (rope_multi_params != nullptr);

                if (param_count != 1) {
                    throw std::runtime_error("Exactly ONE RoPE parameter struct must be provided");
                }

                // ---------------- Select backend ----------------
                ggml_backend_t backend;
                {
                    std::lock_guard<std::mutex> lock(matrix_backend_llama.backends_mutex);
                    backend = (use_gpu && gpu_id >= 0 &&
                            gpu_id < (int)matrix_backend_llama.ggml_backends.size())
                        ? matrix_backend_llama.ggml_backends[gpu_id]
                        : matrix_backend_llama.ggml_backends.back();
                }

                // ---------------- Prepare input ----------------
                const float* a_ptr = matrixA.data->data();
                int colsA  = matrixA.cols_A;
                int rowsA  = matrixA.rows_A;
                int depthA = matrixA.depthA;
                int batchA = matrixA.batchA;

                std::unique_ptr<float[]> a_tmp;

                if (transposeA) {
                    a_tmp = (depthA > 1 || batchA > 1)
                        ? matrix_backend_llama.transpose_4d(
                            const_cast<float*>(a_ptr), batchA, depthA, rowsA, colsA)
                        : matrix_backend_llama.transpose_2d(
                            const_cast<float*>(a_ptr), rowsA, colsA);
                    a_ptr = a_tmp.get();
                    std::swap(rowsA, colsA);
                }

                int dimsA[4] = { colsA, rowsA, depthA, batchA };

                // ---------------- Execute ----------------
                void* rope_params = nullptr;
                if (rope_base_params)  rope_params = (void*)rope_base_params;
                if (rope_ext_params)   rope_params = (void*)rope_ext_params;
                if (rope_multi_params) rope_params = (void*)rope_multi_params;

                if (op_type != "rope" && op_type != "rope_ext" && op_type != "rope_multi") {
                    throw std::runtime_error("Invalid RoPE op_type: " + op_type);
                }

                MatrixResult result = matrix_backend_llama.matrix_op_nd(
                    const_cast<float*>(a_ptr),
                    dimsA,
                    nullptr,
                    nullptr,
                    backend,
                    op_type,
                    rope_params
                );

                // ---------------- CPU fallback ----------------
                if (!result.data && use_gpu) {
                    ggml_backend_t cpu_backend;
                    {
                        std::lock_guard<std::mutex> lock(matrix_backend_llama.backends_mutex);
                        cpu_backend = matrix_backend_llama.ggml_backends.back();
                    }

                    result = matrix_backend_llama.matrix_op_nd(
                        const_cast<float*>(a_ptr),
                        dimsA,
                        nullptr,
                        nullptr,
                        cpu_backend,
                        op_type,
                        rope_params
                    );
                }

                if (!result.data) {
                    throw std::runtime_error("RoPE computation failed");
                }

                // ---------------- Output filename (SAFE) ----------------
                std::string output_filename = matrixA.base_file_name;

                int shard_num = shard_index_override;
                if (shard_num < 0) {
                    std::lock_guard<std::mutex> lock(output_shard_mutex);
                    shard_num = output_shard_counters[output_filename]++;
                }

                // If the original filename does not already contain a shard suffix, add it
                if (output_filename.find("_shard_") == std::string::npos) {
                    output_filename += "_shard_" + std::to_string(shard_num) + ".bin";
                }

                // ---------------- Store / send ----------------
                store_matrix_result_to_shard_list(
                    output_filename,
                    result,
                    matrixA.output_dtype_tag
                );

                if (send_back != 0) {
                    send_back_file(
                        output_filename,
                        output_filename,
                        result,
                        send_back,
                        "llama",
                        matrixA.output_dtype_tag
                    );
                }

                op_success = true;

            }
            catch (const std::exception& ex) {
                std::cerr << "Error in rope_openartion: " << ex.what() << std::endl;
                op_success = false;
            }

            try { send_ack("ACK_matrixOp_complete"); } catch (...) {}

            return op_success;
        }

        bool reshape_matrix(  
            const std::string& backend_type,  
            const matrix_shard_object& matrixA,  
            bool transposeA,  
            bool use_gpu,  
            int gpu_id,  
            int send_back,  
            const int output_dims[4],  
            int shard_index_override)  
        {  
            bool op_success = false;  
        
            try {  
                if (backend_type != "llama") {  
                    throw std::runtime_error("Reshape only implemented for llama backend");  
                }  
        
                if (!matrixA.data) {  
                    throw std::runtime_error("Missing matrix data for reshape operation");  
                }  
        
                // ---------------- Validate output dimensions (torch order) ----------------  
                int inferred = 0;
                for (int i = 0; i < 4; i++) {  
                    if (output_dims[i] == 0) {  
                        throw std::runtime_error("Invalid output dimension at index " + std::to_string(i));  
                    }  
                    if (output_dims[i] < 0) {
                        inferred += 1;
                    }
                }  
                if (inferred > 1) {
                    throw std::runtime_error("Only one output dimension can be inferred (-1)");
                }
        
                // ---------------- Select backend ----------------  
                ggml_backend_t backend;  
                {  
                    std::lock_guard<std::mutex> lock(matrix_backend_llama.backends_mutex);  
                    backend = (use_gpu && gpu_id >= 0 &&  
                            gpu_id < (int)matrix_backend_llama.ggml_backends.size())  
                        ? matrix_backend_llama.ggml_backends[gpu_id]  
                        : matrix_backend_llama.ggml_backends.back();  
                }  
        
                // ---------------- Prepare input ----------------  
                const float* a_ptr = matrixA.data->data();  
                int colsA  = matrixA.cols_A;  
                int rowsA  = matrixA.rows_A;  
                int depthA = matrixA.depthA;  
                int batchA = matrixA.batchA;  
        
                std::unique_ptr<float[]> a_tmp;  
        
                if (transposeA) {  
                    a_tmp = (depthA > 1 || batchA > 1)  
                        ? matrix_backend_llama.transpose_4d(  
                            const_cast<float*>(a_ptr), batchA, depthA, rowsA, colsA)  
                        : matrix_backend_llama.transpose_2d(  
                            const_cast<float*>(a_ptr), rowsA, colsA);  
                    a_ptr = a_tmp.get();  
                    std::swap(rowsA, colsA);  
                }  
        
                int input_dims[4] = { colsA, rowsA, depthA, batchA };  

                // Convert output dims from torch order (batch, depth, rows, cols)
                // to ggml order (cols, rows, depth, batch)
                int output_dims_ggml[4] = {
                    output_dims[3],
                    output_dims[2],
                    output_dims[1],
                    output_dims[0]
                };
        
                // ---------------- Execute reshape ----------------  
                MatrixResult result = matrix_backend_llama.reshape_nd(  
                    const_cast<float*>(a_ptr),  
                    input_dims,  
                    output_dims_ggml,  
                    backend  
                );  
        
                // ---------------- CPU fallback ----------------  
                if (!result.data && use_gpu) {  
                    ggml_backend_t cpu_backend;  
                    {  
                        std::lock_guard<std::mutex> lock(matrix_backend_llama.backends_mutex);  
                        cpu_backend = matrix_backend_llama.ggml_backends.back();  
                    }  
        
                    result = matrix_backend_llama.reshape_nd(  
                        const_cast<float*>(a_ptr),  
                        input_dims,  
                        output_dims_ggml,  
                        cpu_backend  
                    );  
                }  
        
                if (!result.data) {  
                    throw std::runtime_error("Reshape computation failed");  
                }  
        
                // ---------------- Output filename (SAFE) ----------------  
                std::string output_filename = matrixA.base_file_name;  
        
                int shard_num = shard_index_override;  
                if (shard_num < 0) {  
                    std::lock_guard<std::mutex> lock(output_shard_mutex);  
                    shard_num = output_shard_counters[output_filename]++;  
                }  
        
                // If the original filename does not already contain a shard suffix, add it  
                if (output_filename.find("_shard_") == std::string::npos) {  
                    output_filename += "_shard_" + std::to_string(shard_num) + ".bin";  
                }  
        
                // ---------------- Store / send ----------------  
                store_matrix_result_to_shard_list(  
                    output_filename,  
                    result,  
                    matrixA.output_dtype_tag  
                );  
        
                if (send_back != 0) {  
                    send_back_file(  
                        output_filename,  
                        output_filename,  
                        result,  
                        send_back,  
                        "llama",  
                        matrixA.output_dtype_tag  
                    );  
                }  
        
                op_success = true;  
        
            }  
            catch (const std::exception& ex) {  
                std::cerr << "Error in reshape_matrix: " << ex.what() << std::endl;  
                op_success = false;  
            }  
        
            try { send_ack("ACK_matrixOp_complete"); } catch (...) {}  
        
            return op_success;  
        }

        bool repeat_matrix(  
            const std::string& backend_type,  
            const matrix_shard_object& matrixA,  
            bool transposeA,  
            bool use_gpu,  
            int gpu_id,  
            int send_back,  
            const int repeat_dims[4],  
            int shard_index_override)  
        {  
            bool op_success = false;  
        
            try {  
                if (backend_type != "llama") {  
                    throw std::runtime_error("Repeat only implemented for llama backend");  
                }  
        
                if (!matrixA.data) {  
                    throw std::runtime_error("Missing matrix data for repeat operation");  
                }  
        
                // ---------------- Validate repeat dimensions ----------------  
                for (int i = 0; i < 4; i++) {  
                    if (repeat_dims[i] <= 0) {  
                        throw std::runtime_error("Invalid repeat dimension at index " + std::to_string(i));  
                    }  
                }  
        
                // ---------------- Select backend ----------------  
                ggml_backend_t backend;  
                {  
                    std::lock_guard<std::mutex> lock(matrix_backend_llama.backends_mutex);  
                    backend = (use_gpu && gpu_id >= 0 &&  
                            gpu_id < (int)matrix_backend_llama.ggml_backends.size())  
                        ? matrix_backend_llama.ggml_backends[gpu_id]  
                        : matrix_backend_llama.ggml_backends.back();  
                }  
        
                // ---------------- Prepare input ----------------  
                const float* a_ptr = matrixA.data->data();  
                int colsA  = matrixA.cols_A;  
                int rowsA  = matrixA.rows_A;  
                int depthA = matrixA.depthA;  
                int batchA = matrixA.batchA;  
        
                std::unique_ptr<float[]> a_tmp;  
        
                if (transposeA) {  
                    a_tmp = (depthA > 1 || batchA > 1)  
                        ? matrix_backend_llama.transpose_4d(  
                            const_cast<float*>(a_ptr), batchA, depthA, rowsA, colsA)  
                        : matrix_backend_llama.transpose_2d(  
                            const_cast<float*>(a_ptr), rowsA, colsA);  
                    a_ptr = a_tmp.get();  
                    std::swap(rowsA, colsA);  
                }  
        
                int input_dims[4] = { colsA, rowsA, depthA, batchA };  
        
                // ---------------- Execute repeat ----------------  
                // repeat_dims are provided in torch order (batch, depth, rows, cols)
                // Convert to ggml order (cols, rows, depth, batch)
                int repeat_dims_ggml[4] = {
                    repeat_dims[3],
                    repeat_dims[2],
                    repeat_dims[1],
                    repeat_dims[0]
                };

                MatrixResult result = matrix_backend_llama.repeat_nd(  
                    const_cast<float*>(a_ptr),  
                    input_dims,  
                    repeat_dims_ggml,  
                    backend  
                );  
        
                // ---------------- CPU fallback ----------------  
                if (!result.data && use_gpu) {  
                    ggml_backend_t cpu_backend;  
                    {  
                        std::lock_guard<std::mutex> lock(matrix_backend_llama.backends_mutex);  
                        cpu_backend = matrix_backend_llama.ggml_backends.back();  
                    }  
        
                    result = matrix_backend_llama.repeat_nd(  
                        const_cast<float*>(a_ptr),  
                        input_dims,  
                        repeat_dims_ggml,  
                        cpu_backend  
                    );  
                }  
        
                if (!result.data) {  
                    throw std::runtime_error("Repeat computation failed");  
                }  
        
                // ---------------- Output filename (SAFE) ----------------  
                std::string output_filename = matrixA.base_file_name;  
        
                int shard_num = shard_index_override;  
                if (shard_num < 0) {  
                    std::lock_guard<std::mutex> lock(output_shard_mutex);  
                    shard_num = output_shard_counters[output_filename]++;  
                }  
        
                // If the original filename does not already contain a shard suffix, add it  
                if (output_filename.find("_shard_") == std::string::npos) {  
                    output_filename += "_shard_" + std::to_string(shard_num) + ".bin";  
                }  
        
                // ---------------- Store / send ----------------  
                store_matrix_result_to_shard_list(  
                    output_filename,  
                    result,  
                    matrixA.output_dtype_tag  
                );  
        
                if (send_back != 0) {  
                    send_back_file(  
                        output_filename,  
                        output_filename,  
                        result,  
                        send_back,  
                        "llama",  
                        matrixA.output_dtype_tag  
                    );  
                }  
        
                op_success = true;  
        
            }  
            catch (const std::exception& ex) {  
                std::cerr << "Error in repeat_matrix: " << ex.what() << std::endl;  
                op_success = false;  
            }  
        
            try { send_ack("ACK_matrixOp_complete"); } catch (...) {}  
        
            return op_success;  
        }
};

int main()
{
    llama_zmq_server server;
    server.run_server();
}
