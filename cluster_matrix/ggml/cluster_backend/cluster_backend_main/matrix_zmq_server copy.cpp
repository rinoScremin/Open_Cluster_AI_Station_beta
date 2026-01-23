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
#include <torch/torch.h>


struct combined_matrix_shards
{
    int total_shards_reserved = 0;        // Number of shards currently received
    int number_of_shards_needed = 0;      // Total shards expected for this matrix
    std::string file_name;                // Base filename (without shard index)
    
    std::vector<int> shard_numbers;       // List of received shard indices
    std::list<std::vector<uint8_t>> received_matrix_data;  // Raw binary data of each shard
    std::list<std::vector<int>> dims_list;                 // Dimensions of each shard [batch, depth, rows, cols]
    
    // System marker:
    // - System 1: concatenate shards along join_dim
    // - System 2: 2D grid tile assembly
    //
    // IMPORTANT: some sent_back shards arrive via `save_file_handler()` which currently
    // calls `handle_combine_matrix_shard_list(..., total_shards=0)`, so we must persist
    // whether this matrix is System-2 when we first learn it (from any shard that carries
    // the signed total_shards encoding).
    bool is_system2 = false;

    // Output dtype tag for the final combined matrix (v2 header):
    //   -1 = float32, -2 = float16, -3 = bfloat16
    // This is derived from input dtypes and propagated from worker/head results.
    int output_dtype_tag = -1;

    // Dedupe: avoid counting the same shard index twice if it is retransmitted.
    std::set<int> received_shard_numbers;

    int join_dim = 0; // << for now you only will join dim=0 but join based off this 
    // Note: Using std::list for received data allows efficient insertion
    //       as shards arrive in potentially non-sequential order from workers
};

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

        std::vector<combined_matrix_shards> combined_matrix_shards_list;

        // In your class member variables:
        std::vector<std::string> matrix_file_paths;

        std::vector<std::string> received_data_eth_linux_command;
        std::vector<std::string> received_data_wifi_linux_command;
        std::vector<std::string> received_data_eth_server_command;
        std::vector<std::string> received_data_wifi_server_command;
        
        // Thread-safe mutexes (ADD THESE)
        std::mutex linux_commands_mutex;
        std::mutex server_commands_mutex;
        std::mutex file_data_mutex;
        std::mutex wifi_commands_mutex;
        
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
            // Avoid hardcoding absolute paths; default to current working directory.
            std::string default_project_folder = std::filesystem::current_path().string();
            if (!default_project_folder.empty() && default_project_folder.back() != '/') {
                default_project_folder.push_back('/');
            }
            project_folder = get_env("OPEN_CLUSTER_PROJECT_DIRECTORY", default_project_folder.c_str());



            matrix_shard_folder = get_env("OPEN_CLUSTER_MATRIX_SHARD_DIRECTORY", 
                                        "/dev/shm/matrix_shards/");
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
            
            // Setup Python front-end ACK communication
            std::string python_frontend_ip = get_env("HEAD_NODE_IP", "192.168.2.100");
            std::string python_frontend_port = get_env("PYTHON_FRONT_END_CLUSTER_PORT", "7790");
            
            ack_sender = zmq::socket_t(zmq_context, zmq::socket_type::push);
            ack_sender.connect("tcp://" + python_frontend_ip + ":" + python_frontend_port);
            
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
            
            std::cout << "\nServer initialization complete" << std::endl;
            std::cout << "==============================\n" << std::endl;
        }

        void send_ack(std::string ack_msg = "ACK") 
        {
            zmq::message_t ack(ack_msg.data(), ack_msg.size());
            ack_sender.send(ack, zmq::send_flags::none);
        }

        void run_server() 
        {
            std::cout << "🚀 C++ ZMQ Node Server starting..." << std::endl;
            
            // Start network listener threads for dual-interface operation
            std::thread eth_thread(&llama_zmq_server::listen_interface, this, "Ethernet");
            std::thread wifi_thread(&llama_zmq_server::listen_interface, this, "WiFi");
            std::thread process_command_thread(&llama_zmq_server::process_command, this);
            
            // Detach threads to run as daemon processes (background services)
            eth_thread.detach();
            wifi_thread.detach();
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
                        operation_success = matrix_operation(
                            command_type,
                            command_args[3].c_str(),   // Matrix B path
                            transposeB,
                            command_args[1].c_str(),   // Matrix A path
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
                        operation_success = matrix_operation(
                            command_type,
                            command_args[1].c_str(),   // Matrix A path
                            transposeA,
                            command_args[3].c_str(),   // Matrix B path
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

            // Iterate through each reserved file entry and handle cases:
            // - parallel (ETH + WiFi halves)
            // - single-interface ETH
            // - single-interface WiFi
            for (auto &rf : local_reserved_files)
            {
                std::string filename = rf.save_parallel_file_name.empty() ? std::string("unknown") : rf.save_parallel_file_name[0];
                // Helper lambda to write raw bytes to path
                auto write_raw = [&](const std::filesystem::path &path, const std::vector<uint8_t> &bytes) -> bool {
                    std::filesystem::create_directories(path.parent_path());
                    std::ofstream file(path, std::ios::binary);
                    if (!file.is_open()) return false;
                    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                    file.close();
                    return true;
                };

                // If we have both halves (parallel) -> combine
                if ((rf.is_parallel) || (!rf.received_data_eth_file.empty() && !rf.received_data_wifi_file.empty()))
                {
                    std::vector<uint8_t> combined;
                    combined.reserve(rf.received_data_eth_file.size() + rf.received_data_wifi_file.size());
                    combined.insert(combined.end(), rf.received_data_eth_file.begin(), rf.received_data_eth_file.end());
                    combined.insert(combined.end(), rf.received_data_wifi_file.begin(), rf.received_data_wifi_file.end());

                    size_t sent_back_pos = filename.find("sent_back=");
                    if (sent_back_pos != std::string::npos)
                    {
                        std::string actual_filename = filename.substr(sent_back_pos + 10);
                        std::filesystem::path save_path = std::filesystem::path(matrix_results_folder) / actual_filename;
                        if (write_raw(save_path, combined))
                            std::cout << "PARALLEL saved to RESULTS: " << save_path << " (" << combined.size() << " bytes)" << std::endl;
                        else
                            std::cerr << "Failed to save PARALLEL sent_back: " << save_path << std::endl;

                        // Head node-specific processing for combined sent_back (attempt to parse 4D tensor)
                        if (local_IP_eth == head_node_ip_eth)
                        {
                            auto bf16_to_f32 = [](uint16_t v) -> float {
                                uint32_t bits = uint32_t(v) << 16;
                                float out;
                                std::memcpy(&out, &bits, sizeof(out));
                                return out;
                            };

                            auto fp16_to_f32 = [](uint16_t h) -> float {
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
                            };

                            const uint8_t* p = combined.data();
                            int tag_or_ndim = *reinterpret_cast<const int*>(p);
                            p += sizeof(int);
                            const int dtype_tag = (tag_or_ndim < 0) ? tag_or_ndim : -1;
                            const int ndim = (tag_or_ndim < 0) ? 4 : tag_or_ndim;
                            if (ndim == 4)
                            {
                                int dims[4];
                                for (int i = 0; i < 4; ++i) { dims[i] = *reinterpret_cast<const int*>(p); p += sizeof(int); }
                                int batch = dims[0];
                                int depth = dims[1];
                                int rows = dims[2];
                                int cols = dims[3];
                                size_t total_elements = static_cast<size_t>(batch) * depth * rows * cols;
                                auto shard_data = std::make_unique<float[]>(total_elements);
                                if (dtype_tag == -1)
                                {
                                    std::memcpy(shard_data.get(), p, total_elements * sizeof(float));
                                }
                                else if (dtype_tag == -2 || dtype_tag == -3)
                                {
                                    const uint16_t* u16 = reinterpret_cast<const uint16_t*>(p);
                                    for (size_t i = 0; i < total_elements; ++i)
                                    {
                                        shard_data[i] = (dtype_tag == -2) ? fp16_to_f32(u16[i]) : bf16_to_f32(u16[i]);
                                    }
                                }
                                else
                                {
                                    std::cerr << "ERROR: Unsupported dtype_tag in sent_back payload: " << dtype_tag << std::endl;
                                    return;
                                }

                                handle_combine_matrix_shard_list(actual_filename, std::move(shard_data), rows, cols, 0, dtype_tag);
                            }
                        }
                    }
                    else
                    {
                        std::filesystem::path save_path = std::filesystem::path(matrix_shard_folder) / filename;
                        // Try to validate as 4D binary and save via save_matrix_bin; otherwise write raw
                        bool saved = false;
                        if (combined.size() >= static_cast<size_t>(5 * sizeof(int)))
                        {
                            const uint8_t* p = combined.data();
                            int tag_or_ndim = *reinterpret_cast<const int*>(p);
                            // v2 dtype-tagged files must be saved as raw bytes to preserve dtype.
                            if (tag_or_ndim >= 0 && tag_or_ndim == 4)
                            {
                                MatrixResult result;
                                result.dims[0] = *reinterpret_cast<const int*>(p + sizeof(int));
                                result.dims[1] = *reinterpret_cast<const int*>(p + 2 * sizeof(int));
                                result.dims[2] = *reinterpret_cast<const int*>(p + 3 * sizeof(int));
                                result.dims[3] = *reinterpret_cast<const int*>(p + 4 * sizeof(int));
                                size_t total_elements = static_cast<size_t>(result.dims[0]) * result.dims[1] * result.dims[2] * result.dims[3];
                                result.data = std::make_unique<float[]>(total_elements);
                                std::memcpy(result.data.get(), p + 5 * sizeof(int), total_elements * sizeof(float));
                                if (save_matrix_bin(save_path.c_str(), result))
                                {
                                    saved = true;
                                    std::cout << "PARALLEL saved to SHARDS: " << save_path << " (" << combined.size() << " bytes)" << std::endl;
                                }
                            }
                        }

                        if (!saved)
                        {
                            if (write_raw(save_path, combined))
                                std::cout << "PARALLEL saved (raw): " << save_path << " (" << combined.size() << " bytes)" << std::endl;
                            else
                                std::cerr << "Failed to save PARALLEL file: " << save_path << std::endl;
                        }
                    }
                }
                // ETH single-interface file
                else if (!rf.received_data_eth_file.empty())
                {
                    const auto &data = rf.received_data_eth_file;
                    size_t sent_back_pos = filename.find("sent_back=");
                    if (sent_back_pos != std::string::npos)
                    {
                        std::string actual_filename = filename.substr(sent_back_pos + 10);
                        std::filesystem::path save_path = std::filesystem::path(matrix_results_folder) / actual_filename;
                        if (write_raw(save_path, data))
                            std::cout << "ETH sent_back saved to RESULTS: " << save_path << " (" << data.size() << " bytes)" << std::endl;
                        else
                            std::cerr << "Failed to save ETH sent_back: " << save_path << std::endl;

                        // Head node-specific processing for shard combination
                        if (local_IP_eth == head_node_ip_eth)
                        {
                            auto bf16_to_f32 = [](uint16_t v) -> float {
                                uint32_t bits = uint32_t(v) << 16;
                                float out;
                                std::memcpy(&out, &bits, sizeof(out));
                                return out;
                            };

                            auto fp16_to_f32 = [](uint16_t h) -> float {
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
                            };

                            const uint8_t* p = data.data();
                            int tag_or_ndim = *reinterpret_cast<const int*>(p);
                            p += sizeof(int);
                            const int dtype_tag = (tag_or_ndim < 0) ? tag_or_ndim : -1;
                            const int ndim = (tag_or_ndim < 0) ? 4 : tag_or_ndim;
                            if (ndim == 4)
                            {
                                int dims[4];
                                for (int i = 0; i < 4; ++i) { dims[i] = *reinterpret_cast<const int*>(p); p += sizeof(int); }
                                int batch = dims[0];
                                int depth = dims[1];
                                int rows = dims[2];
                                int cols = dims[3];
                                size_t total_elements = static_cast<size_t>(batch) * depth * rows * cols;
                                auto shard_data = std::make_unique<float[]>(total_elements);
                                if (dtype_tag == -1)
                                {
                                    std::memcpy(shard_data.get(), p, total_elements * sizeof(float));
                                }
                                else if (dtype_tag == -2 || dtype_tag == -3)
                                {
                                    const uint16_t* u16 = reinterpret_cast<const uint16_t*>(p);
                                    for (size_t i = 0; i < total_elements; ++i)
                                    {
                                        shard_data[i] = (dtype_tag == -2) ? fp16_to_f32(u16[i]) : bf16_to_f32(u16[i]);
                                    }
                                }
                                else
                                {
                                    std::cerr << "ERROR: Unsupported dtype_tag in sent_back payload: " << dtype_tag << std::endl;
                                    return;
                                }

                                handle_combine_matrix_shard_list(actual_filename, std::move(shard_data), rows, cols, 0, dtype_tag);
                            }
                        }
                    }
                    else
                    {
                        // Regular ETH file: for v2 dtype-tagged files, save raw bytes to preserve dtype.
                        std::filesystem::path save_path = std::filesystem::path(matrix_shard_folder) / filename;
                        if (data.size() >= static_cast<size_t>(5 * sizeof(int)))
                        {
                            const uint8_t* p = data.data();
                            int tag_or_ndim = *reinterpret_cast<const int*>(p);
                            if (tag_or_ndim < 0)
                            {
                                if (write_raw(save_path, data))
                                    std::cout << "ETH saved (raw) to SHARDS: " << save_path << " (" << data.size() << " bytes)" << std::endl;
                                else
                                    std::cerr << "Failed to save ETH file: " << save_path << std::endl;
                            }
                            else if (tag_or_ndim != 4)
                            {
                                std::cerr << "ERROR: Worker sent non-4D tensor: " << filename << " (ndim=" << tag_or_ndim << ")" << std::endl;
                            }
                            else
                            {
                                MatrixResult result;
                                result.dims[0] = *reinterpret_cast<const int*>(p + sizeof(int));
                                result.dims[1] = *reinterpret_cast<const int*>(p + 2 * sizeof(int));
                                result.dims[2] = *reinterpret_cast<const int*>(p + 3 * sizeof(int));
                                result.dims[3] = *reinterpret_cast<const int*>(p + 4 * sizeof(int));
                                size_t total_elements = static_cast<size_t>(result.dims[0]) * result.dims[1] * result.dims[2] * result.dims[3];
                                result.data = std::make_unique<float[]>(total_elements);
                                std::memcpy(result.data.get(), p + 5 * sizeof(int), total_elements * sizeof(float));

                                if (save_matrix_bin(save_path.c_str(), result))
                                    std::cout << "ETH saved to SHARDS: " << save_path << " (" << data.size() << " bytes)" << std::endl;
                                else
                                    std::cerr << "Failed to save ETH file: " << save_path << std::endl;
                            }
                        }
                        else
                        {
                            if (write_raw(save_path, data))
                                std::cout << "ETH saved (raw) to SHARDS: " << save_path << " (" << data.size() << " bytes)" << std::endl;
                            else
                                std::cerr << "Failed to save ETH file: " << save_path << std::endl;
                        }
                    }
                }
                // WiFi single-interface file
                else if (!rf.received_data_wifi_file.empty())
                {
                    const auto &data = rf.received_data_wifi_file;
                    size_t sent_back_pos = filename.find("sent_back=");
                    if (sent_back_pos != std::string::npos)
                    {
                        std::string actual_filename = filename.substr(sent_back_pos + 10);
                        std::filesystem::path save_path = std::filesystem::path(matrix_results_folder) / actual_filename;
                        if (write_raw(save_path, data))
                            std::cout << "WiFi sent_back saved to RESULTS: " << save_path << " (" << data.size() << " bytes)" << std::endl;
                        else
                            std::cerr << "Failed to save WiFi sent_back: " << save_path << std::endl;
                    }
                    else
                    {
                        std::filesystem::path save_path = std::filesystem::path(matrix_shard_folder) / filename;
                        if (write_raw(save_path, data))
                            std::cout << "WiFi saved to SHARDS: " << save_path << " (" << data.size() << " bytes)" << std::endl;
                        else
                            std::cerr << "Failed to save WiFi file: " << save_path << std::endl;
                    }
                }
                else
                {
                    std::cout << "Skipping empty ReservedFiles entry for: " << filename << std::endl;
                }

                // Python `cluster_matrix_v1.py` expects the ACK message to match the saved filename
                // (e.g. `small_matrixA.bin`) for stream transfers.
                if (local_IP_eth != head_node_ip_eth)
                {
                    const bool is_sent_back = filename.rfind("sent_back=", 0) == 0;
                    if (!is_sent_back)
                    {
                        send_ack(filename);
                    }
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
            // WORKER NODE → SEND RESULT BACK TO HEAD (HEAD-SEMANTIC FORMAT)
            // ============================================================
            if (!is_head_node)
            {
                std::string send_back_filename = "sent_back=" + filename;
                std::cout << "Worker sending result back to head: " << send_back_filename << std::endl;

                // Send the on-disk v2 .bin bytes as-is so dtype_tag matches the payload.
                std::vector<uint8_t> buffer;
                {
                    std::ifstream f(local_file_path, std::ios::binary | std::ios::ate);
                    if (!f) {
                        std::cerr << "ERROR: Worker cannot open result file to send_back: " << local_file_path << std::endl;
                        return false;
                    }
                    std::streamsize size = f.tellg();
                    if (size <= 0) {
                        std::cerr << "ERROR: Worker result file empty: " << local_file_path << std::endl;
                        return false;
                    }
                    buffer.resize(static_cast<size_t>(size));
                    f.seekg(0, std::ios::beg);
                    if (!f.read(reinterpret_cast<char*>(buffer.data()), size)) {
                        std::cerr << "ERROR: Worker failed to read result file for send_back: " << local_file_path << std::endl;
                        return false;
                    }
                }

                // Send data to head node via ZeroMQ
                zmq::message_t filename_msg(send_back_filename.data(), send_back_filename.size());
                zmq::message_t data_msg(buffer.data(), buffer.size());

                head_node_sender_eth.send(filename_msg, zmq::send_flags::sndmore);
                head_node_sender_eth.send(data_msg, zmq::send_flags::none);

                std::cout << "Result sent to head node: "
                        << send_back_filename << " (" << buffer.size() << " bytes)" << std::endl;

                return true;
            }

            // ============================================================
            // HEAD NODE → SAVE FILE + TRACK SHARDS
            // ============================================================
            if (is_head_node)
            {
                // Extract shard dimensions
                int shard_rows = save_result.dims[2];
                int shard_cols = save_result.dims[3];
                size_t data_size = shard_rows * shard_cols * sizeof(float);

                // Copy data into unique_ptr for shard processing
                auto shard_data = std::make_unique<float[]>(shard_rows * shard_cols);
                std::memcpy(shard_data.get(),
                            save_result.data.get(),
                            data_size);


                std::cout << "**send_back_file** total_shards: " << total_shards;
                // Process shard through combination handler
                bool result = handle_combine_matrix_shard_list(
                    filename,
                    std::move(shard_data),
                    shard_rows,
                    shard_cols,
                    total_shards,
                    output_dtype_tag
                );

                std::cout << "Head node processed shard: " << filename 
                        << " (" << data_size << " bytes)" << std::endl;

                return result;
            }

            return false;
        }

        bool handle_combine_matrix_shard_list(
            const std::string& filename,
            std::unique_ptr<float[]> data,
            int shard_rows,
            int shard_cols,
            int total_shards,
            int output_dtype_tag
        )
        {
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

                        std::cout << "DEBUG: ALL SHARDS RECEIVED! Combining..."
                                << std::endl;

                        MatrixResult full = is_system2
                            ? combine_matrix_shards_grid_2d(combined)
                            : combine_matrix_shards_2d(combined);

                        if (full.data)
                        {
                            std::string final_path =
                                std::filesystem::path(matrix_shard_folder) /
                                (matrix_name + "_combined.bin");

                            save_matrix_bin(final_path.c_str(), full, combined.output_dtype_tag);
                            send_ack("ACK_combined_matrix_saved");

                            std::cout << "Combined matrix saved: "
                                    << final_path << std::endl;
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
        }

        MatrixResult combine_matrix_shards_2d(const combined_matrix_shards& combined)
        {
            MatrixResult result;

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
                // Save shard bytes to temp file OR directly load from memory
                // You already have bin layout → reuse loader
                const std::vector<uint8_t>& bytes = *data_it;

                // Write to temp buffer-backed stream
                std::string tmp_path = "/dev/shm/tmp_shard_" + std::to_string(*shard_num_it) + ".bin";
                {
                    std::ofstream f(tmp_path, std::ios::binary);
                    f.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                }

                torch::Tensor t = load_matrix_bin_as_torch_view(tmp_path);

                shards.push_back({*shard_num_it, t});
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
        }

        MatrixResult combine_matrix_shards_grid_2d(
            const combined_matrix_shards& combined
        )
        {
            MatrixResult result;

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
        }

        bool matrix_operation(
            const std::string& backend_type,
            const char* matrix_pathA,
            bool transposeA,
            const char* matrix_pathB,
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
                std::string output_filename = get_matrix_output_filename(matrix_pathA, matrix_pathB);
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
                std::string output_path = std::filesystem::path(matrix_shard_folder) / output_filename;

                // Determine output dtype tag from input file headers (v2 dtype_tag or legacy float32).
                const int dtype_tag_A = read_dtype_tag_from_bin_file(matrix_pathA);
                const int dtype_tag_B = read_dtype_tag_from_bin_file(matrix_pathB);
                const int output_dtype_tag = merge_output_dtype_tag(dtype_tag_A, dtype_tag_B);

                // ============================================================
                // BACKEND: LLAMA / GGML / VULKAN
                // ============================================================
                if (backend_type == "llama")
                {
                    std::unique_ptr<float[]> matrix_A = nullptr;
                    std::unique_ptr<float[]> matrix_B = nullptr;
                    int rows_A, cols_A, rows_B, cols_B;
                    int depthA = 1, batchA = 1;
                    int depthB = 1, batchB = 1;

                    // Load matrices
                    matrix_A = load_matrix_bin(matrix_pathA, rows_A, cols_A, batchA, depthA);
                    matrix_B = load_matrix_bin(matrix_pathB, rows_B, cols_B, batchB, depthB);
                    
                    if (!matrix_A || !matrix_B) {
                        std::cerr << "❌ Failed to load input matrices" << std::endl;
                        return false;
                    }

                    // Apply transposes
                    if (transposeA) {
                        matrix_A = (depthA > 1 || batchA > 1)
                            ? matrix_backend_llama.transpose_4d(matrix_A.get(), batchA, depthA, rows_A, cols_A)
                            : matrix_backend_llama.transpose_2d(matrix_A.get(), rows_A, cols_A);
                        std::swap(rows_A, cols_A);
                    }
                    
                    if (transposeB) {
                        matrix_B = (depthB > 1 || batchB > 1)
                            ? matrix_backend_llama.transpose_4d(matrix_B.get(), batchB, depthB, rows_B, cols_B)
                            : matrix_backend_llama.transpose_2d(matrix_B.get(), rows_B, cols_B);
                        std::swap(rows_B, cols_B);
                    }

                    // GGML format: {cols, rows, depth, batch}
                    int dims_a[4] = { cols_A, rows_A, depthA, batchA };
                    int dims_b[4] = { cols_B, rows_B, depthB, batchB };

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

                    // Execute
                    MatrixResult result = matrix_backend_llama.matrix_op_nd(
                        matrix_A.get(), dims_a,
                        matrix_B.get(), dims_b,
                        backend, operation_type
                    );

                    if (result.dims[0] == 0 && result.dims[1] == 0) {
                        result.dims[0] = 1;
                        result.dims[1] = 1;
                    }

                    if (!result.data) {
                        std::cerr << "❌ LLAMA operation failed" << std::endl;
                        op_success = false;
                    } else {
                        // Common save/send_back
                        if (!save_matrix_bin(output_path.c_str(), result, output_dtype_tag)) {
                            std::cerr << "❌ Failed to save result" << std::endl;
                            op_success = false;
                        } else {
                            if (send_back > 0 || send_back < 0)
                            {
                                std::cout << "**matrix_operation** send_back: " << send_back << std::endl;
                                send_back_file(output_path, output_filename, result, send_back, "llama", output_dtype_tag);
                            }
                            
                            op_success = true;
                        }
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

                    // Load tensors
                    torch::Tensor A = load_matrix_bin_as_torch_view(matrix_pathA);
                    torch::Tensor B = load_matrix_bin_as_torch_view(matrix_pathB);
                    
                    if (!A.defined() || !B.defined()) {
                        std::cerr << "❌ Failed to load matrices" << std::endl;
                        op_success = false;
                        throw std::runtime_error("load fail");
                    }

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

                    // Common save/send_back
                    if (!save_matrix_bin(output_path.c_str(), result, output_dtype_tag)) {
                        std::cerr << "❌ Failed to save result" << std::endl;
                        op_success = false;
                    } else {
                        if (send_back > 0)
                            
                            send_back_file(output_path, output_filename, result, send_back, "torch", output_dtype_tag);
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

                    // Load via Torch (I/O only)
                    torch::Tensor tensorA = load_matrix_bin_as_torch_view(matrix_pathA);
                    torch::Tensor tensorB = load_matrix_bin_as_torch_view(matrix_pathB);

                    if (!tensorA.defined() || !tensorB.defined()) {
                        std::cerr << "❌ Failed to load matrices" << std::endl;
                        return false;
                    }

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

                    // Common save/send_back
                    if (!save_matrix_bin(output_path.c_str(), result, output_dtype_tag)) {
                        std::cerr << "❌ Failed to save result" << std::endl;
                        op_success = false;
                    } else {
                        if (send_back > 0)
                            send_back_file(output_path, output_filename, result, send_back, "opencl", output_dtype_tag);
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

        // OLD FUNCTIONS REMOVED - Now using unified matrix_operation() for all backends
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

};

int main()
{
    llama_zmq_server server;
    server.run_server();
}

/*
int main()
{
    const char* path_to_A = "/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/test_2d_a.bin";
    const char* path_to_B = "/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/test_2d_a.bin";

    llama_zmq_server server;
    /*
    // ==========================================
    // Torch timing
    // ==========================================
    auto torch_start = std::chrono::high_resolution_clock::now();

    server.matrix_operation_torch(
        path_to_A,
        false,          // transposeA
        path_to_B,
        true,           // transposeB
        false,          // use_gpu
        0,              // gpu_id
        false,          // send_back
        "mul"
    );

    auto torch_end = std::chrono::high_resolution_clock::now();
    auto torch_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            torch_end - torch_start
        ).count();

    std::cout << "🔥 Torch took "
              << torch_us << " µs ("
              << torch_us / 1e6 << " s)\n";


    // ==========================================
    // GGML timing per backend
    // ==========================================
    for (int gpu_id = 0; gpu_id <= 2; gpu_id++) {

        auto start = std::chrono::high_resolution_clock::now();

        server.matrix_operation_llama(
            path_to_A,
            true,           // transposeA
            path_to_B,
            true,           // transposeB
            true,           // use_gpu
            gpu_id,         // gpu_id
            false,          // send_back
            "mul",
            2               // dim
        );

        auto end = std::chrono::high_resolution_clock::now();
        auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                end - start
            ).count();

        std::cout << "⚙️  GGML gpu_id " << gpu_id
                  << " took " << us << " µs ("
                  << us / 1e6 << " s)\n";
    }

}
*/

/*
int main()
{
    const char* path_to_B_shard_1 = "/dev/shm/matrix_shards/test_2d_a_shard_1.bin";
    const char* path_to_A        = "/dev/shm/matrix_shards/test_2d_a.bin";
    const char* path_to_B        = "/dev/shm/matrix_shards/test_2d_a.bin";

    llama_matrix_backend server;
    {
        std::unique_ptr<float[]> matrix_A = nullptr;
        std::unique_ptr<float[]> matrix_B = nullptr;
        std::unique_ptr<float[]> matrix_B_shard_1 = nullptr;

        int rows_A, cols_A;
        int rows_B, cols_B;
        int rows_B_shard_1, cols_B_shard_1;

        int depthA = 0, batchA = 0;
        int depthB = 0, batchB = 0;
        int depthB_s = 0, batchB_s = 0;

        matrix_A = load_matrix_bin(path_to_A, rows_A, cols_A, batchA, depthA);
        matrix_B = load_matrix_bin(path_to_B, rows_B, cols_B, batchB, depthB);
        matrix_B_shard_1 = load_matrix_bin(
            path_to_B_shard_1,
            rows_B_shard_1, cols_B_shard_1,
            batchB_s, depthB_s
        );

        // GGML dims: [cols, rows, depth, batch]
        int dims2d_a[4] = { cols_A, rows_A, 1, 1 };
        int dims2d_b[4] = { cols_B, rows_B, 1, 1 };
        int dims2d_b_shard_1[4] = { cols_B_shard_1, rows_B_shard_1, 1, 1 };

        std::cout << "Original A: " << rows_A << "x" << cols_A << std::endl;
        std::cout << "Original B (full): " << rows_B << "x" << cols_B << std::endl;
        std::cout << "Original B (shard_1): " << rows_B_shard_1 << "x" << cols_B_shard_1 << std::endl;

        //print_bin_from_torch("/dev/shm/matrix_shards/test_2d_a_shard_1.bin",10,10);
        //print_bin_from_torch("/dev/shm/matrix_shards/test_2d_a.bin",10,10);



        // ---- A @ B (shard_1) ----
        {
            MatrixResult r = server.matrix_op_nd(
                matrix_A.get(), dims2d_a,
                matrix_B_shard_1.get(), dims2d_b_shard_1,
                server.ggml_backends[0], "mul"
            );
            save_matrix_bin("/dev/shm/matrix_shards/shard_AB_out.bin",r);
            torch::Tensor shard_AB = load_matrix_bin_as_torch_view("/dev/shm/matrix_shards/shard_AB_out.bin");
            std::cout << "\nSHARD A@B flat:\n";
            print_tensor_start_flat(shard_AB, "fuck" ,40);
        }
        
    }

    return 0;
}
*/
