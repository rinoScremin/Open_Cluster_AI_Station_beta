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


struct ParallelFile
{
    std::vector<std::string> save_parallel_file_name;      // Filename(s) for parallel save operations
    std::vector<uint8_t> received_data_eth_file;          // Data received via Ethernet interface
    std::vector<uint8_t> received_data_wifi_file;         // Data received via WiFi interface
    
    // Note: The intention is to split single file transfers across both
    //       network interfaces and recombine at destination. Current
    //       implementation status: UNSTABLE/EXPERIMENTAL
};

struct combined_matrix_shards
{
    int total_shards_reserved = 0;        // Number of shards currently received
    int number_of_shards_needed = 0;      // Total shards expected for this matrix
    std::string file_name;                // Base filename (without shard index)
    
    std::vector<int> shard_numbers;       // List of received shard indices
    std::list<std::vector<uint8_t>> received_matrix_data;  // Raw binary data of each shard
    std::list<std::vector<int>> dims_list;                 // Dimensions of each shard [batch, depth, rows, cols]
    
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

        // In your class member variables (public section):
        ParallelFile ParallelFile_struct;
        std::vector<ParallelFile> ParallelFile_struct_list;

        std::vector<combined_matrix_shards> combined_matrix_shards_list;

        std::vector<std::string> save_file_names_eth;    // Ethernet file names
        std::vector<std::string> save_file_names_wifi;   // WiFi file names (NOTE: you wrote "wif" but used "wifi" in code)
        std::vector<std::string> save_file_names_Parallel; // Parallel file names

        // In your class member variables:
        std::vector<std::string> matrix_file_paths;

        std::vector<std::vector<uint8_t>> received_data_eth_files;
        std::vector<std::vector<uint8_t>> received_data_wifi_files;
        std::vector<std::vector<uint8_t>> received_data_parallel_files;

        std::vector<uint8_t> received_data_eth_file;
        std::vector<uint8_t> received_data_wifi_file;
        std::vector<uint8_t> received_data_parallel_file;

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
            // Load configuration from environment variables with defaults
            project_folder = get_env("OPEN_CLUSTER_PROJECT_DIRECTORY", 
                                "/home/rino/Desktop/Open_Cluster_AI_Station_beta/");
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
            
            // Initialize parallel file structures
            ParallelFile_struct = ParallelFile();
            ParallelFile_struct_list = std::vector<ParallelFile>();
            
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

        void send_ack() 
        {
            std::string ack_msg = "ACK";
            zmq::message_t ack(ack_msg.data(), ack_msg.size());
            ack_sender.send(ack, zmq::send_flags::none);
        }

        void run_server() 
        {
            std::cout << "🚀 C++ ZMQ Node Server starting..." << std::endl;
            
            // Start network listener threads for dual-interface operation
            std::thread eth_thread(&llama_zmq_server::listen_ethernet, this);
            std::thread wifi_thread(&llama_zmq_server::listen_wifi, this);
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

        void listen_ethernet() 
        {
            std::cout << "🔌 Ethernet listener thread started" << std::endl;
            
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
                        auto result = file_receiver_eth.recv(message, zmq::recv_flags::dontwait);
                        
                        if (result) 
                        {
                            // Check if more parts are coming in this message
                            more_parts = message.more();
                            parts.push_back(std::move(message));
                        } 
                        else 
                        {
                            // No data available, sleep briefly to prevent CPU spinning
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                            break;
                        }
                    }
                    
                    // Skip if no data was received
                    if (parts.empty()) continue;
                    
                    // Handle single-part messages (typically commands)
                    if (parts.size() == 1) 
                    {
                        std::string command = parts[0].to_string();
                        size_t server_cmd_pos = command.find("server_command=");
                        
                        if (server_cmd_pos != std::string::npos) 
                        {
                            // Extract and store server control command
                            std::string server_cmd = command.substr(server_cmd_pos + 15);
                            std::lock_guard<std::mutex> lock(server_commands_mutex);
                            received_data_eth_server_command.push_back(server_cmd);
                            
                            std::cout << "📋 Ethernet: Received server command" << std::endl;
                        } 
                        else 
                        {
                            // Store as regular Linux/system command
                            std::lock_guard<std::mutex> lock(linux_commands_mutex);
                            received_data_eth_linux_command.push_back(command);
                            
                            std::cout << "💻 Ethernet: Received Linux command" << std::endl;
                        }
                    }
                    // Handle two-part messages (typically file transfers)
                    else if (parts.size() == 2) 
                    {
                        std::string filename_header = parts[0].to_string();
                        size_t parallel_send_pos = filename_header.find("P_SEND_");   
                        
                        // Handle parallel file transfers (Ethernet half of WiFi+Ethernet split)
                        if (parallel_send_pos != std::string::npos) 
                        {
                            std::string actual_filename = filename_header.substr(parallel_send_pos + 7);
                            
                            // Extract file data from second message part
                            const uint8_t* data = static_cast<const uint8_t*>(parts[1].data());
                            size_t data_size = parts[1].size();
                            
                            std::lock_guard<std::mutex> lock(file_data_mutex);
                            
                            // Check for existing parallel file structs
                            if (ParallelFile_struct_list.empty())
                            {
                                // Create new parallel file structure for WiFi+Ethernet split file
                                ParallelFile new_parallel_file;
                                new_parallel_file.received_data_eth_file.assign(data, data + data_size);
                                new_parallel_file.save_parallel_file_name.push_back(actual_filename);
                                ParallelFile_struct_list.push_back(new_parallel_file);
                                
                                std::cout << "📂 Ethernet: Started parallel file '" 
                                        << actual_filename << "' (Ethernet half)" << std::endl;
                            }
                            else
                            {
                                // Find existing parallel file structure and add Ethernet data
                                bool file_found = false;
                                for (ParallelFile& pf : ParallelFile_struct_list)
                                {
                                    if(!pf.save_parallel_file_name.empty() && 
                                    pf.save_parallel_file_name[0] == actual_filename)
                                    {
                                        pf.received_data_eth_file.assign(data, data + data_size);
                                        file_found = true;
                                        
                                        std::cout << "📂 Ethernet: Added to parallel file '" 
                                                << actual_filename << "'" << std::endl;
                                        break;
                                    }
                                }
                                
                                if (!file_found)
                                {
                                    std::cout << "⚠️ Ethernet: Parallel file not found '" 
                                            << actual_filename << "'" << std::endl;
                                }
                            }
                        }
                        // Handle regular file transfers (complete file over Ethernet)
                        else
                        {
                            std::string filename = parts[0].to_string();
                            
                            // Extract file data
                            std::vector<uint8_t> file_data;
                            const uint8_t* data_ptr = static_cast<const uint8_t*>(parts[1].data());
                            size_t data_size = parts[1].size();
                            file_data.assign(data_ptr, data_ptr + data_size);
                            
                            // Store file metadata and data
                            {
                                std::lock_guard<std::mutex> lock(file_data_mutex);
                                save_file_names_eth.push_back(filename);
                                received_data_eth_files.push_back(file_data);
                            }
                            
                            std::cout << "📁 Ethernet: Received file '" 
                                    << filename << "' (" << data_size << " bytes)" << std::endl;
                        }
                        
                        // Process any completed files
                        save_file_handler();
                    }
                    // Handle unexpected message formats
                    else 
                    {
                        std::cout << "⚠️ Ethernet: Unexpected message format - " 
                                << parts.size() << " parts received" << std::endl;
                    }
                } 
                catch (const std::exception& e) 
                {
                    std::cerr << "❌ Ethernet listener error: " << e.what() << std::endl;
                    // Brief pause on error to prevent tight error loops
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
            
            std::cout << "🔌 Ethernet listener thread stopping" << std::endl;
        }

        void listen_wifi() 
        {
            while (server_running) 
            {
                try 
                {
                    std::vector<zmq::message_t> parts;
                    bool more = true;
                    
                    while (more && server_running) 
                    {
                        zmq::message_t message;
                        auto result = file_receiver_wifi.recv(message, zmq::recv_flags::dontwait);
                        
                        if (result) 
                        {
                            more = message.more();
                            parts.push_back(std::move(message));
                        } 
                        else 
                        {
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                            break;
                        }
                    }
                    if (parts.empty()) continue;
                    
                    if (parts.size() == 1) 
                    {
                        std::string received_data_wifi_command = parts[0].to_string();
                        size_t pos = received_data_wifi_command.find("server_command=");
                        
                        if (pos != std::string::npos) 
                        {
                            std::lock_guard<std::mutex> lock(server_commands_mutex);
                            received_data_wifi_server_command.push_back(received_data_wifi_command.substr(pos + 15));
                        } 
                        else 
                        {
                            std::lock_guard<std::mutex> lock(wifi_commands_mutex);
                            received_data_wifi_linux_command.push_back(received_data_wifi_command);
                        }
                    }
                    else if (parts.size() == 2) 
                    {
                        std::string fileName = parts[0].to_string();
                        size_t p_send_pos = fileName.find("P_SEND_");   
                        
                        if (p_send_pos != std::string::npos) 
                        {
                            std::string actual_filename = fileName.substr(p_send_pos + 7);
                            
                            // Get the data from part 1
                            const uint8_t* data = static_cast<const uint8_t*>(parts[1].data());
                            size_t size = parts[1].size();
                            
                            std::lock_guard<std::mutex> lock(file_data_mutex);
                            
                            // Check if we have both halves (Ethernet + WiFi)
                            if (ParallelFile_struct_list.size() == 0)
                            {
                                // if struct list is zero then make new ParallelFile struct 
                                // add the data and the file name 
                                ParallelFile ParallelFile_struct = ParallelFile(); 
                                ParallelFile_struct.received_data_wifi_file.assign(data, data + size);
                                ParallelFile_struct.save_parallel_file_name.push_back(actual_filename);
                                ParallelFile_struct_list.push_back(ParallelFile_struct);
                            }
                            else
                            {
                                for (ParallelFile& ParallelFile_struct : ParallelFile_struct_list)
                                {
                                    if(ParallelFile_struct.save_parallel_file_name[0] == actual_filename)
                                    {
                                        ParallelFile_struct.received_data_wifi_file.assign(data, data + size);
                                    }
                                } 
                            }
                        }
                        else
                        {
                            std::string filename = parts[0].to_string();
                            std::vector<uint8_t> file_data;
                            std::lock_guard<std::mutex> lock(file_data_mutex);
                            const uint8_t* data_ptr = static_cast<const uint8_t*>(parts[1].data());
                            size_t data_size = parts[1].size();
                            file_data.assign(data_ptr, data_ptr + data_size);
                            save_file_names_wifi.push_back(filename);
                             received_data_wifi_files.push_back(file_data);
                        }
                        
                        if (p_send_pos != std::string::npos)
                        {
                            std::cout << "save_file_handler: save_file_handler() run from listen_wifi() function\n";
                        }
                        else
                        {
                            save_file_handler();
                        }            
                    }
                    else 
                    {
                        std::cerr << "⚠️ WiFi: Received message with " << parts.size() 
                                << " parts, expected 1 or 2" << std::endl;
                    }
                } 
                catch (const std::exception& e) 
                {
                    std::cerr << "❌ WiFi receiver error: " << e.what() << std::endl;
                }
            }
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
                    
                    // Store for later use in result distribution
                    send_back_number_of_shards = send_back;
                    
                    bool operation_success = false;
                    std::string backend_name;

                    // Dispatch to appropriate backend implementation
                    if (command_type == "llama") 
                    {
                        backend_name = "LLaMA/Vulkan";
                        operation_success = matrix_operation_llama(
                            command_args[1].c_str(),  // Matrix A path
                            transposeA,
                            command_args[3].c_str(),  // Matrix B path
                            transposeB,
                            use_gpu, 
                            gpu_id, 
                            send_back,
                            operation_type, 
                            n_dims
                        );
                    } 
                    else if (command_type == "opencl") 
                    {
                        backend_name = "OpenCL";
                        operation_success = matrix_operation_openCL(
                            command_args[1].c_str(),  // Matrix A path
                            transposeA,
                            command_args[3].c_str(),  // Matrix B path
                            transposeB,
                            gpu_id, 
                            send_back,
                            operation_type,
                            openCL_kernel_matmul  // Kernel selection
                        );
                    } 
                    else if (command_type == "torch") 
                    {
                        backend_name = "PyTorch";
                        
                        // Validate GPU availability for PyTorch
                        bool torch_gpu_available = false;
                        #ifdef USE_CUDA
                        torch_gpu_available = torch::cuda::is_available();
                        #endif
                        
                        // Adjust GPU usage if requested but unavailable
                        if (use_gpu && !torch_gpu_available) {
                            std::cout << "⚠️  GPU requested for PyTorch but CUDA unavailable. Falling back to CPU." << std::endl;
                            use_gpu = false;
                            gpu_id = 0;  // Reset GPU ID for CPU mode
                        }

                        operation_success = matrix_operation_torch(
                            command_args[1].c_str(),  // Matrix A path
                            transposeA,
                            command_args[3].c_str(),  // Matrix B path
                            transposeB,
                            use_gpu, 
                            gpu_id,                    // Only used if use_gpu == true
                            send_back,
                            operation_type
                        );
                    }

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
            size_t shard_pos = filename.find("_shard_");
            
            if (shard_pos != std::string::npos) {
                // Extract the base matrix name (everything before "_shard_")
                // Example: "matrixA_shard_42.bin" -> "matrixA"
                std::string matrix_name = filename.substr(0, shard_pos);
                
                // Extract the shard number portion (everything after "_shard_")
                // 7 = length of "_shard_"
                std::string shard_part = filename.substr(shard_pos + 7);
                
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
            // Local copies for thread-safe processing
            std::vector<std::string> local_eth_file_names;
            std::vector<std::vector<uint8_t>> local_file_eth_data;
            std::vector<std::string> local_wifi_file_names;
            std::vector<std::vector<uint8_t>> local_file_wifi_data;
            std::vector<ParallelFile> local_parallel_structs;

            // Critical section: acquire lock and move data to local variables
            {   
                std::lock_guard<std::mutex> lock(file_data_mutex);

                // Check if there's any data to process
                if (save_file_names_eth.empty() && received_data_eth_files.empty() 
                    && save_file_names_wifi.empty() && received_data_wifi_files.empty() 
                    && ParallelFile_struct_list.empty()) 
                {
                    std::cout << "No files to save" << std::endl;
                    return;
                }

                // Move data to local copies (thread-safe transfer)
                local_eth_file_names = std::move(save_file_names_eth);
                local_file_eth_data = std::move(received_data_eth_files);
                local_wifi_file_names = std::move(save_file_names_wifi);
                local_file_wifi_data = std::move(received_data_wifi_files);
                local_parallel_structs = std::move(ParallelFile_struct_list);

                // Clear the original containers
                save_file_names_eth.clear(); 
                received_data_eth_files.clear();
                save_file_names_wifi.clear();
                received_data_wifi_files.clear();
                ParallelFile_struct_list.clear();

                std::cout << "Processing: " 
                        << local_eth_file_names.size() << " ETH files, "
                        << local_wifi_file_names.size() << " WiFi files, "
                        << local_parallel_structs.size() << " parallel files" << std::endl;
            }

            // Process ETH files
            for (size_t i = 0; i < local_eth_file_names.size(); i++)
            {
                const std::string& filename = local_eth_file_names[i];
                const std::vector<uint8_t>& data = local_file_eth_data[i];

                size_t sent_back_pos = filename.find("sent_back=");
                
                if (sent_back_pos != std::string::npos)
                {
                    // Handle sent_back files (worker-to-head communication)
                    std::string actual_filename = filename.substr(sent_back_pos + 10);
                    std::filesystem::path save_path = std::filesystem::path(matrix_results_folder) / actual_filename;
                    
                    std::cout << "ETH sent_back: " << actual_filename 
                            << " (" << data.size() << " bytes)" << std::endl;
                    
                    std::filesystem::create_directories(save_path.parent_path());
                    
                    // Save to results folder
                    std::ofstream file(save_path, std::ios::binary);
                
                    if (file.is_open())
                    {
                        file.write(reinterpret_cast<const char*>(data.data()), data.size());
                        file.close();
                        std::cout << "ETH saved to RESULTS: " << save_path << std::endl;

                        // Head node-specific processing for matrix shard combination
                        if (local_IP_eth == head_node_ip_eth)
                        {
                            const uint8_t* p = data.data();

                            // Read tensor dimensions
                            int ndim = *reinterpret_cast<const int*>(p);
                            p += sizeof(int);

                            if (ndim != 4)
                            {
                                std::cerr << "ERROR: Expected 4D tensor, got ndim=" << ndim << std::endl;
                                return;
                            }

                            // Read all 4 dimensions
                            int dims[4];
                            for (int i = 0; i < 4; ++i)
                            {
                                dims[i] = *reinterpret_cast<const int*>(p);
                                p += sizeof(int);
                            }

                            int batch = dims[0];
                            int depth = dims[1];
                            int rows  = dims[2];
                            int cols  = dims[3];

                            // Compute total elements
                            size_t total_elements =
                                static_cast<size_t>(batch) *
                                static_cast<size_t>(depth) *
                                static_cast<size_t>(rows) *
                                static_cast<size_t>(cols);

                            // Allocate buffer and copy data
                            auto shard_data = std::make_unique<float[]>(total_elements);
                            std::memcpy(shard_data.get(), p, total_elements * sizeof(float));

                            // Hand off to shard combination logic
                            handle_combine_matrix_shard_list(
                                actual_filename,
                                std::move(shard_data),
                                rows,
                                cols,
                                0   // shard index
                            );
                        }
                    }
                    else
                    {
                        std::cerr << "Failed to save ETH sent_back: " << save_path << std::endl;
                    }
                }
                else
                {
                    // Regular ETH file - validate and normalize format
                    std::filesystem::path save_path = std::filesystem::path(matrix_shard_folder) / filename;  
                    
                    // Validate tensor format
                    const uint8_t* p = data.data();  
                    int ndim = *reinterpret_cast<const int*>(p);  
                    
                    if (ndim != 4) {  
                        std::cerr << "ERROR: Worker sent non-4D tensor: " << filename   
                                << " (ndim=" << ndim << ")" << std::endl;  
                        return;  
                    }  
                    
                    // Create MatrixResult from received data  
                    MatrixResult result;  
                    result.dims[0] = *reinterpret_cast<const int*>(p + sizeof(int));  
                    result.dims[1] = *reinterpret_cast<const int*>(p + 2 * sizeof(int));  
                    result.dims[2] = *reinterpret_cast<const int*>(p + 3 * sizeof(int));  
                    result.dims[3] = *reinterpret_cast<const int*>(p + 4 * sizeof(int));  
                    
                    size_t total_elements = static_cast<size_t>(result.dims[0]) *   
                                            static_cast<size_t>(result.dims[1]) *   
                                            static_cast<size_t>(result.dims[2]) *   
                                            static_cast<size_t>(result.dims[3]);  
                    
                    result.data = std::make_unique<float[]>(total_elements);  
                    std::memcpy(result.data.get(), p + 5 * sizeof(int), total_elements * sizeof(float));  
                    
                    // Use save_matrix_bin for consistent 4D format  
                    if (save_matrix_bin(save_path.c_str(), result)) {  
                        std::cout << "ETH saved to SHARDS: " << save_path  
                                << " (" << data.size() << " bytes)" << std::endl;  
                    } else {  
                        std::cerr << "Failed to save ETH file: " << save_path << std::endl;  
                    }  
                }
            }

            // Process WiFi files
            for (size_t i = 0; i < local_wifi_file_names.size(); i++)
            {
                const std::string& filename = local_wifi_file_names[i];
                const std::vector<uint8_t>& data = local_file_wifi_data[i];

                size_t sent_back_pos = filename.find("sent_back=");
                
                if (sent_back_pos != std::string::npos)
                {
                    // Handle sent_back files
                    std::string actual_filename = filename.substr(sent_back_pos + 10);
                    std::filesystem::path save_path = std::filesystem::path(matrix_results_folder) / actual_filename;
                    
                    std::cout << "WiFi sent_back: " << actual_filename 
                            << " (" << data.size() << " bytes)" << std::endl;
                    
                    std::filesystem::create_directories(save_path.parent_path());
                    
                    std::ofstream file(save_path, std::ios::binary);
                    if (file.is_open())
                    {
                        file.write(reinterpret_cast<const char*>(data.data()), data.size());
                        file.close();
                        std::cout << "WiFi saved to RESULTS: " << save_path << std::endl;
                    }
                    else
                    {
                        std::cerr << "Failed to save WiFi sent_back: " << save_path << std::endl;
                    }
                }
                else
                {
                    // Regular WiFi file
                    std::filesystem::path save_path = std::filesystem::path(matrix_shard_folder) / filename;
                    std::filesystem::create_directories(save_path.parent_path());

                    std::ofstream file(save_path, std::ios::binary);
                    if (file.is_open())
                    {
                        file.write(reinterpret_cast<const char*>(data.data()), data.size());
                        file.close();
                        std::cout << "WiFi saved to SHARDS: " << save_path
                                << " (" << data.size() << " bytes)" << std::endl;
                    }
                    else
                    {
                        std::cerr << "Failed to save WiFi file: " << save_path << std::endl;
                    }
                }
            }

            // TODO: Parallel file processing - experiments not yet finished
            // Process parallel files (ETH + WiFi combined)
            // Process parallel files (ETH + WiFi combined)
            if (!local_parallel_structs.empty())
            {
                for (size_t i = 0; i < local_parallel_structs.size(); i++)
                {
                    ParallelFile& pf = local_parallel_structs[i];

                    // Validate struct has required data
                    if (!pf.save_parallel_file_name.empty() &&
                        !pf.received_data_eth_file.empty() &&
                        !pf.received_data_wifi_file.empty())
                    {
                        std::string filename = pf.save_parallel_file_name[0];

                        // Combine ETH + WiFi data
                        std::vector<uint8_t> combined_data;
                        combined_data.reserve(pf.received_data_eth_file.size() + pf.received_data_wifi_file.size());
                        combined_data.insert(combined_data.end(),
                                            pf.received_data_eth_file.begin(),
                                            pf.received_data_eth_file.end());
                        combined_data.insert(combined_data.end(),
                                            pf.received_data_wifi_file.begin(),
                                            pf.received_data_wifi_file.end());

                        // Determine save path based on file type
                        size_t sent_back_pos = filename.find("sent_back=");
                        std::filesystem::path save_path;
                        
                        if (sent_back_pos != std::string::npos)
                        {
                            std::string actual_filename = filename.substr(sent_back_pos + 10);
                            save_path = std::filesystem::path(matrix_results_folder) / actual_filename;
                            std::cout << "PARALLEL sent_back: " << actual_filename << std::endl;
                        }
                        else
                        {
                            save_path = std::filesystem::path(matrix_shard_folder) / filename;
                            std::cout << "PARALLEL regular: " << filename << std::endl;
                        }
                        
                        std::filesystem::create_directories(save_path.parent_path());

                        std::ofstream file(save_path, std::ios::binary);
                        if (file.is_open())
                        {
                            file.write(reinterpret_cast<const char*>(combined_data.data()), combined_data.size());
                            file.close();
                            std::cout << "PARALLEL saved: " << save_path
                                    << " (" << combined_data.size() << " bytes)" << std::endl;
                        }
                        else
                        {
                            std::cerr << "Failed to save PARALLEL file: " << save_path << std::endl;
                        }
                    }
                }
            }

            // Send acknowledgment if this is not the head node
            if (local_IP_eth != head_node_ip_eth)
            {
                send_ack();
            }

            std::cout << "Save file handler completed" << std::endl;
        }

        bool send_back_file(const std::string& local_file_path,
                            const std::string& filename,
                            MatrixResult& save_result,
                            int total_shards,
                            const std::string& selected_backend)
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

                std::vector<uint8_t> buffer;

                // Network contract: Always send logical 4D tensor
                int ndim = 4;
                buffer.insert(buffer.end(),
                    reinterpret_cast<uint8_t*>(&ndim),
                    reinterpret_cast<uint8_t*>(&ndim) + sizeof(int));

                // Normalize dimensions according to backend format
                int batch, depth, shard_rows, shard_cols;
                if (selected_backend == "llama")  // for incase things go fucked does nothing
                {  
                    // GGML format: {cols, rows, depth, batch}
                    batch      = save_result.dims[0];  // batch is index 3
                    depth      = save_result.dims[1];  // depth is index 2
                    shard_rows = save_result.dims[2];  // rows is index 1
                    shard_cols = save_result.dims[3];  // cols is index 0
                }  
                else if (selected_backend == "torch")  // for incase things go fucked does nothing 
                {  
                    // Torch format: {batch, depth, rows, cols}
                    batch      = save_result.dims[0];
                    depth      = save_result.dims[1];
                    shard_rows = save_result.dims[2];
                    shard_cols = save_result.dims[3];
                }

                // IMPORTANT: save_result.dims[] must already reflect the logical shape
                // If not, this indicates a backend issue, not a network issue
                int dims[4] = { batch, depth, shard_rows, shard_cols };

                // Insert all 4 dimensions into buffer
                for (int i = 0; i < 4; i++)
                {
                    buffer.insert(buffer.end(),
                        reinterpret_cast<uint8_t*>(&dims[i]),
                        reinterpret_cast<uint8_t*>(&dims[i]) + sizeof(int));
                }

                // Calculate and insert data payload
                size_t total_elements =
                    static_cast<size_t>(batch) *
                    static_cast<size_t>(depth) *
                    static_cast<size_t>(shard_rows) *
                    static_cast<size_t>(shard_cols);

                buffer.insert(buffer.end(),
                    reinterpret_cast<uint8_t*>(save_result.data.get()),
                    reinterpret_cast<uint8_t*>(save_result.data.get()) +
                    total_elements * sizeof(float));

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

                // Process shard through combination handler
                bool result = handle_combine_matrix_shard_list(
                    filename,
                    std::move(shard_data),
                    shard_rows,
                    shard_cols,
                    total_shards
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
            int total_shards  
        )  
        {  
            // Extract base matrix name and shard index from filename
            // Example: "matrix_0.bin" → "matrix", 0
            auto [matrix_name, shard_num] = get_matrix_name_and_shard_number(filename);  

            // ============================================================
            // BUILD SHARD BYTES WITH METADATA
            // ============================================================
            // Create buffer containing: [ndim, dims[4], data]
            std::vector<uint8_t> shard_bytes;  

            // Always use 4D format: batch, depth, rows, cols
            int ndim = 4; 
            shard_bytes.insert(  
                shard_bytes.end(),  
                reinterpret_cast<uint8_t*>(&ndim),  
                reinterpret_cast<uint8_t*>(&ndim) + sizeof(int)  
            );  

            // Dimensions for a single shard: batch=1, depth=1, rows, cols
            int dims[4] = {1, 1, shard_rows, shard_cols};  
            for (int i = 0; i < 4; ++i)  
            {  
                shard_bytes.insert(  
                    shard_bytes.end(),  
                    reinterpret_cast<uint8_t*>(&dims[i]),  
                    reinterpret_cast<uint8_t*>(&dims[i]) + sizeof(int)  
                );  
            }  

            // Append actual matrix data
            size_t data_size = static_cast<size_t>(shard_rows) * shard_cols * sizeof(float);  
            shard_bytes.insert(  
                shard_bytes.end(),  
                reinterpret_cast<uint8_t*>(data.get()),  
                reinterpret_cast<uint8_t*>(data.get()) + data_size  
            );  

            // ============================================================
            // TRACK SHARD (check if we're already collecting for this matrix)
            // ============================================================
            for (auto& combined : combined_matrix_shards_list)  
            {  
                // Extract base name from the tracked entry
                auto [combined_name, _] = get_matrix_name_and_shard_number(combined.file_name);  

                // Found existing entry for this matrix
                if (combined_name == matrix_name)  
                {  
                    // Update tracking information
                    combined.total_shards_reserved++;  
                    combined.shard_numbers.push_back(shard_num);  
                    combined.received_matrix_data.push_back(std::move(shard_bytes));  
                    
                    // Store dimensions for this shard
                    std::vector<int> shard_dims = {1, 1, shard_rows, shard_cols};  
                    combined.dims_list.push_back(shard_dims);  

                    // ============================================================
                    // CHECK IF ALL SHARDS HAVE ARRIVED
                    // ============================================================
                    if (combined.total_shards_reserved == combined.number_of_shards_needed)  
                    {  
                        std::cout << "All shards received. Combining matrix: " << matrix_name << std::endl;  

                        // Combine all shards into full matrix
                        MatrixResult full = combine_matrix_shards_2d(combined);  

                        if (!full.data)  
                        {  
                            std::cerr << "ERROR: Failed to combine matrix shards for: " << matrix_name << std::endl;  
                        }  
                        else  
                        {  
                            // Save combined matrix to file
                            std::string final_path =  
                                std::filesystem::path(matrix_shard_folder) /  
                                (combined_name + "_combined.bin");  

                            save_matrix_bin(final_path.c_str(), full);  
                            std::cout << "Combined matrix saved: " << final_path << std::endl;  
                        }  

                        // Remove completed entry from tracking list
                        combined_matrix_shards_list.erase(  
                            std::remove_if(  
                                combined_matrix_shards_list.begin(),  
                                combined_matrix_shards_list.end(),  
                                [&](const combined_matrix_shards& c)  
                                {  
                                    auto [n, __] = get_matrix_name_and_shard_number(c.file_name);  
                                    return n == combined_name;  
                                }),  
                            combined_matrix_shards_list.end()  
                        );  
                    }  

                    return true;  
                }  
            }  

            // ============================================================
            // FIRST SHARD FOR THIS MATRIX (create new tracking entry)
            // ============================================================
            combined_matrix_shards combined;  
            combined.file_name = matrix_name;  
            combined.number_of_shards_needed = total_shards;  
            combined.total_shards_reserved = 1;  
            combined.shard_numbers.push_back(shard_num);  
            combined.received_matrix_data.push_back(std::move(shard_bytes));  
            
            // Store dimensions for the first shard  
            std::vector<int> shard_dims = {1, 1, shard_rows, shard_cols};  
            combined.dims_list.push_back(shard_dims);  

            // Add new entry to tracking list
            combined_matrix_shards_list.push_back(std::move(combined));  

            std::cout << "Started tracking new matrix: " << matrix_name 
                    << " (shard " << shard_num << " of " << total_shards << ")" << std::endl;
            
            return true;  
        }

        MatrixResult combine_matrix_shards_2d(const combined_matrix_shards& combined)
        {
            MatrixResult result;

            // Early return if no data received
            if (combined.received_matrix_data.empty()) {
                return result;
            }

            // ============================================================
            // STEP 1: SORT SHARDS BY SHARD NUMBER
            // ============================================================
            // Create vector of (shard_number, shard_data_pointer) pairs
            std::vector<std::pair<int, const std::vector<uint8_t>*>> sorted_shards;
            
            auto shard_num_it = combined.shard_numbers.begin();
            auto data_it      = combined.received_matrix_data.begin();

            // Pair each shard number with its corresponding data
            for (; shard_num_it != combined.shard_numbers.end() &&
                data_it      != combined.received_matrix_data.end();
                ++shard_num_it, ++data_it)
            {
                sorted_shards.emplace_back(*shard_num_it, &(*data_it));
            }

            // Sort by shard number to ensure correct concatenation order
            std::sort(sorted_shards.begin(), sorted_shards.end(),
                    [](auto& a, auto& b){ return a.first < b.first; });

            // ============================================================
            // STEP 2: COMPUTE TOTAL DIMENSIONS
            // ============================================================
            // Calculate total rows (sum of all shard rows) 
            // and total cols (maximum shard columns)
            int total_rows = 0;
            int total_cols = 0;

            for (const auto& [shard_num, shard_bytes] : sorted_shards) {
                const uint8_t* p = shard_bytes->data();

                // Read metadata from shard
                int ndim = *reinterpret_cast<const int*>(p);
                p += sizeof(int);

                std::vector<int> dims(ndim);
                for (int i = 0; i < ndim; ++i) {
                    dims[i] = *reinterpret_cast<const int*>(p);
                    p += sizeof(int);
                }

                int rows = dims[2];  // rows dimension
                int cols = dims[3];  // cols dimension

                total_rows += rows;
                total_cols = std::max(total_cols, cols);
            }

            std::cout << "Combining " << sorted_shards.size() << " shards into "
                    << total_rows << "x" << total_cols << " matrix" << std::endl;

            // ============================================================
            // STEP 3: ALLOCATE OUTPUT MATRIX
            // ============================================================
            // Set dimensions: single batch and depth, combined rows and cols
            result.dims[0] = 1;  // batch
            result.dims[1] = 1;  // depth
            result.dims[2] = total_rows;
            result.dims[3] = total_cols;

            // Allocate memory for combined matrix
            result.data = std::make_unique<float[]>(
                static_cast<size_t>(total_rows) * total_cols
            );

            // ============================================================
            // STEP 4: COPY SHARDS (ROW-MAJOR CONCATENATION)
            // ============================================================
            int row_offset = 0;  // Track where to place next shard

            for (const auto& [shard_num, shard_bytes] : sorted_shards) {
                const uint8_t* p = shard_bytes->data();

                // Read shard metadata
                int ndim = *reinterpret_cast<const int*>(p);
                p += sizeof(int);

                std::vector<int> dims(ndim);
                for (int i = 0; i < ndim; ++i) {
                    dims[i] = *reinterpret_cast<const int*>(p);
                    p += sizeof(int);
                }

                int rows = dims[2];
                int cols = dims[3];
                
                const float* shard_data = reinterpret_cast<const float*>(p);

                std::cout << "  Copying shard " << shard_num << ": " 
                        << rows << "x" << cols << std::endl;

                // Row-major copy: concatenate shards vertically
                for (int r = 0; r < rows; ++r) {
                    std::memcpy(
                        result.data.get() + (row_offset + r) * total_cols,
                        shard_data + r * cols,
                        sizeof(float) * cols
                    );
                }

                row_offset += rows;
            }

            std::cout << "Matrix combination complete" << std::endl;
            
            return result;
        }

        bool matrix_operation_llama(
            const char* matrix_pathA,
            bool transposeA,
            const char* matrix_pathB,
            bool transposeB,
            bool use_gpu,
            int gpu_id,
            int send_back,
            const std::string& matrix_operation_select,
            int dim
        )
        {
            std::cout << "MATRIX OPERATION LLAMA BACKEND" << std::endl;

            std::unique_ptr<float[]> matrix_A = nullptr;
            std::unique_ptr<float[]> matrix_B = nullptr;
            int rows_A, cols_A, rows_B, cols_B;
            int depthA = 1, batchA = 1;
            int depthB = 1, batchB = 1;

            // ============================================================
            // STEP 1: LOAD MATRICES FROM FILES
            // ============================================================
            matrix_A = load_matrix_bin(matrix_pathA, rows_A, cols_A, batchA, depthA);
            matrix_B = load_matrix_bin(matrix_pathB, rows_B, cols_B, batchB, depthB);
            
            if (!matrix_A || !matrix_B) {
                std::cerr << "ERROR: Failed to load input matrices" << std::endl;
                return false;
            }

            std::cout << "Matrix A loaded: " << rows_A << "x" << cols_A 
                    << " (batch=" << batchA << ", depth=" << depthA << ")" << std::endl;
            std::cout << "Matrix B loaded: " << rows_B << "x" << cols_B 
                    << " (batch=" << batchB << ", depth=" << depthB << ")" << std::endl;

            // ============================================================
            // STEP 2: APPLY OPTIONAL TRANSPOSE OPERATIONS
            // ============================================================
            if (transposeA) {
                std::cout << "Transposing Matrix A" << std::endl;
                matrix_A = (depthA > 1 || batchA > 1)
                    ? matrix_backend_llama.transpose_4d(matrix_A.get(), batchA, depthA, rows_A, cols_A)
                    : matrix_backend_llama.transpose_2d(matrix_A.get(), rows_A, cols_A);
                std::swap(rows_A, cols_A);
                std::cout << "Matrix A transposed to: " << rows_A << "x" << cols_A << std::endl;
            }
            
            if (transposeB) {
                std::cout << "Transposing Matrix B" << std::endl;
                matrix_B = (depthB > 1 || batchB > 1)
                    ? matrix_backend_llama.transpose_4d(matrix_B.get(), batchB, depthB, rows_B, cols_B)
                    : matrix_backend_llama.transpose_2d(matrix_B.get(), rows_B, cols_B);
                std::swap(rows_B, cols_B);
                std::cout << "Matrix B transposed to: " << rows_B << "x" << cols_B << std::endl;
            }

            // ============================================================
            // STEP 3: CONFIGURE GGML DIMENSIONS
            // ============================================================
            // GGML uses format: {cols, rows, depth, batch}
            int dims_a[4] = { cols_A, rows_A, depthA, batchA };
            int dims_b[4] = { cols_B, rows_B, depthB, batchB };

            // Acquire lock for thread-safe backend access
            std::lock_guard<std::mutex> lock(matrix_backend_llama.backends_mutex);
            
            // Select appropriate backend (GPU if specified, otherwise default)
            ggml_backend_t backend =
                (use_gpu && gpu_id >= 0 &&
                gpu_id < (int)matrix_backend_llama.ggml_backends.size())
                ? matrix_backend_llama.ggml_backends[gpu_id]
                : matrix_backend_llama.ggml_backends.back();

            std::cout << "Using backend: " << (use_gpu ? "GPU #" + std::to_string(gpu_id) : "CPU")
                    << " for operation: " << matrix_operation_select << std::endl;

            // ============================================================
            // STEP 4: PERFORM MATRIX OPERATION
            // ============================================================
            MatrixResult result = matrix_backend_llama.matrix_op_nd(
                matrix_A.get(), dims_a,
                matrix_B.get(), dims_b,
                backend, matrix_operation_select
            );

            // Ensure result dimensions are in consistent 4D format
            // Some backends may return 2D dims {rows, cols, 0, 0}
            if (result.dims[0] == 0 && result.dims[1] == 0) {
                // Convert to 4D: {1, 1, rows, cols}
                result.dims[0] = 1;  // batch
                result.dims[1] = 1;  // depth
                // dims[2] and dims[3] already contain rows and cols
            }

            if (!result.data) {
                std::cerr << "ERROR: Matrix operation failed" << std::endl;
                return false;
            }

            std::cout << "Operation result dimensions: " 
                    << result.dims[0] << "x" << result.dims[1] << "x"
                    << result.dims[2] << "x" << result.dims[3] << std::endl;

            // ============================================================
            // STEP 5: SAVE RESULT LOCALLY
            // ============================================================
            std::string output_filename = get_matrix_output_filename(matrix_pathA, matrix_pathB);
            std::string output_path = std::filesystem::path(matrix_shard_folder) / output_filename;

            if (!save_matrix_bin(output_path.c_str(), result)) {
                std::cerr << "ERROR: Failed to save result to: " << output_path << std::endl;
                return false;
            }
            
            std::cout << "Result saved locally: " << output_path << std::endl;

            // ============================================================
            // STEP 6: HANDLE SEND-BACK OPERATIONS
            // ============================================================
            if (send_back > 0) {
                std::cout << "Sending result back to head node (level 1)" << std::endl;
                send_back_file(output_path, output_filename, result, send_back, "llama");
            }
            
            if (send_back < 0) {
                std::cout << "Sending result back to head node (level 2)" << std::endl;
                send_back_level2(output_path, output_filename, result, worker_percentages, send_back);
            }

            std::cout << "Matrix operation completed successfully" << std::endl;
            
            return true;
        }

        bool matrix_operation_torch(
            const char* matrix_pathA,
            bool transposeA,
            const char* matrix_pathB,
            bool transposeB,
            bool use_gpu,
            int gpu_id,
            int send_back,
            const std::string& matrix_operation_select
        )
        {
            std::cout << "MATRIX OPERATION TORCH BACKEND" << std::endl;
            
            try {
                // ============================================================
                // STEP 1: LOAD TENSORS FROM BINARY FILES
                // ============================================================
                torch::Tensor A = load_matrix_bin_as_torch_view(matrix_pathA);
                torch::Tensor B = load_matrix_bin_as_torch_view(matrix_pathB);
                
                if (!A.defined() || !B.defined()) {
                    std::cerr << "ERROR: Torch failed to load matrix files" << std::endl;
                    return false;
                }

                std::cout << "Tensor A loaded: " << A.sizes() << std::endl;
                std::cout << "Tensor B loaded: " << B.sizes() << std::endl;

                // ============================================================
                // STEP 2: APPLY TRANSPOSE OPERATIONS (LAST 2 DIMENSIONS ONLY)
                // ============================================================
                if (transposeA) {
                    std::cout << "Transposing Tensor A" << std::endl;
                    A = A.transpose(-2, -1).contiguous();
                }
                
                if (transposeB) {
                    std::cout << "Transposing Tensor B" << std::endl;
                    B = B.transpose(-2, -1).contiguous();
                }

                // ============================================================
                // STEP 3: SELECT COMPUTATION DEVICE (CPU/GPU)
                // ============================================================
                torch::Device device = torch::kCPU;
                
                if (use_gpu) {
                    if (!torch::cuda::is_available()) {
                        std::cerr << "ERROR: CUDA requested but not available" << std::endl;
                        return false;
                    }
                    device = torch::Device(torch::kCUDA, gpu_id);
                    std::cout << "Using GPU device: " << gpu_id << std::endl;
                } else {
                    std::cout << "Using CPU device" << std::endl;
                }

                // Move tensors to selected device
                A = A.to(device);
                B = B.to(device);

                // ============================================================
                // STEP 4: PERFORM MATRIX OPERATION
                // ============================================================
                torch::Tensor C;
                
                if (matrix_operation_select == "mul") {
                    std::cout << "Performing matrix multiplication" << std::endl;
                    C = torch::matmul(A, B);
                } else if (matrix_operation_select == "add") {
                    std::cout << "Performing matrix addition" << std::endl;
                    C = A + B;
                } else if (matrix_operation_select == "sub") {
                    std::cout << "Performing matrix subtraction" << std::endl;
                    C = A - B;
                } else {
                    std::cerr << "ERROR: Unsupported operation: " << matrix_operation_select << std::endl;
                    return false;
                }

                // Move result back to CPU and ensure contiguous memory
                C = C.contiguous().to(torch::kCPU);
                std::cout << "Result tensor: " << C.sizes() << std::endl;

                // ============================================================
                // STEP 5: CONVERT TORCH TENSOR → MatrixResult
                // ============================================================
                MatrixResult save_result;
                auto sizes = C.sizes();
                int ndim = sizes.size();
                
                // Default dimensions (4D format)
                int batch = 1, depth = 1, rows = 1, cols = 1;
                
                // Handle different tensor ranks
                if (ndim == 2) {
                    rows = sizes[0];
                    cols = sizes[1];
                } else if (ndim == 3) {
                    batch = sizes[0];
                    rows  = sizes[1];
                    cols  = sizes[2];
                } else if (ndim == 4) {
                    batch = sizes[0];
                    depth = sizes[1];
                    rows  = sizes[2];
                    cols  = sizes[3];
                } else {
                    std::cerr << "ERROR: Unsupported tensor rank: " << ndim << std::endl;
                    return false;
                }

                // Set dimensions in MatrixResult
                save_result.dims[0] = batch;
                save_result.dims[1] = depth;
                save_result.dims[2] = rows;
                save_result.dims[3] = cols;

                // Allocate memory and copy tensor data
                int64_t total_elements = C.numel();
                save_result.data = std::make_unique<float[]>(total_elements);
                
                memcpy(
                    save_result.data.get(),
                    C.data_ptr<float>(),
                    total_elements * sizeof(float)
                );

                // ============================================================
                // STEP 6: DEBUG OUTPUT (OPTIONAL)
                // ============================================================
                // Uncomment for debugging:
                // print_matrix(save_result.data.get(), save_result.dims, 12);

                // ============================================================
                // STEP 7: SAVE RESULT TO FILE
                // ============================================================
                std::string output_filename = get_matrix_output_filename(matrix_pathA, matrix_pathB);
                std::string local_full_output_path = 
                    std::filesystem::path(matrix_shard_folder) / output_filename;
                
                if (!save_matrix_bin(local_full_output_path.c_str(), save_result)) {
                    std::cerr << "ERROR: Failed to save Torch result to: " << local_full_output_path << std::endl;
                    return false;
                }
                
                std::cout << "Result saved locally: " << local_full_output_path << std::endl;

                // ============================================================
                // STEP 8: HANDLE SEND-BACK TO HEAD NODE
                // ============================================================
                if (send_back > 0) {
                    std::cout << "Sending result back to head node" << std::endl;
                    send_back_file(
                        local_full_output_path,
                        output_filename,
                        save_result,
                        send_back,
                        "torch"
                    );
                }

                std::cout << "Torch matrix operation completed successfully" << std::endl;
                
                return true;
            }
            catch (const std::exception& e) {
                std::cerr << "ERROR: Torch matrix operation failed: " << e.what() << std::endl;
                return false;
            }
        }

        bool matrix_operation_openCL(
            const char* matrix_pathA,
            bool transposeA,
            const char* matrix_pathB,
            bool transposeB,
            int gpu_id,
            int send_back,
            const std::string& matrix_operation_select,
            std::string kernel_code
        )
        {
            // Commence the ancient ritual of OpenCL programming
            // You know who uses OpenCL? The kind of people who think
            // "family reunions" are a good place to meet their next date.
            // Seriously, you have to be pretty desperate to choose OpenCL
            // over actual GPU libraries. Like choosing to assemble your own
            // furniture from IKEA when there's a perfectly good chair right there.
            
            std::cout << "INITIATING OPENCL OPERATION (because apparently we like pain)" << std::endl;

            // ------------------------------
            // VALIDATE GPU ID (because apparently some people can't count)
            // ------------------------------
            if (gpu_id < 0 || gpu_id >= (int)openCL_GPU_select_list.size()) {
                std::cerr << "ERROR: Invalid OpenCL gpu_id - did you run out of fingers to count on?" << std::endl;
                return false;
            }

            // ------------------------------
            // LOAD MATRICES (using Torch because OpenCL can't even load files by itself)
            // ------------------------------
            // OpenCL is like that cousin who needs help with everything.
            // "Can you load this file for me? Can you manage my memory for me?"
            // Meanwhile CUDA is out there actually getting work done.
            std::cout << "Loading matrices (using Torch, because OpenCL needs adult supervision)" << std::endl;
            
            torch::Tensor tensorA = load_matrix_bin_as_torch_view(matrix_pathA);
            torch::Tensor tensorB = load_matrix_bin_as_torch_view(matrix_pathB);

            if (!tensorA.defined() || !tensorB.defined()) {
                std::cerr << "ERROR: Failed to load matrices - did you forget to feed the OpenCL hamster?" << std::endl;
                return false;
            }

            std::cout << "Tensor A loaded: " << tensorA.sizes() << std::endl;
            std::cout << "Tensor B loaded: " << tensorB.sizes() << std::endl;

            // ------------------------------
            // TRANSPOSE ON CPU (because OpenCL can't handle complexity)
            // ------------------------------
            // Fun fact: OpenCL developers have a family tree that looks like
            // a telephone pole. Just straight vertical.
            if (transposeA) {
                std::cout << "Transposing Matrix A (because OpenCL can't read unless everything is just right)" << std::endl;
                tensorA = tensorA.transpose(-2, -1).contiguous();
            }
            if (transposeB) {
                std::cout << "Transposing Matrix B (don't worry, we'll hold OpenCL's hand through this)" << std::endl;
                tensorB = tensorB.transpose(-2, -1).contiguous();
            }

            float* A_ptr = tensorA.data_ptr<float>();
            float* B_ptr = tensorB.data_ptr<float>();
            int M = tensorA.size(-2);
            int K = tensorA.size(-1);
            int N = tensorB.size(-1);

            if (K != tensorB.size(-2)) {
                std::cerr << "ERROR: Dimension mismatch - typical OpenCL family dynamics" << std::endl;
                return false;
            }

            std::cout << "Matrix dimensions: " << M << "x" << K << " * " << K << "x" << N << std::endl;

            // ------------------------------
            // OPENCL SETUP (the part where we regret our life choices)
            // ------------------------------
            // Setting up OpenCL is like arranging a marriage between your sister
            // and your brother. It's probably going to work, but everyone will judge you.
            std::cout << "Initializing OpenCL (brace yourself for disappointment)" << std::endl;
            
            cl::Device device = openCL_GPU_select_list[gpu_id];
            cl::Context context(device);
            cl::CommandQueue queue(context, device);
            
            // Create buffers (because OpenCL needs everything in its special format)
            cl::Buffer bufA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * tensorA.numel(), A_ptr);
            cl::Buffer bufB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * tensorB.numel(), B_ptr);
            cl::Buffer bufC(context, CL_MEM_WRITE_ONLY, sizeof(float) * M * N);
            
            // Build program (this is where the OpenCL compiler cries)
            std::cout << "Building OpenCL kernel (this may take a while, like a family reunion)" << std::endl;
            cl::Program program(context, kernel_code);
            program.build({device});
            
            // Create kernel (because apparently we're making our own graphics driver now)
            cl::Kernel kernel(program, "matmul");

            // ------------------------------
            // SET KERNEL ARGUMENTS (the part where we pray to the GPU gods)
            // ------------------------------
            // Each argument is like another branch on the family tree.
            // More arguments, more problems.
            std::cout << "Setting kernel arguments (6 arguments, just like the 6 degrees of separation in your family tree)" << std::endl;
            
            kernel.setArg(0, bufA);
            kernel.setArg(1, bufB);
            kernel.setArg(2, bufC);
            kernel.setArg(3, M);
            kernel.setArg(4, N);
            kernel.setArg(5, K);

            // ------------------------------
            // LAUNCH KERNEL (the moment of truth)
            // ------------------------------
            // This is it. The moment where all our questionable life choices
            // either pay off or crash spectacularly. Just like Thanksgiving dinner.
            std::cout << "Launching OpenCL kernel (cross your fingers, toes, and any other appendages)" << std::endl;
            
            cl::NDRange global(M, N);
            cl::NDRange local(16, 16); // Magic numbers, because OpenCL is basically witchcraft
            
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
            queue.finish();  // Wait for completion (and possibly for regrets to set in)

            // ------------------------------
            // READ BACK RESULT (hope it's not gibberish)
            // ------------------------------
            std::cout << "Reading back results (please don't be corrupted, please don't be corrupted)" << std::endl;
            
            MatrixResult result;
            result.dims[0] = M;
            result.dims[1] = N;
            result.dims[2] = 1;
            result.dims[3] = 1;
            
            result.data = std::make_unique<float[]>(M * N);
            queue.enqueueReadBuffer(bufC, CL_TRUE, 0, sizeof(float) * M * N, result.data.get());

            // ------------------------------
            // SAVE RESULT (so we have proof this actually worked)
            // ------------------------------
            // Saving the file is like taking a family photo. We need evidence
            // that this whole disaster actually happened.
            std::string output_filename = get_matrix_output_filename(
                std::filesystem::path(matrix_pathA).filename().string(),
                matrix_pathB
            );
            
            std::string output_path = std::filesystem::path(matrix_shard_folder) / output_filename;
            
            if (!save_matrix_bin(output_path.c_str(), result)) {
                std::cerr << "ERROR: Failed to save OpenCL result - figures, right?" << std::endl;
                return false;
            }

            std::cout << "Result saved: " << output_path << std::endl;

            // ------------------------------
            // SEND BACK TO HEAD NODE (if requested)
            // ------------------------------
            // Sharing OpenCL results is like telling your family about your new
            // hobby. They'll pretend to be interested, but really they're just
            // waiting for you to stop talking.
            if (send_back > 0) {
                std::cout << "Sending result back to head node (don't tell them we used OpenCL)" << std::endl;
                send_back_file(output_path, output_filename, result, send_back, "opencl");
            }

            std::cout << "OpenCL operation completed (against all odds)" << std::endl;
            
            return true;
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

            if (shard_num != -1)
                output_filename += "_shard_" + std::to_string(shard_num);

            output_filename += ".bin";

            return output_filename;
        }

        std::vector<torch::Tensor> convert_to_cluster_matrix_shards(
            const MatrixResult& result,
            const std::vector<float>& node_percentages,
            int number_of_shards,
            int split_dim = 0
        ) 
        {
            // ============================================================
            // STEP 1: CONVERT MatrixResult TO TORCH TENSOR
            // ============================================================
            // Convert raw matrix data into a PyTorch tensor for manipulation
            torch::Tensor matrix = convert_matrix_to_torch(result.data, result.dims);
            std::cout << "Loaded matrix: " << matrix.sizes() 
                    << " (total elements: " << matrix.numel() << ")" << std::endl;

            // ============================================================
            // STEP 2: DETERMINE SPLIT DIMENSION
            // ============================================================
            // For 4D tensors [batch, depth, rows, cols], default to splitting rows
            // This handles cases where split_dim=0 (batch) might not be desired
            if (matrix.dim() == 4 && split_dim == 0) {
                split_dim = 2; // Split along rows dimension
                std::cout << "Adjusted split dimension to rows (dimension " << split_dim << ")" << std::endl;
            }

            // ============================================================
            // STEP 3: VALIDATE AND ADJUST NUMBER OF SHARDS
            // ============================================================
            // Ensure we don't request more shards than the matrix dimension supports
            int64_t matrix_dim_size = matrix.size(split_dim);
            if (matrix_dim_size < number_of_shards) {
                number_of_shards = static_cast<int>(matrix_dim_size);
                std::cout << "WARNING: Adjusted shards from " << number_of_shards 
                        << " to " << number_of_shards 
                        << " (matrix dimension size is " << matrix_dim_size << ")" << std::endl;
            }

            std::cout << "Splitting matrix of size " << matrix_dim_size 
                    << " into " << number_of_shards << " shards" << std::endl;

            // ============================================================
            // STEP 4: INITIAL SHARD SPLITTING
            // ============================================================
            // Split the matrix into equal-sized shards along the specified dimension
            std::vector<torch::Tensor> matrix_shards = torch::chunk(matrix, number_of_shards, split_dim);
            std::cout << "Split into " << matrix_shards.size() << " initial equal shards" << std::endl;

            // ============================================================
            // STEP 5: MERGE SHARDS BASED ON NODE PERCENTAGES
            // ============================================================
            // Allocate shards to nodes according to their percentage share
            // Example: 3 nodes with percentages [0.5, 0.3, 0.2] and 10 shards:
            // Node 0 gets 5 shards, Node 1 gets 3 shards, Node 2 gets 2 shards
            std::vector<torch::Tensor> node_matrices;
            int start_index = 0;
            int total_shards = static_cast<int>(matrix_shards.size());

            std::cout << "Distributing " << total_shards << " shards to " 
                    << node_percentages.size() << " nodes:" << std::endl;

            for (size_t i = 0; i < node_percentages.size(); ++i) {
                // Calculate number of shards this node gets based on its percentage
                int shards_to_take = std::max(1, static_cast<int>(node_percentages[i] * total_shards));
                int end_index = std::min(start_index + shards_to_take, total_shards);

                if (start_index < end_index) {
                    // Collect shards for this node
                    std::vector<torch::Tensor> shards_to_merge;
                    for (int j = start_index; j < end_index; ++j) {
                        shards_to_merge.push_back(matrix_shards[j]);
                    }
                    // Merge collected shards back together
                    node_matrices.push_back(torch::cat(shards_to_merge, split_dim));
                } else {
                    // Node gets no shards (edge case for very small percentages)
                    node_matrices.push_back(torch::empty({0}));
                }

                std::cout << "  Node " << i << ": " 
                        << node_matrices.back().sizes() 
                        << " (" << (node_percentages[i] * 100.0f) << "% = " 
                        << (end_index - start_index) << " shards)" << std::endl;

                start_index = end_index;
                if (start_index >= total_shards) break;
            }

            // ============================================================
            // STEP 6: HANDLE LEFTOVER SHARDS
            // ============================================================
            // Due to rounding, some shards might not be allocated.
            // Add any remaining shards to the last node.
            if (start_index < total_shards && !node_matrices.empty()) {
                std::vector<torch::Tensor> leftover_shards;
                int leftover_count = 0;
                
                for (int j = start_index; j < total_shards; ++j) {
                    leftover_shards.push_back(matrix_shards[j]);
                    leftover_count++;
                }
                
                // Concatenate leftovers to the last node's matrix
                node_matrices.back() = torch::cat({node_matrices.back(), torch::cat(leftover_shards, split_dim)}, split_dim);
                
                std::cout << "Added " << leftover_count << " leftover shards to last node" << std::endl;
            }

            // ============================================================
            // STEP 7: FINAL SUMMARY
            // ============================================================
            std::cout << "Shard distribution complete: " 
                    << node_matrices.size() << " node matrices created" << std::endl;
            
            for (size_t i = 0; i < node_matrices.size(); ++i) {
                if (node_matrices[i].numel() > 0) {
                    std::cout << "  Node " << i << " final size: " 
                            << node_matrices[i].sizes() 
                            << " (" << node_matrices[i].numel() << " elements)" << std::endl;
                }
            }

            return node_matrices;
        }

        torch::Tensor merged_matrix(
            const std::vector<torch::Tensor>& shards,
            int start_index,
            int end_index,
            int dim = 0)
        {
            if (start_index < 0 || end_index <= start_index || start_index >= shards.size()) {
                throw std::runtime_error("Invalid start/end indices for merged_matrix");
            }

            int actual_end = std::min(end_index, (int)shards.size());
            std::vector<torch::Tensor> to_merge;
            for (int i = start_index; i < actual_end; ++i) {
                to_merge.push_back(shards[i]);
            }

            return torch::cat(to_merge, dim);
        }

        void send_back_level2(
            const std::string& local_file_path,
            const std::string& filename,
            const MatrixResult& result,
            const std::vector<float>& worker_percentages,
            int send_back,
            int split_dim = 0,
            int base_number_of_shards = 100
        )
        {
            std::cout << "####################--Send Back Level 2--#####################\n";
            int num_nodes = std::abs(send_back);
            
            // Check sizes match
            if ((int)worker_ip_list.size() < num_nodes) {
                std::cerr << "ERROR: worker_ip_list size (" << worker_ip_list.size() 
                        << ") is less than num_nodes (" << num_nodes << ")\n";
                return;
            }
            
            std::cout << "\n########DEBUG send_back_filename before processing############\n"
                    << filename << "\n";
            
            // Process filename
            std::string filename_copy = filename;
            size_t pos = filename_copy.rfind(".bin");
            if (pos != std::string::npos) {
                filename_copy = filename_copy.substr(0, pos);
            }
            
            // Generate shards
            std::vector<torch::Tensor> shards = convert_to_cluster_matrix_shards(
                result,
                worker_percentages,
                base_number_of_shards,
                split_dim
            );
            
            // Check if we have enough shards
            if ((int)shards.size() < num_nodes) {
                std::cout << "Warning: Only generated " << shards.size() 
                        << " shards for " << num_nodes << " nodes\n";
            }
            
            // ------------------------------------------------------------
            // 2️⃣ Send shards worker ↔ worker
            // ------------------------------------------------------------
            for (int i = 0; i < num_nodes; ++i) {
                const std::string& ip = worker_ip_list[i];
                
                // Check if this node is us (local node)
                if (ip == local_IP_eth || ip == local_IP_wifi) {
                    // Save shard locally using save_matrix_bin function
                    if (i < (int)shards.size() && shards[i].numel() > 0) {
                        std::string local_save_name =
                            filename_copy + "_shard_" + std::to_string(i) + ".bin";
                        std::string local_save_path =
                            std::filesystem::path(matrix_shard_folder) / local_save_name;
                        
                        // Convert tensor to MatrixResult
                        MatrixResult local_result;
                        auto shard_sizes = shards[i].sizes();
                        local_result.dims[0] = shard_sizes.size() > 0 ? shard_sizes[0] : 1;
                        local_result.dims[1] = shard_sizes.size() > 1 ? shard_sizes[1] : 1;
                        local_result.dims[2] = shard_sizes.size() > 2 ? shard_sizes[2] : 1;
                        local_result.dims[3] = shard_sizes.size() > 3 ? shard_sizes[3] : 1;
                        
                        int64_t total_elements = shards[i].numel();
                        local_result.data = std::make_unique<float[]>(total_elements);
                        
                        // Copy tensor data
                        torch::Tensor cpu_shard = shards[i].contiguous().to(torch::kCPU);
                        std::memcpy(
                            local_result.data.get(),
                            cpu_shard.data_ptr<float>(),
                            total_elements * sizeof(float)
                        );
                        
                        // Save locally
                        if (save_matrix_bin(local_save_path.c_str(), local_result)) {
                            std::cout << "Saved local shard " << i
                                    << " to: " << local_save_path << "\n";
                        } else {
                            std::cerr << "Failed to save local shard " << i << "\n";
                        }
                    } else {
                        std::cout << "Node " << i
                                << " (local): No shard allocated or empty tensor\n";
                    }
                }
                else if (i < (int)shards.size() && shards[i].numel() > 0) {
                    // Send to remote node
                    zmq::socket_t sender(zmq_context, zmq::socket_type::push);
                    sender.connect("tcp://" + ip + ":" + worker_peer_port);
                    
                    torch::Tensor& shard = shards[i];
                    torch::Tensor cpu_tensor = shard.contiguous().to(torch::kCPU);
                    size_t byte_size = cpu_tensor.numel() * cpu_tensor.element_size();
                    zmq::message_t msg(byte_size);
                    std::memcpy(msg.data(), cpu_tensor.data_ptr(), byte_size);
                    
                    std::string send_back_filename =
                        filename_copy + "_shard_" + std::to_string(i) + ".bin";
                    std::cout << "\n########DEBUG send_back_filename after processing############\n"
                            << send_back_filename << "\n";
                    
                    zmq::message_t filename_msg(
                        send_back_filename.data(),
                        send_back_filename.size()
                    );
                    sender.send(filename_msg, zmq::send_flags::sndmore);
                    sender.send(msg, zmq::send_flags::none);
                    
                    std::cout << "Sent Level-2 shard " << i << " → " << ip
                            << " (" << byte_size << " bytes) as "
                            << send_back_filename << "\n";
                }
                else {
                    std::cout << "Node " << i << " (" << ip
                            << "): No shard allocated or empty tensor\n";
                }
            }
            
            // ------------------------------------------------------------
            // 3️⃣ ACK back to Python frontend (single ACK)
            // ------------------------------------------------------------
            send_ack();
            std::cout << "Level-2 send-back complete. ACK sent.\n";
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