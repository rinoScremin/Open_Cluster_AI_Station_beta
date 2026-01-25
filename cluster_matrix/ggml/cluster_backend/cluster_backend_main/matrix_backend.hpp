#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"  // <-- Add this for ggml_backend_cpu_init()
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdlib>
#include <set>
#include <thread>
#include <torch/torch.h>
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <atomic>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <mutex>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <sys/mman.h>



std::vector<cl::Device> openCL_GPU_select_list;

struct MatrixResult 
{  
    std::unique_ptr<float[]> data;  
    int dims[4]; // ne0, ne1, ne2, ne3  
};  

static inline uint16_t float_to_bf16_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static inline uint16_t float_to_fp16_bits(float f) {
    // IEEE 754 float -> half (round to nearest even)
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));

    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = int32_t((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x007FFFFFu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign; // too small -> 0
        // subnormal
        mant |= 0x00800000u;
        uint32_t shift = (uint32_t)(1 - exp);
        uint32_t mant16 = mant >> (shift + 13);
        // round-to-nearest-even
        uint32_t round_bit = (mant >> (shift + 12)) & 1u;
        uint32_t sticky = mant & ((1u << (shift + 12)) - 1u);
        mant16 += (round_bit && (sticky || (mant16 & 1u))) ? 1u : 0u;
        return (uint16_t)(sign | mant16);
    }

    if (exp >= 31) {
        // inf/nan
        if ((x & 0x7FFFFFFFu) == 0x7F800000u) {
            return (uint16_t)(sign | 0x7C00u);
        }
        uint16_t nan_mant = (uint16_t)(mant >> 13);
        if (nan_mant == 0) nan_mant = 1;
        return (uint16_t)(sign | 0x7C00u | nan_mant);
    }

    uint32_t mant16 = mant >> 13;
    uint32_t round_bit = (mant >> 12) & 1u;
    uint32_t sticky = mant & 0xFFFu;
    mant16 += (round_bit && (sticky || (mant16 & 1u))) ? 1u : 0u;
    if (mant16 == 0x400u) { // mantissa overflow
        mant16 = 0;
        exp += 1;
        if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    }

    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant16 & 0x03FFu));
}

// ============================================================================
// OPENCL KERNELS FOR MATRIX MULTIPLICATION
// ============================================================================
// These kernels implement matrix multiplication using OpenCL.
// 
// LEGEND HAS IT that OpenCL was conceived during a legendary hackathon
// where the developers were so high on Adderall and Mountain Dew that
// they thought "let's make CUDA, but for people who hate themselves."
// 
// The design meeting apparently went:
// "What if we took everything good about CUDA... and removed it?"
// "Genius! And let's make the API documentation read like IKEA instructions
//  translated through 5 different languages!"
// 
// True story: The first OpenCL compiler was actually just a developer
// vomiting code into a terminal while his coworkers cheered him on.
// Every segmentation fault is a tribute to that sacred moment.
// ============================================================================

// Basic matrix multiplication kernel (naïve implementation)
// This kernel was written by an intern who'd never seen a GPU before.
// He was told "make the threads go brrr" and this is what he came up with.
// It's so inefficient that it actually heats your room in winter.
const char* openCL_kernel_matmul = R"(
__kernel void matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K)
{
    int row = get_global_id(0); // row index
    int col = get_global_id(1); // col index

    if(row < M && col < N) {
        float sum = 0.0f;
        for(int k = 0; k < K; ++k) {
            sum += A[row*K + k] * B[k*N + col];
        }
        C[row*N + col] = sum;
    }
}
)";

// Tiled matrix multiplication kernel (optimized)
// This kernel was written AFTER the developers took a shower together
// (platonic, obviously - they high-fived about cache lines, not each other)
// 
// The breakthrough happened when Senior Engineer Chad realized:
// "What if... we used local memory?" 
// The team erupted in applause. Promotions were handed out on the spot.
// Someone's mom brought cookies to celebrate.
const char* openCL_kernel_matmul_tiled = R"(
__kernel void matmul_tiled(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K)
{
    const int TILE_SIZE = 16;
    __local float A_tile[TILE_SIZE][TILE_SIZE];
    __local float B_tile[TILE_SIZE][TILE_SIZE];

    int row = get_global_id(0);
    int col = get_global_id(1);

    float sum = 0.0f;

    for(int t = 0; t < (K + TILE_SIZE - 1)/TILE_SIZE; t++) {
        int tiled_col = t * TILE_SIZE + get_local_id(1);
        int tiled_row = t * TILE_SIZE + get_local_id(0);

        // Load tiles into local memory
        if(row < M && tiled_col < K)
            A_tile[get_local_id(0)][get_local_id(1)] = A[row*K + tiled_col];
        else
            A_tile[get_local_id(0)][get_local_id(1)] = 0.0f;

        if(tiled_row < K && col < N)
            B_tile[get_local_id(0)][get_local_id(1)] = B[tiled_row*N + col];
        else
            B_tile[get_local_id(0)][get_local_id(1)] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < TILE_SIZE; k++)
            sum += A_tile[get_local_id(0)][k] * B_tile[k][get_local_id(1)];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(row < M && col < N)
        C[row*N + col] = sum;
}
)";

void init_openCL_GPUS() 
{
    std::cout << "🎬 SCENE: OPENCL INITIALIZATION - TAKE 47" << std::endl;
    std::cout << "Our hero (you) attempts to tame the wild OpenCL beast" << std::endl;
    std::cout << "Spoiler: It doesn't end well" << std::endl << std::endl;
    
    // Clear the list from previous attempts
    // Like wiping the whiteboard after a failed math proof
    openCL_GPU_select_list.clear();

    std::cout << "STEP 1: Query platforms (ask politely for GPUs)" << std::endl;
    // This is the equivalent of yelling "HELLO?" into a dark cave
    // Sometimes you get an echo, sometimes you get a bear
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    
    if(platforms.empty()) {
        std::cout << "🚨 PLOT TWIST: No OpenCL platforms found!" << std::endl;
        std::cout << "Your computer is either:" << std::endl;
        std::cout << "  a) A toaster" << std::endl;
        std::cout << "  b) Too advanced for this peasant technology" << std::endl;
        std::cout << "  c) Judging you for even trying" << std::endl;
        throw std::runtime_error("System has reached maximum disappointment capacity");
    }

    std::cout << "Found " << platforms.size() 
              << " platform(s) with names that sound like bad sci-fi movies" << std::endl;

    // The platform discovery loop
    // Each iteration is like opening a mystery box
    // Will it contain a shiny GPU? Or another CPU pretending to be special?
    for(size_t i = 0; i < platforms.size(); i++) {
        std::string platformName = platforms[i].getInfo<CL_PLATFORM_NAME>();
        
        // Platform names usually contain words like:
        // "Advanced", "Accelerated", "Parallel", "Disappointment"
        platformName.erase(std::remove(platformName.begin(), platformName.end(), '\n'), platformName.end());
        platformName.erase(std::remove(platformName.begin(), platformName.end(), '\r'), platformName.end());
        
        std::cout << std::endl << "🔍 Investigating Platform " << i << ": \"" 
                  << platformName << "\"" << std::endl;
        std::cout << "   (Probably from 2012, judging by the naming convention)" << std::endl;

        // Get devices - this is where the real comedy begins
        // CL_DEVICE_TYPE_ALL means "give me everything, including the kitchen sink"
        std::vector<cl::Device> devices;
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        
        if(devices.empty()) {
            std::cout << "   💀 Platform is a ghost town. Moving on..." << std::endl;
            continue;
        }

        std::cout << "   Found " << devices.size() << " device(s). Let's meet them!" << std::endl;

        // Device introduction ceremony
        // Each device gets a moment in the spotlight
        for(size_t j = 0; j < devices.size(); j++) {
            std::string deviceName = devices[j].getInfo<CL_DEVICE_NAME>();
            
            // Clean up the name - GPU vendors love adding "\n" and "\r"
            // because they think it makes their hardware look fancy
            deviceName.erase(std::remove(deviceName.begin(), deviceName.end(), '\n'), deviceName.end());
            deviceName.erase(std::remove(deviceName.begin(), deviceName.end(), '\r'), deviceName.end());
            
            std::cout << "   👉 Device " << j << ": \"" << deviceName << "\"";
            
            // Check device type for comedy/ tragedy
            cl_device_type type = devices[j].getInfo<CL_DEVICE_TYPE>();
            if(type & CL_DEVICE_TYPE_GPU) {
                std::cout << " [GPU 🎉]";
                std::cout << " - An actual graphics processor! Praise the silicon gods!";
            } else if(type & CL_DEVICE_TYPE_CPU) {
                std::cout << " [CPU 😴]";
                std::cout << " - Oh look, it's the thing we're already using. How... special.";
            } else if(type & CL_DEVICE_TYPE_ACCELERATOR) {
                std::cout << " [ACCELERATOR 🚀]";
                std::cout << " - Sounds fast! Probably isn't.";
            } else if(type & CL_DEVICE_TYPE_DEFAULT) {
                std::cout << " [DEFAULT 🤷]";
                std::cout << " - The OpenCL equivalent of 'idk, something'";
            } else {
                std::cout << " [MYSTERY BOX ❓]";
                std::cout << " - Could be a GPU, could be a smart fridge. Who knows?";
            }
            std::cout << std::endl;

            // Add to global list
            // This is the "collectible cards" phase of OpenCL programming
            openCL_GPU_select_list.push_back(devices[j]);
        }
    }

    // The moment of truth: did we find anything useful?
    std::cout << std::endl << "📊 FINAL TALLY:" << std::endl;
    std::cout << "Total OpenCL devices collected: " << openCL_GPU_select_list.size() << std::endl;
    
    if(openCL_GPU_select_list.empty()) {
        std::cout << "🎭 TRAGEDY: The list is emptier than our protagonist's social life." << std::endl;
        throw std::runtime_error("No OpenCL devices found. Try sacrificing a goat to the silicon gods.");
    }

    // Count actual GPUs vs impostors
    int gpu_count = 0;
    int cpu_count = 0;
    int weird_count = 0;
    
    for(const auto& device : openCL_GPU_select_list) {
        cl_device_type type = device.getInfo<CL_DEVICE_TYPE>();
        if(type & CL_DEVICE_TYPE_GPU) gpu_count++;
        else if(type & CL_DEVICE_TYPE_CPU) cpu_count++;
        else weird_count++;
    }
    
    std::cout << "📈 Breakdown:" << std::endl;
    std::cout << "  Real GPUs: " << gpu_count << " (the good stuff)" << std::endl;
    std::cout << "  CPUs: " << cpu_count << " (OpenCL's participation trophy)" << std::endl;
    std::cout << "  Weird stuff: " << weird_count << " (probably FPGAs crying for attention)" << std::endl;
    
    if(gpu_count == 0) {
        std::cout << std::endl << "⚠️  WARNING: No actual GPUs detected!" << std::endl;
        std::cout << "Your OpenCL experience will be powered by:" << std::endl;
        std::cout << "  • Wishful thinking" << std::endl;
        std::cout << "  • The ghosts of forgotten benchmarks" << std::endl;
        std::cout << "  • Whatever thermal paste is left on your CPU" << std::endl;
        std::cout << "Good luck! You'll need it." << std::endl;
    } else {
        std::cout << std::endl << "🎊 CONGRATULATIONS!" << std::endl;
        std::cout << "You found " << gpu_count << " GPU(s) that might actually work!" << std::endl;
        std::cout << "Now the real suffering begins: writing kernels that don't segfault!" << std::endl;
    }
    
    std::cout << std::endl << "🏁 OPENCL INITIALIZATION COMPLETE" << std::endl;
    std::cout << "Now go take a shower and high-five yourself in the mirror." << std::endl;
    std::cout << "You've earned it. (The high-five, not the shower.)" << std::endl;
}

bool save_matrix_bin(const char* path, const MatrixResult& result)    
{    
    // Create directory if it doesn't exist    
    std::string path_str = path;    
    size_t last_slash = path_str.find_last_of('/');    
    if (last_slash != std::string::npos)   
    {    
        std::string dir_path = path_str.substr(0, last_slash);    
        std::filesystem::create_directories(dir_path);    
    }    
    std::ofstream file(path, std::ios::binary);    
    if (!file)  
    {    
        std::cerr << "Cannot create file: " << path << std::endl;    
        return false;    
    }    
    // Default: write float32 payload (v2 header).
    // Use the overload `save_matrix_bin(path, result, dtype_tag)` to request float16/bfloat16 output.
    const int dtype_tag = -1;
    file.write(reinterpret_cast<const char*>(&dtype_tag), sizeof(int));
    const int ndim = 4;
    for (int i = 0; i < ndim; i++) {
        file.write(reinterpret_cast<const char*>(&result.dims[i]), sizeof(int));
    }
    // Calculate total elements and write data    
    size_t total_elements = 1;  
    for (int i = 0; i < ndim; i++)   
    {  
        total_elements *= result.dims[i];  
    }    
    file.write(reinterpret_cast<const char*>(result.data.get()), sizeof(float) * total_elements);
    file.close();      
    // Print info (matching Python reference)    
    std::cout << "  Saved to " << path << std::endl;    
    // Calculate expected file size
    size_t file_size = 4 + 4 * 4 + total_elements * 4;  // 5 ints + float32 payload
    std::cout << "  Shape: (";    
    for (int i = 0; i < ndim; i++)  
    {  
        std::cout << result.dims[i];  
        if (i < ndim - 1) std::cout << ", ";  
    }  
    std::cout << "), Size: " << file_size << " bytes" << std::endl;       
    return true;    
}

bool save_matrix_bin(const char* path, const MatrixResult& result, int dtype_tag)
{
    // Create directory if it doesn't exist
    std::string path_str = path;
    size_t last_slash = path_str.find_last_of('/');
    if (last_slash != std::string::npos)
    {
        std::string dir_path = path_str.substr(0, last_slash);
        std::filesystem::create_directories(dir_path);
    }

    std::ofstream file(path, std::ios::binary);
    if (!file)
    {
        std::cerr << "Cannot create file: " << path << std::endl;
        return false;
    }

    // Header (v2): dtype_tag + fixed 4D dims
    if (dtype_tag != -1 && dtype_tag != -2 && dtype_tag != -3) {
        std::cerr << "Unsupported dtype_tag for save_matrix_bin: " << dtype_tag << std::endl;
        return false;
    }

    file.write(reinterpret_cast<const char*>(&dtype_tag), sizeof(int));
    const int ndim = 4;
    for (int i = 0; i < ndim; i++) {
        file.write(reinterpret_cast<const char*>(&result.dims[i]), sizeof(int));
    }

    size_t total_elements = 1;
    for (int i = 0; i < ndim; i++) {
        total_elements *= result.dims[i];
    }

    if (dtype_tag == -1) {
        file.write(reinterpret_cast<const char*>(result.data.get()), sizeof(float) * total_elements);
    } else if (dtype_tag == -2) {
        std::vector<uint16_t> out(total_elements);
        for (size_t i = 0; i < total_elements; ++i) {
            out[i] = float_to_fp16_bits(result.data[i]);
        }
        file.write(reinterpret_cast<const char*>(out.data()), sizeof(uint16_t) * total_elements);
    } else { // -3 bfloat16
        std::vector<uint16_t> out(total_elements);
        for (size_t i = 0; i < total_elements; ++i) {
            out[i] = float_to_bf16_bits(result.data[i]);
        }
        file.write(reinterpret_cast<const char*>(out.data()), sizeof(uint16_t) * total_elements);
    }

    file.close();

    const size_t elem_bytes = (dtype_tag == -1) ? 4 : 2;
    size_t file_size = 4 + 4 * 4 + total_elements * elem_bytes;
    std::cout << "  Saved to " << path << std::endl;
    std::cout << "  Shape: (" << result.dims[0] << ", " << result.dims[1] << ", " << result.dims[2] << ", " << result.dims[3]
              << "), Size: " << file_size << " bytes" << std::endl;
    return true;
}

std::unique_ptr<float[]> load_matrix_bin(const char* path, int& rows, int& cols,   
                                        int& depth, int& batch)   
{  
    std::ifstream file(path, std::ios::binary);  
    if (!file) 
    {  
        std::cerr << "Cannot open file: " << path << std::endl;  
        return nullptr;  
    }  

    // Read first header word:
    // - v2: dtype_tag (negative)
    // - v1: ndim (positive)
    int tag_or_ndim = 0;
    file.read(reinterpret_cast<char*>(&tag_or_ndim), sizeof(int));
    if (!file) {
        std::cerr << "Cannot read header tag from file: " << path << std::endl;
        return nullptr;
    }

    int ndim = 0;
    int dtype_tag = -1; // legacy default: float32
    std::vector<int> dims;

    if (tag_or_ndim < 0) {
        dtype_tag = tag_or_ndim;
        ndim = 4;
        dims.resize(4);
        for (int i = 0; i < 4; i++) {
            file.read(reinterpret_cast<char*>(&dims[i]), sizeof(int));
        }
    } else {
        ndim = tag_or_ndim;
        dims.resize(ndim);
        for (int i = 0; i < ndim; i++) {
            file.read(reinterpret_cast<char*>(&dims[i]), sizeof(int));
        }
    }
                
    for (int i = 0; i < ndim; i++)
    {
        std::cout << "  Dim[" << i << "] = " << dims[i] << "\n";
    }
                
                // Map dimensions correctly based on ndim
    if (ndim == 2) 
    {
                    // [rows, cols]
        rows = dims[0];
        cols = dims[1];
        depth = 1;
        batch = 1;
    } 
    else if 
    (ndim == 3) 
    {
                    // [batch, rows, cols] 
        batch = dims[0];   // batch comes first!
        rows = dims[1];    // then rows
        cols = dims[2];    // then cols
        depth = 1;
    } 
    else if (ndim == 4) 
    {
                    // [outer_batch, inner_batch, rows, cols]
        batch = dims[0];   // outer_batch
        depth = dims[1];   // inner_batch
        rows = dims[2];    // rows
        cols = dims[3];    // cols
    } 
    else
    {
        std::cerr << "Error: Unsupported number of dimensions: " << ndim << std::endl;
        return nullptr;
    }
                
    //std::cout << "DEBUG: Mapped to - batch=" << batch << ", depth=" << depth 
    //        << ", rows=" << rows << ", cols=" << cols << "\n";
                
                // Calculate total elements  
    size_t total_elements = 1;  
    for (int dim : dims) 
    {  
        total_elements *= dim;  
    }  
                
    //std::cout << "DEBUG: Total elements = " << total_elements << "\n";
                
    // Allocate and read data (convert to float32 as the internal representation)
    auto matrix = std::make_unique<float[]>(total_elements);

    auto bf16_to_f32 = [](uint16_t v) -> float {
        uint32_t bits = uint32_t(v) << 16;
        float out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    };

    auto fp16_to_f32 = [](uint16_t h) -> float {
        // IEEE 754 half -> float
        const uint32_t sign = (uint32_t(h & 0x8000u) << 16);
        uint32_t exp = (h & 0x7C00u) >> 10;
        uint32_t mant = (h & 0x03FFu);

        uint32_t f_bits = 0;
        if (exp == 0) {
            if (mant == 0) {
                f_bits = sign; // zero
            } else {
                // subnormal
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
            // inf/nan
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

    if (dtype_tag == -1) {
        file.read(reinterpret_cast<char*>(matrix.get()), sizeof(float) * total_elements);
    } else if (dtype_tag == -2 || dtype_tag == -3) {
        std::vector<uint16_t> tmp(total_elements);
        file.read(reinterpret_cast<char*>(tmp.data()), sizeof(uint16_t) * total_elements);
        if (!file) {
            std::cerr << "Failed to read 16-bit payload from: " << path << std::endl;
            return nullptr;
        }
        for (size_t i = 0; i < total_elements; ++i) {
            matrix[i] = (dtype_tag == -2) ? fp16_to_f32(tmp[i]) : bf16_to_f32(tmp[i]);
        }
    } else {
        std::cerr << "Error: Unsupported dtype_tag: " << dtype_tag << std::endl;
        return nullptr;
    }
                
                // Print first few values
    //std::cout << "DEBUG: First 5 values: ";
    for (int i = 0; i < std::min(5, (int)total_elements); i++) 
    {
        std::cout << matrix[i] << " ";
    }
    std::cout << "\n";
                
    file.close();  
    return matrix;  
}

void print_matrix(float* matrix, const int dims[4], int max_print = 4) {
    if (!matrix) 
    {
        std::cout << "Matrix is null!" << std::endl;
        return;
    }
    // Determine actual dimensions (skip trailing ones)
    int actual_dims = 4;
    while (actual_dims > 1 && dims[actual_dims-1] == 1) 
    {
        actual_dims--;
    }
    std::cout << "Shape: (";
    for (int i = 0; i < actual_dims; i++) 
    {
        std::cout << dims[i];
        if (i < actual_dims - 1) std::cout << "x";
    }
    std::cout << ")\n";
    // Calculate total elements
    int total = dims[0] * dims[1] * dims[2] * dims[3];
    // Print first few elements
    int print_count = std::min(max_print, total);
    std::cout << "First " << print_count << " values: ";
    for (int i = 0; i < print_count; i++) 
    {
        std::cout << matrix[i] << " ";
    }
    std::cout << "\n\n";
}

torch::Tensor load_matrix_bin_as_torch_view(const std::string& filepath) 
{  
    int rows = 0, cols = 0, depth = 1, batch = 1;  
        
    auto data = load_matrix_bin(filepath.c_str(), rows, cols, depth, batch);  
    if (!data) 
    {  
        throw std::runtime_error("Failed to load matrix: " + filepath);  
    }  
        
    std::vector<int64_t> sizes;  
    if (batch > 1 && depth > 1) 
    {  
        sizes = {batch, depth, rows, cols};  
    } 
    else if (batch > 1)
    {  
        sizes = {batch, rows, cols};  
    }
    else 
    {  
        sizes = {rows, cols};      
    }  
        
    auto options = torch::TensorOptions()  
        .dtype(torch::kFloat32)  
        .device(torch::kCPU);  
        
            // Release ownership so torch owns it  
    float* raw_ptr = data.release();  
        
    return torch::from_blob(  
        raw_ptr,  
        sizes,  
                // Custom deleter called when tensor is freed  
        [](void* ptr) {  
            delete[] static_cast<float*>(ptr);  
        },  
        options  
    );  
}

// Helper: Convert raw float data into a Torch tensor
at::Tensor convert_matrix_to_torch(
    const std::unique_ptr<float[]>& data,
    const int dims[4])
{
    std::vector<int64_t> sizes = {
        dims[0], dims[1], dims[2], dims[3]
    };

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCPU);

    return torch::from_blob(
        data.get(),   // safe: read-only
        sizes,
        options
    );
}

void print_tensor_start_flat(
    const torch::Tensor& t,
    const std::string& name,
    int64_t count = 100
) {
    auto flat = t.reshape({-1}).cpu();

    std::cout << "\n" << name << " - first " << count
              << " values (flat order):\n";

    for (int64_t i = 0; i < std::min(count, flat.numel()); i++) {
        std::cout << flat[i].item<float>() << " ";
    }
    std::cout << "\n";
}

int get_physical_cores() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    std::set<std::string> core_ids;
    
    while (std::getline(cpuinfo, line)) {
        if (line.find("core id") != std::string::npos) {
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                std::string core_id = line.substr(pos + 1);
                core_id.erase(0, core_id.find_first_not_of(" \t"));
                core_id.erase(core_id.find_last_not_of(" \t") + 1);
                core_ids.insert(core_id);
            }
        }
    }
    
    if (!core_ids.empty()) {
        return core_ids.size();
    }
    
    // Simple fallback
    unsigned int logical_cores = std::thread::hardware_concurrency();
    if (logical_cores == 0) {
        return 4;
    }
    return logical_cores;
}

class llama_matrix_backend
{
    public:
        int number_of_physical_cores; 
        std::vector<ggml_backend_t> ggml_backends;
        std::mutex backends_mutex;

        // NEW: persistent per-device GGML contexts and memory buffers
        std::vector<struct ggml_context*> ggml_ctxs;     // one context per backend/device
        std::vector<void*> ggml_mem_buffers;             // raw buffer pointers for ggml contexts
        std::vector<size_t> ggml_mem_sizes;              // sizes of the above buffers
        std::vector<std::unique_ptr<std::mutex>> device_mutexes;          // per-device mutex to reduce contention
        std::vector<std::unique_ptr<std::atomic<uint64_t>>> device_load_ptrs;  // simple load counters for scheduling

        // Host-side reusable buffer pool for transposes/results to avoid frequent alloc/free
        struct HostBufferPool {
            std::unordered_map<size_t, std::vector<void*>> pool; // keyed by size
            std::mutex mtx;
            bool use_pinned = false;

            void* get(size_t size) {
                std::lock_guard<std::mutex> lk(mtx);
                auto it = pool.find(size);
                if (it != pool.end() && !it->second.empty()) {
                    void* p = it->second.back();
                    it->second.pop_back();
                    return p;
                }
                // allocate aligned buffer; prefer posix_memalign
#ifdef _POSIX_C_SOURCE
                void* ptr = nullptr;
                if (posix_memalign(&ptr, 64, size) != 0) ptr = nullptr;
                if (!ptr) ptr = malloc(size);
#else
                void* ptr = malloc(size);
#endif
                if (use_pinned && ptr) {
                    // try to mlock for pinned behavior; ignore failures
                    mlock(ptr, size);
                }
                return ptr;
            }

            void put(void* ptr, size_t size) {
                if (!ptr) return;
                std::lock_guard<std::mutex> lk(mtx);
                pool[size].push_back(ptr);
            }

            ~HostBufferPool() {
                for (auto &kv : pool) {
                    for (void* p : kv.second) free(p);
                }
            }
        } host_pool;

        // Simple instrumentation
        std::atomic<uint64_t> stat_ops{0};
        std::atomic<uint64_t> stat_allocs{0};
        
    public: 
        llama_matrix_backend() : number_of_physical_cores(get_physical_cores()) {
            Initialize_backend();
        }

        void Initialize_backend()
        {
                // Clear any existing backends
            ggml_backends.clear();
            ggml_ctxs.clear();
            ggml_mem_buffers.clear();
            ggml_mem_sizes.clear();
            device_mutexes.clear();
            device_load_ptrs.clear();
                
                // Load all available backends
            ggml_backend_load_all();
                
                // Get count of available devices
            size_t device_count = ggml_backend_dev_count();  
            std::cout << "Found " << device_count << " devices:" << std::endl;  
                
                // Reserve capacity to avoid reallocations of non-movable types
            ggml_backends.reserve(device_count + 1);
            ggml_ctxs.reserve(device_count + 1);
            ggml_mem_buffers.reserve(device_count + 1);
            ggml_mem_sizes.reserve(device_count + 1);
            device_mutexes.reserve(device_count + 1);
            device_load_ptrs.reserve(device_count + 1);

            // Add all GPU backends to the vector
            for (size_t i = 0; i < device_count; i++) 
            {  
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);  
                const char* name = ggml_backend_dev_name(dev);  
                const char* desc = ggml_backend_dev_description(dev);  
                std::cout << "  Device " << i << ": " << name << " (" << desc << ")" << std::endl;  
                    
                    // Initialize and add this backend to the vector
                    // Reserve a persistent context buffer for each GPU device; size can be tuned
                    // Increase default to larger value to avoid OOM for large matrix ops.
                    // Can be overridden per-system via the `GGML_CTX_MEM_MB` env var.
                    size_t ctx_mb = 512; // default 512MB per device (was 128MB)
                    const char* env = std::getenv("GGML_CTX_MEM_MB");
                    if (env) ctx_mb = std::stoul(env);

                    void* mem_buf = malloc(ctx_mb * 1024 * 1024);
                    if (!mem_buf) {
                        std::cerr << "Warning: Failed to allocate ggml mem buffer for device " << i << std::endl;
                    }

                    ggml_init_params params;
                    params.mem_size = ctx_mb * 1024 * 1024;
                    params.mem_buffer = mem_buf;
                    // Use no_alloc=true so backend allocation routines are used
                    // for persistent contexts (we rely on backend buffers).
                    params.no_alloc = true;
                    struct ggml_context* ctx = ggml_init(params);

                    ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
                    if (backend) {
                        ggml_backends.push_back(backend);
                        ggml_ctxs.push_back(ctx);
                        ggml_mem_buffers.push_back(mem_buf);
                        ggml_mem_sizes.push_back(params.mem_size);
                        device_mutexes.emplace_back(std::make_unique<std::mutex>());
                        device_load_ptrs.emplace_back(std::make_unique<std::atomic<uint64_t>>(0));
                        std::cout << "    ✓ Added to backends vector and created persistent context" << std::endl;
                    } else {
                        // if backend init failed, free temp ctx/mem
                        if (ctx) ggml_free(ctx);
                        if (mem_buf) free(mem_buf);
                    }
            }  
                
                // Always add CPU backend as fallback
            ggml_backend_t cpu_backend = ggml_backend_cpu_init();
            if (cpu_backend) {
                // Create a persistent context for CPU backend as well
                // Increase default to reduce chances of running out of CPU context memory.
                // Can be overridden via `GGML_CPU_CTX_MEM_MB` env var.
                size_t cpu_ctx_mb = 256; // default 256MB (was 64MB)
                const char* env_cpu = std::getenv("GGML_CPU_CTX_MEM_MB");
                if (env_cpu) cpu_ctx_mb = std::stoul(env_cpu);
                void* cpu_buf = malloc(cpu_ctx_mb * 1024 * 1024);
                ggml_init_params cpu_params;
                cpu_params.mem_size = cpu_ctx_mb * 1024 * 1024;
                cpu_params.mem_buffer = cpu_buf;
                // For CPU persistent context use no_alloc=true as well
                cpu_params.no_alloc = true;
                struct ggml_context* cpu_ctx = ggml_init(cpu_params);

                ggml_backends.push_back(cpu_backend);
                ggml_ctxs.push_back(cpu_ctx);
                ggml_mem_buffers.push_back(cpu_buf);
                ggml_mem_sizes.push_back(cpu_params.mem_size);
                device_mutexes.emplace_back(std::make_unique<std::mutex>());
                device_load_ptrs.emplace_back(std::make_unique<std::atomic<uint64_t>>(0));

                std::cout << "✓ Added CPU backend with persistent context" << std::endl;
            }
                
            std::cout << "Total backends initialized: " << ggml_backends.size() << std::endl;
            // configure host buffer pool behavior from env
            const char* env_pinned = std::getenv("GGML_USE_PINNED_HOST_MEM");
            if (env_pinned && std::string(env_pinned) == "1") host_pool.use_pinned = true;
        }
            
        // 2D Transpose: ✓ WORKING
        std::unique_ptr<float[]> transpose_2d(float* matrix, int rows, int cols)  
        {  
            // Blocked transpose for better cache locality
            auto result = std::make_unique<float[]>(rows * cols);
            const int B = 64; // tile size (tunable)
            for (int i = 0; i < rows; i += B) {
                for (int j = 0; j < cols; j += B) {
                    int ib = std::min(B, rows - i);
                    int jb = std::min(B, cols - j);
                    for (int ii = 0; ii < ib; ++ii) {
                        for (int jj = 0; jj < jb; ++jj) {
                            result[(j + jj) * rows + (i + ii)] = matrix[(i + ii) * cols + (j + jj)];
                        }
                    }
                }
            }
            return result;
        }  

        // 3D Transpose: Transpose each 2D slice (batch, rows, cols) -> (batch, cols, rows)
        std::unique_ptr<float[]> transpose_3d(float* matrix, int batch, int rows, int cols)  
        {  
            auto result = std::make_unique<float[]>(batch * rows * cols);
            int slice_size = rows * cols;
            const int B = 64;
            for (int b = 0; b < batch; b++) {
                float* src_slice = matrix + b * slice_size;
                float* dst_slice = result.get() + b * slice_size;
                for (int i = 0; i < rows; i += B) {
                    for (int j = 0; j < cols; j += B) {
                        int ib = std::min(B, rows - i);
                        int jb = std::min(B, cols - j);
                        for (int ii = 0; ii < ib; ++ii) {
                            for (int jj = 0; jj < jb; ++jj) {
                                dst_slice[(j + jj) * rows + (i + ii)] = src_slice[(i + ii) * cols + (j + jj)];
                            }
                        }
                    }
                }
            }
            return result;  
        }  


        std::unique_ptr<float[]> transpose_4d(float* matrix, int batch, int depth, int rows, int cols)  
        {  
            auto result = std::make_unique<float[]>(batch * depth * rows * cols);
            int depth_size = rows * cols;
            int batch_size = depth * depth_size;
            const int B = 64;

            for (int b = 0; b < batch; b++) {
                for (int d = 0; d < depth; d++) {
                    float* src_slice = matrix + b * batch_size + d * depth_size;
                    float* dst_slice = result.get() + b * batch_size + d * depth_size;
                    for (int i = 0; i < rows; i += B) {
                        for (int j = 0; j < cols; j += B) {
                            int ib = std::min(B, rows - i);
                            int jb = std::min(B, cols - j);
                            for (int ii = 0; ii < ib; ++ii) {
                                for (int jj = 0; jj < jb; ++jj) {
                                    dst_slice[(j + jj) * rows + (i + ii)] = src_slice[(i + ii) * cols + (j + jj)];
                                }
                            }
                        }
                    }
                }
            }
            return result;
        }

        ggml_backend_t pick_least_loaded_backend() {
            // return backend with minimal reported load (simple scheduler)
            size_t best = 0;
            uint64_t best_load = UINT64_MAX;
            for (size_t i = 0; i < ggml_backends.size(); ++i) {
                uint64_t load = 0;
                if (i < device_load_ptrs.size() && device_load_ptrs[i]) load = device_load_ptrs[i]->load(std::memory_order_relaxed);
                if (load < best_load) { best_load = load; best = i; }
            }
            return ggml_backends.empty() ? (ggml_backend_t)nullptr : ggml_backends[best];
        }

        MatrixResult matrix_op_nd(  
            float* matrix_a, const int dims_a[4],  
            float* matrix_b, const int dims_b[4],  
            ggml_backend_t backend,  
            const std::string& op = "mul")  
        {  
            stat_ops.fetch_add(1, std::memory_order_relaxed);
                // Validate dimensions - NO ZERO DIMENSIONS ALLOWED  
            for (int i = 0; i < 4; i++) {  
                if (dims_a[i] == 0 || dims_b[i] == 0) {  
                    std::cerr << "Error: Zero dimension detected at index " << i << std::endl;  
                    return {nullptr, {0,0,0,0}};  
                }  
            }  
                
                // Initialize GGML context (use persistent per-device context when available)
            int backend_index = -1;
            if (!backend) {
                backend = pick_least_loaded_backend();
            }
            for (size_t bi = 0; bi < ggml_backends.size(); ++bi) {
                if (ggml_backends[bi] == backend) { backend_index = (int)bi; break; }
            }

            struct ggml_context* ctx = nullptr;
            bool ctx_is_persistent = false;
            if (backend_index >= 0 && backend_index < (int)ggml_ctxs.size() && ggml_ctxs[backend_index]) {
                ctx = ggml_ctxs[backend_index];
                ctx_is_persistent = true;
            } else {
                struct ggml_init_params params;
                params.mem_size = 16 * 1024 * 1024;
                params.mem_buffer = NULL;
                // temporary contexts must allow allocations for tensor/back-end work
                params.no_alloc = false;
                ctx = ggml_init(params);
            }

            // RAII: only free ctx if it was a temporary allocation
            auto ctx_deleter = [&](struct ggml_context* p) { if (p && !ctx_is_persistent) ggml_free(p); };
            std::unique_ptr<struct ggml_context, decltype(ctx_deleter)> ctx_holder(ctx, ctx_deleter);
                
                // Validate dimensions for matrix multiplication  
                if (op == "mul") {  
                if (dims_a[0] != dims_b[0]) {  
                    std::cerr << "Error: For matrix multiplication, dims_a[0] must equal dims_b[0]" << std::endl;  
                    return {nullptr, {0,0,0,0}};  
                }  
                if ((dims_b[2] % dims_a[2] != 0) || (dims_b[3] % dims_a[3] != 0)) {  
                    std::cerr << "Error: Broadcasting rules violated" << std::endl;  
                    return {nullptr, {0,0,0,0}};  
                }  
            }  
                
                // Create tensors (data pointers are NULL at this point)  
            struct ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dims_a[0], dims_a[1], dims_a[2], dims_a[3]);  
            struct ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dims_b[0], dims_b[1], dims_b[2], dims_b[3]);  
                
                // Build computation graph  
            struct ggml_tensor* result;  
            if (op == "mul") {  
                result = ggml_mul_mat(ctx, a, b);  
            } else if (op == "add") {  
                result = ggml_add(ctx, a, b);  
            } else if (op == "sub") {  
                result = ggml_sub(ctx, a, b);  
            } else {  
                return {nullptr, {0,0,0,0}};  
            }  
                
                // Create and build computation graph  
            struct ggml_cgraph* gf = ggml_new_graph(ctx);  
            ggml_build_forward_expand(gf, result);  
                
                // Allocate tensors on backend FIRST  
            // Use per-device mutex to serialize allocations/compute per device if available
            std::unique_lock<std::mutex> device_lock;
            bool incremented_load = false;
            if (backend_index >= 0 && backend_index < (int)device_mutexes.size()) {
                device_lock = std::unique_lock<std::mutex>(*device_mutexes[backend_index]);
                // increment load counter
                if (backend_index >= 0 && backend_index < (int)device_load_ptrs.size() && device_load_ptrs[backend_index]) {
                    device_load_ptrs[backend_index]->fetch_add(1, std::memory_order_relaxed);
                    incremented_load = true;
                }
            }

            // Allocate backend buffers for this graph. If allocation fails, return early.
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);  
            if (!buf) {  
                if (incremented_load && backend_index >= 0 && backend_index < (int)device_load_ptrs.size() && device_load_ptrs[backend_index]) {
                    device_load_ptrs[backend_index]->fetch_sub(1, std::memory_order_relaxed);
                }
                return {nullptr, {0,0,0,0}};  
            }  
                
                // Calculate total elements for data transfer  
            int64_t total_a = dims_a[0] * dims_a[1] * dims_a[2] * dims_a[3];  
            int64_t total_b = dims_b[0] * dims_b[1] * dims_b[2] * dims_b[3];  
                
                // NOW copy data to tensors using backend functions  
            ggml_backend_tensor_set(a, matrix_a, 0, sizeof(float) * total_a);  
            ggml_backend_tensor_set(b, matrix_b, 0, sizeof(float) * total_b);  
                
                // Execute computation  
            ggml_backend_graph_compute(backend, gf);  
                
                // Extract result data using backend function  
            MatrixResult output;  
            int64_t total_result = result->ne[0] * result->ne[1] * result->ne[2] * result->ne[3];  
            output.data = std::make_unique<float[]>(total_result);  

            
            // CORRECT - Keep dimensions as-is  
            output.dims[0] = result->ne[3]; // batch  
            output.dims[1] = result->ne[2]; // depth    
            output.dims[2] = result->ne[1]; // rows  
            output.dims[3] = result->ne[0]; // cols

                
            ggml_backend_tensor_get(result, output.data.get(), 0, sizeof(float) * total_result);  
            // so i can just change this to get the correct formaty>
            //make it so the tensor matchs the torch format after the mul so c1=c2

            // Cleanup
            ggml_backend_buffer_free(buf);
            if (incremented_load && backend_index >= 0 && backend_index < (int)device_load_ptrs.size() && device_load_ptrs[backend_index]) {
                device_load_ptrs[backend_index]->fetch_sub(1, std::memory_order_relaxed);
            }
            if (device_lock.owns_lock()) device_lock.unlock();

            stat_allocs.fetch_add(1, std::memory_order_relaxed);

            return output;  
        }

        // Variant: operate using pre-existing GGML tensors (e.g. cached in VRAM).
        // The input tensors must already be initialized on the target backend.
        MatrixResult matrix_op_nd_tensors(
            struct ggml_tensor* a,
            struct ggml_tensor* b,
            ggml_backend_t backend,
            const std::string& op = "mul")
        {
            stat_ops.fetch_add(1, std::memory_order_relaxed);

            if (!a || !b) {
                std::cerr << "Error: matrix_op_nd_tensors got null input tensor(s)" << std::endl;
                return {nullptr, {0, 0, 0, 0}};
            }

            int backend_index = -1;
            if (!backend) {
                backend = pick_least_loaded_backend();
            }
            for (size_t bi = 0; bi < ggml_backends.size(); ++bi) {
                if (ggml_backends[bi] == backend) {
                    backend_index = (int) bi;
                    break;
                }
            }

            ggml_init_params params;
            params.mem_size = 16 * 1024 * 1024;
            params.mem_buffer = NULL;
            params.no_alloc = false;
            struct ggml_context* ctx = ggml_init(params);
            if (!ctx) {
                std::cerr << "Error: ggml_init failed in matrix_op_nd_tensors" << std::endl;
                return {nullptr, {0, 0, 0, 0}};
            }

            auto ctx_deleter = [](struct ggml_context* p) { if (p) ggml_free(p); };
            std::unique_ptr<struct ggml_context, decltype(ctx_deleter)> ctx_holder(ctx, ctx_deleter);

            struct ggml_tensor* result = nullptr;
            if (op == "mul") {
                result = ggml_mul_mat(ctx, a, b);
            } else if (op == "add") {
                result = ggml_add(ctx, a, b);
            } else if (op == "sub") {
                result = ggml_sub(ctx, a, b);
            } else {
                std::cerr << "Error: Unsupported op in matrix_op_nd_tensors: " << op << std::endl;
                return {nullptr, {0, 0, 0, 0}};
            }

            struct ggml_cgraph* gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, result);

            std::unique_lock<std::mutex> device_lock;
            bool incremented_load = false;
            if (backend_index >= 0 && backend_index < (int) device_mutexes.size()) {
                device_lock = std::unique_lock<std::mutex>(*device_mutexes[backend_index]);
                if (backend_index >= 0 && backend_index < (int) device_load_ptrs.size() && device_load_ptrs[backend_index]) {
                    device_load_ptrs[backend_index]->fetch_add(1, std::memory_order_relaxed);
                    incremented_load = true;
                }
            }

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buf) {
                if (incremented_load && backend_index >= 0 && backend_index < (int) device_load_ptrs.size() && device_load_ptrs[backend_index]) {
                    device_load_ptrs[backend_index]->fetch_sub(1, std::memory_order_relaxed);
                }
                return {nullptr, {0, 0, 0, 0}};
            }

            ggml_backend_graph_compute(backend, gf);

            MatrixResult output;
            const int64_t total_result = result->ne[0] * result->ne[1] * result->ne[2] * result->ne[3];
            output.data = std::make_unique<float[]>(total_result);

            output.dims[0] = result->ne[3]; // batch
            output.dims[1] = result->ne[2]; // depth
            output.dims[2] = result->ne[1]; // rows
            output.dims[3] = result->ne[0]; // cols

            ggml_backend_tensor_get(result, output.data.get(), 0, sizeof(float) * total_result);

            ggml_backend_buffer_free(buf);
            if (incremented_load && backend_index >= 0 && backend_index < (int) device_load_ptrs.size() && device_load_ptrs[backend_index]) {
                device_load_ptrs[backend_index]->fetch_sub(1, std::memory_order_relaxed);
            }
            if (device_lock.owns_lock()) device_lock.unlock();

            stat_allocs.fetch_add(1, std::memory_order_relaxed);
            return output;
        }
};

// =====================================================================================
// VRAM buffer/cache setup (best-effort, backend-agnostic via GGML backends)
// =====================================================================================

struct vram_tensor_entry {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    struct ggml_context* ctx = nullptr;
    struct ggml_tensor* tensor = nullptr;
    size_t alloc_bytes = 0;
    int dims[4] = {0, 0, 0, 0}; // {cols, rows, depth, batch} in GGML order
};

struct vram_backend_state {
    bool enabled = false;
    size_t budget_bytes = 0;
    size_t used_bytes = 0;
    ggml_backend_buffer_t scratch = nullptr;
    size_t scratch_bytes = 0;
};

class vram_cache_manager {
public:
    void init(const std::vector<ggml_backend_t>& backends_in) {
        std::lock_guard<std::mutex> lock(mtx);
        if (initialized) return;
        backends = backends_in;
        states.assign(backends.size(), vram_backend_state{});
        initialized = true;
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mtx);
        for (auto& kv : cache) {
            if (kv.second.buffer) ggml_backend_buffer_free(kv.second.buffer);
            if (kv.second.ctx) ggml_free(kv.second.ctx);
        }
        cache.clear();

        for (auto& st : states) {
            if (st.scratch) ggml_backend_buffer_free(st.scratch);
            st.scratch = nullptr;
            st.enabled = false;
            st.budget_bytes = 0;
            st.used_bytes = 0;
            st.scratch_bytes = 0;
        }

        initialized = false;
    }

    void configure_defaults() {
        std::lock_guard<std::mutex> lock(mtx);
        const size_t budget_mb = env_size_mb("OPEN_CLUSTER_VRAM_CACHE_MB", 0);
        const size_t reserve_mb = env_size_mb("OPEN_CLUSTER_VRAM_RESERVE_MB", 2048);
        const size_t scratch_mb = env_size_mb("OPEN_CLUSTER_VRAM_SCRATCH_MB", 64);

        const size_t reserve_bytes = reserve_mb * 1024ull * 1024ull;

        const size_t scratch_bytes = scratch_mb * 1024ull * 1024ull;

        for (size_t i = 0; i < states.size(); ++i) {
            ggml_backend_t backend = backends[i];
            const std::string name = backend_name(i, backend);
            ggml_backend_buffer_type_t buft = backend ? ggml_backend_get_default_buffer_type(backend) : nullptr;
            const bool is_host = buft ? ggml_backend_buft_is_host(buft) : true;

            // Per-backend budget selection:
            // - If OPEN_CLUSTER_VRAM_CACHE_MB is set, use it for every non-host backend.
            // - Else, derive from backend-reported max size minus reserve.
            const size_t backend_max = backend ? ggml_backend_get_max_size(backend) : 0;
            size_t budget_bytes = 0;
            if (budget_mb > 0) {
                budget_bytes = budget_mb * 1024ull * 1024ull;
            } else if (backend_max > reserve_bytes) {
                budget_bytes = backend_max - reserve_bytes;
            }

            states[i].enabled = (!is_host) && (budget_bytes > 0);
            states[i].budget_bytes = budget_bytes;
            states[i].used_bytes = 0;
            states[i].scratch_bytes = 0;
            states[i].scratch = nullptr;

            if (!states[i].enabled) continue;
            if (!backend) { states[i].enabled = false; continue; }

            if (scratch_bytes > 0) {
                ggml_backend_buffer_t scratch = ggml_backend_alloc_buffer(backend, scratch_bytes);
                if (scratch) {
                    states[i].scratch = scratch;
                    states[i].scratch_bytes = scratch_bytes;
                } else {
                    // If scratch alloc fails, keep caching disabled for safety.
                    states[i].enabled = false;
                }
            }
        }
    }

    bool enabled(int backend_index) const {
        std::lock_guard<std::mutex> lock(mtx);
        return initialized &&
            backend_index >= 0 &&
            backend_index < (int) states.size() &&
            states[(size_t) backend_index].enabled;
    }

    size_t budget_bytes(int backend_index) const {
        std::lock_guard<std::mutex> lock(mtx);
        if (backend_index < 0 || backend_index >= (int) states.size()) return 0;
        return states[(size_t) backend_index].budget_bytes;
    }

    size_t used_bytes(int backend_index) const {
        std::lock_guard<std::mutex> lock(mtx);
        if (backend_index < 0 || backend_index >= (int) states.size()) return 0;
        return states[(size_t) backend_index].used_bytes;
    }

    size_t free_budget_bytes(int backend_index) const {
        std::lock_guard<std::mutex> lock(mtx);
        if (backend_index < 0 || backend_index >= (int) states.size()) return 0;
        const auto& st = states[(size_t) backend_index];
        if (!st.enabled) return 0;
        const size_t used = st.used_bytes + st.scratch_bytes;
        return (st.budget_bytes > used) ? (st.budget_bytes - used) : 0;
    }

    ggml_tensor* get_tensor(int backend_index, const std::string& name) const {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = cache.find(make_key(backend_index, name));
        if (it == cache.end()) return nullptr;
        return it->second.tensor;
    }

    bool cache_tensor_f32_4d(
        int backend_index,
        const std::string& name,
        const float* data,
        int cols,
        int rows,
        int depth,
        int batch)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!initialized || backend_index < 0 || backend_index >= (int) states.size()) return false;
        if (!states[(size_t) backend_index].enabled) return false;
        if (!data) return false;

        const std::string key = make_key(backend_index, name);
        if (cache.find(key) != cache.end()) {
            return true; // already cached
        }

        ggml_backend_t backend = backends[(size_t) backend_index];
        if (!backend) return false;

        // Build a dedicated context for this cached tensor (kept alive until shutdown).
        ggml_init_params params;
        params.mem_size = 4 * 1024 * 1024;
        params.mem_buffer = NULL;
        params.no_alloc = true;
        ggml_context* ctx = ggml_init(params);
        if (!ctx) return false;

        ggml_tensor* t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cols, rows, depth, batch);
        if (!t) {
            ggml_free(ctx);
            return false;
        }

        const size_t payload_bytes = ggml_nbytes(t);
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        const size_t alloc_bytes = ggml_backend_buft_get_alloc_size(buft, t);

        const size_t used_now = states[(size_t) backend_index].used_bytes + states[(size_t) backend_index].scratch_bytes;
        if (alloc_bytes == 0 || states[(size_t) backend_index].budget_bytes <= used_now || alloc_bytes > (states[(size_t) backend_index].budget_bytes - used_now)) {
            ggml_free(ctx);
            return false;
        }

        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, alloc_bytes);
        if (!buf) {
            ggml_free(ctx);
            return false;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        // Attach tensor to the backend buffer (sets tensor->buffer and tensor->data).
        // Without this, `ggml_backend_tensor_set()` will assert "tensor buffer not set".
        void* base = ggml_backend_buffer_get_base(buf);
        const ggml_status st = ggml_backend_tensor_alloc(buf, t, base);
        if (st != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return false;
        }

        ggml_backend_tensor_set(t, data, 0, payload_bytes);

        vram_tensor_entry entry;
        entry.backend = backend;
        entry.buffer = buf;
        entry.ctx = ctx;
        entry.tensor = t;
        entry.alloc_bytes = alloc_bytes;
        entry.dims[0] = cols;
        entry.dims[1] = rows;
        entry.dims[2] = depth;
        entry.dims[3] = batch;

        cache.emplace(key, entry);
        states[(size_t) backend_index].used_bytes += alloc_bytes;
        return true;
    }

private:
    static size_t env_size_mb(const char* key, size_t def_mb) {
        const char* v = std::getenv(key);
        if (!v || !*v) return def_mb;
        try {
            return std::stoull(v);
        } catch (...) {
            return def_mb;
        }
    }

    static std::string backend_name(size_t idx, ggml_backend_t backend) {
        (void) idx;
        if (!backend) return "";
        const char* n = ggml_backend_name(backend);
        return n ? std::string(n) : std::string();
    }

    static bool is_cpu_backend(const std::string& name) {
        const auto lower = [](std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
            return s;
        };
        std::string n = lower(name);
        return (n.find("cpu") != std::string::npos);
    }

    static std::string make_key(int backend_index, const std::string& name) {
        return std::to_string(backend_index) + "|" + name;
    }

    mutable std::mutex mtx;
    bool initialized = false;
    std::vector<ggml_backend_t> backends;
    std::vector<vram_backend_state> states;
    std::unordered_map<std::string, vram_tensor_entry> cache;
};

static inline vram_cache_manager& get_vram_cache_manager() {
    static vram_cache_manager mgr;
    return mgr;
}

static inline void set_VRAM_buffers(llama_matrix_backend& llama_backend) {
    auto& mgr = get_vram_cache_manager();
    mgr.init(llama_backend.ggml_backends);
    mgr.configure_defaults();
    // Ensure we always attempt to clean up even if server exits via std::exit().
    static bool registered = false;
    if (!registered) {
        registered = true;
        std::atexit([]() {
            try {
                get_vram_cache_manager().shutdown();
            } catch (...) {
            }
        });
    }
}

/*
int main() 
{
    std::cout << "Testing GGML matrix loading..." << std::endl;
    
    // 2D test files
    const char* test_2d_a = "matrix_shards/test_2d_a.bin";
    const char* test_2d_b = "matrix_shards/test_2d_b.bin";
    
    // 3D test files
    const char* test_3d_a = "matrix_shards/test_3d_a.bin";
    const char* test_3d_b = "matrix_shards/test_3d_b.bin";
    
    // 4D test files
    const char* test_4d_a = "matrix_shards/test_4d_a.bin";
    const char* test_4d_b = "matrix_shards/test_4d_b.bin";
    
    llama_matrix_backend server;
    
    // =========================================================================
    // 2D MATRIX MULTIPLICATION TEST
    // =========================================================================
    //std::cout << "\n" << "="*60 << std::endl;
    std::cout << "2D MATRIX MULTIPLICATION TEST" << std::endl;
    //std::cout << "="*60 << std::endl;
    
    {
        std::unique_ptr<float[]> matrix_A = nullptr;
        std::unique_ptr<float[]> matrix_B = nullptr;
        int rows_A, cols_A, rows_B, cols_B, depthA=0, batchA=0, depthB=0, batchB=0;  
        
        matrix_A = server.load_matrix_bin(test_2d_a, rows_A, cols_A, batchA, depthA);   
        matrix_B = server.load_matrix_bin(test_2d_b, rows_B, cols_B, batchB, depthB);  
        
        if (!matrix_A || !matrix_B) {
            std::cerr << "Failed to load 2D matrices" << std::endl;
        } else {
            // Transpose B for GGML convention
            auto matrix_B_T = server.transpose_2d(matrix_B.get(), rows_B, cols_B);
            int rows_B_T = cols_B;
            int cols_B_T = rows_B;
            
            std::cout << "Original A: " << rows_A << "x" << cols_A << std::endl;
            std::cout << "Original B: " << rows_B << "x" << cols_B << std::endl;
            std::cout << "Transposed B: " << rows_B_T << "x" << cols_B_T << std::endl;
            
            int dims2d_a[4] = {cols_A, rows_A, 1, 1};
            int dims2d_b_T[4] = {rows_B, cols_B, 1, 1};
            
            std::cout << "\nMatrix A:" << std::endl;
            server.print_matrix(matrix_A.get(), dims2d_a, 20);
            std::cout << "Matrix B (transposed):" << std::endl;
            server.print_matrix(matrix_B_T.get(), dims2d_b_T, 15);
            
            llama_matrix_backend::MatrixResult result = server.matrix_op_nd(
                matrix_A.get(), dims2d_a, 
                matrix_B_T.get(), dims2d_b_T, 
                server.ggml_backends[0], "mul"
            );
            
            std::cout << "2D Result (A @ B):" << std::endl;
            server.print_matrix(result.data.get(), result.dims, 12);
        }
    }
    
    // =========================================================================
    // 3D MATRIX MULTIPLICATION TEST
    // =========================================================================
    //std::cout << "\n" << "="*60 << std::endl;
    std::cout << "3D MATRIX MULTIPLICATION TEST" << std::endl;
    //std::cout << "="*60 << std::endl;
    
    {
        std::unique_ptr<float[]> matrix_A = nullptr;
        std::unique_ptr<float[]> matrix_B = nullptr;
        int rows_A, cols_A, rows_B, cols_B, depthA=0, batchA=0, depthB=0, batchB=0;  
        
        matrix_A = server.load_matrix_bin(test_3d_a, rows_A, cols_A, batchA, depthA);   
        matrix_B = server.load_matrix_bin(test_3d_b, rows_B, cols_B, batchB, depthB);  
        
        if (!matrix_A || !matrix_B) {
            std::cerr << "Failed to load 3D matrices" << std::endl;
        } else {
            std::cout << "Original A: batch=" << batchA << ", rows=" << rows_A << ", cols=" << cols_A << std::endl;
            std::cout << "Original B: batch=" << batchB << ", rows=" << rows_B << ", cols=" << cols_B << std::endl;
            
            // Transpose each batch in B for GGML convention
            auto matrix_B_T = server.transpose_3d(matrix_B.get(), batchB, rows_B, cols_B);
            
            // For GGML: dims = [cols, rows, 1, batch]
            int dims3d_a[4] = {cols_A, rows_A, 1, batchA};
            int dims3d_b_T[4] = {rows_B, cols_B, 1, batchB};  // B is transposed
            
            std::cout << "\nMatrix A (first batch):" << std::endl;
            server.print_matrix(matrix_A.get(), dims3d_a, 6);
            std::cout << "Matrix B transposed (first batch):" << std::endl;
            server.print_matrix(matrix_B_T.get(), dims3d_b_T, 6);
            
            llama_matrix_backend::MatrixResult result = server.matrix_op_nd(
                matrix_A.get(), dims3d_a, 
                matrix_B_T.get(), dims3d_b_T, 
                server.ggml_backends[0], "mul"
            );
            
            std::cout << "3D Result (batch matmul):" << std::endl;
            server.print_matrix(result.data.get(), result.dims, std::min(12, result.dims[0]*result.dims[1]*result.dims[2]*result.dims[3]));
        }
    }
    
    // =========================================================================
    // 4D MATRIX MULTIPLICATION TEST
    // =========================================================================
    //std::cout << "\n" << "="*60 << std::endl;
    std::cout << "4D MATRIX MULTIPLICATION TEST" << std::endl;
    //std::cout << "="*60 << std::endl;
    
    {
        std::unique_ptr<float[]> matrix_A = nullptr;
        std::unique_ptr<float[]> matrix_B = nullptr;
        int rows_A, cols_A, rows_B, cols_B, depthA=0, batchA=0, depthB=0, batchB=0;  
        
        matrix_A = server.load_matrix_bin(test_4d_a, rows_A, cols_A, batchA, depthA);   
        matrix_B = server.load_matrix_bin(test_4d_b, rows_B, cols_B, batchB, depthB);  
        
        if (!matrix_A || !matrix_B) {
            std::cerr << "Failed to load 4D matrices" << std::endl;
        } else {
            std::cout << "Original A: batch=" << batchA << ", depth=" << depthA 
                      << ", rows=" << rows_A << ", cols=" << cols_A << std::endl;
            std::cout << "Original B: batch=" << batchB << ", depth=" << depthB 
                      << ", rows=" << rows_B << ", cols=" << cols_B << std::endl;
            
            // Transpose each 2D slice in B for GGML convention
            auto matrix_B_T = server.transpose_4d(matrix_B.get(), batchB, depthB, rows_B, cols_B);
            
            // For GGML: dims = [cols, rows, depth, batch]
            int dims4d_a[4] = {cols_A, rows_A, depthA, batchA};
            int dims4d_b_T[4] = {rows_B, cols_B, depthB, batchB};  // B is transposed
            
            std::cout << "\nMatrix A (first batch, first depth):" << std::endl;
            server.print_matrix(matrix_A.get(), dims4d_a, 4);
            std::cout << "Matrix B transposed (first batch, first depth):" << std::endl;
            server.print_matrix(matrix_B_T.get(), dims4d_b_T, 4);
            
            llama_matrix_backend::MatrixResult result = server.matrix_op_nd(
                matrix_A.get(), dims4d_a, 
                matrix_B_T.get(), dims4d_b_T, 
                server.ggml_backends[2], "mul"
            );
            
            std::cout << "4D Result (batch of batch matmul):" << std::endl;
            server.print_matrix(result.data.get(), result.dims, std::min(8, result.dims[0]*result.dims[1]*result.dims[2]*result.dims[3]));
        }
    }
    
    //std::cout << "\n" << "="*60 << std::endl;
    std::cout << "ALL TESTS COMPLETED" << std::endl;
    //std::cout << "="*60 << std::endl;
    
    return 0;
}
*/
