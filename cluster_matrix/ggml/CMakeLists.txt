cmake_minimum_required(VERSION 3.18)
project(ggml_project)

# --- Standard C++ ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Threads ---
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# --- BLAS ---
find_package(OpenBLAS REQUIRED)

# --- Vulkan/OpenCL (optional) ---
find_package(Vulkan)
find_package(OpenCL)

# --- Include directories for ggml root ---
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# --- Add subdirectory for backend ---
add_subdirectory(src)
add_subdirectory(cluster_backend)

