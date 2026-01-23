cmake_minimum_required(VERSION 3.14)
project(ggml_cluster_backend)

# --- Standard C++ ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# --- Threads ---
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# --- Include directories ---
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# --- Add subdirectory for cluster_backend_main ---
add_subdirectory(cluster_backend_main)
