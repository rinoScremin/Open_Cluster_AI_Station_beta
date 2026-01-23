cmake_minimum_required(VERSION 3.18)
project(matrix_zmq_server LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Torch ----
set(Torch_DIR "${CMAKE_SOURCE_DIR}/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# ---- Executable ----
add_executable(matrix_zmq_server
    matrix_zmq_server.cpp
)

# ---- ggml ----
target_include_directories(matrix_zmq_server PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(matrix_zmq_server PRIVATE
    ggml
)

# ---- Torch includes + libs ----
target_include_directories(matrix_zmq_server PRIVATE
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(matrix_zmq_server PRIVATE
    ${TORCH_LIBRARIES}
)

find_package(OpenCL REQUIRED)

target_link_libraries(matrix_zmq_server
    PRIVATE ggml
    PRIVATE OpenCL::OpenCL
)


# ---- ZMQ ----
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

target_include_directories(matrix_zmq_server PRIVATE
    ${ZMQ_INCLUDE_DIRS}
)

target_link_libraries(matrix_zmq_server PRIVATE
    ${ZMQ_LIBRARIES}
)

# ---- Threads ----
find_package(Threads REQUIRED)
target_link_libraries(matrix_zmq_server PRIVATE Threads::Threads)
