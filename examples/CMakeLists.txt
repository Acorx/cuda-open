cmake_minimum_required(VERSION 3.18)

# Example 1: Basic memory operations
add_executable(example_01_memory 01_basic_memory.cpp)
target_link_libraries(example_01_memory PRIVATE cuda_open)

# Example 2: Kernel execution
add_executable(example_02_kernel 02_kernel_execution.cpp)
target_link_libraries(example_02_kernel PRIVATE cuda_open)

# Example 3: Quantization
add_executable(example_03_quantization 03_quantization.cpp)
target_link_libraries(example_03_quantization PRIVATE cuda_open)

# Example 4: BitNet matrix multiplication
add_executable(example_04_bitnet 04_bitnet_matmul.cpp)
target_link_libraries(example_04_bitnet PRIVATE cuda_open)

# Example 5: Revolutionary optimizations
add_executable(example_05_revolutionary 05_revolutionary_optimizations.cpp)
target_link_libraries(example_05_revolutionary PRIVATE cuda_open)
