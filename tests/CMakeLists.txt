cmake_minimum_required(VERSION 3.18)

# Find testing framework
include(CTest)

# Test executables
add_executable(test_quantization test_quantization.cpp)
target_link_libraries(test_quantization PRIVATE cuda_open)

add_executable(test_memory test_memory.cpp)
target_link_libraries(test_memory PRIVATE cuda_open)

add_executable(test_device test_device.cpp)
target_link_libraries(test_device PRIVATE cuda_open)

add_executable(test_bitnet_7b test_bitnet_7b.cpp)
target_link_libraries(test_bitnet_7b PRIVATE cuda_open)

# Register tests
add_test(NAME QuantizationTest COMMAND test_quantization)
add_test(NAME MemoryTest COMMAND test_memory)
add_test(NAME DeviceTest COMMAND test_device)
add_test(NAME BitNet7BTest COMMAND test_bitnet_7b)

# Set test properties
set_tests_properties(QuantizationTest MemoryTest DeviceTest BitNet7BTest
    PROPERTIES TIMEOUT 60
)
