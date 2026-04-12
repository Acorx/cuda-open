cmake_minimum_required(VERSION 3.18)
project(cuda_open VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(CUDA_OPEN_BUILD_TESTS "Build tests" ON)
option(CUDA_OPEN_BUILD_EXAMPLES "Build examples" ON)
option(CUDA_OPEN_ENABLE_CUDA "Enable CUDA/GPU support" OFF)
option(CUDA_OPEN_OPTIMIZE "Enable optimizations" ON)

# Find CUDA (optional)
if(CUDA_OPEN_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
        add_definitions(-DHAVE_CUDA)
        enable_language(CUDA)
    else()
        message(WARNING "CUDA not found, building without GPU support")
        set(CUDA_OPEN_ENABLE_CUDA OFF)
    endif()
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CUDA_OPEN_OPTIMIZE)
        add_compile_options(-O3 -march=native)
    endif()
endif()

if(MSVC)
    add_compile_options(/W4)
    if(CUDA_OPEN_OPTIMIZE)
        add_compile_options(/O2)
    endif()
endif()

# Source files
set(CUDA_OPEN_SOURCES
    src/device.cpp
    src/quantization.cpp
    src/kernel.cpp
    src/tensor.cpp
    src/kernel_optimizer.cpp
)

# Add CUDA sources if available
if(CUDA_OPEN_ENABLE_CUDA)
    list(APPEND CUDA_OPEN_SOURCES
        src/device_gpu.cu
    )
endif()

# Create library
add_library(cuda_open ${CUDA_OPEN_SOURCES})

# Include directories
target_include_directories(cuda_open
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link threading library
find_package(Threads REQUIRED)
target_link_libraries(cuda_open PUBLIC Threads::Threads)

# Link CUDA if available
if(CUDA_OPEN_ENABLE_CUDA)
    target_link_libraries(cuda_open PUBLIC CUDA::cudart)
endif()

# Tests
if(CUDA_OPEN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(CUDA_OPEN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS cuda_open
    EXPORT cuda_open-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/cuda_open
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT cuda_open-targets
    FILE cuda_open-config.cmake
    NAMESPACE cuda_open::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuda_open
)

# Print configuration summary
message(STATUS "")
message(STATUS "CUDA Open Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Support: ${CUDA_OPEN_ENABLE_CUDA}")
message(STATUS "  Build Tests: ${CUDA_OPEN_BUILD_TESTS}")
message(STATUS "  Build Examples: ${CUDA_OPEN_BUILD_EXAMPLES}")
message(STATUS "  Optimizations: ${CUDA_OPEN_OPTIMIZE}")
message(STATUS "")
