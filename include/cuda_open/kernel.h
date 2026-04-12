/**
 * @file kernel.h
 * @brief Kernel execution framework with CPU/GPU abstraction
 * 
 * Provides a unified interface for defining and executing compute kernels
 * on both CPU and GPU devices.
 */

#ifndef CUDA_OPEN_KERNEL_H
#define CUDA_OPEN_KERNEL_H

#include "device.h"
#include "memory.h"
#include <functional>
#include <vector>
#include <string>
#include <any>

namespace cuda_open {

// Kernel launch configuration
struct LaunchConfig {
    int grid_dim_x = 1;
    int grid_dim_y = 1;
    int grid_dim_z = 1;
    int block_dim_x = 1;
    int block_dim_y = 1;
    int block_dim_z = 1;
    size_t shared_memory = 0;
    
    // Helper to compute block/grid dimensions for 1D problems
    static LaunchConfig make_1d_config(size_t num_elements, int block_size = 256);
    
    // Helper to compute block/grid dimensions for 2D problems
    static LaunchConfig make_2d_config(size_t width, size_t height, 
                                       int block_dim_x = 16, int block_dim_y = 16);
    
    // Helper to compute block/grid dimensions for 3D problems
    static LaunchConfig make_3d_config(size_t width, size_t height, size_t depth,
                                       int block_dim_x = 8, int block_dim_y = 8, int block_dim_z = 8);
};

// Kernel arguments wrapper
class KernelArgs {
public:
    template<typename T>
    void add(const T& arg) {
        args_.push_back(arg);
    }
    
    template<typename T>
    void add_ptr(T* ptr) {
        ptrs_.push_back(static_cast<void*>(ptr));
    }
    
    void add_device_ptr(void* ptr) {
        ptrs_.push_back(ptr);
    }
    
    const std::vector<void*>& get_pointers() const { return ptrs_; }
    
private:
    std::vector<std::any> args_;
    std::vector<void*> ptrs_;
};

// Kernel launcher
class KernelLauncher {
public:
    KernelLauncher(Device& device);
    
    // Launch a kernel
    void launch(
        std::function<void(void**)> kernel_func,
        const LaunchConfig& config,
        KernelArgs& args
    );
    
    // Launch with simple function
    void launch_simple(
        std::function<void()> kernel_func,
        const LaunchConfig& config
    );
    
    // Synchronize device
    void synchronize();
    
private:
    Device& device_;
};

// Kernel thread ID simulation
struct ThreadId {
    int block_idx_x;
    int block_idx_y;
    int block_idx_z;
    int thread_idx_x;
    int thread_idx_y;
    int thread_idx_z;
    
    // Global thread ID
    int global_id_x() const {
        return block_idx_x * block_dim_x + thread_idx_x;
    }
    
    int global_id_y() const {
        return block_idx_y * block_dim_y + thread_idx_y;
    }
    
    int global_id_z() const {
        return block_idx_z * block_dim_z + thread_idx_z;
    }
    
    int block_dim_x;
    int block_dim_y;
    int block_dim_z;
    int grid_dim_x;
    int grid_dim_y;
    int grid_dim_z;
};

} // namespace cuda_open

#endif // CUDA_OPEN_KERNEL_H
