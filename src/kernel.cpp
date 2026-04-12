/**
 * @file kernel.cpp
 * @brief Kernel execution framework implementation
 */

#include "cuda_open/kernel.h"
#include <algorithm>

namespace cuda_open {

// ============================================================================
// LaunchConfig Helpers
// ============================================================================

LaunchConfig LaunchConfig::make_1d_config(size_t num_elements, int block_size) {
    LaunchConfig config;
    config.block_dim_x = block_size;
    config.grid_dim_x = static_cast<int>((num_elements + block_size - 1) / block_size);
    return config;
}

LaunchConfig LaunchConfig::make_2d_config(size_t width, size_t height, 
                                          int block_dim_x, int block_dim_y) {
    LaunchConfig config;
    config.block_dim_x = block_dim_x;
    config.block_dim_y = block_dim_y;
    config.grid_dim_x = static_cast<int>((width + block_dim_x - 1) / block_dim_x);
    config.grid_dim_y = static_cast<int>((height + block_dim_y - 1) / block_dim_y);
    return config;
}

LaunchConfig LaunchConfig::make_3d_config(size_t width, size_t height, size_t depth,
                                          int block_dim_x, int block_dim_y, int block_dim_z) {
    LaunchConfig config;
    config.block_dim_x = block_dim_x;
    config.block_dim_y = block_dim_y;
    config.block_dim_z = block_dim_z;
    config.grid_dim_x = static_cast<int>((width + block_dim_x - 1) / block_dim_x);
    config.grid_dim_y = static_cast<int>((height + block_dim_y - 1) / block_dim_y);
    config.grid_dim_z = static_cast<int>((depth + block_dim_z - 1) / block_dim_z);
    return config;
}

// ============================================================================
// KernelLauncher Implementation
// ============================================================================

KernelLauncher::KernelLauncher(Device& device) : device_(device) {}

void KernelLauncher::launch(
    std::function<void(void**)> kernel_func,
    const LaunchConfig& config,
    KernelArgs& args
) {
    device_.launch_kernel(
        kernel_func,
        const_cast<void**>(args.get_pointers().data()),
        config.grid_dim_x, config.grid_dim_y, config.grid_dim_z,
        config.block_dim_x, config.block_dim_y, config.block_dim_z,
        config.shared_memory
    );
}

void KernelLauncher::launch_simple(
    std::function<void()> kernel_func,
    const LaunchConfig& config
) {
    // Wrapper for simple kernels without arguments
    auto wrapper = [&kernel_func](void** args) {
        kernel_func();
    };
    
    KernelArgs empty_args;
    launch(wrapper, config, empty_args);
}

void KernelLauncher::synchronize() {
    device_.synchronize();
}

} // namespace cuda_open
