/**
 * @file device.cpp
 * @brief Device abstraction implementation with CPU backend
 */

#include "cuda_open/device.h"
#include <iostream>
#include <thread>
#include <algorithm>
#include <cstring>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace cuda_open {

// ============================================================================
// CPU Device Implementation
// ============================================================================

class CPUDevice : public Device {
public:
    CPUDevice() {
        info_.type = DeviceType::CPU;
        info_.name = "CPU";
        info_.num_cores = std::thread::hardware_concurrency();
        info_.compute_capability_major = 0;
        info_.compute_capability_minor = 0;
        
        // Get approximate system memory
        info_.memory_total = 0;  // Would need platform-specific code
        info_.memory_free = 0;
        
#if defined(__linux__)
        FILE* file = fopen("/proc/meminfo", "r");
        if (file) {
            char line[256];
            while (fgets(line, sizeof(line), file)) {
                if (sscanf(line, "MemTotal: %zu kB", &info_.memory_total) == 1) {
                    info_.memory_total *= 1024;  // Convert to bytes
                }
                if (sscanf(line, "MemAvailable: %zu kB", &info_.memory_free) == 1) {
                    info_.memory_free *= 1024;
                }
            }
            fclose(file);
        }
#endif
    }
    
    DeviceInfo get_info() const override { return info_; }
    DeviceType get_type() const override { return DeviceType::CPU; }
    std::string get_name() const override { return info_.name; }
    
    void* allocate(size_t size) override {
        void* ptr = std::malloc(size);
        if (!ptr) throw DeviceError("Failed to allocate CPU memory");
        return ptr;
    }
    
    void free(void* ptr) override {
        std::free(ptr);
    }
    
    void memcpy_host_to_device(void* dst, const void* src, size_t size) override {
        std::memcpy(dst, src, size);
    }
    
    void memcpy_device_to_host(void* dst, const void* src, size_t size) override {
        std::memcpy(dst, src, size);
    }
    
    void memcpy_device_to_device(void* dst, const void* src, size_t size) override {
        std::memcpy(dst, src, size);
    }
    
    void launch_kernel(
        std::function<void(void**)> kernel_func,
        void** args,
        int grid_dim_x, int grid_dim_y, int grid_dim_z,
        int block_dim_x, int block_dim_y, int block_dim_z,
        size_t shared_memory
    ) override {
        // CPU execution: iterate through all blocks and threads
        for (int bz = 0; bz < grid_dim_z; ++bz) {
            for (int by = 0; by < grid_dim_y; ++by) {
                for (int bx = 0; bx < grid_dim_x; ++bx) {
                    for (int tz = 0; tz < block_dim_z; ++tz) {
                        for (int ty = 0; ty < block_dim_y; ++ty) {
                            for (int tx = 0; tx < block_dim_x; ++tx) {
                                // Store thread IDs in thread-local storage
                                thread_block_idx_x = bx;
                                thread_block_idx_y = by;
                                thread_block_idx_z = bz;
                                thread_thread_idx_x = tx;
                                thread_thread_idx_y = ty;
                                thread_thread_idx_z = tz;
                                thread_block_dim_x = block_dim_x;
                                thread_block_dim_y = block_dim_y;
                                thread_block_dim_z = block_dim_z;
                                thread_grid_dim_x = grid_dim_x;
                                thread_grid_dim_y = grid_dim_y;
                                thread_grid_dim_z = grid_dim_z;
                                
                                kernel_func(args);
                            }
                        }
                    }
                }
            }
        }
    }
    
    void synchronize() override {
        // CPU is already synchronous
    }
    
    void* create_stream() override {
        return nullptr;  // Not needed for CPU
    }
    
    void destroy_stream(void* stream) override {
        // Nothing to destroy
    }
    
    // Thread ID accessors (thread-local)
    static thread_local int thread_block_idx_x;
    static thread_local int thread_block_idx_y;
    static thread_local int thread_block_idx_z;
    static thread_local int thread_thread_idx_x;
    static thread_local int thread_thread_idx_y;
    static thread_local int thread_thread_idx_z;
    static thread_local int thread_block_dim_x;
    static thread_local int thread_block_dim_y;
    static thread_local int thread_block_dim_z;
    static thread_local int thread_grid_dim_x;
    static thread_local int thread_grid_dim_y;
    static thread_local int thread_grid_dim_z;
    
private:
    DeviceInfo info_;
};

// Thread-local storage initialization
thread_local int CPUDevice::thread_block_idx_x = 0;
thread_local int CPUDevice::thread_block_idx_y = 0;
thread_local int CPUDevice::thread_block_idx_z = 0;
thread_local int CPUDevice::thread_thread_idx_x = 0;
thread_local int CPUDevice::thread_thread_idx_y = 0;
thread_local int CPUDevice::thread_thread_idx_z = 0;
thread_local int CPUDevice::thread_block_dim_x = 1;
thread_local int CPUDevice::thread_block_dim_y = 1;
thread_local int CPUDevice::thread_block_dim_z = 1;
thread_local int CPUDevice::thread_grid_dim_x = 1;
thread_local int CPUDevice::thread_grid_dim_y = 1;
thread_local int CPUDevice::thread_grid_dim_z = 1;

// ============================================================================
// GPU Device Implementation (CUDA)
// ============================================================================

#ifdef HAVE_CUDA

class GPUDevice : public Device {
public:
    GPUDevice(int device_id) {
        cudaSetDevice(device_id);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop);
        
        info_.type = DeviceType::GPU;
        info_.name = prop.name;
        info_.memory_total = prop.totalGlobalMem;
        info_.memory_free = prop.totalGlobalMem;  // Approximate
        info_.compute_capability_major = prop.major;
        info_.compute_capability_minor = prop.minor;
        info_.num_cores = prop.multiProcessorCount;
    }
    
    DeviceInfo get_info() const override { return info_; }
    DeviceType get_type() const override { return DeviceType::GPU; }
    std::string get_name() const override { return info_.name; }
    
    void* allocate(size_t size) override {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            throw DeviceError(std::string("CUDA malloc failed: ") + cudaGetErrorString(err));
        }
        return ptr;
    }
    
    void free(void* ptr) override {
        cudaFree(ptr);
    }
    
    void memcpy_host_to_device(void* dst, const void* src, size_t size) override {
        cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    }
    
    void memcpy_device_to_host(void* dst, const void* src, size_t size) override {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }
    
    void memcpy_device_to_device(void* dst, const void* src, size_t size) override {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    }
    
    void launch_kernel(
        std::function<void(void**)> kernel_func,
        void** args,
        int grid_dim_x, int grid_dim_y, int grid_dim_z,
        int block_dim_x, int block_dim_y, int block_dim_z,
        size_t shared_memory
    ) override {
        // Note: This is a simplified wrapper
        // Real CUDA kernel launch requires more complex handling
        throw DeviceError("Direct kernel launch not supported - use CUDA kernel wrapper");
    }
    
    void synchronize() override {
        cudaDeviceSynchronize();
    }
    
    void* create_stream() override {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        return static_cast<void*>(stream);
    }
    
    void destroy_stream(void* stream) override {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    }
    
private:
    DeviceInfo info_;
};

#endif // HAVE_CUDA

// ============================================================================
// DeviceManager Implementation
// ============================================================================

class DeviceManager::Impl {
public:
    std::vector<std::unique_ptr<Device>> devices_;
    int current_device_ = -1;
};

DeviceManager::DeviceManager() : impl_(std::make_unique<Impl>()) {}

DeviceManager::~DeviceManager() = default;

DeviceManager& DeviceManager::instance() {
    static DeviceManager instance;
    return instance;
}

void DeviceManager::initialize() {
    impl_->devices_.clear();
    
    // Always add CPU
    impl_->devices_.push_back(std::make_unique<CPUDevice>());
    impl_->current_device_ = 0;
    
    // Check for GPU
#ifdef HAVE_CUDA
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    for (int i = 0; i < device_count; ++i) {
        try {
            impl_->devices_.push_back(std::make_unique<GPUDevice>(i));
        } catch (...) {
            std::cerr << "Warning: Failed to initialize GPU " << i << std::endl;
        }
    }
#endif
}

std::vector<DeviceInfo> DeviceManager::get_available_devices() const {
    std::vector<DeviceInfo> devices;
    for (auto& device : impl_->devices_) {
        devices.push_back(device->get_info());
    }
    return devices;
}

void DeviceManager::set_device(int device_id) {
    if (device_id < 0 || device_id >= static_cast<int>(impl_->devices_.size())) {
        throw DeviceError("Invalid device ID");
    }
    impl_->current_device_ = device_id;
}

Device& DeviceManager::get_current_device() {
    if (impl_->current_device_ < 0) {
        throw DeviceError("No device selected");
    }
    return *impl_->devices_[impl_->current_device_];
}

const Device& DeviceManager::get_current_device() const {
    if (impl_->current_device_ < 0) {
        throw DeviceError("No device selected");
    }
    return *impl_->devices_[impl_->current_device_];
}

bool DeviceManager::is_gpu_available() const {
    for (auto& device : impl_->devices_) {
        if (device->get_type() == DeviceType::GPU) {
            return true;
        }
    }
    return false;
}

// Helper to get thread IDs from CPU device
struct ThreadIdStruct {
    int block_idx_x;
    int block_idx_y;
    int block_idx_z;
    int thread_idx_x;
    int thread_idx_y;
    int thread_idx_z;
    int block_dim_x;
    int block_dim_y;
    int block_dim_z;
    int grid_dim_x;
    int grid_dim_y;
    int grid_dim_z;
};

ThreadIdStruct get_thread_id() {
    ThreadIdStruct tid;
    tid.block_idx_x = CPUDevice::thread_block_idx_x;
    tid.block_idx_y = CPUDevice::thread_block_idx_y;
    tid.block_idx_z = CPUDevice::thread_block_idx_z;
    tid.thread_idx_x = CPUDevice::thread_thread_idx_x;
    tid.thread_idx_y = CPUDevice::thread_thread_idx_y;
    tid.thread_idx_z = CPUDevice::thread_thread_idx_z;
    tid.block_dim_x = CPUDevice::thread_block_dim_x;
    tid.block_dim_y = CPUDevice::thread_block_dim_y;
    tid.block_dim_z = CPUDevice::thread_block_dim_z;
    tid.grid_dim_x = CPUDevice::thread_grid_dim_x;
    tid.grid_dim_y = CPUDevice::thread_grid_dim_y;
    tid.grid_dim_z = CPUDevice::thread_grid_dim_z;
    return tid;
}

} // namespace cuda_open
