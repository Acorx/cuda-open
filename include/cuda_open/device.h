/**
 * @file device.h
 * @brief Core device abstraction layer for CPU/GPU unified execution
 * 
 * This module provides a unified interface for executing computations
 * on both CPU and GPU devices, with automatic fallback and device detection.
 */

#ifndef CUDA_OPEN_DEVICE_H
#define CUDA_OPEN_DEVICE_H

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>

namespace cuda_open {

// Device types supported
enum class DeviceType {
    CPU,
    GPU,
    FPGA  // Future expansion
};

// Quantization types
enum class QuantizationType {
    FP32,      // Full precision 32-bit
    FP16,      // Half precision
    INT8,      // 8-bit integer
    INT4,      // 4-bit integer
    INT2,      // 2-bit integer
    BITNET158  // 1.58-bit (ternary: -1, 0, 1)
};

// Device information structure
struct DeviceInfo {
    DeviceType type;
    std::string name;
    size_t memory_total;
    size_t memory_free;
    int compute_capability_major;
    int compute_capability_minor;
    int num_cores;
};

// Forward declaration
class Device;

// Global device management
class DeviceManager {
public:
    static DeviceManager& instance();
    
    // Initialize all available devices
    void initialize();
    
    // Get available devices
    std::vector<DeviceInfo> get_available_devices() const;
    
    // Set active device
    void set_device(int device_id);
    
    // Get current device
    Device& get_current_device();
    const Device& get_current_device() const;
    
    // Check if GPU is available
    bool is_gpu_available() const;
    
private:
    DeviceManager();
    ~DeviceManager();
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;
    
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Base device class
class Device {
public:
    virtual ~Device() = default;
    
    // Device information
    virtual DeviceInfo get_info() const = 0;
    virtual DeviceType get_type() const = 0;
    virtual std::string get_name() const = 0;
    
    // Memory management
    virtual void* allocate(size_t size) = 0;
    virtual void free(void* ptr) = 0;
    virtual void memcpy_host_to_device(void* dst, const void* src, size_t size) = 0;
    virtual void memcpy_device_to_host(void* dst, const void* src, size_t size) = 0;
    virtual void memcpy_device_to_device(void* dst, const void* src, size_t size) = 0;
    
    // Kernel execution
    virtual void launch_kernel(
        std::function<void(void**)> kernel_func,
        void** args,
        int grid_dim_x, int grid_dim_y, int grid_dim_z,
        int block_dim_x, int block_dim_y, int block_dim_z,
        size_t shared_memory = 0
    ) = 0;
    
    // Synchronization
    virtual void synchronize() = 0;
    
    // Stream management (for async operations)
    virtual void* create_stream() = 0;
    virtual void destroy_stream(void* stream) = 0;
    
protected:
    Device() = default;
};

// Exception for device errors
class DeviceError : public std::runtime_error {
public:
    explicit DeviceError(const std::string& msg) : std::runtime_error(msg) {}
};

} // namespace cuda_open

#endif // CUDA_OPEN_DEVICE_H
