# CUDA Open API Documentation

## Overview

CUDA Open provides a unified interface for executing computations on both CPU and GPU devices. This document describes the core API.

## Core Classes

### DeviceManager

Singleton class for managing compute devices.

```cpp
// Initialize all available devices
static DeviceManager& instance();
void initialize();

// Get information about available devices
std::vector<DeviceInfo> get_available_devices() const;

// Select active device (0 = CPU, 1+ = GPU if available)
void set_device(int device_id);

// Get current device
Device& get_current_device();

// Check if GPU is available
bool is_gpu_available() const;
```

### Device

Abstract base class for compute devices.

```cpp
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
virtual void launch_kernel(...) = 0;

// Synchronization
virtual void synchronize() = 0;
```

### DevicePtr<T>

Template class for device memory management.

```cpp
// Allocate on device
explicit DevicePtr(Device& device, size_t count);

// Copy operations
void copy_from_host(const T* host_data);
void copy_to_host(T* host_data) const;

// Accessors
pointer get() const;
Device* device() const;
size_t size() const;
bool valid() const;
```

### UnifiedBuffer<T>

Automatically manages host and device memory copies.

```cpp
// Access host data
T* host_data();

// Access device pointer
DevicePtr<T>& device_ptr();

// Synchronization
void sync_to_device();
void sync_from_device();

size_t size() const;
```

### Quantization

Static utilities for quantization operations.

```cpp
// Get packing factor (values per byte)
static int get_packing_factor(QuantizationType type);

// Calculate storage size
static size_t get_storage_size(size_t num_elements, QuantizationType type);

// Quantization functions
static std::vector<uint8_t> quantize_fp32_to_int8(const float* data, size_t size);
static std::vector<float> dequantize_int8_to_fp32(const uint8_t* data, size_t size, float scale);

static std::vector<uint8_t> quantize_fp32_to_int4(const float* data, size_t size);
static std::vector<float> dequantize_int4_to_fp32(const uint8_t* data, size_t size, float scale);

static std::vector<uint8_t> quantize_fp32_to_int2(const float* data, size_t size);
static std::vector<float> dequantize_int2_to_fp32(const uint8_t* data, size_t size, float scale);

// BitNet 1.58-bit (ternary: -1, 0, 1)
static std::vector<uint8_t> quantize_fp32_to_bitnet158(const float* data, size_t size);
static std::vector<float> dequantize_bitnet158_to_fp32(const uint8_t* data, size_t size);

// Scale computation
static float compute_scale(const float* data, size_t size);
```

### QuantizedTensor<T>

Manages quantized tensor data.

```cpp
// Create quantized tensor
QuantizedTensor(size_t num_elements, QuantizationType qtype);

// Accessors
T* data();
size_t size() const;
QuantizationType quant_type() const;
float scale() const;
void set_scale(float s);

// Quantization
void quantize_from_fp32(const float* fp32_data);
std::vector<float> dequantize_to_fp32() const;
```

### LaunchConfig

Configuration for kernel launches.

```cpp
int grid_dim_x, grid_dim_y, grid_dim_z;
int block_dim_x, block_dim_y, block_dim_z;
size_t shared_memory;

// Helper functions
static LaunchConfig make_1d_config(size_t num_elements, int block_size = 256);
static LaunchConfig make_2d_config(size_t width, size_t height, ...);
static LaunchConfig make_3d_config(size_t width, size_t height, size_t depth, ...);
```

### KernelLauncher

Launches kernels on devices.

```cpp
KernelLauncher(Device& device);

void launch(
    std::function<void(void**)> kernel_func,
    const LaunchConfig& config,
    KernelArgs& args
);

void synchronize();
```

## Quantization Types

```cpp
enum class QuantizationType {
    FP32,      // Full precision 32-bit float
    FP16,      // Half precision 16-bit float
    INT8,      // 8-bit integer (4x compression)
    INT4,      // 4-bit integer (8x compression)
    INT2,      // 2-bit integer (16x compression)
    BITNET158  // 1.58-bit ternary (-1, 0, 1) (~16x compression)
};
```

## Thread ID Access (CPU Backend)

When executing kernels on CPU, thread IDs are stored in thread-local storage:

```cpp
// Inside kernel function, access via static members:
CPUDevice::thread_block_idx_x
CPUDevice::thread_thread_idx_x
CPUDevice::thread_block_dim_x
CPUDevice::thread_grid_dim_x
// ... and similar for Y, Z dimensions
```

## Error Handling

All errors throw `DeviceError` exceptions:

```cpp
try {
    // CUDA Open operations
} catch (const cuda_open::DeviceError& e) {
    std::cerr << "Device error: " << e.what() << std::endl;
}
```

## Complete Example

```cpp
#include <cuda_open/cuda_open.h>

int main() {
    using namespace cuda_open;
    
    // Initialize
    initialize();
    
    // Get device
    auto& device = DeviceManager::instance().get_current_device();
    
    // Allocate memory
    const size_t size = 1024;
    DevicePtr<float> input(device, size);
    DevicePtr<float> output(device, size);
    
    // Prepare data
    std::vector<float> host_data(size, 1.0f);
    input.copy_from_host(host_data.data());
    
    // Launch kernel
    KernelLauncher launcher(device);
    LaunchConfig config = LaunchConfig::make_1d_config(size, 256);
    
    KernelArgs args;
    args.add_device_ptr(input.get());
    args.add_device_ptr(output.get());
    
    launcher.launch(my_kernel, config, args);
    launcher.synchronize();
    
    // Retrieve results
    std::vector<float> results(size);
    output.copy_to_host(results.data());
    
    return 0;
}
```
