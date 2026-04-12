/**
 * @file tensor.h
 * @brief Tensor operations with CPU/GPU support
 * 
 * Common tensor operations (matmul, add, etc.) optimized for different
 * quantization formats and execution devices.
 */

#ifndef CUDA_OPEN_TENSOR_H
#define CUDA_OPEN_TENSOR_H

#include "device.h"
#include "memory.h"
#include "quantization.h"
#include <vector>
#include <memory>

namespace cuda_open {
namespace tensor {

// Basic tensor structure
template<typename T = float>
class Tensor {
public:
    Tensor() : data_(nullptr), shape_({}), size_(0) {}
    
    Tensor(Device& device, const std::vector<size_t>& shape)
        : device_(&device), shape_(shape), size_(1) {
        for (auto s : shape) size_ *= s;
        data_ = std::make_unique<DevicePtr<T>>(device, size_);
    }
    
    // Shape access
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return size_; }
    size_t dim(size_t i) const { return i < shape_.size() ? shape_[i] : 1; }
    size_t ndim() const { return shape_.size(); }
    
    // Data access
    DevicePtr<T>& data() { return *data_; }
    const DevicePtr<T>& data() const { return *data_; }
    
    Device& device() const { return *device_; }
    
private:
    Device* device_;
    std::unique_ptr<DevicePtr<T>> data_;
    std::vector<size_t> shape_;
    size_t size_;
};

// Matrix multiplication
template<typename T = float>
void matmul(
    const Tensor<T>& A,
    const Tensor<T>& B,
    Tensor<T>& C,
    bool transpose_a = false,
    bool transpose_b = false
);

// Element-wise addition
template<typename T = float>
void add(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C);

// Element-wise multiplication
template<typename T = float>
void multiply(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C);

// ReLU activation
template<typename T = float>
void relu(const Tensor<T>& input, Tensor<T>& output);

// Softmax
template<typename T = float>
void softmax(const Tensor<T>& input, Tensor<T>& output, int dim = -1);

// Layer normalization
template<typename T = float>
void layer_norm(const Tensor<T>& input, Tensor<T>& output, 
                const Tensor<T>& weight, const Tensor<T>& bias,
                float epsilon = 1e-5);

// Quantized matrix multiplication (for BitNet and other quantized models)
void matmul_quantized(
    const QuantizedTensor<>& A,
    const QuantizedTensor<>& B,
    Tensor<float>& C,
    const std::vector<size_t>& shape_a,
    const std::vector<size_t>& shape_b,
    QuantizationType qtype
);

} // namespace tensor
} // namespace cuda_open

#endif // CUDA_OPEN_TENSOR_H
