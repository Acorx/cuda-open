/**
 * @file tensor.cpp
 * @brief Tensor operations implementation
 */

#include "cuda_open/tensor.h"
#include "cuda_open/kernel.h"
#include <cmath>
#include <algorithm>

namespace cuda_open {
namespace tensor {

// ============================================================================
// CPU Kernel Functions (can be executed on device)
// ============================================================================

// Matrix multiplication kernel (CPU execution)
void matmul_cpu_kernel(
    const float* A, const float* B, float* C,
    size_t M, size_t K, size_t N,
    bool transpose_a, bool transpose_b
) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float a_val = transpose_a ? A[k * M + i] : A[i * K + k];
                float b_val = transpose_b ? B[j * K + k] : B[k * N + j];
                sum += a_val * b_val;
            }
            C[i * N + j] = sum;
        }
    }
}

// Element-wise operations kernel
void elementwise_kernel(
    const float* A, const float* B, float* C, size_t size,
    std::function<float(float, float)> op
) {
    for (size_t i = 0; i < size; ++i) {
        C[i] = op(A[i], B[i]);
    }
}

// ReLU kernel
void relu_kernel(const float* input, float* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

// Softmax kernel
void softmax_kernel(const float* input, float* output, size_t size, size_t dim_size) {
    size_t num_rows = size / dim_size;
    
    for (size_t i = 0; i < num_rows; ++i) {
        const float* row = input + i * dim_size;
        float* out_row = output + i * dim_size;
        
        // Find max for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < dim_size; ++j) {
            max_val = std::max(max_val, row[j]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (size_t j = 0; j < dim_size; ++j) {
            out_row[j] = std::exp(row[j] - max_val);
            sum += out_row[j];
        }
        
        // Normalize
        for (size_t j = 0; j < dim_size; ++j) {
            out_row[j] /= sum;
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

template<>
void matmul<float>(
    const Tensor<float>& A,
    const Tensor<float>& B,
    Tensor<float>& C,
    bool transpose_a,
    bool transpose_b
) {
    // Get dimensions
    size_t M = transpose_a ? A.dim(1) : A.dim(0);
    size_t K = transpose_a ? A.dim(0) : A.dim(1);
    size_t N = transpose_b ? B.dim(0) : B.dim(1);
    
    // Copy data to host for computation
    std::vector<float> h_A(A.size()), h_B(B.size()), h_C(C.size());
    A.data().copy_to_host(h_A.data());
    B.data().copy_to_host(h_B.data());
    
    // Execute on CPU
    matmul_cpu_kernel(h_A.data(), h_B.data(), h_C.data(), M, K, N, 
                      transpose_a, transpose_b);
    
    // Copy result back to device
    C.data().copy_from_host(h_C.data());
}

template<>
void add<float>(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C) {
    if (A.size() != B.size() || A.size() != C.size()) {
        throw DeviceError("Tensor size mismatch for addition");
    }
    
    std::vector<float> h_A(A.size()), h_B(B.size()), h_C(C.size());
    A.data().copy_to_host(h_A.data());
    B.data().copy_to_host(h_B.data());
    
    elementwise_kernel(h_A.data(), h_B.data(), h_C.data(), A.size(),
                       [](float a, float b) { return a + b; });
    
    C.data().copy_from_host(h_C.data());
}

template<>
void multiply<float>(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C) {
    if (A.size() != B.size() || A.size() != C.size()) {
        throw DeviceError("Tensor size mismatch for multiplication");
    }
    
    std::vector<float> h_A(A.size()), h_B(B.size()), h_C(C.size());
    A.data().copy_to_host(h_A.data());
    B.data().copy_to_host(h_B.data());
    
    elementwise_kernel(h_A.data(), h_B.data(), h_C.data(), A.size(),
                       [](float a, float b) { return a * b; });
    
    C.data().copy_from_host(h_C.data());
}

template<>
void relu<float>(const Tensor<float>& input, Tensor<float>& output) {
    if (input.size() != output.size()) {
        throw DeviceError("Tensor size mismatch for ReLU");
    }
    
    std::vector<float> h_input(input.size()), h_output(output.size());
    input.data().copy_to_host(h_input.data());
    
    relu_kernel(h_input.data(), h_output.data(), input.size());
    
    output.data().copy_from_host(h_output.data());
}

template<>
void softmax<float>(const Tensor<float>& input, Tensor<float>& output, int dim) {
    if (input.size() != output.size()) {
        throw DeviceError("Tensor size mismatch for softmax");
    }
    
    std::vector<float> h_input(input.size()), h_output(output.size());
    input.data().copy_to_host(h_input.data());
    
    size_t dim_size = input.dim(input.ndim() + dim);
    softmax_kernel(h_input.data(), h_output.data(), input.size(), dim_size);
    
    output.data().copy_from_host(h_output.data());
}

// Quantized matrix multiplication (BitNet support)
void matmul_quantized(
    const QuantizedTensor<>& A,
    const QuantizedTensor<>& B,
    Tensor<float>& C,
    const std::vector<size_t>& shape_a,
    const std::vector<size_t>& shape_b,
    QuantizationType qtype
) {
    // Dequantize inputs
    auto fp32_A = A.dequantize_to_fp32();
    auto fp32_B = B.dequantize_to_fp32();
    
    size_t M = shape_a[0];
    size_t K = shape_a[1];
    size_t N = shape_b[1];
    
    // Copy to device
    DevicePtr<float> dev_A(C.device(), fp32_A.size());
    DevicePtr<float> dev_B(C.device(), fp32_B.size());
    dev_A.copy_from_host(fp32_A.data());
    dev_B.copy_from_host(fp32_B.data());
    
    // Create temporary tensors
    Tensor<float> tensor_A(C.device(), shape_a);
    Tensor<float> tensor_B(C.device(), shape_b);
    
    // Perform matmul
    matmul(tensor_A, tensor_B, C);
}

} // namespace tensor
} // namespace cuda_open
