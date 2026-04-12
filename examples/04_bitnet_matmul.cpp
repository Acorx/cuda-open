/**
 * @example 04_bitnet_matmul.cpp
 * @brief Demonstrates BitNet 1.58-bit matrix multiplication
 */

#include <cuda_open/cuda_open.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

void print_matrix(const std::string& name, const std::vector<float>& mat, size_t rows, size_t cols) {
    std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (size_t i = 0; i < std::min(rows, (size_t)5); ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < std::min(cols, (size_t)5); ++j) {
            std::cout << mat[i * cols + j];
            if (j < std::min(cols, (size_t)5) - 1) std::cout << ", ";
        }
        if (cols > 5) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }
    if (rows > 5) std::cout << "  ..." << std::endl;
}

int main() {
    try {
        cuda_open::initialize();
        
        std::cout << "=== CUDA Open - BitNet Matrix Multiplication ===\n" << std::endl;
        
        auto& device = cuda_open::DeviceManager::instance().get_current_device();
        std::cout << "Using device: " << device.get_name() << std::endl;
        
        // Matrix dimensions
        const size_t M = 64;  // rows of A
        const size_t K = 128; // cols of A, rows of B
        const size_t N = 64;  // cols of B
        
        std::cout << "Matrix A: " << M << "x" << K << std::endl;
        std::cout << "Matrix B: " << K << "x" << N << std::endl;
        std::cout << "Result C: " << M << "x" << N << std::endl;
        
        // Generate random matrices
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        std::vector<float> A(M * K);
        std::vector<float> B(K * N);
        
        for (size_t i = 0; i < A.size(); ++i) A[i] = dist(gen);
        for (size_t i = 0; i < B.size(); ++i) B[i] = dist(gen);
        
        print_matrix("A (sample)", A, M, K);
        print_matrix("B (sample)", B, K, N);
        
        // Quantize to BitNet format
        std::cout << "\n--- Quantizing to BitNet 1.58-bit ---" << std::endl;
        
        cuda_open::QuantizedTensor<> qA(A.size(), cuda_open::QuantizationType::BITNET158);
        cuda_open::QuantizedTensor<> qB(B.size(), cuda_open::QuantizationType::BITNET158);
        
        qA.quantize_from_fp32(A.data());
        qB.quantize_from_fp32(B.data());
        
        std::cout << "A quantized: " << A.size() * sizeof(float) << " -> " 
                  << (A.size() + 3) / 4 << " bytes" << std::endl;
        std::cout << "B quantized: " << B.size() * sizeof(float) << " -> " 
                  << (B.size() + 3) / 4 << " bytes" << std::endl;
        
        // Dequantize for computation
        std::cout << "\n--- Dequantizing for computation ---" << std::endl;
        auto A_dequant = qA.dequantize_to_fp32();
        auto B_dequant = qB.dequantize_to_fp32();
        
        // Perform matrix multiplication (FP32 after dequantization)
        std::cout << "\n--- Computing C = A @ B ---" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<float> C(M * N, 0.0f);
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += A_dequant[i * K + k] * B_dequant[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Computation time: " << duration.count() << " ms" << std::endl;
        print_matrix("C (sample)", C, M, N);
        
        // Compare with full precision
        std::cout << "\n--- Comparison with full precision ---" << std::endl;
        
        std::vector<float> C_full(M * N, 0.0f);
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C_full[i * N + j] = sum;
            }
        }
        
        // Compute error
        float max_error = 0.0f;
        float mse = 0.0f;
        for (size_t i = 0; i < C.size(); ++i) {
            float error = std::abs(C_full[i] - C[i]);
            max_error = std::max(max_error, error);
            mse += error * error;
        }
        mse /= C.size();
        
        std::cout << "  Max error: " << max_error << std::endl;
        std::cout << "  MSE: " << mse << std::endl;
        
        // Memory savings
        size_t original_bytes = (A.size() + B.size()) * sizeof(float);
        size_t quantized_bytes = (A.size() + 3) / 4 + (B.size() + 3) / 4;
        std::cout << "\n--- Memory Savings ---" << std::endl;
        std::cout << "  Original: " << original_bytes << " bytes" << std::endl;
        std::cout << "  Quantized: " << quantized_bytes << " bytes" << std::endl;
        std::cout << "  Compression: " << static_cast<float>(original_bytes) / quantized_bytes << "x" << std::endl;
        
        std::cout << "\nExample completed successfully!" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
