/**
 * @example 03_quantization.cpp
 * @brief Demonstrates quantization operations
 */

#include <cuda_open/cuda_open.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

void test_quantization(
    const std::string& name,
    std::vector<uint8_t> (*quantize)(const float*, size_t),
    std::vector<float> (*dequantize)(const uint8_t*, size_t, float),
    const std::vector<float>& data
) {
    std::cout << "\n--- " << name << " ---" << std::endl;
    
    // Quantize
    auto quantized = quantize(data.data(), data.size());
    float scale = cuda_open::Quantization::compute_scale(data.data(), data.size());
    size_t original_size = data.size() * sizeof(float);
    size_t compressed_size = quantized.size();
    float ratio = static_cast<float>(original_size) / compressed_size;
    
    std::cout << "  Original size: " << original_size << " bytes" << std::endl;
    std::cout << "  Compressed size: " << compressed_size << " bytes" << std::endl;
    std::cout << "  Compression ratio: " << ratio << "x" << std::endl;
    std::cout << "  Scale: " << scale << std::endl;
    
    // Dequantize
    auto dequantized = dequantize(quantized.data(), data.size(), scale);
    
    // Compute error
    float max_error = 0.0f;
    float mse = 0.0f;
    for (size_t i = 0; i < data.size(); ++i) {
        float error = std::abs(data[i] - dequantized[i]);
        max_error = std::max(max_error, error);
        mse += error * error;
    }
    mse /= data.size();
    
    std::cout << "  Max error: " << max_error << std::endl;
    std::cout << "  MSE: " << mse << std::endl;
}

int main() {
    try {
        cuda_open::initialize();
        
        std::cout << "=== CUDA Open - Quantization Example ===\n" << std::endl;
        
        // Generate test data
        const size_t size = 1000;
        std::vector<float> data(size);
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
        
        std::cout << "Generated " << size << " random values (mean=0, std=1)" << std::endl;
        std::cout << "Sample values: ";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
        
        // Test different quantization methods
        test_quantization(
            "INT8 Quantization",
            cuda_open::Quantization::quantize_fp32_to_int8,
            cuda_open::Quantization::dequantize_int8_to_fp32,
            data
        );
        
        test_quantization(
            "INT4 Quantization",
            cuda_open::Quantization::quantize_fp32_to_int4,
            cuda_open::Quantization::dequantize_int4_to_fp32,
            data
        );
        
        test_quantization(
            "INT2 Quantization",
            cuda_open::Quantization::quantize_fp32_to_int2,
            cuda_open::Quantization::dequantize_int2_to_fp32,
            data
        );
        
        test_quantization(
            "BitNet 1.58-bit Quantization",
            cuda_open::Quantization::quantize_fp32_to_bitnet158,
            [](const uint8_t* data, size_t size, float) {
                return cuda_open::Quantization::dequantize_bitnet158_to_fp32(data, size);
            },
            data
        );
        
        // Test QuantizedTensor
        std::cout << "\n--- QuantizedTensor Example ---" << std::endl;
        
        cuda_open::QuantizedTensor<> tensor(size, cuda_open::QuantizationType::BITNET158);
        tensor.quantize_from_fp32(data.data());
        
        auto reconstructed = tensor.dequantize_to_fp32();
        
        std::cout << "  Quantized " << size << " values to BitNet format" << std::endl;
        std::cout << "  First 5 original: ";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "  First 5 reconstructed: ";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << reconstructed[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "\nExample completed successfully!" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
