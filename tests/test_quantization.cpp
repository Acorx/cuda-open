/**
 * @file test_quantization.cpp
 * @brief Unit tests for quantization operations
 */

#include <cuda_open/quantization.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>

using namespace cuda_open;

// Test helpers
void test_int8_quantization() {
    std::cout << "Testing INT8 quantization... ";
    
    std::vector<float> data = {0.5f, -0.3f, 0.8f, -1.0f, 0.0f, 0.2f};
    
    // Quantize
    auto quantized = Quantization::quantize_fp32_to_int8(data.data(), data.size());
    assert(quantized.size() == data.size());
    
    // Dequantize
    float scale = Quantization::compute_scale(data.data(), data.size());
    auto dequantized = Quantization::dequantize_int8_to_fp32(quantized.data(), data.size(), scale);
    assert(dequantized.size() == data.size());
    
    // Check error is reasonable
    float max_error = 0.0f;
    for (size_t i = 0; i < data.size(); ++i) {
        max_error = std::max(max_error, std::abs(data[i] - dequantized[i]));
    }
    assert(max_error < 0.1f);  // INT8 should be fairly accurate
    
    std::cout << "PASSED (max_error=" << max_error << ")" << std::endl;
}

void test_int4_quantization() {
    std::cout << "Testing INT4 quantization... ";
    
    std::vector<float> data = {0.5f, -0.3f, 0.8f, -1.0f, 0.0f, 0.2f};
    
    auto quantized = Quantization::quantize_fp32_to_int4(data.data(), data.size());
    size_t expected_size = (data.size() + 1) / 2;
    assert(quantized.size() == expected_size);
    
    float scale = Quantization::compute_scale(data.data(), data.size());
    auto dequantized = Quantization::dequantize_int4_to_fp32(quantized.data(), data.size(), scale);
    
    float max_error = 0.0f;
    for (size_t i = 0; i < data.size(); ++i) {
        max_error = std::max(max_error, std::abs(data[i] - dequantized[i]));
    }
    assert(max_error < 0.3f);  // INT4 has more error
    
    std::cout << "PASSED (max_error=" << max_error << ")" << std::endl;
}

void test_int2_quantization() {
    std::cout << "Testing INT2 quantization... ";
    
    std::vector<float> data = {0.5f, -0.3f, 0.8f, -1.0f, 0.0f, 0.2f};
    
    auto quantized = Quantization::quantize_fp32_to_int2(data.data(), data.size());
    size_t expected_size = (data.size() + 3) / 4;
    assert(quantized.size() == expected_size);
    
    float scale = Quantization::compute_scale(data.data(), data.size());
    auto dequantized = Quantization::dequantize_int2_to_fp32(quantized.data(), data.size(), scale);
    
    float max_error = 0.0f;
    for (size_t i = 0; i < data.size(); ++i) {
        max_error = std::max(max_error, std::abs(data[i] - dequantized[i]));
    }
    assert(max_error < 1.0f);  // INT2 has significant error
    
    std::cout << "PASSED (max_error=" << max_error << ")" << std::endl;
}

void test_bitnet158_quantization() {
    std::cout << "Testing BitNet 1.58-bit quantization... ";
    
    // Use a multiple of 4 values so packed 2-bit storage reaches the
    // expected 16x compression (vs FP32).
    std::vector<float> data = {0.8f, -0.6f, 0.1f, -0.9f, 0.0f, 0.5f, -0.2f, 0.7f};
    
    auto quantized = Quantization::quantize_fp32_to_bitnet158(data.data(), data.size());
    size_t expected_size = (data.size() + 3) / 4;
    assert(quantized.size() == expected_size);
    
    auto dequantized = Quantization::dequantize_bitnet158_to_fp32(quantized.data(), data.size());
    assert(dequantized.size() == data.size());
    
    // Check values are ternary
    for (auto val : dequantized) {
        assert(val == -1.0f || val == 0.0f || val == 1.0f);
    }
    
    // Check compression ratio
    size_t original_bytes = data.size() * sizeof(float);
    size_t compressed_bytes = quantized.size();
    float ratio = static_cast<float>(original_bytes) / compressed_bytes;
    assert(ratio > 15.0f);  // Should be around 16x
    
    std::cout << "PASSED (compression=" << ratio << "x)" << std::endl;
}

void test_quantized_tensor() {
    std::cout << "Testing QuantizedTensor... ";
    
    const size_t size = 1000;
    std::vector<float> data(size);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    
    // Test different quantization types
    std::vector<QuantizationType> types = {
        QuantizationType::INT8,
        QuantizationType::INT4,
        QuantizationType::INT2,
        QuantizationType::BITNET158
    };
    
    for (auto qtype : types) {
        QuantizedTensor<> tensor(size, qtype);
        tensor.quantize_from_fp32(data.data());
        
        auto reconstructed = tensor.dequantize_to_fp32();
        assert(reconstructed.size() == size);
    }
    
    std::cout << "PASSED" << std::endl;
}

void test_scale_computation() {
    std::cout << "Testing scale computation... ";
    
    std::vector<float> data = {1.0f, -2.0f, 3.0f, -4.0f, 0.5f};
    float scale = Quantization::compute_scale(data.data(), data.size());
    assert(std::abs(scale - 4.0f) < 0.001f);
    
    std::cout << "PASSED" << std::endl;
}

void test_storage_size() {
    std::cout << "Testing storage size calculation... ";
    
    size_t num_elements = 100;
    
    assert(Quantization::get_storage_size(num_elements, QuantizationType::FP32) == num_elements * 4);
    assert(Quantization::get_storage_size(num_elements, QuantizationType::INT8) == num_elements);
    assert(Quantization::get_storage_size(num_elements, QuantizationType::INT4) == (num_elements + 1) / 2);
    assert(Quantization::get_storage_size(num_elements, QuantizationType::INT2) == (num_elements + 3) / 4);
    assert(Quantization::get_storage_size(num_elements, QuantizationType::BITNET158) == (num_elements + 3) / 4);
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "=== Running Quantization Tests ===\n" << std::endl;
    
    try {
        test_scale_computation();
        test_storage_size();
        test_int8_quantization();
        test_int4_quantization();
        test_int2_quantization();
        test_bitnet158_quantization();
        test_quantized_tensor();
        
        std::cout << "\nAll quantization tests PASSED!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED: " << e.what() << std::endl;
        return 1;
    }
}
