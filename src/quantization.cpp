/**
 * @file quantization.cpp
 * @brief Quantization implementation for various bit-widths
 */

#include "cuda_open/quantization.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace cuda_open {

// ============================================================================
// Quantization Utilities
// ============================================================================

int Quantization::get_packing_factor(QuantizationType type) {
    switch (type) {
        case QuantizationType::FP32: return 1;
        case QuantizationType::FP16: return 2;
        case QuantizationType::INT8: return 4;
        case QuantizationType::INT4: return 8;  // 8 values per byte (4 bits each, packed in pairs)
        case QuantizationType::INT2: return 16; // Actually 4 values per byte (2 bits each)
        case QuantizationType::BITNET158: return 16; // Packed ternary values
        default: return 1;
    }
}

size_t Quantization::get_storage_size(size_t num_elements, QuantizationType type) {
    switch (type) {
        case QuantizationType::FP32:
            return num_elements * sizeof(float);
        case QuantizationType::FP16:
            return num_elements * sizeof(uint16_t);
        case QuantizationType::INT8:
            return num_elements * sizeof(int8_t);
        case QuantizationType::INT4:
            return (num_elements + 1) / 2;  // 2 values per byte
        case QuantizationType::INT2:
            return (num_elements + 3) / 4;  // 4 values per byte
        case QuantizationType::BITNET158:
            return (num_elements + 3) / 4;  // 4 values per byte (2 bits per value)
        default:
            return num_elements * sizeof(float);
    }
}

// ============================================================================
// INT8 Quantization
// ============================================================================

std::vector<uint8_t> Quantization::quantize_fp32_to_int8(const float* data, size_t size) {
    std::vector<uint8_t> quantized(size);
    float scale = compute_scale(data, size);
    float inv_scale = 127.0f / scale;
    
    for (size_t i = 0; i < size; ++i) {
        int32_t val = static_cast<int32_t>(std::round(data[i] * inv_scale));
        val = std::clamp(val, -128, 127);
        quantized[i] = static_cast<uint8_t>(val);
    }
    
    return quantized;
}

std::vector<float> Quantization::dequantize_int8_to_fp32(const uint8_t* data, size_t size, float scale) {
    std::vector<float> dequantized(size);
    float norm_scale = scale / 127.0f;
    
    for (size_t i = 0; i < size; ++i) {
        int8_t val = static_cast<int8_t>(data[i]);
        dequantized[i] = static_cast<float>(val) * norm_scale;
    }
    
    return dequantized;
}

// ============================================================================
// INT4 Quantization
// ============================================================================

std::vector<uint8_t> Quantization::quantize_fp32_to_int4(const float* data, size_t size) {
    size_t packed_size = (size + 1) / 2;
    std::vector<uint8_t> quantized(packed_size, 0);
    float scale = compute_scale(data, size);
    float inv_scale = 7.0f / scale;  // INT4 range: -8 to 7
    
    for (size_t i = 0; i < size; ++i) {
        int32_t val = static_cast<int32_t>(std::round(data[i] * inv_scale));
        val = std::clamp(val, -8, 7);
        
        // Pack two INT4 values into one byte
        size_t byte_idx = i / 2;
        size_t nibble_idx = i % 2;
        
        uint8_t packed_val = static_cast<uint8_t>(val + 8);  // Shift to 0-15 range
        if (nibble_idx == 0) {
            quantized[byte_idx] = packed_val;
        } else {
            quantized[byte_idx] |= (packed_val << 4);
        }
    }
    
    return quantized;
}

std::vector<float> Quantization::dequantize_int4_to_fp32(const uint8_t* data, size_t size, float scale) {
    std::vector<float> dequantized(size);
    float norm_scale = scale / 7.0f;
    
    for (size_t i = 0; i < size; ++i) {
        size_t byte_idx = i / 2;
        size_t nibble_idx = i % 2;
        
        uint8_t val;
        if (nibble_idx == 0) {
            val = data[byte_idx] & 0x0F;
        } else {
            val = (data[byte_idx] >> 4) & 0x0F;
        }
        
        dequantized[i] = static_cast<float>(static_cast<int32_t>(val) - 8) * norm_scale;
    }
    
    return dequantized;
}

// ============================================================================
// INT2 Quantization
// ============================================================================

std::vector<uint8_t> Quantization::quantize_fp32_to_int2(const float* data, size_t size) {
    size_t packed_size = (size + 3) / 4;
    std::vector<uint8_t> quantized(packed_size, 0);
    float scale = compute_scale(data, size);
    float inv_scale = 1.0f / scale;  // INT2 range: -2 to 1 (or -1 to 1 for ternary)
    
    for (size_t i = 0; i < size; ++i) {
        int32_t val = static_cast<int32_t>(std::round(data[i] * inv_scale));
        val = std::clamp(val, -2, 1);
        
        // Pack four INT2 values into one byte
        size_t byte_idx = i / 4;
        size_t offset = (i % 4) * 2;
        
        uint8_t packed_val = static_cast<uint8_t>(val + 2);  // Shift to 0-3 range
        quantized[byte_idx] |= (packed_val << offset);
    }
    
    return quantized;
}

std::vector<float> Quantization::dequantize_int2_to_fp32(const uint8_t* data, size_t size, float scale) {
    std::vector<float> dequantized(size);
    float norm_scale = scale / 1.0f;
    
    for (size_t i = 0; i < size; ++i) {
        size_t byte_idx = i / 4;
        size_t offset = (i % 4) * 2;
        
        uint8_t val = (data[byte_idx] >> offset) & 0x03;
        dequantized[i] = static_cast<float>(static_cast<int32_t>(val) - 2) * norm_scale;
    }
    
    return dequantized;
}

// ============================================================================
// BitNet 1.58-bit Quantization (Ternary: -1, 0, 1)
// ============================================================================

std::vector<uint8_t> Quantization::quantize_fp32_to_bitnet158(const float* data, size_t size) {
    size_t packed_size = (size + 3) / 4;
    std::vector<uint8_t> quantized(packed_size, 0);
    
    for (size_t i = 0; i < size; ++i) {
        // Ternary quantization: -1, 0, 1
        int32_t val;
        if (data[i] > 0.5f) {
            val = 1;
        } else if (data[i] < -0.5f) {
            val = -1;
        } else {
            val = 0;
        }
        
        // Pack four ternary values into one byte (2 bits each)
        size_t byte_idx = i / 4;
        size_t offset = (i % 4) * 2;
        
        // Encode: -1 -> 0b01, 0 -> 0b00, 1 -> 0b10
        uint8_t encoded;
        if (val == -1) encoded = 0x01;
        else if (val == 1) encoded = 0x02;
        else encoded = 0x00;
        
        quantized[byte_idx] |= (encoded << offset);
    }
    
    return quantized;
}

std::vector<float> Quantization::dequantize_bitnet158_to_fp32(const uint8_t* data, size_t size) {
    std::vector<float> dequantized(size);
    
    for (size_t i = 0; i < size; ++i) {
        size_t byte_idx = i / 4;
        size_t offset = (i % 4) * 2;
        
        uint8_t encoded = (data[byte_idx] >> offset) & 0x03;
        
        // Decode: 0b01 -> -1, 0b00 -> 0, 0b10 -> 1
        float val;
        if (encoded == 0x01) val = -1.0f;
        else if (encoded == 0x02) val = 1.0f;
        else val = 0.0f;
        
        dequantized[i] = val;
    }
    
    return dequantized;
}

// ============================================================================
// Scale Computation
// ============================================================================

float Quantization::compute_scale(const float* data, size_t size) {
    if (size == 0) return 1.0f;
    
    float max_abs = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        max_abs = std::max(max_abs, std::abs(data[i]));
    }
    
    return max_abs > 0 ? max_abs : 1.0f;
}

std::vector<float> Quantization::compute_per_channel_scale(
    const float* data, size_t size, size_t channels) {
    std::vector<float> scales(channels, 1.0f);
    size_t channel_size = size / channels;
    
    for (size_t c = 0; c < channels; ++c) {
        float max_abs = 0.0f;
        for (size_t i = 0; i < channel_size; ++i) {
            max_abs = std::max(max_abs, std::abs(data[c * channel_size + i]));
        }
        scales[c] = max_abs > 0 ? max_abs : 1.0f;
    }
    
    return scales;
}

} // namespace cuda_open
