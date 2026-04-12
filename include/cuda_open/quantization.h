/**
 * @file quantization.h
 * @brief Quantization support for various bit-widths
 * 
 * Supports FP32, FP16, INT8, INT4, INT2, and BitNet 1.58-bit quantization
 * with efficient packing and unpacking operations.
 */

#ifndef CUDA_OPEN_QUANTIZATION_H
#define CUDA_OPEN_QUANTIZATION_H

#include "device.h"
#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring>

namespace cuda_open {

// Quantization utilities
class Quantization {
public:
    // Get number of values that can be packed in one byte
    static int get_packing_factor(QuantizationType type);
    
    // Calculate required storage size
    static size_t get_storage_size(size_t num_elements, QuantizationType type);
    
    // Quantization functions
    static std::vector<uint8_t> quantize_fp32_to_int8(const float* data, size_t size);
    static std::vector<float> dequantize_int8_to_fp32(const uint8_t* data, size_t size, float scale);
    
    static std::vector<uint8_t> quantize_fp32_to_int4(const float* data, size_t size);
    static std::vector<float> dequantize_int4_to_fp32(const uint8_t* data, size_t size, float scale);
    
    static std::vector<uint8_t> quantize_fp32_to_int2(const float* data, size_t size);
    static std::vector<float> dequantize_int2_to_fp32(const uint8_t* data, size_t size, float scale);
    
    // BitNet 1.58-bit quantization (ternary: -1, 0, 1)
    static std::vector<uint8_t> quantize_fp32_to_bitnet158(const float* data, size_t size);
    static std::vector<float> dequantize_bitnet158_to_fp32(const uint8_t* data, size_t size);
    
    // Scale computation
    static float compute_scale(const float* data, size_t size);
    static std::vector<float> compute_per_channel_scale(const float* data, size_t size, size_t channels);
};

// Quantized tensor storage
template<typename T = uint8_t>
class QuantizedTensor {
public:
    QuantizedTensor() 
        : data_(nullptr), size_(0), quant_type_(QuantizationType::FP32), 
          scale_(1.0f), zero_point_(0) {}
    
    QuantizedTensor(size_t num_elements, QuantizationType qtype)
        : size_(num_elements), quant_type_(qtype) {
        size_t storage_size = Quantization::get_storage_size(num_elements, qtype);
        data_ = new T[storage_size]();
        scale_ = 1.0f;
        zero_point_ = 0;
    }
    
    ~QuantizedTensor() {
        delete[] data_;
    }
    
    // Move semantics
    QuantizedTensor(QuantizedTensor&& other) noexcept
        : data_(other.data_), size_(other.size_), quant_type_(other.quant_type_),
          scale_(other.scale_), zero_point_(other.zero_point_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    QuantizedTensor& operator=(QuantizedTensor&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            quant_type_ = other.quant_type_;
            scale_ = other.scale_;
            zero_point_ = other.zero_point_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Accessors
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    QuantizationType quant_type() const { return quant_type_; }
    float scale() const { return scale_; }
    void set_scale(float s) { scale_ = s; }
    int zero_point() const { return zero_point_; }
    void set_zero_point(int zp) { zero_point_ = zp; }
    
    // Quantize from FP32
    void quantize_from_fp32(const float* fp32_data) {
        switch (quant_type_) {
            case QuantizationType::INT8: {
                scale_ = Quantization::compute_scale(fp32_data, size_);
                auto quantized = Quantization::quantize_fp32_to_int8(fp32_data, size_);
                std::memcpy(data_, quantized.data(), quantized.size());
                break;
            }
            case QuantizationType::INT4: {
                scale_ = Quantization::compute_scale(fp32_data, size_);
                auto quantized = Quantization::quantize_fp32_to_int4(fp32_data, size_);
                std::memcpy(data_, quantized.data(), quantized.size());
                break;
            }
            case QuantizationType::INT2: {
                scale_ = Quantization::compute_scale(fp32_data, size_);
                auto quantized = Quantization::quantize_fp32_to_int2(fp32_data, size_);
                std::memcpy(data_, quantized.data(), quantized.size());
                break;
            }
            case QuantizationType::BITNET158: {
                auto quantized = Quantization::quantize_fp32_to_bitnet158(fp32_data, size_);
                std::memcpy(data_, quantized.data(), quantized.size());
                break;
            }
            default:
                throw DeviceError("Unsupported quantization type");
        }
    }
    
    // Dequantize to FP32
    std::vector<float> dequantize_to_fp32() const {
        switch (quant_type_) {
            case QuantizationType::INT8:
                return Quantization::dequantize_int8_to_fp32(data_, size_, scale_);
            case QuantizationType::INT4:
                return Quantization::dequantize_int4_to_fp32(data_, size_, scale_);
            case QuantizationType::INT2:
                return Quantization::dequantize_int2_to_fp32(data_, size_, scale_);
            case QuantizationType::BITNET158:
                return Quantization::dequantize_bitnet158_to_fp32(data_, size_);
            default:
                throw DeviceError("Unsupported quantization type");
        }
    }
    
private:
    T* data_;
    size_t size_;
    QuantizationType quant_type_;
    float scale_;
    int zero_point_;
};

} // namespace cuda_open

#endif // CUDA_OPEN_QUANTIZATION_H
