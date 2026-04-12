# BitNet 1.58-bit Quantization Guide

## What is BitNet 1.58-bit?

BitNet 1.58-bit is an extreme quantization technique that represents weights using ternary values {-1, 0, 1}, achieving approximately 16x compression compared to FP32 while maintaining reasonable accuracy.

The "1.58-bit" name comes from log2(3) ≈ 1.58, as we need ~1.58 bits to represent 3 distinct values.

## When to Use BitNet

### Good Use Cases
- **Large Language Models**: Reduce memory footprint dramatically
- **Inference-only**: Quantization loss acceptable for inference
- **Edge Deployment**: Run large models on constrained hardware
- **Batch Processing**: Throughput over precision

### Not Recommended For
- **Training**: Gradient updates require higher precision
- **Small Models**: Accuracy loss may be too significant
- **Precision-critical Applications**: Use INT8 instead

## How It Works

### Encoding Scheme

Each value is mapped to one of three levels:
- Values > threshold → +1
- Values < -threshold → -1  
- Otherwise → 0

Default threshold: 0.5

### Packing Format

Four ternary values are packed into one byte using 2 bits per value:
- `0b00` → 0
- `0b01` → -1
- `0b10` → +1
- `0b11` → unused

## Usage Example

```cpp
#include <cuda_open/quantization.h>

using namespace cuda_open;

// Original FP32 data
std::vector<float> weights = {0.8f, -0.6f, 0.1f, -0.9f, 0.0f, 0.5f};

// Quantize to BitNet format
auto quantized = Quantization::quantize_fp32_to_bitnet158(
    weights.data(), 
    weights.size()
);

// Storage size comparison
size_t original_size = weights.size() * sizeof(float);  // 24 bytes
size_t compressed_size = quantized.size();              // 2 bytes
std::cout << "Compression: " << (float)original_size / compressed_size << "x\n";

// Dequantize when needed
auto reconstructed = Quantization::dequantize_bitnet158_to_fp32(
    quantized.data(),
    weights.size()
);

// reconstructed will be: {1.0, -1.0, 0.0, -1.0, 0.0, 0.0}
```

## Using QuantizedTensor

```cpp
// Create quantized tensor
QuantizedTensor<> tensor(num_weights, QuantizationType::BITNET158);

// Quantize from FP32
tensor.quantize_from_fp32(weight_data);

// Access raw quantized data
const uint8_t* raw_data = tensor.data();
size_t storage_size = Quantization::get_storage_size(
    num_weights, 
    QuantizationType::BITNET158
);

// Dequantize when needed for computation
auto fp32_weights = tensor.dequantize_to_fp32();
```

## BitNet Matrix Multiplication

```cpp
// Quantize both matrices
QuantizedTensor<> qA(A.size(), QuantizationType::BITNET158);
QuantizedTensor<> qB(B.size(), QuantizationType::BITNET158);

qA.quantize_from_fp32(A.data());
qB.quantize_from_fp32(B.data());

// For actual computation, dequantize first
auto A_dequant = qA.dequantize_to_fp32();
auto B_dequant = qB.dequantize_to_fp32();

// Then perform matrix multiplication
// Note: The actual matmul happens in dequantized space
// The benefit is reduced memory bandwidth requirements
```

## Performance Characteristics

### Memory Savings
- **FP32**: 32 bits per value
- **BitNet**: 2 bits per value
- **Compression**: 16x reduction

### Accuracy Trade-offs

| Metric | FP32 | INT8 | INT4 | BitNet |
|--------|------|------|------|--------|
| Bits/value | 32 | 8 | 4 | 2 |
| Compression | 1x | 4x | 8x | 16x |
| Typical Accuracy | 100% | 95-99% | 85-95% | 70-90% |

### When BitNet Works Well

BitNet performs best when:
1. Weight distribution is approximately normal
2. Model has been trained with quantization awareness
3. Values naturally cluster around {-1, 0, 1}
4. Model is overparameterized (redundancy helps)

## Custom Threshold

You can adjust the quantization threshold:

```cpp
// Use custom threshold instead of default 0.5
float custom_threshold = 0.3f;

// Manual quantization with custom threshold
std::vector<uint8_t> custom_quantize(const float* data, size_t size) {
    std::vector<uint8_t> result((size + 3) / 4, 0);
    
    for (size_t i = 0; i < size; ++i) {
        int val;
        if (data[i] > custom_threshold) val = 1;
        else if (data[i] < -custom_threshold) val = -1;
        else val = 0;
        
        // Pack into byte (same as standard BitNet)
        size_t byte_idx = i / 4;
        size_t offset = (i % 4) * 2;
        
        uint8_t encoded = (val == -1) ? 0x01 : (val == 1) ? 0x02 : 0x00;
        result[byte_idx] |= (encoded << offset);
    }
    
    return result;
}
```

## Tips for Best Results

1. **Quantization-Aware Training**: Train with simulated quantization for best accuracy
2. **Per-Channel Scaling**: Use different scales for different channels
3. **Outlier Handling**: Clip extreme values before quantization
4. **Calibration**: Use representative data to choose optimal threshold
5. **Mixed Precision**: Keep critical layers in higher precision

## Limitations

- **Irreversible**: Information loss is significant
- **Not for Gradients**: Too imprecise for training
- **Model-dependent**: Some architectures quantize better than others
- **Requires Testing**: Always validate accuracy after quantization

## References

- Wang, H. et al. "BitNet: Scaling 1-bit Transformers for Large Language Models" (2023)
- Ma, S. et al. "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" (2024)
