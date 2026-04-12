# CUDA Open - Quick Start Guide

## Project Structure

```
CUDA OPEN/
├── include/cuda_open/           # Header files
│   ├── cuda_open.h              # Main include
│   ├── device.h                 # Device abstraction
│   ├── memory.h                 # Memory management
│   ├── quantization.h           # Quantization support
│   ├── kernel.h                 # Kernel execution
│   └── tensor.h                 # Tensor operations
├── src/                         # Implementation files
│   ├── device.cpp               # CPU device backend
│   ├── quantization.cpp         # Quantization algorithms
│   ├── kernel.cpp               # Kernel launcher
│   └── tensor.cpp               # Tensor operations
├── examples/                    # Example programs
│   ├── 01_basic_memory.cpp      # Memory operations
│   ├── 02_kernel_execution.cpp  # Kernel launches
│   ├── 03_quantization.cpp      # Quantization demo
│   └── 04_bitnet_matmul.cpp     # BitNet matrix mult
├── tests/                       # Unit tests
│   ├── test_device.cpp
│   ├── test_memory.cpp
│   └── test_quantization.cpp
├── docs/                        # Documentation
│   ├── API.md
│   └── BITNET_GUIDE.md
└── CMakeLists.txt               # Build configuration
```

## Building

### Basic Build (CPU only)
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### With Tests and Examples
```bash
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_OPEN_BUILD_TESTS=ON \
    -DCUDA_OPEN_BUILD_EXAMPLES=ON
make -j$(nproc)
```

### With CUDA GPU Support
```bash
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_OPEN_ENABLE_CUDA=ON
make -j$(nproc)
```

## Running Tests

```bash
cd build
ctest --output-on-failure
```

Individual tests:
```bash
./tests/test_quantization
./tests/test_memory
./tests/test_device
```

## Running Examples

```bash
cd build

# Basic memory operations
./examples/example_01_memory

# Kernel execution
./examples/example_02_kernel

# Quantization (INT8, INT4, INT2, BitNet 1.58)
./examples/example_03_quantization

# BitNet matrix multiplication
./examples/example_04_bitnet
```

## Key Features

### 1. Unified CPU/GPU Interface
```cpp
cuda_open::initialize();
auto& device = cuda_open::DeviceManager::instance().get_current_device();
// Code works on both CPU and GPU automatically
```

### 2. Memory Management
```cpp
// Device pointer
cuda_open::DevicePtr<float> dev_data(device, 1024);
dev_data.copy_from_host(host_data);

// Unified buffer (auto-sync)
cuda_open::UnifiedBuffer<float> buffer(device, 1024);
buffer.sync_to_device();
```

### 3. Quantization Support
```cpp
// BitNet 1.58-bit (ternary: -1, 0, 1)
auto quantized = Quantization::quantize_fp32_to_bitnet158(data, size);
auto reconstructed = Quantization::dequantize_bitnet158_to_fp32(quantized, size);

// Other formats: INT8, INT4, INT2
```

### 4. Kernel Execution
```cpp
KernelLauncher launcher(device);
LaunchConfig config = LaunchConfig::make_1d_config(1024, 256);

KernelArgs args;
args.add_device_ptr(input.get());
args.add_device_ptr(output.get());

launcher.launch(my_kernel, config, args);
```

## Compression Ratios

| Format | Bits/Value | Compression | Max Error |
|--------|-----------|-------------|-----------|
| FP32 | 32 | 1x | 0 |
| INT8 | 8 | 4x | ~0.02 |
| INT4 | 4 | 8x | ~0.3 |
| INT2 | 2 | 16x | ~2.0 |
| BitNet | 2 | 16x | ~3.0 |

## When to Use BitNet 1.58

**Good for:**
- Large Language Models (LLMs)
- Inference-only deployment
- Memory-constrained environments
- Edge devices

**Not recommended for:**
- Training (use INT8 minimum)
- Small models
- High-precision requirements

## Next Steps

1. Read `docs/API.md` for complete API reference
2. Read `docs/BITNET_GUIDE.md` for BitNet details
3. Study the examples in `examples/`
4. Check the tests in `tests/` for usage patterns

## Troubleshooting

### Build fails
- Ensure CMake 3.18+ is installed
- Check C++17 compiler support
- For CUDA: verify CUDA toolkit installation

### Tests fail
- All tests should pass on CPU
- GPU tests require CUDA-capable GPU

### Performance issues
- Build with `-DCUDA_OPEN_OPTIMIZE=ON`
- Use Release build type
- Enable CUDA for GPU acceleration

## License

MIT License - Free to use in commercial and academic projects
