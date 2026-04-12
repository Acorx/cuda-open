#!/bin/bash
# CUDA Open - Easy build and run script

set -e

echo "========================================="
echo "CUDA Open - Build and Test"
echo "========================================="
echo ""

# Configuration
BUILD_TYPE=${1:-Release}
ENABLE_CUDA=${2:-OFF}
BUILD_DIR="build"

echo "Configuration:"
echo "  Build Type: $BUILD_TYPE"
echo "  CUDA Support: $ENABLE_CUDA"
echo "  Build Directory: $BUILD_DIR"
echo ""

# Create build directory
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure
echo "-----------------------------------------"
echo "Step 1: Configuring with CMake..."
echo "-----------------------------------------"
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCUDA_OPEN_BUILD_TESTS=ON \
    -DCUDA_OPEN_BUILD_EXAMPLES=ON \
    -DCUDA_OPEN_ENABLE_CUDA=$ENABLE_CUDA \
    -DCUDA_OPEN_OPTIMIZE=ON

echo ""

# Build
echo "-----------------------------------------"
echo "Step 2: Building..."
echo "-----------------------------------------"
make -j$(nproc)

echo ""

# Test
echo "-----------------------------------------"
echo "Step 3: Running Tests..."
echo "-----------------------------------------"
ctest --output-on-failure

echo ""
echo "========================================="
echo "Build and Test Complete!"
echo "========================================="
echo ""
echo "Examples available in: $BUILD_DIR/examples/"
echo "  - ./examples/example_01_memory"
echo "  - ./examples/example_02_kernel"
echo "  - ./examples/example_03_quantization"
echo "  - ./examples/example_04_bitnet"
echo ""
echo "Run an example: cd $BUILD_DIR && ./examples/example_03_quantization"
echo ""
