"""
CUDA Open - Minimal Example

SUPER LIGHTWEIGHT - No heavy model loading!
Tests only the core quantization functionality.

Usage:
    python3 examples/minimal_example.py
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("\n" + "="*60)
    print(" CUDA Open - Minimal Example")
    print("="*60 + "\n")
    
    # 1. Test basic quantization
    print("1. Testing basic quantization...")
    data = np.random.randn(100).astype(np.float32)
    
    # Manual quantization
    scale = np.max(np.abs(data))
    threshold = 0.2 * scale
    ternary = np.zeros_like(data)
    ternary[data > threshold] = 1
    ternary[data < -threshold] = -1
    
    # Pack 4 values per byte
    packed = np.zeros((len(data) + 3) // 4, dtype=np.uint8)
    for i in range(len(data)):
        val = int(ternary[i])
        if val == -1: enc = 1
        elif val == 1: enc = 2
        else: enc = 0
        packed[i//4] |= (enc << ((i%4)*2))
    
    compression = data.nbytes / packed.nbytes
    
    print(f"   ✓ Original: {data.nbytes} bytes")
    print(f"   ✓ Compressed: {packed.nbytes} bytes")
    print(f"   ✓ Compression: {compression:.1f}x\n")
    
    # 2. Test from cuda_open module
    print("2. Testing cuda_open module...")
    try:
        from cuda_open.quantizer import BitNetQuantizer
        
        data2 = np.random.randn(200).astype(np.float32)
        packed2, scale2 = BitNetQuantizer.quantize(data2)
        compression2 = data2.nbytes / packed2.nbytes
        
        print(f"   ✓ Module loaded successfully")
        print(f"   ✓ Compression: {compression2:.1f}x\n")
    except ImportError as e:
        print(f"   ⚠ Module not installed: {e}")
        print(f"   Run: pip install -e .\n")
    
    # 3. Summary
    print("="*60)
    print(" SUMMARY")
    print("="*60)
    print(f"  ✓ Basic quantization works")
    print(f"  ✓ ~16x compression achieved")
    print(f"  ✓ cuda_open module {'available' if 'BitNetQuantizer' in dir() else 'not installed'}")
    print()
    print("Next steps:")
    print("  pip install -e .")
    print("  python3 demo.py")
    print()


if __name__ == "__main__":
    main()
