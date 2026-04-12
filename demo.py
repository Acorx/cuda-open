"""
CUDA Open - Quick Demo

Test the cuda_open PyTorch module with a real model.

Usage:
    python3 demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import cuda_open


def main():
    print("\n" + "="*60)
    print(" CUDA Open - PyTorch Module Demo")
    print("="*60 + "\n")
    
    # 1. Load model
    print("1. Loading GPT-2 model...")
    model = cuda_open.BitNetForCausalLM.from_pretrained("gpt2")
    print(f"   ✓ Model loaded\n")
    
    # 2. Check size before quantization
    print("2. Model size (before quantization):")
    size_info = model.get_model_size()
    print(f"   Parameters: {size_info['total_params']:,}")
    print(f"   Size: {size_info['original_size_mb']:.1f} MB")
    print(f"   Quantized: {size_info['quantized']}\n")
    
    # 3. Generate text before quantization
    print("3. Generating text (FP32)...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    output_fp32 = model.generate(
        "Hello, how are",
        tokenizer=tokenizer,
        max_length=40,
        do_sample=True,
        temperature=0.8
    )
    print(f"   Output: {output_fp32[:80]}...\n")
    
    # 4. Quantize to BitNet
    print("4. Quantizing to BitNet 1.58-bit...")
    stats = model.quantize_to_bitnet()
    print(f"   Compression: {stats['compression_ratio']:.1f}x\n")
    
    # 5. Check size after quantization
    print("5. Model size (after quantization):")
    size_info = model.get_model_size()
    print(f"   Compressed size: {size_info['compressed_size_mb']:.1f} MB")
    print(f"   Compression: {size_info['compression_ratio']:.1f}x")
    print(f"   Quantized: {size_info['quantized']}\n")
    
    # 6. Generate text after quantization
    print("6. Generating text (BitNet 1.58-bit)...")
    output_bitnet = model.generate(
        "Hello, how are",
        tokenizer=tokenizer,
        max_length=40,
        do_sample=True,
        temperature=0.8
    )
    print(f"   Output: {output_bitnet[:80]}...\n")
    
    # 7. Summary
    print("="*60)
    print(" SUMMARY")
    print("="*60)
    print(f"  ✓ Model loaded and quantized")
    print(f"  ✓ Compression: {stats['compression_ratio']:.1f}x")
    print(f"  ✓ Generation works with BitNet weights")
    print(f"  ✓ Module ready for use!\n")
    
    print("Next steps:")
    print("  from cuda_open import BitNetForCausalLM")
    print("  model = BitNetForCausalLM.from_pretrained('gpt2')")
    print("  model.quantize_to_bitnet()")
    print("  model.generate('Hello')")
    print()


if __name__ == "__main__":
    main()
