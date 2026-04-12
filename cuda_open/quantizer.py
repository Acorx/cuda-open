"""
CUDA Open - BitNet Quantizer

Quantize arrays to BitNet 1.58-bit format.
Works with numpy arrays - no PyTorch required for basic quantization.
"""

import numpy as np
from typing import Tuple, Dict, Optional


class BitNetQuantizer:
    """
    Quantize PyTorch models to BitNet 1.58-bit ternary format.
    
    Supports:
    - Post-training quantization
    - Quantization-aware training (QAT)
    - Gradual quantization rampup
    """
    
    @staticmethod
    def quantize(tensor) -> Tuple:
        """
        Quantize a tensor/array to ternary {-1, 0, 1}.
        
        Works with both numpy arrays and PyTorch tensors.
        
        Returns:
            ternary: Ternary array
            scale: Quantization scale
        """
        # Handle both numpy and torch
        is_torch = False
        try:
            import torch
            if isinstance(tensor, any):
                is_torch = True
                device = tensor.device
                tensor_np = tensor.cpu().numpy()
        except:
            tensor_np = np.asarray(tensor)
        
        # Compute scale
        scale = np.max(np.abs(tensor_np))
        if scale < 1e-8:
            scale = 1.0
        
        # Threshold for ternary
        threshold = 0.2 * scale
        
        # Quantize to ternary
        ternary = np.zeros_like(tensor_np)
        ternary[tensor_np > threshold] = 1.0
        ternary[tensor_np < -threshold] = -1.0
        
        # Convert back to torch if needed
        if is_torch:
            import torch
            ternary = torch.from_numpy(ternary).to(device)
            scale = torch.tensor(scale, device=device)
        
        return ternary, scale
    
    @staticmethod
    def pack_ternary(ternary: any) -> any:
        """
        Pack ternary values into uint8 (4 values per byte).
        
        Encoding: 0b00=0, 0b01=-1, 0b10=1
        """
        ternary_flat = ternary.flatten().cpu().numpy()
        n = len(ternary_flat)
        packed_size = (n + 3) // 4
        
        packed = np.zeros(packed_size, dtype=np.uint8)
        
        for i in range(n):
            val = int(ternary_flat[i])
            if val == -1:
                encoded = 0x01
            elif val == 1:
                encoded = 0x02
            else:
                encoded = 0x00
            
            byte_idx = i // 4
            bit_offset = (i % 4) * 2
            packed[byte_idx] |= (encoded << bit_offset)
        
        return torch.from_numpy(packed)
    
    @staticmethod
    def unpack_ternary(packed: any, shape: tuple, device='cpu') -> any:
        """Unpack ternary values from uint8 format."""
        packed_np = packed.cpu().numpy()
        n_elements = int(np.prod(shape))
        
        lut = np.array([0, -1, 1, 0], dtype=np.float32)
        ternary = np.zeros(n_elements, dtype=np.float32)
        
        for i in range(n_elements):
            byte_idx = i // 4
            bit_offset = (i % 4) * 2
            encoded = (packed_np[byte_idx] >> bit_offset) & 0x03
            ternary[i] = lut[encoded]
        
        return torch.from_numpy(ternary.reshape(shape)).to(device)
    
    @classmethod
    def quantize_model(cls, model: any, verbose: bool = True) -> Dict:
        """
        Quantize entire model to BitNet 1.58-bit.
        
        Returns:
            stats: Quantization statistics
        """
        stats = {
            'total_params': 0,
            'quantized_params': 0,
            'original_size_bytes': 0,
            'compressed_size_bytes': 0,
            'layers': []
        }
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Quantize weights
                weight = module.weight.data
                ternary, scale = cls.quantize_tensor(weight)
                packed = cls.pack_ternary(ternary)
                
                # Store quantized weights
                module.weight.data = ternary * scale
                
                # Update stats
                num_params = weight.numel()
                stats['total_params'] += num_params
                stats['quantized_params'] += packed.numel() * 4  # 4 values per byte
                stats['original_size_bytes'] += num_params * 4  # FP32
                stats['compressed_size_bytes'] += packed.numel()
                
                stats['layers'].append({
                    'name': name,
                    'params': num_params,
                    'scale': scale.item()
                })
        
        # Compute compression ratio
        if stats['compressed_size_bytes'] > 0:
            stats['compression_ratio'] = stats['original_size_bytes'] / stats['compressed_size_bytes']
        else:
            stats['compression_ratio'] = 1.0
        
        if verbose:
            print(f"\n{'='*60}")
            print(f" BitNet Quantization Complete")
            print(f"{'='*60}")
            print(f"  Layers quantized: {len(stats['layers'])}")
            print(f"  Total params: {stats['total_params']:,}")
            print(f"  Original size: {stats['original_size_bytes']/1024/1024:.1f} MB")
            print(f"  Compressed size: {stats['compressed_size_bytes']/1024/1024:.1f} MB")
            print(f"  Compression ratio: {stats['compression_ratio']:.1f}x")
            print(f"{'='*60}\n")
        
        return stats


def quantize_model(model: any, **kwargs) -> Dict:
    """Convenience function to quantize model."""
    return BitNetQuantizer.quantize_model(model, **kwargs)


def dequantize_model(model: any) -> None:
    """Restore model to FP32 (if master weights available)."""
    for name, module in model.named_modules():
        if hasattr(module, 'master_weight'):
            module.weight.data = module.master_weight.data.clone()
