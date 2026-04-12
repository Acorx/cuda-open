"""
CUDA Open - BitNet Quantizer (ULTRA RAPIDE)

Quantize arrays to BitNet 1.58-bit format.
100% numpy - ZERO dépendance lourde.
"""

import numpy as np
from typing import Tuple


class BitNetQuantizer:
    """Quantize numpy arrays to BitNet 1.58-bit ternary format."""

    @staticmethod
    def quantize(data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Quantize numpy array to packed ternary format.
        
        Args:
            data: numpy float32 array
            
        Returns:
            packed: uint8 array (4 ternary values per byte)
            scale: float scale factor
        """
        # Ensure float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # Compute scale
        scale = float(np.max(np.abs(data)))
        if scale < 1e-8:
            scale = 1.0
        
        # Threshold for ternary (20% of max)
        threshold = 0.2 * scale
        
        # Quantize to ternary: -1, 0, 1
        ternary = np.zeros_like(data, dtype=np.int8)
        ternary[data > threshold] = 1
        ternary[data < -threshold] = -1
        
        # Pack 4 ternary values per byte
        packed = BitNetQuantizer._pack_fast(ternary)
        
        return packed, scale
    
    @staticmethod
    def _pack_fast(ternary: np.ndarray) -> np.ndarray:
        """Fast packing: 4 ternary values → 1 byte"""
        # Flatten
        flat = ternary.flatten()
        n = len(flat)
        
        # Map -1→1, 0→0, 1→2
        vals = (flat + 1).astype(np.uint8)
        
        # Pad to multiple of 4
        remainder = n % 4
        if remainder > 0:
            vals = np.pad(vals, (0, 4 - remainder), mode='constant')
        
        # Reshape to groups of 4
        vals = vals.reshape(-1, 4)
        
        # Pack: [a, b, c, d] → a | (b<<2) | (c<<4) | (d<<6)
        packed = (vals[:, 0] | 
                  (vals[:, 1] << 2) | 
                  (vals[:, 2] << 4) | 
                  (vals[:, 3] << 6)).astype(np.uint8)
        
        return packed
    
    @staticmethod
    def unpack(packed: np.ndarray, original_size: int, scale: float) -> np.ndarray:
        """Unpack ternary values back to float32."""
        # Lookup table
        lut = np.array([0.0, -1.0, 1.0, 0.0], dtype=np.float32)
        
        # Unpack each byte into 4 values
        vals = np.zeros(original_size, dtype=np.float32)
        
        for i in range(original_size):
            byte_idx = i // 4
            offset = (i % 4) * 2
            encoded = (packed[byte_idx] >> offset) & 0x03
            vals[i] = lut[encoded] * scale
        
        return vals


# Fonctions de commodité
def quantize(data: np.ndarray) -> Tuple[np.ndarray, float]:
    """Quantize data to BitNet format."""
    return BitNetQuantizer.quantize(data)


def compress(data: np.ndarray) -> dict:
    """Compress data to BitNet format with metadata."""
    packed, scale = BitNetQuantizer.quantize(data)
    return {
        'packed': packed,
        'scale': scale,
        'original_shape': data.shape,
        'original_dtype': str(data.dtype),
        'compression_ratio': data.nbytes / packed.nbytes
    }


def decompress(metadata: dict) -> np.ndarray:
    """Decompress BitNet data back to float32."""
    return BitNetQuantizer.unpack(
        metadata['packed'],
        int(np.prod(metadata['original_shape'])),
        metadata['scale']
    ).reshape(metadata['original_shape'])
