"""
CUDA Open - Wrapper Python pour les Kernels CUDA BitNet

Charge dynamiquement le kernel CUDA compilé et expose une API Python propre.
Fallback automatique sur numpy si CUDA n'est pas disponible.

Usage:
    from cuda_open.cuda_kernels import BitNetMatMul
    
    matmul = BitNetMatMul()  # Détecte automatiquement CUDA
    C = matmul.execute(A_packed, B, scale, M, K, N)
"""

import numpy as np
import ctypes
import os
import sys
from pathlib import Path
from typing import Optional


class BitNetCUDAKernels:
    """Wrapper vers les kernels CUDA BitNet compilés."""
    
    def __init__(self, lib_path: Optional[str] = None):
        self.lib = None
        self.cuda_available = False
        self.gpu_name = "N/A"
        
        # Cherche la bibliothèque
        if lib_path is None:
            # Cherche dans le dossier kernels/
            project_root = Path(__file__).parent.parent.parent
            possible_paths = [
                project_root / "kernels" / "libbitnet.so",
                project_root / "kernels" / "libbitnet.dylib",  # macOS
                project_root / "kernels" / "bitnet_kernels.dll",  # Windows
            ]
            for p in possible_paths:
                if p.exists():
                    lib_path = str(p)
                    break
        
        if lib_path and Path(lib_path).exists():
            try:
                self.lib = ctypes.CDLL(lib_path)
                self._setup_c_api()
                self.cuda_available = self.lib.cuda_available()
                if self.cuda_available:
                    # Récupère le nom du GPU
                    name_buffer = ctypes.create_string_buffer(256)
                    self.lib.get_gpu_name(name_buffer, 256)
                    self.gpu_name = name_buffer.value.decode('utf-8')
            except Exception as e:
                print(f"⚠️ Erreur chargement CUDA kernel: {e}")
                self.lib = None
    
    def _setup_c_api(self):
        """Configure les signatures des fonctions C."""
        if self.lib is None:
            return
        
        # bitnet_matmul_launch
        self.lib.bitnet_matmul_launch.argtypes = [
            ctypes.c_void_p,  # A_packed
            ctypes.c_void_p,  # B
            ctypes.c_void_p,  # C
            ctypes.c_float,   # scale
            ctypes.c_int,     # M
            ctypes.c_int,     # K
            ctypes.c_int,     # N
            ctypes.c_int      # use_shared_memory
        ]
        self.lib.bitnet_matmul_launch.restype = ctypes.c_int
        
        # bitnet_batch_matmul_launch
        self.lib.bitnet_batch_matmul_launch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self.lib.bitnet_batch_matmul_launch.restype = ctypes.c_int
        
        # cuda_available
        self.lib.cuda_available.argtypes = []
        self.lib.cuda_available.restype = ctypes.c_int
        
        # get_gpu_name
        self.lib.get_gpu_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.get_gpu_name.restype = None
        
        # benchmark_bitnet_matmul
        self.lib.benchmark_bitnet_matmul.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self.lib.benchmark_bitnet_matmul.restype = ctypes.c_float
    
    def matmul(self, A_packed: np.ndarray, B: np.ndarray, 
               scale: float, M: int, K: int, N: int,
               use_shared_memory: bool = False) -> np.ndarray:
        """
        Exécute C = A_ternary @ B * scale sur GPU.
        
        Args:
            A_packed: uint8 array [M, K/4] ternary packed
            B: float32 array [K, N]
            scale: float scale factor
            M, K, N: dimensions
            use_shared_memory: utilise la mémoire partagée
        
        Returns:
            C: float32 array [M, N]
        """
        if not self.cuda_available or self.lib is None:
            # Fallback numpy
            return self._fallback_matmul(A_packed, B, scale, M, K, N)
        
        # Alloue la sortie
        C = np.zeros((M, N), dtype=np.float32)
        
        # Appelle le kernel
        err = self.lib.bitnet_matmul_launch(
            A_packed.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_float(scale),
            ctypes.c_int(M),
            ctypes.c_int(K),
            ctypes.c_int(N),
            ctypes.c_int(1 if use_shared_memory else 0)
        )
        
        if err != 0:
            raise RuntimeError("CUDA kernel execution failed")
        
        return C
    
    def benchmark(self, M: int, K: int, N: int, iterations: int = 100) -> float:
        """Benchmark le kernel CUDA. Retourne le temps moyen en ms."""
        if not self.cuda_available or self.lib is None:
            return -1.0
        
        return self.lib.benchmark_bitnet_matmul(M, K, N, iterations)
    
    @staticmethod
    def _fallback_matmul(A_packed: np.ndarray, B: np.ndarray,
                         scale: float, M: int, K: int, N: int) -> np.ndarray:
        """Fallback numpy si CUDA non disponible."""
        # Dépacke A
        lut = np.array([0.0, -1.0, 1.0, 0.0], dtype=np.float32)
        A = np.zeros((M, K), dtype=np.float32)
        k_packed = K // 4
        
        for i in range(M):
            for p in range(k_packed):
                packed = A_packed[i, p]
                for j in range(4):
                    if p * 4 + j < K:
                        encoded = (packed >> (j * 2)) & 0x03
                        A[i, p * 4 + j] = lut[encoded] * scale
        
        return A @ B


# Singleton global
_kernels: Optional[BitNetCUDAKernels] = None

def get_kernels() -> BitNetCUDAKernels:
    """Retourne l'instance singleton des kernels CUDA."""
    global _kernels
    if _kernels is None:
        _kernels = BitNetCUDAKernels()
    return _kernels


def is_cuda_available() -> bool:
    """Vérifie si les kernels CUDA sont disponibles."""
    return get_kernels().cuda_available


def get_gpu_info() -> str:
    """Retourne les infos du GPU."""
    k = get_kernels()
    if k.cuda_available:
        return f"CUDA: {k.gpu_name}"
    return "CUDA: Non disponible (fallback numpy)"
