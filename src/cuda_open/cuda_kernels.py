"""
CUDA Open - Wrapper Python pour les Kernels CUDA BitNet

Gère le chargement des bibliothèques CUDA et expose l'API Python.
Supporte les kernels standards ET les kernels fusionnés (la méthode pour battre CUDA).
"""

import numpy as np
import ctypes
import os
import sys
from pathlib import Path
from typing import Optional


class BitNetCUDAKernels:
    """Interface vers les kernels CUDA compilés."""
    
    def __init__(self, lib_path: Optional[str] = None):
        self.lib = None
        self.cuda_available = False
        self.gpu_name = "N/A"
        self.fused_kernel_available = False
        
        # Cherche la bibliothèque
        if lib_path is None:
            project_root = Path(__file__).parent.parent.parent
            possible_paths = [
                project_root / "kernels" / "libbitnet.so",
                project_root / "kernels" / "libbitnet.dylib",
                project_root / "kernels" / "bitnet_kernels.dll",
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
                    name_buffer = ctypes.create_string_buffer(256)
                    self.lib.get_gpu_name(name_buffer, 256)
                    self.gpu_name = name_buffer.value.decode('utf-8')
                    
                    # Vérifie si le kernel fusionné est dispo
                    if hasattr(self.lib, 'launch_fused_bitnet_relu'):
                        self.fused_kernel_available = True
            except Exception as e:
                print(f"⚠️ Erreur chargement CUDA kernel: {e}")
                self.lib = None
    
    def _setup_c_api(self):
        """Configure les signatures des fonctions C."""
        if not self.lib: return
        
        # launch_fused_bitnet_relu (Le tueur de performance)
        self.lib.launch_fused_bitnet_relu.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_float,
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self.lib.launch_fused_bitnet_relu.restype = ctypes.c_int
        
        # ... (autres signatures standard) ...
        self.lib.bitnet_matmul_launch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self.lib.bitnet_matmul_launch.restype = ctypes.c_int
        
        self.lib.cuda_available.restype = ctypes.c_int
        self.lib.get_gpu_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.benchmark_bitnet_matmul.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.benchmark_bitnet_matmul.restype = ctypes.c_float
    
    def matmul_fused(self, A_packed, B, Bias, scale, M, K, N):
        """
        Exécute: Output = ReLU( (A @ B) + Bias )
        C'est cette méthode qui bat les librairies standards en évitant les accès mémoire inutiles.
        """
        if not self.cuda_available or not self.lib:
            # Fallback numpy pour la compatibilité
            # Note: Le fallback numpy simule le ReLU
            C = self.matmul(A_packed, B, scale, M, K, N)
            if Bias is not None:
                C += Bias[:, np.newaxis].T # Bias broadcast
            return np.maximum(0, C) # ReLU
        
        C = np.zeros((M, N), dtype=np.float32)
        bias_ptr = Bias.ctypes.data_as(ctypes.c_void_p) if Bias is not None else ctypes.c_void_p(0)
        
        err = self.lib.launch_fused_bitnet_relu(
            A_packed.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            bias_ptr,
            C.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_float(scale),
            M, K, N
        )
        if err != 0: raise RuntimeError("Fused kernel failed")
        return C

    def matmul(self, A_packed, B, scale, M, K, N, use_shared_memory=False):
        """Multiplication matricielle standard (pour référence)."""
        if not self.cuda_available or not self.lib:
            return self._fallback_matmul(A_packed, B, scale, M, K, N)
        
        C = np.zeros((M, N), dtype=np.float32)
        err = self.lib.bitnet_matmul_launch(
            A_packed.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_float(scale),
            M, K, N,
            1 if use_shared_memory else 0
        )
        if err != 0: raise RuntimeError("Standard kernel failed")
        return C

    def _fallback_matmul(self, A_packed, B, scale, M, K, N):
        """Implémentation numpy pure (toujours fonctionnelle)."""
        lut = np.array([0.0, -1.0, 1.0, 0.0], dtype=np.float32)
        A = np.zeros((M, K), dtype=np.float32)
        k_packed = K // 4
        
        # Dépacke en vectorisant numpy (plus rapide que la boucle)
        # Note: C'est une simplification, le unpacking exact dépend de l'endianness
        for i in range(4):
            # Extrait les bits pour la position i
            vals = (A_packed >> (i*2)) & 0x03
            # Utilise le lookup table
            decoded = lut[vals]
            # Place au bon endroit dans A (chaque byte correspond à 4 colonnes de K)
            A[:, i::4] = decoded[:M].reshape(M, -1)[:,:A.shape[1]//4] 
            
        # Correction pour le fallback simple (pour la démo on fait une boucle propre)
        A = np.zeros((M, K), dtype=np.float32)
        for r in range(M):
            for p in range(k_packed):
                packed = A_packed[r, p]
                for i in range(4):
                    if p*4+i < K:
                        A[r, p*4+i] = lut[(packed >> (i*2)) & 0x03] * scale
                        
        return A @ B


# Singleton
_kernels: Optional[BitNetCUDAKernels] = None

def get_kernels() -> BitNetCUDAKernels:
    global _kernels
    if _kernels is None:
        _kernels = BitNetCUDAKernels()
    return _kernels

def is_cuda_available() -> bool:
    return get_kernels().cuda_available

def get_gpu_info() -> str:
    k = get_kernels()
    return f"CUDA: {k.gpu_name}" if k.cuda_available else "CUDA: N/A (Numpy Fallback)"
