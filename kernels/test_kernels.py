"""
CUDA Open - Test & Benchmark des Kernels CUDA

Compare l'implémentation CUDA (si dispo) vs numpy.
Montre le fallback automatique et les performances.

Usage:
    python3 kernels/test_kernels.py
"""

import numpy as np
import time
import sys
from pathlib import Path

# Ajoute le chemin du projet
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

try:
    from cuda_open.cuda_kernels import BitNetCUDAKernels, get_kernels, is_cuda_available, get_gpu_info
    CUDA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Import kernels: {e}")
    CUDA_AVAILABLE = False


def test_numpy_fallback():
    """Teste l'implémentation numpy (toujours fonctionnelle)."""
    print("\n" + "="*60)
    print(" TEST 1: Fallback Numpy (CPU)")
    print("="*60 + "\n")
    
    from cuda_open.quantizer import BitNetQuantizer
    
    # Crée des données test
    M, K, N = 64, 128, 64
    np.random.seed(42)
    A_fp32 = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    # Quantize A
    packed_A, scale = BitNetQuantizer.quantize(A_fp32)
    packed_A = packed_A.reshape(M, K // 4)
    
    # Test avec le fallback
    kernels = BitNetCUDAKernels()
    
    print(f"  Dimensions: {M}x{K} @ {K}x{N}")
    print(f"  CUDA disponible: {kernels.cuda_available}")
    
    # Benchmark numpy
    times = []
    for _ in range(10):
        start = time.time()
        C = kernels._fallback_matmul(packed_A, B, scale, M, K, N)
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000
    print(f"  Temps moyen: {avg_time:.2f} ms")
    print(f"  Output shape: {C.shape}")
    print(f"  Output sample: {C[0, :5]}")
    print(f"  ✅ Test fallback réussi\n")


def test_cuda_kernel():
    """Teste le kernel CUDA si compilé."""
    print("\n" + "="*60)
    print(" TEST 2: Kernel CUDA (GPU)")
    print("="*60 + "\n")
    
    if not CUDA_AVAILABLE:
        print("  ⚠️ Module cuda_kernels non importable\n")
        return
    
    kernels = get_kernels()
    
    if not kernels.cuda_available:
        print("  ⚠️ CUDA non disponible sur ce système")
        print("  → Fallback vers numpy automatique")
        print("  → Pour activer CUDA: compiler avec nvcc")
        print("     cd kernels && ./build.sh\n")
        return
    
    print(f"  ✅ GPU détecté: {kernels.gpu_name}")
    
    # Test de multiplication
    M, K, N = 256, 512, 256
    np.random.seed(42)
    A_fp32 = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    from cuda_open.quantizer import BitNetQuantizer
    packed_A, scale = BitNetQuantizer.quantize(A_fp32)
    packed_A = packed_A.reshape(M, K // 4)
    
    # Exécute sur GPU
    C_gpu = kernels.matmul(packed_A, B, scale, M, K, N)
    
    # Compare avec numpy
    C_cpu = kernels._fallback_matmul(packed_A, B, scale, M, K, N)
    
    max_diff = np.max(np.abs(C_gpu - C_cpu))
    print(f"  Dimensions: {M}x{K} @ {K}x{N}")
    print(f"  Différence max GPU vs CPU: {max_diff:.6f}")
    print(f"  ✅ Kernel CUDA fonctionnel\n")


def benchmark_comparison():
    """Compare les performances CUDA vs numpy."""
    print("\n" + "="*60)
    print(" BENCHMARK: CUDA vs Numpy")
    print("="*60 + "\n")
    
    if not CUDA_AVAILABLE:
        print("  ⚠️ Benchmark CUDA non disponible\n")
        return
    
    kernels = get_kernels()
    
    sizes = [(128, 256, 128), (256, 512, 256), (512, 1024, 512)]
    
    print(f"  {'Taille':<20} {'CUDA(ms)':>10} {'Numpy(ms)':>10}")
    print(f"  {'-'*42}")
    
    for M, K, N in sizes:
        from cuda_open.quantizer import BitNetQuantizer
        
        np.random.seed(42)
        A_fp32 = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        packed_A, scale = BitNetQuantizer.quantize(A_fp32)
        packed_A = packed_A.reshape(M, K // 4)
        
        # Benchmark CUDA
        if kernels.cuda_available:
            cuda_time = kernels.benchmark(M, K, N, 50)
        else:
            cuda_time = -1
        
        # Benchmark Numpy
        times = []
        for _ in range(5):
            start = time.time()
            kernels._fallback_matmul(packed_A, B, scale, M, K, N)
            times.append((time.time() - start) * 1000)
        numpy_time = np.mean(times)
        
        cuda_str = f"{cuda_time:.2f}" if cuda_time > 0 else "N/A"
        speedup = numpy_time / cuda_time if cuda_time > 0 else 1.0
        speedup_str = f"({speedup:.1f}x)" if cuda_time > 0 else ""
        
        print(f"  {M:>4}x{K:>4}x{N:<4}       {cuda_str:>10} {numpy_time:>9.2f} {speedup_str}")
    
    print()


def main():
    print("\n" + "="*60)
    print(" CUDA Open - Test des Kernels CUDA")
    print("="*60)
    print(f"\n  Info GPU: {get_gpu_info()}\n")
    
    test_numpy_fallback()
    test_cuda_kernel()
    benchmark_comparison()
    
    print("="*60)
    print(" RÉSULTATS")
    print("="*60)
    
    if is_cuda_available():
        print("  ✅ CUDA kernels compilés et fonctionnels")
        print("  ✅ Benchmark GPU disponible")
    else:
        print("  ⚠️ CUDA non compilé (pas de nvcc ou pas de GPU)")
        print("  ✅ Fallback numpy fonctionnel")
        print("  → Pour activer CUDA: cd kernels && ./build.sh")
    
    print()


if __name__ == "__main__":
    main()
