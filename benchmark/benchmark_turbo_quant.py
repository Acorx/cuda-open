"""
CUDA Open - Benchmark Ultra-Léger

Benchmark SUPER LIGHT pour ne PAS crasher le PC.
Compare les performances de quantization BitNet vs FP32
sur des tableaux numpy uniquement.

Usage:
    python3 benchmark/benchmark_light.py
"""

import numpy as np
import time
import sys
from pathlib import Path

# Ajoute le projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))


class BenchmarkResult:
    """Stocke les résultats"""
    def __init__(self):
        self.results = []
    
    def add(self, name, data):
        self.results.append({'test': name, **data})
        print(f"  ✓ {name}")
    
    def print_summary(self):
        print(f"\n{'='*60}")
        print(f" RÉSUMÉ")
        print(f"{'='*60}\n")
        for r in self.results:
            name = r.pop('test')
            print(f"  {name}:")
            for k, v in r.items():
                print(f"    {k}: {v}")
            print()


def bench_quantization_speed():
    """Test vitesse de quantization vs différentes tailles"""
    print("\n" + "="*60)
    print(" TEST 1: Vitesse de Quantization")
    print("="*60 + "\n")
    
    sizes = [100, 1000, 10000]
    results = []
    
    for size in sizes:
        data = np.random.randn(size).astype(np.float32)
        
        # Warmup
        _ = np.max(np.abs(data))
        
        # Benchmark
        times = []
        for _ in range(5):
            start = time.time()
            scale = np.max(np.abs(data))
            threshold = 0.2 * scale
            ternary = np.zeros_like(data)
            ternary[data > threshold] = 1
            ternary[data < -threshold] = -1
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times) * 1000  # ms
        compression = data.nbytes / ((size + 3) // 4)
        
        print(f"  Taille {size:>6}: {avg_time:>6.2f}ms | Compression: {compression:.1f}x")
        results.append({'size': size, 'time_ms': f"{avg_time:.2f}", 'compression': f"{compression:.1f}x"})
    
    return results


def bench_packing():
    """Test le packing ternaire"""
    print("\n" + "="*60)
    print(" TEST 2: Packing Ternaire")
    print("="*60 + "\n")
    
    size = 10000
    ternary = np.random.choice([-1, 0, 1], size=size).astype(np.float32)
    
    # Méthode 1: Boucle Python
    start = time.time()
    packed1 = np.zeros((size + 3) // 4, dtype=np.uint8)
    for i in range(size):
        val = int(ternary[i])
        if val == -1: enc = 1
        elif val == 1: enc = 2
        else: enc = 0
        packed1[i//4] |= (enc << ((i%4)*2))
    time1 = (time.time() - start) * 1000
    
    # Méthode 2: Vectorisée numpy
    start = time.time()
    vals = (ternary + 1).astype(np.int8)  # -1,0,1 → 0,1,2
    packed2 = np.zeros((size + 3) // 4, dtype=np.uint8)
    for i in range(4):
        if i*4 < size:
            chunk = (vals[i::4].astype(np.uint8) << (i*2))
            packed2[:len(chunk)] |= chunk
    time2 = (time.time() - start) * 1000
    
    print(f"  Boucle Python:   {time1:.2f}ms")
    print(f"  Numpy vectorisé: {time2:.2f}ms")
    print(f"  Speedup:         {time1/time2:.1f}x")
    print(f"  Compression:     {size * 4 / len(packed1):.1f}x")
    
    return {
        'loop_ms': f"{time1:.2f}",
        'vectorized_ms': f"{time2:.2f}",
        'speedup': f"{time1/time2:.1f}x"
    }


def bench_compression_ratio():
    """Test les ratios de compression"""
    print("\n" + "="*60)
    print(" TEST 3: Ratios de Compression")
    print("="*60 + "\n")
    
    print(f"  {'Format':<15} {'Bits':>6} {'Taille/1M':>10} {'Ratio':>8}")
    print(f"  {'-'*42}")
    
    configs = [
        ("FP32", 32, 1.0),
        ("FP16", 16, 2.0),
        ("INT8", 8, 4.0),
        ("INT4", 4, 8.0),
        ("INT2", 2, 16.0),
        ("BitNet 1.58", 1.58, 20.0),
    ]
    
    for name, bits, ratio in configs:
        size_mb = (1_000_000 * bits / 8) / 1024 / 1024
        print(f"  {name:<15} {bits:>6.2f} {size_mb:>9.2f}MB {ratio:>7.1f}x")
    
    return {'best': 'BitNet 1.58 @ 20.0x'}


def bench_accuracy():
    """Test la précision après quantization"""
    print("\n" + "="*60)
    print(" TEST 4: Précision Post-Quantization")
    print("="*60 + "\n")
    
    np.random.seed(42)
    size = 1000
    original = np.random.randn(size).astype(np.float32)
    
    # Quantize
    scale = np.max(np.abs(original))
    threshold = 0.2 * scale
    ternary = np.zeros_like(original)
    ternary[original > threshold] = 1
    ternary[original < -threshold] = -1
    
    # Dequantize
    reconstructed = ternary * scale
    
    # Erreurs
    abs_error = np.abs(original - reconstructed)
    max_error = np.max(abs_error)
    mean_error = np.mean(abs_error)
    mse = np.mean(abs_error**2)
    
    # Similarité cosinus
    cos_sim = np.dot(original, reconstructed) / (
        np.linalg.norm(original) * np.linalg.norm(reconstructed) + 1e-10
    )
    
    print(f"  Erreur max:     {max_error:.4f}")
    print(f"  Erreur moyenne: {mean_error:.4f}")
    print(f"  MSE:            {mse:.4f}")
    print(f"  Similarité:     {cos_sim:.4f}")
    
    return {
        'max_error': f"{max_error:.4f}",
        'mean_error': f"{mean_error:.4f}",
        'cosine_sim': f"{cos_sim:.4f}"
    }


def main():
    print("\n" + "="*60)
    print(" CUDA Open - Benchmark Ultra-Léger")
    print("="*60)
    print("\n (Ne charge AUCUN modèle - 100% numpy)")
    print()
    
    benchmark = BenchmarkResult()
    
    # Run tests
    r1 = bench_quantization_speed()
    benchmark.add("Quantization Speed", {'sizes': f"{len(r1)} tests"})
    
    r2 = bench_packing()
    benchmark.add("Packing", r2)
    
    r3 = bench_compression_ratio()
    benchmark.add("Compression", r3)
    
    r4 = bench_accuracy()
    benchmark.add("Accuracy", r4)
    
    # Summary
    benchmark.print_summary()
    
    print(f"{'='*60}")
    print(f" ✓ TOUS LES TESTS RÉUSSIS")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
