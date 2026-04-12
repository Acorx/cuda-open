"""
CUDA Open - Optimisation du Seuil BitNet

Le benchmark montre que BitNet a une cosine similarity de 0.33 (faible).
Ce script optimise le seuil pour maximiser la qualité tout en gardant 16x compression.

Résultat attendu: Cosine similarity > 0.85 avec compression 16x maintenue.

Usage:
    python3 benchmark/optimize_bitnet_threshold.py
"""

import numpy as np
import time
from pathlib import Path


def quantize_with_threshold(data: np.ndarray, threshold_ratio: float):
    """Quantize avec un ratio de seuil custom."""
    scale = np.max(np.abs(data))
    if scale < 1e-8:
        scale = 1.0
    
    threshold = threshold_ratio * scale
    ternary = np.zeros_like(data, dtype=np.int8)
    ternary[data > threshold] = 1
    ternary[data < -threshold] = -1
    
    # Pack 4 valeurs par byte
    flat = ternary.flatten()
    vals = (flat + 1).astype(np.uint8)
    remainder = len(vals) % 4
    if remainder > 0:
        vals = np.pad(vals, (0, 4 - remainder), mode='constant')
    vals = vals.reshape(-1, 4)
    packed = (vals[:, 0] | (vals[:, 1] << 2) | (vals[:, 2] << 4) | (vals[:, 3] << 6)).astype(np.uint8)
    
    return packed, scale, ternary


def dequantize(packed, scale, original_size):
    """Déquantize ternary."""
    lut = np.array([0.0, -1.0, 1.0, 0.0], dtype=np.float32)
    unpacked = np.zeros(original_size, dtype=np.float32)
    for i in range(original_size):
        byte_idx = i // 4
        offset = (i % 4) * 2
        encoded = (packed[byte_idx] >> offset) & 0x03
        unpacked[i] = lut[encoded] * scale
    return unpacked


def main():
    print("\n" + "="*70)
    print(" CUDA Open - Optimisation du Seuil BitNet")
    print("="*70 + "\n")

    np.random.seed(42)
    data = np.random.randn(100000).astype(np.float32)
    
    print(f"Test sur {len(data):,} valeurs (distribution normale)")
    print(f"Échelle des données: min={data.min():.3f}, max={data.max():.3f}\n")

    # Tester différents ratios de seuil
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
    
    results = []
    
    print(f"{'Seuil':>8} {'% Zéros':>8} {'% -1':>8} {'% +1':>8} {'CosSim':>8} {'MSE':>10} {'SNR(dB)':>8}")
    print("-" * 70)
    
    for ratio in thresholds:
        packed, scale, ternary = quantize_with_threshold(data, ratio)
        reconstructed = dequantize(packed, scale, len(data))
        
        # Métriques
        cos_sim = np.dot(data, reconstructed) / (
            np.linalg.norm(data) * np.linalg.norm(reconstructed) + 1e-10
        )
        mse = np.mean((data - reconstructed) ** 2)
        snr_db = 10 * np.log10(np.mean(data**2) / (mse + 1e-10))
        
        # Statistiques ternaires
        pct_zeros = np.mean(ternary == 0) * 100
        pct_neg = np.mean(ternary == -1) * 100
        pct_pos = np.mean(ternary == 1) * 100
        
        # Compression
        compression = data.nbytes / packed.nbytes
        
        results.append({
            'ratio': ratio,
            'cos_sim': cos_sim,
            'mse': mse,
            'snr_db': snr_db,
            'pct_zeros': pct_zeros,
            'compression': compression
        })
        
        print(f"{ratio:>8.2f} {pct_zeros:>7.1f}% {pct_neg:>7.1f}% {pct_pos:>7.1f}% {cos_sim:>8.4f} {mse:>9.4f} {snr_db:>7.1f}")

    # Trouver le seuil optimal
    # On cherche le meilleur compromis: cos_sim > 0.80 ET compression >= 15x
    print("\n" + "="*70)
    print(" RÉSULTATS D'OPTIMISATION")
    print("="*70)
    
    # Trouver le meilleur seuil pour cos_sim > 0.85
    best_for_quality = max(results, key=lambda x: x['cos_sim'] if x['cos_sim'] < 0.99 else 0)
    best_ratio = best_for_quality['ratio']
    
    # Trouver le seuil pour compression maximale avec cos_sim > 0.80
    valid_results = [r for r in results if r['cos_sim'] >= 0.80]
    if valid_results:
        best_efficient = max(valid_results, key=lambda x: x['compression'])
        print(f"\n  🎯 SEUIL OPTIMAL POUR QUALITÉ > 0.80:")
        print(f"     Ratio: {best_efficient['ratio']:.2f}")
        print(f"     Cosine Similarity: {best_efficient['cos_sim']:.4f}")
        print(f"     Compression: {best_efficient['compression']:.1f}x")
        print(f"     SNR: {best_efficient['snr_db']:.1f} dB")
        print(f"     % de zéros: {best_efficient['pct_zeros']:.1f}%")
    
    # Recommandation finale
    print(f"\n  📊 RECOMMANDATION:")
    print(f"     Seuil actuel (0.20): CosSim = {next(r['cos_sim'] for r in results if r['ratio'] == 0.20):.4f}")
    print(f"     Seuil optimisé ({best_ratio:.2f}): CosSim = {best_for_quality['cos_sim']:.4f}")
    print(f"     Amélioration: {(best_for_quality['cos_sim'] / next(r['cos_sim'] for r in results if r['ratio'] == 0.20) - 1) * 100:.1f}%")
    
    # Sauvegarder le résultat
    output = Path(__file__).parent / "bitnet_optimization_results.json"
    import json
    with open(output, 'w') as f:
        json.dump({
            'optimal_threshold': best_ratio,
            'cosine_similarity': best_for_quality['cos_sim'],
            'compression': best_for_quality['compression'],
            'snr_db': best_for_quality['snr_db'],
            'all_results': [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()} for r in results]
        }, f, indent=2)
    
    print(f"\n  💾 Résultats sauvés: {output}")
    print()


if __name__ == "__main__":
    main()
