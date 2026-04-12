"""
CUDA Open - Benchmark TurboQuant vs SOTA

Compare :
1. Naive Quantization (INT4/INT2 simple)
2. Rotation (QuaRot style)
3. SmoothQuant (Lissage des outliers)
4. TurboQuant (Rotation + Group-Wise)

Usage:
    python3 simulation/benchmark_turbo_quant.py
"""

import numpy as np
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from cuda_open.turbo_quant import turbo_quantize_pipeline, smooth_quant, rotate_weights


def benchmark_turbo():
    print("\n" + "="*70)
    print(" CUDA Open - Benchmark TurboQuant vs SOTA")
    print("="*70 + "\n")

    # Setup : Matrice de poids réaliste (simule une couche de LLM avec outliers)
    torch.manual_seed(42)
    # On crée une distribution "Heavy Tailed" typique des LLMs
    weights = torch.randn(256, 1024)
    # Ajout d'outliers artificiels
    weights[0, 0] = 50.0 
    weights[1, 1] = -45.0
    
    print(f"📊 Matrice: {weights.shape} (256x1024)")
    print(f"   Max outlier: {torch.max(torch.abs(weights)):.2f}")
    print()

    results = []

    # 1. Naive INT4
    print("⏳ Test 1: Naive INT4")
    res_naive = turbo_quantize_pipeline(weights, bits=4, use_rotation=False, use_smooth=False)
    print(f"   MSE: {res_naive['mse']:.5f} | CosSim: {res_naive['mse']:.5f}") # MSE is key here
    results.append({'Method': 'Naive INT4', 'MSE': res_naive['mse'], 'CosSim': res_naive['cos_sim']})

    # 2. Naive INT2 (BitNet style)
    print("⏳ Test 2: Naive INT2 (1.58-bit approx)")
    res_naive2 = turbo_quantize_pipeline(weights, bits=2, use_rotation=False, use_smooth=False)
    print(f"   MSE: {res_naive2['mse']:.5f} | CosSim: {res_naive2['cos_sim']:.5f}")
    results.append({'Method': 'Naive INT2', 'MSE': res_naive2['mse'], 'CosSim': res_naive2['cos_sim']})

    # 3. Rotation Only (QuaRot/SpinQuant technique)
    print("⏳ Test 3: Rotation (QuaRot/SpinQuant style)")
    res_rot = turbo_quantize_pipeline(weights, bits=4, use_rotation=True, use_smooth=False)
    print(f"   MSE: {res_rot['mse']:.5f} | CosSim: {res_rot['cos_sim']:.5f}")
    results.append({'Method': 'INT4 + Rotation', 'MSE': res_rot['mse'], 'CosSim': res_rot['cos_sim']})

    # 4. SmoothQuant Approx (Lissage)
    print("⏳ Test 4: SmoothQuant Approx")
    # Simule un lissage en réduisant les outliers manuellement pour la démo
    w_smooth = weights.clone()
    w_smooth[torch.abs(w_smooth) > 10] *= 0.1 
    res_smooth = turbo_quantize_pipeline(w_smooth, bits=4, use_rotation=False, use_smooth=False)
    print(f"   MSE: {res_smooth['mse']:.5f} | CosSim: {res_smooth['cos_sim']:.5f}")
    results.append({'Method': 'SmoothQuant INT4', 'MSE': res_smooth['mse'], 'CosSim': res_smooth['cos_sim']})

    # 5. TURBOQUANT (Le Combo Ultime)
    print("⏳ Test 5: 🔥 TURBOQUANT (Rotation + Group-Wise)")
    res_turbo = turbo_quantize_pipeline(weights, bits=4, use_rotation=True, use_smooth=True)
    print(f"   MSE: {res_turbo['mse']:.5f} | CosSim: {res_turbo['cos_sim']:.5f}")
    results.append({'Method': '🔥 TurboQuant INT4', 'MSE': res_turbo['mse'], 'CosSim': res_turbo['cos_sim']})

    # Résumé
    print("\n" + "="*70)
    print(" RÉSULTATS COMPARATIFS")
    print("="*70)
    print(f"  {'Méthode':<25} {'MSE':>10} {'CosSim':>10}")
    print(f"  {'-'*45}")
    for r in results:
        print(f"  {r['Method']:<25} {r['MSE']:>9.5f} {r['CosSim']:>9.5f}")
    
    print(f"\n  🏆 CONCLUSION:")
    best = min(results, key=lambda x: x['MSE'])
    print(f"     {best['Method']} est le gagnant.")
    print(f"     Réduction d'erreur vs Naive INT4: {(1 - best['MSE']/results[0]['MSE'])*100:.1f}%")
    print()

if __name__ == "__main__":
    benchmark_turbo()
