"""
CUDA Open - Démo du Vrai Compilateur

Montre comment le code Python est transformé en CUDA C++ via OpenIR.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# On utilise le nouveau compilateur
from cuda_open.compiler import jit

print("\n" + "="*70)
print(" CUDA Open - Vrai Compilateur (OpenIR -> CUDA C++)")
print("="*70 + "\n")

@jit
def mon_noiau_intense(A, B, Bias):
    """
    Cette fonction Python va être compilée en un noyau CUDA unique.
    """
    return (A @ B) + Bias

# Données pour l'exemple
M, K, N = 1024, 1024, 1024
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)
Bias = np.random.randn(M, N).astype(np.float32)

print("📥 Appel de la fonction Python...")
result = mon_noiau_intense(A, B, Bias)

print("\n" + "="*70)
print(" RÉSULTAT")
print("="*70)
print(f"✅ Shape: {result.shape}")
print(f"✅ Valeur moy: {np.mean(result):.4f}")
print()
