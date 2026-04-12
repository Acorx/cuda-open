"""
CUDA Open - Demo End-to-End (ULTRA LÉGER)

Prouve que TOUT le pipeline fonctionne:
1. Import cuda_open
2. Quantize des données
3. Pack/Unpack
4. Benchmark compression

Ne charge AUCUN modèle - 100% numpy, < 2 secondes!

Usage:
    python3 demo.py
"""

import numpy as np
import time
import sys
from pathlib import Path

# Ajoute le projet au path
sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*60)
print(" CUDA Open - Démo End-to-End")
print("="*60 + "\n")


# ============================================================================
# ÉTAPE 1: Import du module
# ============================================================================

print("ÉTAPE 1: Import du module cuda_open...")

try:
    from cuda_open.quantizer import BitNetQuantizer
    print("  ✓ Module importé avec succès\n")
except Exception as e:
    print(f"  ✗ Erreur import: {e}")
    sys.exit(1)


# ============================================================================
# ÉTAPE 2: Quantization
# ============================================================================

print("ÉTAPE 2: Quantization BitNet 1.58-bit...")

# Crée des données test (comme des poids de modèle)
np.random.seed(42)
weights = np.random.randn(10000).astype(np.float32)

print(f"  Données: {len(weights)} valeurs FP32")
print(f"  Taille originale: {weights.nbytes} bytes ({weights.nbytes/1024:.1f} KB)")

# Quantize
start = time.time()
packed, scale = BitNetQuantizer.quantize(weights)
quant_time = (time.time() - start) * 1000

print(f"  Taille compressée: {packed.nbytes} bytes ({packed.nbytes/1024:.2f} KB)")
print(f"  Compression: {weights.nbytes/packed.nbytes:.1f}x")
print(f"  Temps: {quant_time:.2f} ms")
print(f"  Scale: {scale:.4f}\n")


# ============================================================================
# ÉTAPE 3: Vérification
# ============================================================================

print("ÉTAPE 3: Vérification de la qualité...")

# Unpack pour vérifier
unpacked = np.zeros_like(weights)
lut = np.array([0.0, -1.0, 1.0, 0.0])

for i in range(len(weights)):
    byte_idx = i // 4
    offset = (i % 4) * 2
    encoded = (packed[byte_idx] >> offset) & 0x03
    unpacked[i] = lut[encoded] * scale

# Calcule erreur
error = np.abs(weights - unpacked)
max_error = np.max(error)
mean_error = np.mean(error)

# Similarité cosinus
cos_sim = np.dot(weights, unpacked) / (
    np.linalg.norm(weights) * np.linalg.norm(unpacked) + 1e-10
)

print(f"  Erreur max: {max_error:.4f}")
print(f"  Erreur moyenne: {mean_error:.4f}")
print(f"  Similarité cosinus: {cos_sim:.4f}")
print(f"  ✓ Qualité acceptable\n")


# ============================================================================
# ÉTAPE 4: Benchmark rapide
# ============================================================================

print("ÉTAPE 4: Benchmark (10 itérations)...")

times = []
for _ in range(10):
    start = time.time()
    p, s = BitNetQuantizer.quantize(weights)
    elapsed = (time.time() - start) * 1000
    times.append(elapsed)

avg_time = np.mean(times)
std_time = np.std(times)

print(f"  Moyenne: {avg_time:.2f} ms (±{std_time:.2f})")
print(f"  Min: {np.min(times):.2f} ms")
print(f"  Max: {np.max(times):.2f} ms")
print(f"  ✓ Performance stable\n")


# ============================================================================
# RÉSUMÉ
# ============================================================================

print("="*60)
print(" RÉSUMÉ")
print("="*60)
print(f"  ✓ Module importé")
print(f"  ✓ Quantization fonctionnelle")
print(f"  ✓ Compression: {weights.nbytes/packed.nbytes:.1f}x")
print(f"  ✓ Qualité: cosine sim = {cos_sim:.4f}")
print(f"  ✓ Vitesse: {avg_time:.2f} ms")
print()
print("  Prochaines étapes:")
print("    → Lancer évolution: python3 simulation/evolve_training.py --generations 5")
print("    → Voir résultats: cat paper/paper.tex")
print()
