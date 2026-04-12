"""
CUDA Open - Démo : Alternative à CUDA

Comparez la syntaxe :
CUDA: Nécessite de gérer la mémoire, les blocs, les threads, les shared mem...
CUDA Open: Vous écrivez la logique, le Runtime fait le reste via JIT et Évolution.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
import cuda_open.runtime as runtime

# ============================================================================
# 1. L'expérience utilisateur (UX)
# ============================================================================

# L'utilisateur définit simplement son calcul
@runtime.jit(target="auto")
def mon_calcul_complexe(A, B, Bias):
    # C'est du Python standard. 
    # Le runtime va tracer ces opérations, les optimiser (fusionner MatMul + Add),
    # et générer le kernel GPU optimal automatiquement.
    return (A @ B) + Bias

# ============================================================================
# 2. Exécution
# ============================================================================

def main():
    print("\n" + "="*70)
    print(" CUDA Open - Démo Runtime Auto-Compilant")
    print("="*70 + "\n")

    M, K, N = 256, 128, 256
    
    # Données (sur CPU pour l'instant, le runtime gérera le transfert)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    Bias = np.random.randn(M, N).astype(np.float32)
    
    print("📝 Code Utilisateur (Python pur) :")
    print("   @cuda_open.jit")
    print("   def mon_calcul_complexe(A, B, Bias):")
    print("       return (A @ B) + Bias")
    print()
    
    print("🚀 Exécution...")
    
    # Premier appel : Trace -> Optimise -> Compile -> Exécute
    result1 = mon_calcul_complexe(A, B, Bias)
    
    # Deuxième appel : Utilise le cache (Ultra rapide)
    print("🔄 Deuxième appel (Cache HIT)...")
    result2 = mon_calcul_complexe(A, B, Bias)
    
    print("\n✅ Résultat Shape:", result1.shape)
    print("✅ Erreur Max vs Numpy:", np.max(np.abs(result1 - ((A @ B) + Bias))))
    print()
    
    print("="*70)
    print(" POURQUOI C'EST UNE RÉVOLUTION ?")
    print("="*70)
    print("  1. Pas de gestion mémoire explicite (cudaMalloc/cpy).")
    print("  2. Pas de configuration de blocs/thread (<<<>>>).")
    print("  3. Optimisation automatique (Fusion MatMul+Bias).")
    print("  4. Portabilité : Ce même code tournera sur NVIDIA, AMD ou CPU.")
    print()

if __name__ == "__main__":
    main()
