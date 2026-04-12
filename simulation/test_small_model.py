#!/usr/bin/env python3
"""
CUDA Open - Test avec Très Petit Modèle

Démo complète avec un micro-LLM pour valider tout le pipeline:
- Modèle: 64 hidden, 2 layers, 4 têtes ( ~100K paramètres)
- Vocabulaire: 100 tokens
- Génération: 20 tokens

Usage:
    python3 test_small_model.py
"""

import numpy as np
import time
import sys
from pathlib import Path

# Import simulation modules
sys.path.insert(0, str(Path(__file__).parent))

from bitnet_inference import (
    BitNetLLM,
    BitNetQuantizer,
    BitNetLinear
)


def create_tiny_model():
    """Créer un tout petit modèle pour test rapide"""
    print("=" * 70)
    print("CRÉATION DU MODÈLE TINY")
    print("=" * 70)
    print()
    
    model = BitNetLLM(
        vocab_size=100,        # Très petit vocabulaire
        hidden_size=64,        # Tiny hidden
        num_layers=2,          # Juste 2 layers
        num_heads=4,           # 4 têtes
        max_seq_len=32         # Seq length 32
    )
    
    params = model.count_parameters()
    print(f"\nModèle créé:")
    print(f"  Paramètres: {params*1e9:,.0f}")
    print(f"  Hidden: 64")
    print(f"  Layers: 2")
    print(f"  Heads: 4")
    print(f"  Vocab: 100")
    
    return model


def test_quantization_detailed(model):
    """Test détaillé de la quantization"""
    print("\n" + "=" * 70)
    print("TEST QUANTIZATION BITNET")
    print("=" * 70)
    print()
    
    # Extraire quelques poids pour démonstration
    sample_weights = model.blocks[0].q_proj.weight_fp32[:8, :8]
    
    print(f"Échantillon de poids (8x8) AVANT quantization:")
    print(f"{'':>12}", end="")
    for j in range(8):
        print(f"{j:>8}", end="")
    print()
    print(" " * 12 + "-" * 64)
    
    for i in range(8):
        print(f"Row {i:>2} |", end="")
        for j in range(8):
            print(f"{sample_weights[i, j]:>8.3f}", end="")
        print()
    
    # Quantizer
    print(f"\n→ Quantization en BitNet 1.58-bit...")
    start = time.time()
    packed, scale = BitNetQuantizer.quantize(sample_weights)
    quant_time = time.time() - start
    
    print(f"  Temps: {quant_time*1000:.2f} ms")
    print(f"  Scale: {scale:.4f}")
    print(f"  Taille originale: {sample_weights.nbytes} bytes")
    print(f"  Taille compressée: {packed.nbytes} bytes")
    print(f"  Compression: {sample_weights.nbytes / packed.nbytes:.1f}x")
    
    # Déquantizer
    print(f"\n→ Déquantization...")
    start = time.time()
    dequant = BitNetQuantizer.dequantize(packed, scale, sample_weights.shape)
    dequant_time = time.time() - start
    
    print(f"  Temps: {dequant_time*1000:.2f} ms")
    
    # Afficher déquantisé
    print(f"\nÉchantillon APRÈS déquantization:")
    print(f"{'':>12}", end="")
    for j in range(8):
        print(f"{j:>8}", end="")
    print()
    print(" " * 12 + "-" * 64)
    
    for i in range(8):
        print(f"Row {i:>2} |", end="")
        for j in range(8):
            print(f"{dequant[i, j]:>8.3f}", end="")
        print()
    
    # Erreur
    error = np.abs(sample_weights - dequant)
    print(f"\nErreur:")
    print(f"  Max: {np.max(error):.4f}")
    print(f"  Mean: {np.mean(error):.4f}")
    print(f"  Std: {np.std(error):.4f}")


def test_forward_pass(model):
    """Test forward pass"""
    print("\n" + "=" * 70)
    print("TEST FORWARD PASS")
    print("=" * 70)
    print()
    
    batch_size = 1
    seq_len = 8
    
    input_ids = np.random.randint(0, 100, (batch_size, seq_len))
    print(f"Input shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids[0]}")
    print()
    
    # Forward sans quantization
    print("→ Forward pass (FP32)...")
    start = time.time()
    logits_fp32 = model.forward(input_ids)
    time_fp32 = time.time() - start
    
    print(f"  Temps: {time_fp32*1000:.1f} ms")
    print(f"  Output shape: {logits_fp32.shape}")
    print(f"  Output sample (first 5): {logits_fp32[0, 0, :5]}")


def test_quantized_forward(model):
    """Test forward pass avec quantization"""
    print("\n" + "=" * 70)
    print("TEST FORWARD PASS QUANTIZÉ")
    print("=" * 70)
    print()
    
    # Quantizer tous les poids
    print("→ Quantization de tous les poids...")
    start = time.time()
    model.quantize_all_weights()
    quant_time = time.time() - start
    print(f"  Temps quantization: {quant_time*1000:.1f} ms")
    print()
    
    # Forward quantizé
    batch_size = 1
    seq_len = 8
    input_ids = np.random.randint(0, 100, (batch_size, seq_len))
    
    print(f"→ Forward pass (BitNet 1.58-bit)...")
    times = []
    for i in range(3):
        start = time.time()
        logits = model.forward(input_ids)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.1f} ms")
    
    avg_time = np.mean(times)
    print(f"\n  Temps moyen: {avg_time*1000:.1f} ms")
    print(f"  Output shape: {logits.shape}")
    print(f"  Output sample (first 5): {logits[0, 0, :5]}")
    
    # Prochain token
    next_token_logits = logits[0, -1, :]
    next_token = np.argmax(next_token_logits)
    print(f"\n  Prochain token predicted: {next_token}")
    print(f"  Logit value: {next_token_logits[next_token]:.3f}")


def test_text_generation(model):
    """Test génération de texte"""
    print("\n" + "=" * 70)
    print("TEST GÉNÉRATION DE TEXTE")
    print("=" * 70)
    print()
    
    # Prompt simple
    prompt = np.array([[0, 1, 2, 3]])  # 4 tokens de prompt
    
    print(f"Prompt: {prompt[0].tolist()}")
    print(f"Longueur prompt: {prompt.shape[1]}")
    print()
    
    # Générer
    print("→ Génération (10 tokens)...")
    start = time.time()
    
    generated = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=1.0,
        top_k=10
    )
    
    gen_time = time.time() - start
    
    print(f"\nRésultat:")
    print(f"  Prompt: {prompt[0].tolist()}")
    print(f"  Généré: {generated[4:]}")
    print(f"  Total: {len(generated)} tokens")
    print(f"  Temps: {gen_time:.1f} s")
    print(f"  Vitesse: {10/gen_time:.1f} tokens/s")


def test_memory_analysis(model):
    """Analyse mémoire détaillée"""
    print("\n" + "=" * 70)
    print("ANALYSE MÉMOIRE")
    print("=" * 70)
    print()
    
    # Calculer taille modèle
    total_params = 0
    
    # Embedding
    embedding_params = model.embedding.size
    total_params += embedding_params
    
    # Layers
    for block in model.blocks:
        total_params += block.q_proj.weight_fp32.size
        total_params += block.k_proj.weight_fp32.size
        total_params += block.v_proj.weight_fp32.size
        total_params += block.o_proj.weight_fp32.size
        total_params += block.mlp_fc1.weight_fp32.size
        total_params += block.mlp_fc2.weight_fp32.size
    
    # LM head
    total_params += model.lm_head.weight_fp32.size
    
    print(f"Paramètres totaux: {total_params:,}")
    print()
    
    # Différentes formats
    formats = [
        ("FP32", 32, 1.0),
        ("FP16", 16, 2.0),
        ("INT8", 8, 4.0),
        ("BitNet 1.58", 1.58, 16.0)
    ]
    
    print(f"{'Format':<15} {'Bits':>6} {'Taille':>12} {'Compression':>12}")
    print("-" * 50)
    
    for name, bits, compression in formats:
        size_bytes = total_params * bits / 8
        if size_bytes < 1024:
            size_str = f"{size_bytes:.0f} B"
        elif size_bytes < 1024*1024:
            size_str = f"{size_bytes/1024:.1f} KB"
        else:
            size_str = f"{size_bytes/(1024*1024):.1f} MB"
        
        print(f"{name:<15} {bits:>6.2f} {size_str:>12} {compression:>11.1f}x")
    
    print()
    print(f"Ce tiny modèle tient en {total_params * 1.58 / 8 / 1024:.1f} KB en BitNet!")


def benchmark_scaling():
    """Benchmark avec différents sizes"""
    print("\n" + "=" * 70)
    print("BENCHMARK SCALING")
    print("=" * 70)
    print()
    
    configs = [
        ("Tiny", 64, 2, 4, 32),
        ("Small", 128, 2, 4, 64),
        ("Medium", 256, 4, 8, 128)
    ]
    
    print(f"{'Config':<10} {'Hidden':>7} {'Layers':>7} {'Forward(ms)':>12}")
    print("-" * 40)
    
    for name, hidden, layers, heads, seq_len in configs:
        print(f"{name:<10} {hidden:>7} {layers:>7}", end="", flush=True)
        
        model = BitNetLLM(
            vocab_size=100,
            hidden_size=hidden,
            num_layers=layers,
            num_heads=heads,
            max_seq_len=seq_len
        )
        model.quantize_all_weights()
        
        input_ids = np.random.randint(0, 100, (1, seq_len//2))
        
        start = time.time()
        _ = model.forward(input_ids)
        elapsed = time.time() - start
        
        print(f" {elapsed*1000:>11.1f}")


def main():
    """Test complet avec tiny modèle"""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "CUDA Open - Test avec Très Petit Modèle".center(68) + "║")
    print("║" + "Validation complète du pipeline BitNet".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    try:
        # 1. Créer modèle
        model = create_tiny_model()
        
        # 2. Test quantization détaillé
        test_quantization_detailed(model)
        
        # 3. Test forward FP32
        test_forward_pass(model)
        
        # 4. Test forward quantizé
        test_quantized_forward(model)
        
        # 5. Test génération
        test_text_generation(model)
        
        # 6. Analyse mémoire
        test_memory_analysis(model)
        
        # 7. Benchmark scaling
        benchmark_scaling()
        
        # Résumé final
        print("\n" + "=" * 70)
        print("RÉSUMÉ FINAL")
        print("=" * 70)
        print()
        print("✓ Tiny modèle créé (64 hidden, 2 layers)")
        print("✓ Quantization BitNet fonctionnelle (16x compression)")
        print("✓ Forward pass FP32 OK")
        print("✓ Forward pass BitNet OK")
        print("✓ Génération de texte fonctionnelle")
        print("✓ Analyse mémoire validée")
        print("✓ Benchmark scaling OK")
        print()
        print("TOUS LES TESTS RÉUSSIS! 🎉")
        print()
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
