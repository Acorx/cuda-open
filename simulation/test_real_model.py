#!/usr/bin/env python3
"""
CUDA Open - Test avec Vrai Modèle sur PC

Télécharge un petit modèle (GPT-2 124M), le quantize en BitNet 1.58-bit,
et fait de l'inférence réelle pour valider tout le pipeline.

Usage:
    python3 test_real_model.py [--model gpt2] [--prompt "Hello world"]
"""

import numpy as np
import time
import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel

# Import notre BitNet quantizer
sys.path.insert(0, str(Path(__file__).parent))
from bitnet_inference import BitNetQuantizer


def print_header(title):
    width = 70
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width + "\n")


def load_model_and_tokenizer(model_name="gpt2"):
    """Charge un vrai modèle depuis HuggingFace"""
    print(f"Chargement du modèle: {model_name}...")
    print("(Cela peut prendre quelques minutes si pas en cache)\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    print(f"✓ Modèle chargé: {model_name}")
    print(f"  Paramètres: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Taille mémoire: {model.get_memory_footprint()/1024/1024:.1f} MB")
    
    return model, tokenizer


def quantize_model_to_bitnet(model):
    """Quantize tous les poids linéaires en BitNet 1.58-bit"""
    print("\n" + "-" * 70)
    print("Quantization BitNet 1.58-bit")
    print("-" * 70 + "\n")
    
    total_params = 0
    quantized_params = 0
    scales = {}
    packed_weights = {}
    
    # Parcourir tous les modules linéaires
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Extraire les poids
            weight = module.weight.data.numpy()
            total_params += weight.size
            
            # Quantizer
            weight_flat = weight.flatten()
            packed, scale = BitNetQuantizer.quantize(weight_flat)
            scales[name] = scale
            packed_weights[name] = packed
            
            quantized_params += len(packed)
    
    # Stats
    original_size = total_params * 4  # FP32 = 4 bytes
    compressed_size = quantized_params  # 1 byte = 4 valeurs ternaires
    compression = original_size / compressed_size
    
    print(f"Statistiques de quantization:")
    print(f"  Poids originaux: {total_params:,} paramètres")
    print(f"  Taille originale: {original_size/1024/1024:.1f} MB")
    print(f"  Taille compressée: {compressed_size/1024/1024:.1f} MB")
    print(f"  Compression: {compression:.1f}x")
    print(f"  ✓ Quantization terminée\n")
    
    return scales, packed_weights, total_params


def generate_with_quantized_model(model, tokenizer, prompt, max_tokens=50, temperature=1.0):
    """Génère du texte avec le modèle quantized (simulation)"""
    print("-" * 70)
    print(f"Génération de texte (modèle quantized)")
    print("-" * 70 + "\n")
    
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}")
    print(f"Temperature: {temperature}\n")
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Prompt tokens: {input_ids.shape[1]}")
    
    # Mesurer temps de génération
    start_time = time.time()
    
    # Générer avec le modèle original (pour comparaison)
    print("Génération avec modèle FP32 (baseline)...")
    start_baseline = time.time()
    
    output_baseline = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    baseline_time = time.time() - start_baseline
    baseline_text = tokenizer.decode(output_baseline[0], skip_special_tokens=True)
    
    print(f"  Temps: {baseline_time:.2f}s")
    print(f"  Tokens générés: {max_tokens}")
    print(f"  Vitesse: {max_tokens/baseline_time:.1f} tokens/sec")
    print(f"\  Texte:\n  {baseline_text}\n")
    
    # Maintenant simuler la quantization en déquantizant et recalculant
    print("Simulation avec quantization BitNet...\n")
    
    # Pour chaque layer linéaire, déquantizer et remplacer temporairement
    original_weights = {}
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Sauvegarder poids original
            original_weights[name] = module.weight.data.clone()
            
            # Quantizer puis déquantizer
            weight_np = module.weight.data.numpy().flatten()
            packed, scale = BitNetQuantizer.quantize(weight_np)
            dequant = BitNetQuantizer.dequantize(packed, scale, weight_np.shape)
            
            # Remplacer
            module.weight.data = torch.tensor(dequant, dtype=torch.float32).reshape(module.weight.shape)
    
    # Générer avec modèle quantized
    start_quantized = time.time()
    
    output_quantized = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    quantized_time = time.time() - start_quantized
    quantized_text = tokenizer.decode(output_quantized[0], skip_special_tokens=True)
    
    print(f"  Temps: {quantized_time:.2f}s")
    print(f"  Tokens générés: {max_tokens}")
    print(f"  Vitesse: {max_tokens/quantized_time:.1f} tokens/sec")
    print(f"\n  Texte:\n  {quantized_text}\n")
    
    # Restaurer poids originaux
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in original_weights:
            module.weight.data = original_weights[name]
    
    # Comparaison
    print("-" * 70)
    print("COMPARAISON")
    print("-" * 70)
    print(f"  FP32 baseline:  {baseline_time:.2f}s ({max_tokens/baseline_time:.1f} tok/s)")
    print(f"  BitNet 1.58:   {quantized_time:.2f}s ({max_tokens/quantized_time:.1f} tok/s)")
    print(f"  Ratio:          {baseline_time/quantized_time:.2f}x")
    print()
    
    return {
        'baseline_time': baseline_time,
        'quantized_time': quantized_time,
        'baseline_text': baseline_text,
        'quantized_text': quantized_text
    }


def benchmark_multiple_prompts(model, tokenizer):
    """Benchmark avec plusieurs prompts"""
    print("-" * 70)
    print("BENCHMARK MULTI-PROMPTS")
    print("-" * 70 + "\n")
    
    prompts = [
        "Once upon a time",
        "The future of AI is",
        "In a galaxy far far away",
        "To be or not to be",
        "The meaning of life is"
    ]
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Prompt: \"{prompt}\"")
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        # Baseline
        start = time.time()
        output = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
        elapsed = time.time() - start
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        speed = 20 / elapsed
        results.append(speed)
        
        print(f"  Temps: {elapsed:.2f}s, Vitesse: {speed:.1f} tok/s")
        print(f"  Résultat: {text[:80]}...\n")
    
    avg_speed = np.mean(results)
    print(f"Vitesse moyenne: {avg_speed:.1f} tokens/sec")
    print(f"Min: {min(results):.1f}, Max: {max(results):.1f}")
    print()
    
    return results


def test_model_sizes():
    """Compare différents modèles"""
    print("=" * 70)
    print("COMPARAISON DE TAILLES DE MODÈLES")
    print("=" * 70 + "\n")
    
    models = [
        ("gpt2", "GPT-2 Small (124M)"),
        ("gpt2-medium", "GPT-2 Medium (355M)"),
        # ("gpt2-large", "GPT-2 Large (774M)"),  # Trop long
    ]
    
    results = []
    
    for model_name, description in models:
        print(f"\nTest: {description}")
        print("-" * 70)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            
            num_params = sum(p.numel() for p in model.parameters())
            memory_mb = model.get_memory_footprint() / 1024 / 1024
            
            print(f"  Paramètres: {num_params/1e6:.1f}M")
            print(f"  Mémoire: {memory_mb:.1f} MB")
            
            # Test rapide
            input_ids = tokenizer.encode("Hello, how are", return_tensors="pt")
            
            start = time.time()
            output = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            elapsed = time.time() - start
            
            speed = 10 / elapsed
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            print(f"  Vitesse: {speed:.1f} tok/s")
            print(f"  Résultat: {text}\n")
            
            results.append({
                'name': description,
                'params': num_params,
                'memory': memory_mb,
                'speed': speed
            })
            
            # Cleanup
            del model
            del tokenizer
            
        except Exception as e:
            print(f"  Erreur: {e}\n")
            continue
    
    # Résumé
    print("=" * 70)
    print("RÉSUMÉ")
    print("=" * 70 + "\n")
    
    print(f"{'Modèle':<25} {'Params':>8} {'Mémoire':>10} {'Vitesse':>10}")
    print("-" * 55)
    
    for r in results:
        print(f"{r['name']:<25} {r['params']/1e6:>7.1f}M {r['memory']:>8.0f}MB {r['speed']:>8.1f} tok/s")
    
    print()


def main():
    """Test complet avec vrai modèle"""
    print_header("CUDA Open - Test avec Vrai Modèle")
    print("Ce test va:")
    print("  1. Charger un vrai modèle (GPT-2)")
    print("  2. Le quantizer en BitNet 1.58-bit")
    print("  3. Faire de l'inférence réelle")
    print("  4. Benchmarker les performances")
    print()
    
    try:
        # 1. Charger modèle
        model, tokenizer = load_model_and_tokenizer("gpt2")
        
        # 2. Quantizer
        scales, packed, total = quantize_model_to_bitnet(model)
        
        # 3. Générer avec prompt custom
        prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
        results = generate_with_quantized_model(
            model, tokenizer,
            prompt=prompt,
            max_tokens=30,
            temperature=0.8
        )
        
        # 4. Benchmark multi-prompts
        benchmark_results = benchmark_multiple_prompts(model, tokenizer)
        
        # 5. Comparaison tailles
        test_model_sizes()
        
        # Résumé final
        print_header("RÉSUMÉ FINAL")
        
        print("✓ Modèle GPT-2 chargé avec succès")
        print("✓ Quantization BitNet 1.58-bit appliquée")
        print(f"✓ Compression: ~16x (théorique)")
        print(f"✓ Inférence fonctionnelle")
        print(f"✓ Vitesse moyenne: {np.mean(benchmark_results):.1f} tok/s")
        print()
        print("Prochaines étapes:")
        print("  1. Tester avec plus gros modèle")
        print("  2. Implémenter kernels CUDA")
        print("  3. Comparer vs vLLM/TGI")
        print()
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTest interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n\nErreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
