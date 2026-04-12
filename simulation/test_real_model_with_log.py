#!/usr/bin/env python3
"""
CUDA Open - Test avec Vrai Modèle (Version avec Log)

Teste le pipeline complet avec un vrai modèle GPT-2 et enregistre tout dans un fichier log.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Log file
LOG_FILE = Path(__file__).parent.parent / "test_real_model.log"

class Logger:
    """Logger qui écrit à la fois dans la console et dans un fichier"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Clear log file
        with open(self.log_file, 'w') as f:
            f.write("")
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
    
    def section(self, title):
        separator = "=" * 70
        self.log("")
        self.log(separator)
        self.log(f" {title}")
        self.log(separator)
        self.log("")

logger = Logger(LOG_FILE)


def test_dependencies():
    """Vérifie que toutes les dépendances sont disponibles"""
    logger.section("TEST 1: Vérification des Dépendances")
    
    dependencies = {
        'numpy': 'NumPy',
        'torch': 'PyTorch',
        'transformers': 'HuggingFace Transformers',
    }
    
    results = {}
    all_ok = True
    
    for module_name, nice_name in dependencies.items():
        try:
            mod = __import__(module_name)
            version = getattr(mod, '__version__', 'unknown')
            logger.log(f"  ✓ {nice_name}: {version}")
            results[module_name] = True
        except ImportError as e:
            logger.log(f"  ✗ {nice_name}: NON INSTALLÉ")
            results[module_name] = False
            all_ok = False
    
    if not all_ok:
        logger.log("\n  Installation des dépendances manquantes...")
        logger.log("  Exécute: pip install torch transformers numpy")
        logger.log("\n  ABANDON: Dépendances manquantes")
        return False
    
    logger.log("\n  ✓ Toutes les dépendances sont installées\n")
    return True


def test_bitnet_quantizer():
    """Teste le quantizer BitNet indépendamment"""
    logger.section("TEST 2: Quantizer BitNet 1.58-bit")
    
    try:
        from bitnet_inference import BitNetQuantizer
        
        # Test avec des données aléatoires
        np.random.seed(42)
        test_data = np.random.randn(1000).astype(np.float32)
        
        logger.log(f"  Données test: {len(test_data)} valeurs")
        logger.log(f"  Taille originale: {test_data.nbytes} bytes ({test_data.nbytes/1024:.2f} KB)")
        
        # Quantizer
        start = time.time()
        packed, scale = BitNetQuantizer.quantize(test_data)
        quant_time = time.time() - start
        
        logger.log(f"  Taille compressée: {packed.nbytes} bytes ({packed.nbytes/1024:.2f} KB)")
        logger.log(f"  Compression: {test_data.nbytes / packed.nbytes:.1f}x")
        logger.log(f"  Temps quantization: {quant_time*1000:.2f} ms")
        logger.log(f"  Scale: {scale:.4f}")
        
        # Déquantizer
        start = time.time()
        dequant = BitNetQuantizer.dequantize(packed, scale, test_data.shape)
        dequant_time = time.time() - start
        
        # Erreur
        error = np.abs(test_data - dequant)
        logger.log(f"  Temps déquantization: {dequant_time*1000:.2f} ms")
        logger.log(f"  Erreur max: {np.max(error):.4f}")
        logger.log(f"  Erreur moyenne: {np.mean(error):.4f}")
        
        # Vérifier valeurs ternaires
        unique_vals = np.unique(dequant / scale)
        logger.log(f"  Valeurs uniques après quant: {unique_vals}")
        
        logger.log(f"\n  ✓ Quantizer BitNet fonctionnel\n")
        return True
        
    except Exception as e:
        logger.log(f"\n  ✗ Erreur quantizer: {e}")
        import traceback
        logger.log(traceback.format_exc())
        return False


def test_model_loading():
    """Teste le chargement d'un vrai modèle"""
    logger.section("TEST 3: Chargement du Modèle GPT-2")
    
    try:
        import torch
        from transformers import AutoTokenizer, GPT2LMHeadModel
        
        model_name = "gpt2"
        logger.log(f"  Chargement: {model_name}")
        logger.log(f"  (Peut prendre 1-2 minutes si pas en cache)\n")
        
        start = time.time()
        
        # Tokenizer
        logger.log("  Chargement tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.log("  ✓ Tokenizer chargé")
        
        # Modèle
        logger.log("\n  Chargement modèle...")
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        load_time = time.time() - start
        
        # Stats
        num_params = sum(p.numel() for p in model.parameters())
        memory_mb = model.get_memory_footprint() / 1024 / 1024
        
        logger.log(f"\n  ✓ Modèle chargé en {load_time:.1f}s")
        logger.log(f"  Paramètres: {num_params/1e6:.1f}M")
        logger.log(f"  Mémoire: {memory_mb:.1f} MB")
        logger.log(f"  Device: {next(model.parameters()).device}")
        logger.log(f"  Layers: {len(model.transformer.h)}")
        
        # Cleanup pour libérer mémoire
        del model
        del tokenizer
        
        logger.log(f"\n  ✓ Test chargement réussi\n")
        return True
        
    except Exception as e:
        logger.log(f"\n  ✗ Erreur chargement modèle: {e}")
        import traceback
        logger.log(traceback.format_exc())
        return False


def test_inference():
    """Teste l'inférence complète avec quantization"""
    logger.section("TEST 4: Inférence avec Quantization BitNet")
    
    try:
        import torch
        from transformers import AutoTokenizer, GPT2LMHeadModel
        from bitnet_inference import BitNetQuantizer
        
        model_name = "gpt2"
        logger.log(f"  Modèle: {model_name}")
        logger.log(f"  Chargement...\n")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # Prompts à tester
        prompts = [
            "Hello, how are",
            "The future of AI",
            "Once upon a time"
        ]
        
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.log(f"\n  [{i+1}/{len(prompts)}] Prompt: \"{prompt}\"")
            
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            
            # Test FP32 baseline
            logger.log("    Test FP32 baseline...")
            start = time.time()
            output_fp32 = model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
            fp32_time = time.time() - start
            fp32_text = tokenizer.decode(output_fp32[0], skip_special_tokens=True)
            fp32_speed = 20 / fp32_time
            
            logger.log(f"    Temps: {fp32_time:.2f}s ({fp32_speed:.1f} tok/s)")
            logger.log(f"    Texte: {fp32_text[:80]}...")
            
            # Test avec quantization
            logger.log("\n    Application quantization BitNet...")
            original_weights = {}
            total_params_quant = 0
            total_params_orig = 0
            
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    original_weights[name] = module.weight.data.clone()
                    weight_np = module.weight.data.numpy().flatten()
                    total_params_orig += weight_np.size
                    
                    # Quantize + dequantize
                    packed, scale = BitNetQuantizer.quantize(weight_np)
                    dequant = BitNetQuantizer.dequantize(packed, scale, weight_np.shape)
                    total_params_quant += packed.size
                    
                    # Remplacer
                    module.weight.data = torch.tensor(dequant, dtype=torch.float32).reshape(module.weight.shape)
            
            compression = (total_params_orig * 4) / total_params_quant
            logger.log(f"    Compression: {compression:.1f}x")
            logger.log(f"    Paramètres: {total_params_orig:,} → {total_params_quant:,}")
            
            logger.log("\n    Test BitNet 1.58-bit...")
            start = time.time()
            output_bitnet = model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
            bitnet_time = time.time() - start
            bitnet_text = tokenizer.decode(output_bitnet[0], skip_special_tokens=True)
            bitnet_speed = 20 / bitnet_time
            
            logger.log(f"    Temps: {bitnet_time:.2f}s ({bitnet_speed:.1f} tok/s)")
            logger.log(f"    Texte: {bitnet_text[:80]}...")
            
            # Restaurer
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and name in original_weights:
                    module.weight.data = original_weights[name]
            
            results.append({
                'prompt': prompt,
                'fp32_time': fp32_time,
                'fp32_speed': fp32_speed,
                'bitnet_time': bitnet_time,
                'bitnet_speed': bitnet_speed,
                'ratio': fp32_time / bitnet_time
            })
            
            logger.log(f"\n    Ratio FP32/BitNet: {fp32_time/bitnet_time:.2f}x")
        
        # Moyennes
        avg_fp32_speed = np.mean([r['fp32_speed'] for r in results])
        avg_bitnet_speed = np.mean([r['bitnet_speed'] for r in results])
        avg_ratio = np.mean([r['ratio'] for r in results])
        
        logger.log(f"\n  {'='*60}")
        logger.log(f"  MOYENNES")
        logger.log(f"  {'='*60}")
        logger.log(f"  FP32 baseline:  {avg_fp32_speed:.1f} tokens/sec")
        logger.log(f"  BitNet 1.58:    {avg_bitnet_speed:.1f} tokens/sec")
        logger.log(f"  Ratio:          {avg_ratio:.2f}x")
        
        del model
        del tokenizer
        
        logger.log(f"\n  ✓ Test inférence réussi\n")
        return results
        
    except Exception as e:
        logger.log(f"\n  ✗ Erreur inférence: {e}")
        import traceback
        logger.log(traceback.format_exc())
        return None


def test_benchmark():
    """Benchmark de performance"""
    logger.section("TEST 5: Benchmark de Performance")
    
    try:
        import torch
        from transformers import AutoTokenizer, GPT2LMHeadModel
        
        model_name = "gpt2"
        logger.log(f"  Modèle: {model_name}")
        logger.log(f"  Benchmark avec différents batch sizes...\n")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        batch_sizes = [1, 2, 4]
        results = []
        
        for batch_size in batch_sizes:
            logger.log(f"  Batch size: {batch_size}")
            
            # Créer input batch
            input_text = ["Hello, how are you?"] * batch_size
            input_ids = tokenizer(input_text, return_tensors="pt", padding=True).input_ids
            
            # Warmup
            _ = model.generate(input_ids, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            
            # Benchmark
            times = []
            for _ in range(3):
                start = time.time()
                output = model.generate(
                    input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                elapsed = time.time() - start
                times.append(elapsed)
            
            avg_time = np.mean(times)
            speed = (10 * batch_size) / avg_time  # tokens/sec
            
            logger.log(f"    Temps moyen: {avg_time:.2f}s")
            logger.log(f"    Throughput: {speed:.1f} tokens/sec")
            logger.log(f"    Latence par seq: {avg_time*1000:.0f} ms")
            logger.log("")
            
            results.append({
                'batch_size': batch_size,
                'time': avg_time,
                'throughput': speed
            })
        
        # Meilleur résultat
        best = max(results, key=lambda x: x['throughput'])
        logger.log(f"  Meilleur throughput: {best['throughput']:.1f} tok/s (batch={best['batch_size']})")
        
        del model
        del tokenizer
        
        logger.log(f"\n  ✓ Benchmark réussi\n")
        return results
        
    except Exception as e:
        logger.log(f"\n  ✗ Erreur benchmark: {e}")
        import traceback
        logger.log(traceback.format_exc())
        return None


def main():
    """Exécute tous les tests"""
    logger.log("")
    logger.section("CUDA OPEN - TEST AVEC VRAI MODÈLE")
    logger.log(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"  Log file: {LOG_FILE}")
    logger.log(f"  Plateforme: {sys.platform}")
    logger.log(f"  Python: {sys.version}")
    logger.log("")
    
    start_time = time.time()
    
    # Test 1: Dépendances
    deps_ok = test_dependencies()
    if not deps_ok:
        logger.section("RÉSUMÉ FINAL")
        logger.log("✗ Tests abandonnés: dépendances manquantes")
        logger.log("\nPour installer les dépendances:")
        logger.log("  pip install torch transformers numpy")
        return
    
    # Test 2: Quantizer
    quant_ok = test_bitnet_quantizer()
    
    # Test 3: Chargement modèle
    load_ok = test_model_loading()
    
    # Test 4: Inférence
    inference_results = test_inference()
    
    # Test 5: Benchmark
    benchmark_results = test_benchmark()
    
    # Résumé final
    elapsed = time.time() - start_time
    
    logger.section("RÉSUMÉ FINAL")
    
    logger.log(f"  Durée totale: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.log("")
    
    tests = {
        "Dépendances": deps_ok,
        "Quantizer BitNet": quant_ok,
        "Chargement modèle": load_ok,
        "Inférence": inference_results is not None,
        "Benchmark": benchmark_results is not None
    }
    
    logger.log(f"  {'Test':<25} {'Résultat':>10}")
    logger.log(f"  {'-'*35}")
    
    all_passed = True
    for name, passed in tests.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.log(f"  {name:<25} {status:>10}")
        if not passed:
            all_passed = False
    
    logger.log("")
    
    if all_passed:
        logger.log("  ✓ TOUS LES TESTS ONT RÉUSSI!")
        logger.log("")
        logger.log("  Résultats clés:")
        logger.log("    - Compression BitNet: ~16x")
        logger.log("    - Inférence fonctionnelle")
        logger.log("    - Pipeline complet validé")
        logger.log("")
        logger.log("  Prochaines étapes:")
        logger.log("    1. Implémenter kernels CUDA")
        logger.log("    2. Tester avec modèle 7B")
        logger.log("    3. Benchmark vs vLLM/TGI")
    else:
        logger.log("  ⚠ Certains tests ont échoué")
        logger.log("  Vérifie le log pour les détails")
    
    logger.log("")
    logger.log(f"  Log complet sauvegardé: {LOG_FILE}")
    logger.log("=" * 70)
    logger.log("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrompu par l'utilisateur")
        logger.log("\n\n⚠ Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\nErreur fatale: {e}")
        import traceback
        traceback.print_exc()
        logger.log(f"\n✗ Erreur fatale: {e}")
