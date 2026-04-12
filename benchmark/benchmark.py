#!/usr/bin/env python3
"""
CUDA Open - Benchmark Complet

Compare CUDA Open vs PyTorch vs vLLM vs TGI sur:
- Throughput (tokens/sec)
- Latence (TTFT, TPOT)
- Mémoire utilisée
- Qualité de génération

Usage:
    python3 benchmark.py --model gpt2 --batch-sizes 1 2 4 8 --seq-lengths 64 128 256
"""

import torch
import time
import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM


class BenchmarkResult:
    def __init__(self):
        self.results = []
    
    def add(self, name, config, metrics):
        self.results.append({
            'name': name,
            'config': config,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
    
    def to_dict(self):
        return self.results
    
    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"\n✓ Résultats sauvegardés: {filepath}")


def benchmark_throughput(model, tokenizer, batch_sizes, seq_lengths, num_iterations=10):
    """Benchmark de throughput"""
    print("\n" + "="*70)
    print(" BENCHMARK: THROUGHPUT")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            key = f"batch={batch_size},seq={seq_len}"
            print(f"  Test: {key}")
            
            # Créer input
            input_ids = torch.randint(0, model.config.vocab_size, 
                                     (batch_size, seq_len)).to(device)
            
            # Warmup
            with torch.no_grad():
                _ = model.generate(input_ids, max_new_tokens=10, do_sample=False)
            
            # Benchmark
            times = []
            tokens_generated = 0
            
            for _ in range(num_iterations):
                start = time.time()
                with torch.no_grad():
                    output = model.generate(
                        input_ids, 
                        max_new_tokens=20, 
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                elapsed = time.time() - start
                times.append(elapsed)
                tokens_generated += 20 * batch_size
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = tokens_generated / (num_iterations * avg_time)
            
            results[key] = {
                'throughput_tok_s': throughput,
                'avg_time_s': avg_time,
                'std_time_s': std_time,
                'batch_size': batch_size,
                'seq_len': seq_len
            }
            
            print(f"    Throughput: {throughput:.1f} tok/s (±{std_time*1000:.0f}ms)")
    
    return results


def benchmark_latency(model, tokenizer, num_iterations=50):
    """Benchmark de latence (TTFT, TPOT)"""
    print("\n" + "="*70)
    print(" BENCHMARK: LATENCE")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    prompt = "Hello, how are you?"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # TTFT (Time To First Token)
    print("  Mesure TTFT (Time To First Token)...")
    ttft_times = []
    
    for _ in range(num_iterations):
        start = time.time()
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=1, do_sample=False)
        elapsed = time.time() - start
        ttft_times.append(elapsed)
    
    avg_ttft = np.mean(ttft_times)
    p50_ttft = np.percentile(ttft_times, 50)
    p99_ttft = np.percentile(ttft_times, 99)
    
    print(f"    Moyenne: {avg_ttft*1000:.1f}ms")
    print(f"    P50:     {p50_ttft*1000:.1f}ms")
    print(f"    P99:     {p99_ttft*1000:.1f}ms")
    
    # TPOT (Time Per Output Token)
    print("\n  Mesure TPOT (Time Per Output Token)...")
    tpot_times = []
    
    for _ in range(num_iterations):
        start = time.time()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=50, do_sample=False)
        elapsed = time.time() - start
        tpot = elapsed / 50  # Time per token
        tpot_times.append(tpot)
    
    avg_tpot = np.mean(tpot_times)
    p50_tpot = np.percentile(tpot_times, 50)
    p99_tpot = np.percentile(tpot_times, 99)
    
    print(f"    Moyenne: {avg_tpot*1000:.1f}ms")
    print(f"    P50:     {p50_tpot*1000:.1f}ms")
    print(f"    P99:     {p99_tpot*1000:.1f}ms")
    
    return {
        'ttft': {
            'avg_ms': avg_ttft * 1000,
            'p50_ms': p50_ttft * 1000,
            'p99_ms': p99_ttft * 1000
        },
        'tpot': {
            'avg_ms': avg_tpot * 1000,
            'p50_ms': p50_tpot * 1000,
            'p99_ms': p99_tpot * 1000
        }
    }


def benchmark_memory(model, tokenizer):
    """Benchmark d'utilisation mémoire"""
    print("\n" + "="*70)
    print(" BENCHMARK: MÉMOIRE")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Mémoire du modèle
    if torch.cuda.is_available():
        model_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"  Modèle seul: {model_memory:.2f} GB")
    else:
        # Pour CPU, estimer
        model_params = sum(p.numel() for p in model.parameters())
        model_memory = model_params * 4 / 1024**3  # FP32
        print(f"  Modèle estimé (CPU): {model_memory:.2f} GB")
    
    # Mémoire avec KV cache
    input_ids = torch.randint(0, model.config.vocab_size, (1, 1024)).to(device)
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=100, do_sample=False)
    
    if torch.cuda.is_available():
        total_memory = torch.cuda.memory_allocated() / 1024**3
    else:
        total_memory = model_memory  # Estimation
    
    kv_cache_memory = total_memory - model_memory
    
    print(f"  KV cache (1024+100 tokens): {kv_cache_memory:.2f} GB")
    print(f"  Total: {total_memory:.2f} GB")
    
    # Comparaison avec différentes précisions
    print(f"\n  Estimations pour différentes précisions:")
    print(f"    FP32 (100%):    {model_memory:.2f} GB")
    print(f"    FP16 (50%):     {model_memory * 0.5:.2f} GB")
    print(f"    INT8 (25%):     {model_memory * 0.25:.2f} GB")
    print(f"    INT4 (12.5%):   {model_memory * 0.125:.2f} GB")
    print(f"    BitNet 1.58 (6.25%): {model_memory * 0.0625:.2f} GB")
    
    return {
        'model_gb': model_memory,
        'total_gb': total_memory,
        'kv_cache_gb': kv_cache_memory
    }


def benchmark_quality(model, tokenizer, num_samples=10):
    """Benchmark de qualité de génération"""
    print("\n" + "="*70)
    print(" BENCHMARK: QUALITÉ")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    prompts = [
        "The capital of France is",
        "2 + 2 equals",
        "The sun rises in the",
        "Water boils at",
        "The Earth revolves around the",
        "A triangle has",
        "The color of the sky is",
        "Humans have",
        "The largest ocean is",
        "A year has"
    ]
    
    print("  Génération sur prompts factuels...\n")
    
    correct = 0
    total = 0
    
    for prompt in prompts[:num_samples]:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids, 
                max_new_tokens=10, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"  Prompt: {prompt}")
        print(f"  Output: {generated}\n")
        
        total += 1
    
    print(f"  Échantillons générés: {total}")
    print(f"  (Évaluation qualitative manuelle requise)")
    
    return {'samples': total}


def main():
    parser = argparse.ArgumentParser(description='CUDA Open Benchmark')
    parser.add_argument('--model', type=str, default='gpt2', help='Modèle à tester')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--seq-lengths', type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument('--output', type=str, default='benchmark_results.json')
    
    args = parser.parse_args()
    
    print("="*70)
    print(" CUDA OPEN - BENCHMARK COMPLET")
    print("="*70)
    
    # Charger modèle
    print(f"\nChargement du modèle: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32
    )
    
    print(f"✓ Modèle chargé: {args.model}")
    print(f"  Paramètres: {sum(p.numel() for p in model.parameters())/1e6:.1f}M\n")
    
    benchmark = BenchmarkResult()
    
    # 1. Throughput
    throughput_results = benchmark_throughput(
        model, tokenizer, args.batch_sizes, args.seq_lengths
    )
    benchmark.add('throughput', {
        'batch_sizes': args.batch_sizes,
        'seq_lengths': args.seq_lengths
    }, throughput_results)
    
    # 2. Latence
    latency_results = benchmark_latency(model, tokenizer)
    benchmark.add('latency', {}, latency_results)
    
    # 3. Mémoire
    memory_results = benchmark_memory(model, tokenizer)
    benchmark.add('memory', {}, memory_results)
    
    # 4. Qualité
    quality_results = benchmark_quality(model, tokenizer)
    benchmark.add('quality', {}, quality_results)
    
    # Sauvegarder
    benchmark.save(args.output)
    
    # Résumé final
    print("\n" + "="*70)
    print(" RÉSUMÉ FINAL")
    print("="*70 + "\n")
    
    print(f"  Modèle testé: {args.model}")
    print(f"  Meilleur throughput: {max(r['throughput_tok_s'] for r in throughput_results.values()):.1f} tok/s")
    print(f"  TTFT moyen: {latency_results['ttft']['avg_ms']:.1f}ms")
    print(f"  TPOT moyen: {latency_results['tpot']['avg_ms']:.1f}ms")
    print(f"  Mémoire modèle: {memory_results['model_gb']:.2f} GB")
    print(f"  Mémoire totale: {memory_results['total_gb']:.2f} GB")
    print()
    
    print("  Prochaines étapes:")
    print("    1. Comparer avec vLLM/TGI")
    print("    2. Tester avec modèle BitNet")
    print("    3. Benchmarks multi-GPU")
    print()
    print("="*70)


if __name__ == "__main__":
    main()
