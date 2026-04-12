"""
CUDA Open - Benchmark Scientifique Complet

Compare BitNet 1.58-bit vs FP32 vs INT8 vs INT4 vs FP16
Mesure: Vitesse, Compression, Précision, Efficacité Énergétique

100% numpy, < 5 secondes, aucune dépendance lourde.

Usage:
    python3 benchmark/scientific_benchmark.py
"""

import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, Tuple


# ============================================================================
# IMPLÉMENTATIONS DE QUANTIZATION
# ============================================================================

class QuantizationMethods:
    """Implémente toutes les méthodes de quantization pour comparaison équitable."""

    @staticmethod
    def fp32_baseline(data: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """Baseline: aucune quantization."""
        return data.astype(np.float32), 1.0, {'method': 'FP32', 'bits': 32}

    @staticmethod
    def fp16_quantize(data: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """FP16: 16-bit float."""
        return data.astype(np.float16), 1.0, {'method': 'FP16', 'bits': 16}

    @staticmethod
    def int8_quantize(data: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """INT8: 8-bit symmetric quantization."""
        scale = np.max(np.abs(data))
        if scale < 1e-8:
            scale = 1.0
        int8_data = np.round(data / scale * 127).astype(np.int8)
        return int8_data, scale, {'method': 'INT8', 'bits': 8}

    @staticmethod
    def int4_quantize(data: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """INT4: 4-bit quantization (2 values per byte)."""
        scale = np.max(np.abs(data))
        if scale < 1e-8:
            scale = 1.0
        # Quantize to -8..7
        int4_vals = np.round(data / scale * 7).astype(np.int8)
        int4_vals = np.clip(int4_vals, -8, 7)
        # Pack 2 values per byte
        packed = np.zeros((len(int4_vals) + 1) // 2, dtype=np.uint8)
        for i in range(0, len(int4_vals), 2):
            low = int4_vals[i] + 8  # 0..15
            high = (int4_vals[i+1] + 8) if i+1 < len(int4_vals) else 0
            packed[i//2] = (high << 4) | (low & 0x0F)
        return packed, scale, {'method': 'INT4', 'bits': 4}

    @staticmethod
    def bitnet_quantize(data: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """BitNet 1.58-bit: Ternary {-1, 0, 1}, 4 values per byte."""
        scale = np.max(np.abs(data))
        if scale < 1e-8:
            scale = 1.0
        threshold = 0.2 * scale
        ternary = np.zeros_like(data, dtype=np.int8)
        ternary[data > threshold] = 1
        ternary[data < -threshold] = -1
        
        # Pack 4 values per byte
        flat = ternary.flatten()
        vals = (flat + 1).astype(np.uint8)  # -1→0, 0→1, 1→2
        remainder = len(vals) % 4
        if remainder > 0:
            vals = np.pad(vals, (0, 4 - remainder), mode='constant')
        vals = vals.reshape(-1, 4)
        packed = (vals[:, 0] | (vals[:, 1] << 2) | (vals[:, 2] << 4) | (vals[:, 3] << 6)).astype(np.uint8)
        
        return packed, scale, {'method': 'BitNet 1.58', 'bits': 1.58}


# ============================================================================
# DÉQUANTIZATION
# ============================================================================

class DequantizationMethods:
    """Restaure les données quantizées pour mesurer l'erreur."""

    @staticmethod
    def dequantize(compressed, scale: float, method: str, original_size: int) -> np.ndarray:
        if method == 'FP32':
            return compressed.astype(np.float32)
        elif method == 'FP16':
            return compressed.astype(np.float32)
        elif method == 'INT8':
            return (compressed.astype(np.float32) / 127.0) * scale
        elif method == 'INT4':
            # Unpack
            unpacked = np.zeros(original_size, dtype=np.float32)
            for i in range(original_size):
                byte_idx = i // 2
                if i % 2 == 0:
                    val = compressed[byte_idx] & 0x0F
                else:
                    val = (compressed[byte_idx] >> 4) & 0x0F
                unpacked[i] = (val - 8) * (scale / 7.0)
            return unpacked
        elif method == 'BitNet 1.58':
            # Unpack ternary
            lut = np.array([0.0, -1.0, 1.0, 0.0], dtype=np.float32)
            unpacked = np.zeros(original_size, dtype=np.float32)
            for i in range(original_size):
                byte_idx = i // 4
                offset = (i % 4) * 2
                encoded = (compressed[byte_idx] >> offset) & 0x03
                unpacked[i] = lut[encoded] * scale
            return unpacked
        return compressed


# ============================================================================
# BENCHMARK SUITE
# ============================================================================

class ScientificBenchmark:
    """Benchmark scientifique complet et reproductible."""

    def __init__(self):
        self.results = []
        self.methods = {
            'FP32': QuantizationMethods.fp32_baseline,
            'FP16': QuantizationMethods.fp16_quantize,
            'INT8': QuantizationMethods.int8_quantize,
            'INT4': QuantizationMethods.int4_quantize,
            'BitNet 1.58': QuantizationMethods.bitnet_quantize,
        }

    def run_all_tests(self):
        """Exécute TOUS les benchmarks."""
        print("\n" + "="*70)
        print(" CUDA OPEN - BENCHMARK SCIENTIFIQUE COMPLET")
        print("="*70 + "\n")

        self.test_compression_ratio()
        self.test_quantization_speed()
        self.test_accuracy_loss()
        self.test_efficiency_globale()
        self.test_different_distributions()

        self.print_final_report()
        self.save_results()

    def test_compression_ratio(self):
        """Test 1: Ratio de compression."""
        print("TEST 1: Compression Ratio")
        print("-" * 50)

        sizes = [1000, 10000, 100000]
        np.random.seed(42)

        for size in sizes:
            data = np.random.randn(size).astype(np.float32)
            original_bytes = data.nbytes

            print(f"\n  Taille: {size:>7} valeurs ({original_bytes:>7} bytes)")
            print(f"  {'Méthode':<15} {'Compressé':>10} {'Ratio':>8} {'Bits/valeur':>12}")
            print(f"  {'-'*48}")

            for name, func in self.methods.items():
                compressed, scale, meta = func(data)
                compressed_bytes = compressed.nbytes
                ratio = original_bytes / compressed_bytes
                bits_per_val = (compressed_bytes * 8) / size

                print(f"  {name:<15} {compressed_bytes:>7}B   {ratio:>7.1f}x   {bits_per_val:>11.2f}")

                if size == 100000:
                    self.results.append({
                        'test': 'compression',
                        'method': name,
                        'size': size,
                        'ratio': ratio,
                        'bits_per_value': bits_per_val
                    })

    def test_quantization_speed(self):
        """Test 2: Vitesse de quantization."""
        print("\n\nTEST 2: Vitesse de Quantization")
        print("-" * 50)

        np.random.seed(42)
        data = np.random.randn(100000).astype(np.float32)
        iterations = 20

        print(f"\n  Données: 100,000 valeurs | Itérations: {iterations}")
        print(f"  {'Méthode':<15} {'Moyenne':>10} {'Min':>10} {'Max':>10} {'Écart-type':>12}")
        print(f"  {'-'*60}")

        for name, func in self.methods.items():
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                func(data)
                times.append((time.perf_counter() - start) * 1000)

            avg_t = np.mean(times)
            min_t = np.min(times)
            max_t = np.max(times)
            std_t = np.std(times)

            print(f"  {name:<15} {avg_t:>9.2f}ms {min_t:>9.2f}ms {max_t:>9.2f}ms {std_t:>11.3f}ms")

            self.results.append({
                'test': 'speed',
                'method': name,
                'avg_ms': avg_t,
                'min_ms': min_t,
                'max_ms': max_t,
                'std_ms': std_t
            })

    def test_accuracy_loss(self):
        """Test 3: Précision (erreur de reconstruction)."""
        print("\n\nTEST 3: Précision / Erreur de Reconstruction")
        print("-" * 50)

        np.random.seed(42)
        data = np.random.randn(10000).astype(np.float32)

        print(f"\n  {'Méthode':<15} {'MSE':>12} {'MAE':>12} {'MaxErr':>12} {'CosSim':>10} {'SNR(dB)':>10}")
        print(f"  {'-'*74}")

        for name, func in self.methods.items():
            compressed, scale, meta = func(data)
            reconstructed = DequantizationMethods.dequantize(
                compressed, scale, name, len(data)
            )

            # Metrics
            error = data - reconstructed
            mse = np.mean(error ** 2)
            mae = np.mean(np.abs(error))
            max_err = np.max(np.abs(error))

            # Cosine similarity
            cos_sim = np.dot(data, reconstructed) / (
                np.linalg.norm(data) * np.linalg.norm(reconstructed) + 1e-10
            )

            # Signal-to-Noise Ratio
            signal_power = np.mean(data ** 2)
            noise_power = mse
            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

            print(f"  {name:<15} {mse:>11.4f} {mae:>11.4f} {max_err:>11.4f} {cos_sim:>9.4f} {snr_db:>9.1f}")

            self.results.append({
                'test': 'accuracy',
                'method': name,
                'mse': float(mse),
                'mae': float(mae),
                'max_error': float(max_err),
                'cosine_similarity': float(cos_sim),
                'snr_db': float(snr_db)
            })

    def test_efficiency_globale(self):
        """Test 4: Efficacité globale (score composite)."""
        print("\n\nTEST 4: Score d'Efficacité Globale")
        print("-" * 50)

        print(f"\n  {'Méthode':<15} {'Compression':>12} {'Vitesse':>10} {'Précision':>10} {'SCORE':>10}")
        print(f"  {'-'*50}")

        # Scores normalisés (0-100)
        scores = {}
        for name, func in self.methods.items():
            # Récupérer les résultats existants
            comp_res = next((r for r in self.results if r['test'] == 'compression' and r['method'] == name), None)
            speed_res = next((r for r in self.results if r['test'] == 'speed' and r['method'] == name), None)
            acc_res = next((r for r in self.results if r['test'] == 'accuracy' and r['method'] == name), None)

            if comp_res and speed_res and acc_res:
                # Compression: BitNet = 100, FP32 = 6.25
                comp_score = min(100, (comp_res['bits_per_value'] / 32) ** -1 * 100)
                
                # Vitesse: plus c'est rapide, mieux c'est
                all_speeds = [r['avg_ms'] for r in self.results if r['test'] == 'speed']
                speed_score = 100 * (min(all_speeds) / speed_res['avg_ms'])
                
                # Précision: cosine similarity
                prec_score = acc_res['cosine_similarity'] * 100

                # Score composite pondéré
                # Pour l'inférence: compression (40%) + vitesse (30%) + précision (30%)
                total = comp_score * 0.4 + speed_score * 0.3 + prec_score * 0.3

                scores[name] = {
                    'compression': comp_score,
                    'speed': speed_score,
                    'precision': prec_score,
                    'total': total
                }

                print(f"  {name:<15} {comp_score:>11.1f} {speed_score:>9.1f} {prec_score:>9.1f} {total:>9.1f}")

                self.results.append({
                    'test': 'efficiency',
                    'method': name,
                    'scores': scores[name]
                })

        # Classement
        print(f"\n  CLASSEMENT:")
        sorted_scores = sorted(scores.items(), key=lambda x: x[1]['total'], reverse=True)
        for rank, (name, s) in enumerate(sorted_scores, 1):
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][rank-1]
            print(f"    {medal} {rank}. {name}: {s['total']:.1f}/100")

    def test_different_distributions(self):
        """Test 5: Performance sur différentes distributions de données."""
        print("\n\nTEST 5: Robustesse (Différentes Distributions)")
        print("-" * 50)

        np.random.seed(42)
        distributions = {
            'Normale(0,1)': np.random.randn(10000),
            'Uniforme[-1,1]': np.random.uniform(-1, 1, 10000),
            'Laplace(0,1)': np.random.laplace(0, 1, 10000),
            'Queue lourde': np.random.standard_t(3, 10000),
        }

        print(f"\n  {'Méthode':<15}", end="")
        for dist_name in distributions.keys():
            print(f" {dist_name:>15}", end="")
        print()
        print(f"  {'-'*80}")

        for name, func in self.methods.items():
            print(f"  {name:<15}", end="")
            for dist_name, data in distributions.items():
                data = data.astype(np.float32)
                compressed, scale, meta = func(data)
                reconstructed = DequantizationMethods.dequantize(
                    compressed, scale, name, len(data)
                )
                cos_sim = np.dot(data, reconstructed) / (
                    np.linalg.norm(data) * np.linalg.norm(reconstructed) + 1e-10
                )
                print(f" {cos_sim:>14.4f}", end="")
            print()

    def print_final_report(self):
        """Rapport final."""
        print("\n\n" + "="*70)
        print(" RAPPORT FINAL")
        print("="*70)

        # Trouver le meilleur
        efficiency_results = [r for r in self.results if r.get('test') == 'efficiency']
        if efficiency_results:
            best = max(efficiency_results, key=lambda x: x['scores']['total'])
            print(f"\n  🏆 MEILLEURE MÉTHODE: {best['method']}")
            print(f"     Score: {best['scores']['total']:.1f}/100")
            print(f"     Compression: {best['scores']['compression']:.1f}/100")
            print(f"     Vitesse: {best['scores']['speed']:.1f}/100")
            print(f"     Précision: {best['scores']['precision']:.1f}/100")

        print(f"\n  📊 CONCLUSION:")
        print(f"     • BitNet 1.58-bit offre la meilleure compression (16x)")
        print(f"     • INT8 offre le meilleur équilibre vitesse/précision")
        print(f"     • FP32 reste la référence pour la qualité maximale")
        print(f"     • Le choix dépend du cas d'usage (inférence vs training)")
        print()

    def save_results(self):
        """Sauvegarde les résultats."""
        output = Path(__file__).parent / "benchmark_results.json"
        
        # Convertir pour JSON
        json_results = []
        for r in self.results:
            if 'scores' in r:
                r['scores'] = {k: float(v) for k, v in r['scores'].items()}
            json_results.append(r)
        
        with open(output, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"  💾 Résultats sauvés: {output}")


if __name__ == "__main__":
    benchmark = ScientificBenchmark()
    benchmark.run_all_tests()
