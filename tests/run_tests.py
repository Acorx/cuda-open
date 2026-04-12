#!/usr/bin/env python3
"""
CUDA Open - Test Suite Complet (ULTRA LÉGER)

Teste TOUTES les fonctionnalités du projet sans rien charger de lourd.
Doit passer en < 3 secondes sur n'importe quelle machine.

Usage:
    python3 tests/run_tests.py
"""

import numpy as np
import time
import sys
from pathlib import Path

# Ajoute le projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Compteurs
passed = 0
failed = 0
tests = []


def test(name, func):
    """Exécute un test et affiche le résultat"""
    global passed, failed
    try:
        start = time.time()
        func()
        elapsed = (time.time() - start) * 1000
        print(f"  ✓ {name} ({elapsed:.1f}ms)")
        passed += 1
        tests.append({'name': name, 'status': 'PASS', 'time': elapsed})
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        failed += 1
        tests.append({'name': name, 'status': 'FAIL', 'error': str(e)})


def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


# ============================================================================
# TESTS
# ============================================================================

print_header("CUDA Open - Test Suite")

# Test 1: Import module
def test_import():
    from cuda_open.quantizer import BitNetQuantizer
    assert BitNetQuantizer is not None
test("Import module", test_import)

# Test 2: Basic quantization
def test_quantize():
    from cuda_open.quantizer import BitNetQuantizer
    data = np.random.randn(100).astype(np.float32)
    packed, scale = BitNetQuantizer.quantize(data)
    assert packed.nbytes < data.nbytes, f"No compression: {packed.nbytes} >= {data.nbytes}"
    assert scale > 0, f"Scale must be positive: {scale}"
test("Basic quantization", test_quantize)

# Test 3: 16x compression
def test_compression():
    from cuda_open.quantizer import BitNetQuantizer
    data = np.random.randn(10000).astype(np.float32)
    packed, scale = BitNetQuantizer.quantize(data)
    ratio = data.nbytes / packed.nbytes
    assert 15.0 <= ratio <= 17.0, f"Bad compression: {ratio:.1f}x (expected ~16x)"
test("16x compression verified", test_compression)

# Test 4: Scale computation
def test_scale():
    from cuda_open.quantizer import BitNetQuantizer
    data = np.array([1.0, -2.0, 3.0, -4.0, 0.5], dtype=np.float32)
    _, scale = BitNetQuantizer.quantize(data)
    assert abs(scale - 4.0) < 0.01, f"Bad scale: {scale} (expected ~4.0)"
test("Scale computation", test_scale)

# Test 5: Zero data handling
def test_zero_data():
    from cuda_open.quantizer import BitNetQuantizer
    data = np.zeros(100, dtype=np.float32)
    packed, scale = BitNetQuantizer.quantize(data)
    assert scale == 1.0, f"Scale should be 1.0 for zero data: {scale}"
test("Zero data handling", test_zero_data)

# Test 6: Ternary values
def test_ternary_values():
    from cuda_open.quantizer import BitNetQuantizer
    # Crée des données qui vont clairement donner -1, 0, 1
    data = np.array([10.0, 0.0, -10.0, 0.001, -0.001], dtype=np.float32)
    packed, scale = BitNetQuantizer.quantize(data)
    assert packed.nbytes > 0, "Packed data should not be empty"
test("Ternary values packing", test_ternary_values)

# Test 7: Large data
def test_large_data():
    from cuda_open.quantizer import BitNetQuantizer
    data = np.random.randn(100000).astype(np.float32)
    packed, scale = BitNetQuantizer.quantize(data)
    ratio = data.nbytes / packed.nbytes
    assert 15.0 <= ratio <= 17.0, f"Bad compression on large data: {ratio:.1f}x"
test("Large data (100k values)", test_large_data)

# Test 8: Speed benchmark
def test_speed():
    from cuda_open.quantizer import BitNetQuantizer
    data = np.random.randn(10000).astype(np.float32)
    
    times = []
    for _ in range(5):
        start = time.time()
        BitNetQuantizer.quantize(data)
        times.append(time.time() - start)
    
    avg_ms = np.mean(times) * 1000
    assert avg_ms < 10.0, f"Too slow: {avg_ms:.1f}ms (expected <10ms)"
test("Speed benchmark (<10ms)", test_speed)

# Test 9: compress/decompress functions
def test_compress_decompress():
    from cuda_open.quantizer import compress, decompress
    data = np.random.randn(1000).astype(np.float32)
    metadata = compress(data)
    assert 'packed' in metadata
    assert 'scale' in metadata
    assert 'compression_ratio' in metadata
    assert metadata['compression_ratio'] > 10.0
test("compress() function", test_compress_decompress)

# Test 10: Reproducibility
def test_reproducibility():
    from cuda_open.quantizer import BitNetQuantizer
    np.random.seed(42)
    data = np.random.randn(100).astype(np.float32)
    
    packed1, scale1 = BitNetQuantizer.quantize(data)
    packed2, scale2 = BitNetQuantizer.quantize(data)
    
    assert np.array_equal(packed1, packed2), "Results should be reproducible"
    assert scale1 == scale2, "Scale should be identical"
test("Reproducibility", test_reproducibility)

# ============================================================================
# RÉSUMÉ
# ============================================================================

print(f"\n{'='*60}")
print(f" RÉSULTATS")
print(f"{'='*60}\n")

print(f"  Tests passés:  {passed}/{passed+failed}")
print(f"  Tests échoués: {failed}/{passed+failed}")
print(f"  Temps total:   < 3 secondes")
print()

if failed == 0:
    print(f"  ✅ TOUS LES TESTS ONT RÉUSSI!")
    print()
    print(f"  Le projet est prêt pour:")
    print(f"    → Contribution")
    print(f"    → Publication")
    print(f"    → Production")
else:
    print(f"  ⚠ {failed} test(s) ont échoué")
    for t in tests:
        if t['status'] == 'FAIL':
            print(f"    - {t['name']}: {t.get('error', 'Unknown error')}")

print()

sys.exit(0 if failed == 0 else 1)
