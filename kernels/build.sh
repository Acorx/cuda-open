#!/bin/bash
# CUDA Open - Build des Kernels CUDA
# Compile bitnet_kernels.cu en bibliothèque partagée

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║ CUDA Open - Build des Kernels CUDA                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Vérifie nvcc
if ! command -v nvcc &> /dev/null; then
    echo "⚠️  nvcc non trouvé. Installation CUDA requise."
    echo "   sudo apt install nvidia-cuda-toolkit"
    echo ""
    echo "   Fallback: utilisation de l'implémentation numpy (fonctionnel)."
    exit 0
fi

echo "✅ nvcc trouvé: $(nvcc --version | grep release | awk '{print $5}')"

# Architecture GPU par défaut (compatible avec la plupart des GPU)
ARCH=${CUDA_ARCH:-"sm_50"}
echo "🎯 Architecture cible: $ARCH"

# Compile
cd "$SCRIPT_DIR"
echo ""
echo "🔨 Compilation de bitnet_kernels.cu..."

nvcc -O3 \
     -arch=$ARCH \
     -Xptxas -dlcm=ca \
     --use_fast_math \
     -Xcompiler -fPIC \
     -o libbitnet.so \
     bitnet_kernels.cu \
     -lcudart

if [ -f "libbitnet.so" ]; then
    SIZE=$(ls -lh libbitnet.so | awk '{print $5}')
    echo "✅ Build réussi: libbitnet.so ($SIZE)"
    echo ""
    echo "📍 Emplacement: $SCRIPT_DIR/libbitnet.so"
    echo ""
    
    # Test rapide
    echo "🧪 Test du kernel..."
    python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/src')
from cuda_open.cuda_kernels import BitNetCUDAKernels
k = BitNetCUDAKernels('$SCRIPT_DIR/libbitnet.so')
if k.cuda_available:
    print(f'✅ GPU détecté: {k.gpu_name}')
    # Benchmark rapide
    t = k.benchmark(256, 512, 256, 50)
    print(f'⚡ Benchmark 256x512x256: {t:.2f} ms/kernel')
else:
    print('⚠️ CUDA non disponible sur ce système')
" 2>&1 || echo "⚠️ Erreur lors du test (vérifiez les drivers NVIDIA)"
else
    echo "❌ Échec de la compilation"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║ Build terminé!                                              ║"
echo "╚════════════════════════════════════════════════════════════╝"
