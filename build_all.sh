# CUDA Open - Build System Complet

Ce script build tout le projet CUDA Open:
- Bibliothèque C++ CPU
- Kernels CUDA GPU (si disponible)
- Tests
- Exemples
- Benchmarks

Usage:
    ./build_all.sh [options]

Options:
    --release          Build en mode release (défaut)
    --debug            Build en mode debug
    --cuda             Activer le support CUDA
    --no-cuda          Build sans CUDA (CPU only)
    --clean            Nettoyage complet
    --test             Exécuter les tests
    --benchmark        Exécuter les benchmarks
    --all              Build + test + benchmark
"""

set -e

# Couleurs pour output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Variables
BUILD_TYPE="Release"
ENABLE_CUDA="OFF"
RUN_TESTS=0
RUN_BENCHMARK=0
CLEAN=0

# Parser arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --release) BUILD_TYPE="Release"; shift ;;
        --debug) BUILD_TYPE="Debug"; shift ;;
        --cuda) ENABLE_CUDA="ON"; shift ;;
        --no-cuda) ENABLE_CUDA="OFF"; shift ;;
        --clean) CLEAN=1; shift ;;
        --test) RUN_TESTS=1; shift ;;
        --benchmark) RUN_BENCHMARK=1; shift ;;
        --all) RUN_TESTS=1; RUN_BENCHMARK=1; shift ;;
        *) echo "Option inconnue: $1"; exit 1 ;;
    esac
done

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║           CUDA Open - Build System Complet                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Build Type:     $BUILD_TYPE"
echo "  CUDA Support:   $ENABLE_CUDA"
echo "  Run Tests:      $RUN_TESTS"
echo "  Run Benchmark:  $RUN_BENCHMARK"
echo ""

# Nettoyer si demandé
if [ $CLEAN -eq 1 ]; then
    echo -e "${YELLOW}Nettoyage complet...${NC}"
    rm -rf build/
    rm -f kernels/*.o
    rm -f kernels/*.cu.o
    echo -e "${GREEN}✓ Nettoyage terminé${NC}"
fi

# Créer répertoire de build
mkdir -p build
cd build

# ============================================================================
# ÉTAPE 1: Configuration CMake
# ============================================================================

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ ÉTAPE 1: Configuration CMake                               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCUDA_OPEN_BUILD_TESTS=ON \
    -DCUDA_OPEN_BUILD_EXAMPLES=ON \
    -DCUDA_OPEN_ENABLE_CUDA=$ENABLE_CUDA \
    -DCUDA_OPEN_OPTIMIZE=ON

echo -e "${GREEN}✓ Configuration terminée${NC}"
echo ""

# ============================================================================
# ÉTAPE 2: Compilation C++
# ============================================================================

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ ÉTAPE 2: Compilation C++                                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

make -j$(nproc) 2>&1 | tail -20

echo -e "${GREEN}✓ Compilation C++ terminée${NC}"
echo ""

# ============================================================================
# ÉTAPE 3: Compilation CUDA (si activé)
# ============================================================================

if [ "$ENABLE_CUDA" = "ON" ]; then
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ ÉTAPE 3: Compilation CUDA Kernels                          ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    
    # Vérifier NVCC
    if ! command -v nvcc &> /dev/null; then
        echo -e "${RED}✗ NVCC non trouvé${NC}"
        echo "  Installez CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi
    
    cd ../kernels
    
    # Build kernels CUDA
    echo "  Compilation bitnet_kernels.cu..."
    nvcc -O3 \
         -arch=sm_80 \
         -Xptxas -dlcm=ca \
         --use_fast_math \
         -lineinfo \
         bitnet_kernels.cu \
         -o bitnet_kernels.so \
         --shared \
         -Xcompiler -fPIC
    
    echo -e "${GREEN}✓ Compilation CUDA terminée${NC}"
    cd ../build
else
    echo -e "${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║ ÉTAPE 3: SKIP (CUDA désactivé)                             ║${NC}"
    echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"
fi

echo ""

# ============================================================================
# ÉTAPE 4: Tests
# ============================================================================

if [ $RUN_TESTS -eq 1 ]; then
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ ÉTAPE 4: Exécution des Tests                               ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    
    ctest --output-on-failure -V
    
    echo -e "${GREEN}✓ Tests terminés${NC}"
    echo ""
else
    echo -e "${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║ ÉTAPE 4: SKIP (tests non demandés)                         ║${NC}"
    echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
fi

# ============================================================================
# ÉTAPE 5: Benchmarks
# ============================================================================

if [ $RUN_BENCHMARK -eq 1 ]; then
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ ÉTAPE 5: Benchmarks                                        ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    
    if [ -f "examples/example_03_quantization" ]; then
        echo "Running quantization benchmark..."
        ./examples/example_03_quantization
    fi
    
    if [ -f "examples/example_05_revolutionary" ]; then
        echo "Running revolutionary optimizations benchmark..."
        ./examples/example_05_revolutionary
    fi
    
    echo -e "${GREEN}✓ Benchmarks terminés${NC}"
    echo ""
else
    echo -e "${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║ ÉTAPE 5: SKIP (benchmarks non demandés)                    ║${NC}"
    echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
fi

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║ BUILD TERMINÉ AVEC SUCCÈS                                  ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Artefacts créés:${NC}"
echo "  Bibliothèque:    build/libcuda_open.a"
if [ "$ENABLE_CUDA" = "ON" ]; then
    echo "  CUDA Kernels:    kernels/bitnet_kernels.so"
fi
echo "  Tests:           build/tests/*"
echo "  Exemples:        build/examples/*"
echo ""
echo -e "${YELLOW}Pour exécuter:${NC}"
echo "  Tests:           cd build && ctest"
echo "  Exemple 1:       cd build && ./examples/example_01_memory"
echo "  Exemple 3:       cd build && ./examples/example_03_quantization"
echo "  Exemple 5:       cd build && ./examples/example_05_revolutionary"
echo ""
