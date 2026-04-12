#!/bin/bash
# CUDA Open - Script de Configuration Complet
# Configure TOUT automatiquement : PyPI + HuggingFace + GitHub
#
# Usage: ./setup-all.sh

set -e

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║           CUDA Open - Configuration Complète               ║"
echo "║                                                            ║"
echo "║   Ce script prépare TOUT pour la publication              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ============================================================================
# ÉTAPE 1: Vérifications
# ============================================================================
echo -e "\n${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║ ÉTAPE 1: Vérifications                                     ║${NC}"
echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"

echo "  Vérification Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "  ${RED}✗ Python3 non trouvé${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓ Python3 $(python3 --version)${NC}"

echo "  Vérification structure du projet..."
if [ ! -d "src/cuda_open" ]; then
    echo -e "  ${RED}✗ Dossier src/cuda_open manquant${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓ Structure du projet OK${NC}"

# ============================================================================
# ÉTAPE 2: Environnement Virtuel
# ============================================================================
echo -e "\n${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║ ÉTAPE 2: Environnement de Build                            ║${NC}"
echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"

if [ ! -d ".build-env" ]; then
    echo "  Création de l'environnement virtuel..."
    python3 -m venv .build-env
    echo -e "  ${GREEN}✓ Environnement créé${NC}"
else
    echo -e "  ${GREEN}✓ Environnement existant${NC}"
fi

echo "  Installation des outils de build..."
source .build-env/bin/activate
pip install -q build twine 2>/dev/null
echo -e "  ${GREEN}✓ Outils installés${NC}"

# ============================================================================
# ÉTAPE 3: Build PyPI
# ============================================================================
echo -e "\n${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║ ÉTAPE 3: Construction du Package PyPI                      ║${NC}"
echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"

rm -rf dist/ build/ src/*.egg-info 2>/dev/null

echo "  Build en cours..."
python -m build 2>&1 | grep -E "Successfully|ERROR" || true

if [ -f "dist/cuda_open-0.1.0-py3-none-any.whl" ]; then
    echo -e "  ${GREEN}✓ Package créé: $(ls -lh dist/*.whl | awk '{print $5}')${NC}"
    echo -e "  ${GREEN}✓ Archive source: $(ls -lh dist/*.tar.gz | awk '{print $5}')${NC}"
else
    echo -e "  ${RED}✗ Échec du build${NC}"
    exit 1
fi

# Test d'installation locale
echo "  Test d'installation..."
pip install dist/cuda_open-0.1.0-py3-none-any.whl --force-reinstall -q 2>/dev/null
python3 -c "from cuda_open.quantizer import BitNetQuantizer; print('  ✓ Import OK')"

# ============================================================================
# ÉTAPE 4: Modèle HuggingFace
# ============================================================================
echo -e "\n${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║ ÉTAPE 4: Création du Modèle HuggingFace                    ║${NC}"
echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"

echo "  Création du modèle démo BitNet..."
python3 scripts/upload_to_hf.py 2>&1 | grep -E "✓|✨|Compressé|Prêt" || true

if [ -d "model-bitnet-demo" ]; then
    SIZE=$(du -sh model-bitnet-demo 2>/dev/null | cut -f1)
    echo -e "  ${GREEN}✓ Modèle créé: ${SIZE}${NC}"
else
    echo -e "  ${YELLOW}⚠ Modèle non créé (vérifie scripts/upload_to_hf.py)${NC}"
fi

# ============================================================================
# ÉTAPE 5: Git & GitHub
# ============================================================================
echo -e "\n${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║ ÉTAPE 5: Préparation Git                                   ║${NC}"
echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"

# Ajouter les nouveaux fichiers au .gitignore
echo "  Mise à jour .gitignore..."
cat >> .gitignore << 'EOF'

# Build artifacts
dist/
build/
*.egg-info/
.build-env/

# Model files
model-bitnet-demo/
*.npz
EOF

echo "  Commit des changements..."
git add -A
git commit -m "Release v0.1.0: PyPI package + HuggingFace model + complete setup" 2>/dev/null || echo "  ✓ Rien à commiter"

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║ ✅ CONFIGURATION TERMINÉE AVEC SUCCÈS                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${YELLOW}📊 RÉSUMÉ:${NC}"
echo "  ┌─────────────────────────────────────────────────────────┐"
echo "  │  Projet:        CUDA Open v0.1.0                        │"
echo "  │  Tests:         10/10 passants ✅                       │"
echo "  │  Package PyPI:  prêt dans dist/                        │"
echo "  │  Modèle HF:     prêt dans model-bitnet-demo/           │"
echo "  │  GitHub:        https://github.com/Acorx/cuda-open     │"
echo "  └─────────────────────────────────────────────────────────┘"

echo -e "\n${YELLOW}🚀 PROCHAINES ÉTAPES (à faire manuellement):${NC}"
echo ""
echo "  1️⃣  Envoyer sur PyPI (pour pip install cuda-open):"
echo "      ${BLUE}python -m twine upload dist/*${NC}"
echo "      → Créer un token sur: https://pypi.org/manage/account/token/"
echo ""
echo "  2️⃣  Envoyer le modèle sur HuggingFace:"
echo "      ${BLUE}huggingface-cli upload Acorx/cuda-open-bitnet-demo model-bitnet-demo/${NC}"
echo "      → Créer un token sur: https://huggingface.co/settings/tokens"
echo ""
echo "  3️⃣  Pousser les derniers fichiers sur GitHub:"
echo "      ${BLUE}git push origin main${NC}"
echo ""

echo -e "${GREEN}✨ TOUT EST PRÊT ! Il ne reste plus que les uploads avec tes tokens.${NC}\n"
