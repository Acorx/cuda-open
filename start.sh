#!/bin/bash
# CUDA Open - Guide de Démarrage Rapide
# 
# Ce script te guide à travers les premières étapes avec CUDA Open
#
# Usage:
#     ./start.sh

set -e

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║              CUDA OPEN - STARTER GUIDE                     ║"
echo "║                                                            ║"
echo "║   Framework Révolutionnaire par Évolution Neuro-Symbolique║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
echo -e "${YELLOW}Bienvenue dans CUDA Open!${NC}"
echo ""
echo "Ce projet utilise l'évolution neuro-symbolique pour découvrir"
echo "automatiquement des architectures qui surpassent CUDA/PyTorch."
echo ""

# Menu
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ Que veux-tu faire?                                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  1) 📖 Lire la documentation"
echo "  2) 🧪 Lancer les tests"
echo "  3) 🏋️ Exécuter une évolution"
echo "  4) 📊 Lancer un benchmark"
echo "  5) 🎓 Entraîner un modèle BitNet"
echo "  6) 🚀 Build complet du projet"
echo "  0) ❌ Quitter"
echo ""

read -p "Choisis une option (0-6): " choice

case $choice in
    1)
        echo ""
        echo -e "${GREEN}📖 DOCUMENTATION${NC}"
        echo ""
        echo "  1) README principal"
        echo "  2) Guide complet (FR)"
        echo "  3) Résumé ultime"
        echo "  4) Statistiques"
        echo ""
        read -p "Choisis (1-4): " doc_choice
        
        case $doc_choice in
            1) less README_FINAL.md ;;
            2) less GUIDE_COMPLET.md ;;
            3) less RESUME_ULTIME.md ;;
            4) less STATISTIQUES_FINALES.md ;;
            *) echo "Option invalide" ;;
        esac
        ;;
    
    2)
        echo ""
        echo -e "${GREEN}🧪 TESTS${NC}"
        echo ""
        echo "  1) Tests unitaires C++"
        echo "  2) Test avec vrai modèle GPT-2"
        echo ""
        read -p "Choisis (1-2): " test_choice
        
        case $test_choice in
            1)
                echo "Lancement des tests C++..."
                mkdir -p build && cd build
                cmake .. -DCUDA_OPEN_BUILD_TESTS=ON
                make -j$(nproc)
                ctest --output-on-failure
                ;;
            2)
                echo "Lancement du test avec GPT-2..."
                cd simulation
                python3 test_real_model_with_log.py
                ;;
            *) echo "Option invalide" ;;
        esac
        ;;
    
    3)
        echo ""
        echo -e "${GREEN}🏋️ ÉVOLUTION NEURO-SYMBOLIQUE${NC}"
        echo ""
        echo "  1) Évolution architecture hardware"
        echo "  2) Évolution attention mechanisms"
        echo "  3) Évolution training strategy"
        echo "  4) Évolution production"
        echo ""
        read -p "Choisis (1-4): " evo_choice
        
        cd simulation
        
        case $evo_choice in
            1)
                echo "Évolution architecture hardware..."
                python3 neuro_symbolic_evolution.py --generations 20 --population 50
                ;;
            2)
                echo "Évolution attention mechanisms..."
                python3 evolve_attention.py --generations 20 --population 60
                ;;
            3)
                echo "Évolution training strategy..."
                python3 evolve_training.py --generations 30 --population 80
                ;;
            4)
                echo "Évolution production..."
                python3 evolve_production.py --generations 30 --population 80
                ;;
            *) echo "Option invalide" ;;
        esac
        ;;
    
    4)
        echo ""
        echo -e "${GREEN}📊 BENCHMARK${NC}"
        echo ""
        echo "Lancement du benchmark complet..."
        cd benchmark
        python3 benchmark.py --model gpt2 --batch-sizes 1 2 4 --seq-lengths 64 128
        ;;
    
    5)
        echo ""
        echo -e "${GREEN}🎓 TRAINING BITNET${NC}"
        echo ""
        echo "Lancement de l'entraînement BitNet..."
        cd training
        python3 train_bitnet.py --model gpt2 --epochs 3 --batch-size 4 --generate
        ;;
    
    6)
        echo ""
        echo -e "${GREEN}🚀 BUILD COMPLET${NC}"
        echo ""
        echo "Build complet avec tests et benchmarks..."
        ./build_all.sh --all
        ;;
    
    0)
        echo ""
        echo -e "${YELLOW}Au revoir! 🚀${NC}"
        echo ""
        echo "Pour explorer le projet:"
        echo "  cat README_FINAL.md"
        echo "  cat RESUME_ULTIME.md"
        echo ""
        exit 0
        ;;
    
    *)
        echo -e "${RED}Option invalide!${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║ TERMINÉ! 🎉                                                ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
