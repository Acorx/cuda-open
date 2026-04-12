"""
CUDA Open - Evolution of Attention Mechanisms

Utilise l'évolution neuro-symbolique pour trouver les paramètres optimaux de 
Tri-Attention et DFlash (taille de fenêtre, dilatation, taille de bloc).

Objectif : Trouver la configuration qui minimise la perte d'information par rapport
à l'Attention standard (MSE) tout en minimisant la complexité théorique.

Usage:
    python3 simulation/evolve_attention_mechanisms.py
"""

import numpy as np
import torch
import time
import sys
from pathlib import Path

# Ajoute le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from cuda_open.advanced_attention import standard_attention, tri_attention, dflash_attention


def evaluate_attention_config(config, seq_len=64, head_dim=16):
    """
    Évalue une configuration d'attention.
    Fitness = (1 / MSE_vs_Standard) * (1 / Complexity_Penalty)
    """
    # Créer des données aléatoires
    Q = torch.randn(1, 2, seq_len, head_dim)
    K = torch.randn(1, 2, seq_len, head_dim)
    V = torch.randn(1, 2, seq_len, head_dim)
    
    # Attention Standard (Cible)
    start = time.time()
    target = standard_attention(Q, K, V)
    target_time = time.time() - start
    
    # Essayer d'évaluer la config
    try:
        if config['type'] == 'tri':
            # Paramètres: window_size, dilation
            result = tri_attention(Q, K, V, window_size=config['window_size'], dilation=config['dilation'])
            # Complexité estimée : N * (window + window/dilation + 1)
            complexity = seq_len * (config['window_size'] + config['window_size']//config['dilation'] + 1)
        elif config['type'] == 'dflash':
            # Paramètre: block_size
            result = dflash_attention(Q, K, V, block_size=config['block_size'])
            # Complexité : N^2 mais IO-optimal (on considère un surcoût lié aux blocs petits)
            complexity = seq_len**2 + (seq_len // config['block_size']) * 1000
        
        # Calcul de l'erreur
        mse = torch.mean((result - target) ** 2).item()
        
        # Pénalité si l'erreur est trop grande
        if mse > 0.1:
            fitness = 0
        else:
            # On veut minimiser la complexité et l'erreur
            # Fitness = 1 / (Erreur * Complexité_Relative)
            std_complexity = seq_len**2
            fitness = 1.0 / (max(1e-9, mse) * (complexity / std_complexity) + 1e-9)
            
        return fitness, mse, complexity
        
    except Exception as e:
        return 0, float('inf'), float('inf')


def run_evolution(pop_size=50, generations=20):
    print("\n" + "="*70)
    print(" CUDA Open - Évolution des Mécanismes d'Attention")
    print("="*70 + "\n")
    
    np.random.seed(42)
    
    # Population initiale
    population = []
    
    # Seeds
    population.append({'type': 'tri', 'window_size': 16, 'dilation': 4})
    population.append({'type': 'tri', 'window_size': 64, 'dilation': 1})
    population.append({'type': 'dflash', 'block_size': 16})
    population.append({'type': 'dflash', 'block_size': 32})
    
    # Random
    for _ in range(pop_size - len(population)):
        if np.random.random() < 0.5:
            population.append({
                'type': 'tri', 
                'window_size': int(np.random.choice([4, 8, 16, 32, 64])),
                'dilation': int(np.random.choice([1, 2, 4, 8]))
            })
        else:
            population.append({
                'type': 'dflash',
                'block_size': int(np.random.choice([8, 16, 32, 64]))
            })
            
    best_config = None
    best_fitness = 0
    
    print(f"Population: {len(population)} configurations")
    print(f"Objectif: Maximiser la précision tout en réduisant la complexité\n")
    
    for gen in range(generations):
        # Evaluate
        scores = []
        for config in population:
            fit, mse, comp = evaluate_attention_config(config)
            scores.append(fit)
            
            if fit > best_fitness:
                best_fitness = fit
                best_config = config.copy()
                print(f"  ★ Gen {gen+1}: Nouveau meilleur! Type={config['type']}, Fitness={fit:.2e}, MSE={mse:.6f}")
                if config['type'] == 'tri':
                    print(f"    → Params: Window={config['window_size']}, Dilat={config['dilation']}")
                else:
                    print(f"    → Params: Block={config['block_size']}")
        
        # Sélection (Tournoi)
        new_pop = []
        sorted_indices = np.argsort(scores)[::-1]
        # Elitisme
        new_pop.append(population[sorted_indices[0]])
        new_pop.append(population[sorted_indices[1]])
        
        while len(new_pop) < pop_size:
            p1 = population[np.random.choice(np.argsort(scores)[-10:])] # Top 10
            # Mutation
            child = p1.copy()
            if np.random.random() < 0.5:
                # Mutation Tri
                if child['type'] == 'tri':
                    child['window_size'] = int(np.random.choice([4, 8, 16, 32, 64]))
                    child['dilation'] = int(np.random.choice([1, 2, 4, 8]))
                else:
                    child['type'] = 'tri' # Saut de type
                    child['window_size'] = 16
                    child['dilation'] = 4
            else:
                # Mutation DFlash
                if child['type'] == 'dflash':
                    child['block_size'] = int(np.random.choice([8, 16, 32, 64]))
                else:
                    child['type'] = 'dflash' # Saut de type
                    child['block_size'] = 16
            new_pop.append(child)
        
        population = new_pop
        
        avg_score = np.mean(scores)
        print(f"  Gen {gen+1:3d}: Best={best_fitness:.2e}, Avg={avg_score:.2e}\n")

    print(f"{'='*70}")
    print(f" RÉSULTAT FINAL")
    print(f"{'='*70}\n")
    
    if best_config:
        fit, mse, comp = evaluate_attention_config(best_config)
        print(f"  🏆 Mécanisme Gagnant: {best_config['type'].upper()}")
        if best_config['type'] == 'tri':
            print(f"     Fenêtre Locale: {best_config['window_size']}")
            print(f"     Dilatation: {best_config['dilation']}")
            print(f"     → Complexité relative: {(comp / (64**2)) * 100:.1f}% du standard")
        else:
            print(f"     Taille de Bloc: {best_config['block_size']}")
            print(f"     → Optimisation IO maximale")
        
        print(f"     Précision (MSE): {mse:.6f}")
    else:
        print("  Aucune configuration valide trouvée.")


if __name__ == "__main__":
    run_evolution()
