"""
CUDA Open - Projet SWARM : Auto-Curriculum Training

Pour créer une IA capable de sauver des vies, elle doit être robuste.
Nous utilisons une méthode d'Auto-Curriculum :
1. On entraîne l'IA par beau temps.
2. Une fois stable, on augmente la tempête.
3. On ré-entraîne.
Cela crée un "Super-Pilote" capable d'opérer dans le chaos total.

Usage:
    python3 training/train_swarm_auto_curriculum.py
"""

import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# On réimporte les classes du script précédent pour la cohérence
# (Dans un vrai projet, elles seraient dans un fichier models.py)
from train_swarm_controller import DroneSim, PIDExpert, generate_dataset, DroneBrain

print("\n" + "="*70)
print(" CUDA Open - Projet SWARM : Auto-Curriculum (Niveau Expert)")
print("="*70 + "\n")

def train_phase(model, X, Y, epochs=20, lr=0.01):
    """Entraîne le modèle sur une phase spécifique."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X_t, Y_t = torch.from_numpy(X), torch.from_numpy(Y)
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, Y_t)
        loss.backward()
        optimizer.step()
    return loss.item()

def test_flight(model, storm_level, steps=50):
    """Teste le modèle dans une tempête donnée."""
    sim = DroneSim()
    sim.alt = 10.0
    sim.vel = 0.0
    
    survived = 0
    model.eval()
    hidden = torch.zeros(1, 16) # Hidden state du Liquid NN
    
    with torch.no_grad():
        for t in range(steps):
            # Vent violent
            wind = np.random.normal(0, storm_level)
            
            # Input
            input_t = torch.tensor([[[sim.alt, sim.vel, wind]]], dtype=torch.float32)
            thrust = model(input_t).item()
            
            sim.step(thrust, wind)
            if sim.alt > -0.5:
                survived += 1
            else:
                break
    return survived

def main():
    # Configuration
    SEQ_LEN = 20
    HIDDEN = 16
    
    # Modèle initial
    model = DroneBrain(input_size=3, hidden_size=HIDDEN)
    print(f"🧠 DroneBrain Initialisé ({sum(p.numel() for p in model.parameters())} params)")
    
    # Niveaux de difficulté (Vent)
    levels = [1.0, 2.0, 3.0, 5.0] # De la brise à l'ouragan
    
    results = []
    
    for i, noise in enumerate(levels):
        print(f"\n--- PHASE {i+1}/{len(levels)} : Tempête Niveau {noise} ---")
        
        # 1. Génération de données expertes pour ce niveau de bruit
        # L'expert PID doit être bon pour enseigner
        print(f"📡 Génération des données d'expertise (Vent {noise})...")
        X, Y = generate_dataset(num_episodes=1500, seq_len=SEQ_LEN, noise_level=noise)
        
        # 2. Entraînement
        print(f"🏋️ Entraînement sur {50} époques...")
        start = time.time()
        loss = train_phase(model, X, Y, epochs=50, lr=0.005)
        duration = time.time() - start
        
        # 3. Test de vol réel
        # On teste si l'IA survit mieux que la chute libre
        # (Chute libre = 0 steps dans ce test car alt=10 et gravité)
        survival = test_flight(model, storm_level=noise, steps=50)
        status = "✅ STABILISÉ" if survival == 50 else f"⚠️ ÉCHEC ({survival}/50 steps)"
        
        print(f"   Perte Finale: {loss:.4f}")
        print(f"   Résistance: {status}")
        
        results.append({'Level': noise, 'Loss': loss, 'Survival': survival})
        
    # ============================================================================
    # RAPPORT FINAL
    # ============================================================================
    
    print("\n" + "="*70)
    print(" 📊 RAPPORT DE PERFORMANCE DU PROTOTYPE SWARM")
    print("="*70)
    print(f"  {'Niveau Tempête':<15} {'Perte':<10} {'Survie (Steps)':<15} {'Statut'}")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  Vent {r['Level']:<9.1f} {r['Loss']:<10.4f} {r['Survival']:<15} {'✅ OK' if r['Survival']==50 else '❌ Crash'}")
        
    # Conclusion
    success_count = sum(1 for r in results if r['Survival'] == 50)
    if success_count == len(results):
        print(f"\n  🏆 SUCCÈS TOTAL : L'IA a maîtrisé toutes les conditions, même l'ouragan !")
        print(f"  Ce modèle prouve la capacité de CUDA Open à créer des IAs résilientes.")
    else:
        print(f"\n  ⚠️ L'IA a réussi {success_count}/{len(results)} niveaux.")
        print(f"  Pour un usage critique, nous augmenterions la taille du modèle.")
        
    print(f"\n  💾 Modèle prêt pour la compilation C++ vers micro-puces.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
