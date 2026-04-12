"""
CUDA Open - Preuve d'Entraînement BitNet (Tiny Model)

Ce script entraîne un micro-modèle (2 couches) à additionner deux nombres.
Objectif: Prouver que la quantization BitNet 1.58-bit permet l'apprentissage.

100% CPU, < 30 secondes, pas de GPU requis.

Usage:
    python3 training/train_tiny_bitnet.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys
from pathlib import Path

# Ajoute le dossier src au path pour importer cuda_open proprement
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from cuda_open.bitnet_linear import BitNetLinear

class TinyBitNetModel(nn.Module):
    """Un micro-modèle pour apprendre l'addition."""
    def __init__(self):
        super().__init__()
        # Embedding simple : on prend les 2 nombres en entrée
        self.fc1 = BitNetLinear(2, 16)  # Entrée -> Caché (BitNet)
        self.relu = nn.ReLU()
        self.fc2 = BitNetLinear(16, 1)  # Caché -> Sortie (BitNet)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

def generate_batch(batch_size=64):
    """Génère des données d'entraînement (a + b = c)."""
    a = torch.randn(batch_size, 1) * 10
    b = torch.randn(batch_size, 1) * 10
    x = torch.cat([a, b], dim=1)
    y = a + b
    return x, y

def main():
    print("\n" + "="*60)
    print(" CUDA Open - Preuve d'Entraînement BitNet")
    print("="*60 + "\n")
    
    # Configuration
    device = torch.device('cpu') # CPU suffit pour cette démo
    model = TinyBitNetModel().to(device)
    
    # Optimizer (AdamW est standard)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    print("🏋️ Démarrage de l'entraînement...")
    print(f"   Modèle: TinyBitNet (2 couches BitNet 1.58-bit)")
    print(f"   Tâche: Addition (a + b)")
    print(f"   Device: {device}\n")
    
    start_time = time.time()
    
    # Boucle d'entraînement
    epochs = 200
    batch_size = 64
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # 1. Forward
        inputs, targets = generate_batch(batch_size)
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 2. Backward
        loss.backward()
        
        # 3. Step
        optimizer.step()
        
        # Suivi
        if epoch % 20 == 0:
            print(f"   Epoch {epoch:>4}/{epochs} | Loss: {loss.item():.6f} | Time: {time.time()-start_time:.1f}s")
            
            # Test de validation simple
            model.eval()
            with torch.no_grad():
                test_a = torch.tensor([[5.0, 3.0]]).to(device)
                pred = model(test_a).item()
                print(f"   ↳ Test: 5.0 + 3.0 = {pred:.2f} (Vrai: 8.00)")
            
            if loss.item() < best_loss:
                best_loss = loss.item()

    print("\n" + "="*60)
    print(" ✅ ENTRAÎNEMENT TERMINÉ")
    print("="*60)
    print(f"   Meilleur Loss: {best_loss:.6f}")
    print(f"   Temps total:   {time.time()-start_time:.1f}s")
    
    # Vérification finale
    model.eval()
    with torch.no_grad():
        inputs, targets = generate_batch(10)
        preds = model(inputs.to(device))
        
        # Erreur moyenne
        mae = torch.mean(torch.abs(preds - targets.to(device))).item()
        print(f"   Erreur Moyenne (MAE): {mae:.4f}")
        
        if mae < 1.0:
            print(f"\n   🎉 SUCCÈS! Le modèle a appris à additionner avec BitNet 1.58-bit!")
        else:
            print(f"\n   ⚠️ Le modèle est en cours d'apprentissage...")

if __name__ == "__main__":
    main()
