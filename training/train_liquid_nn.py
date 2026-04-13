"""
CUDA Open - Entraînement d'un Liquid Neural Network (LTC) sur CPU

Ce script entraîne un réseau de neurones liquides (Liquid Time-Constant Network)
sur une tâche de prédiction de série temporelle (sinusoïde).
Les LNNs sont inspirés de la biologie et excellent dans les tâches temporelles
avec très peu de paramètres.

Usage:
    python3 training/train_liquid_nn.py
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os
from pathlib import Path

# Pour le tracé, on essaie d'importer matplotlib. Si pas dispo, on sauvegarde en CSV.
try:
    import matplotlib
    matplotlib.use('Agg') # Mode non-interactif pour les serveurs
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("⚠️ Matplotlib non trouvé. Les graphiques seront sauvegardés en CSV.")

print("\n" + "="*70)
print(" CUDA Open - Entraînement Liquid Neural Network (CPU)")
print("="*70 + "\n")

# ============================================================================
# 1. Modèle : Liquid Time-Constant (LTC) Cellule
# ============================================================================

class LTCCell(nn.Module):
    """
    Cellule LTC (Liquid Time-Constant).
    Inspirée par ODE-Net et les réseaux neuronaux liquides.
    Formule simplifiée: h_{t+1} = h_t + 1/tau * (-h_t + f(W*h + U*x))
    """
    def __init__(self, input_size, hidden_size):
        super(LTCCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Poids pour la fonction f (le "squelette" du réseau)
        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(input_size, hidden_size)
        
        # Poids pour la constante de temps tau (adaptive)
        self.W_tau = nn.Linear(hidden_size, hidden_size)
        self.U_tau = nn.Linear(input_size, hidden_size)

        # Activation
        self.act = nn.Tanh()

    def forward(self, input, hidden):
        # Calcul de la dérivée dh/dt
        # f(W*h + U*x)
        state = self.act(self.W(hidden) + self.U(input))
        
        # Calcul de 1/tau
        # tau doit être positif et borné pour la stabilité
        # On utilise une approximation softplus: ln(1 + e^x)
        # Ici on fait simple: on prédit directement l'inverse de tau
        # Pour s'assurer que c'est positif, on passe par un abs ou softplus
        tau_input = self.W_tau(hidden) + self.U_tau(input)
        # tau_inv = torch.sigmoid(tau_input) * 10.0 # tau entre 0.1 et 10
        # Ou plus simple pour la démo :
        tau_inv = torch.sigmoid(tau_input) # Entre 0 et 1
        
        # dh = (1/tau) * (-h + state)
        # Note: Si tau est petit, changement rapide. Si grand, lent.
        dh = tau_inv * (-hidden + state)
        
        # Euler integration step (dt=1 pour simplification dans ce contexte discret)
        new_hidden = hidden + dh
        return new_hidden

class LiquidRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LiquidRNN, self).__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            LTCCell(input_size if i==0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: [batch, seq_len, input_size]
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.num_layers, x.shape[-1] if False else self.cells[0].hidden_size).to(x.device) 
        # Init hidden states (simplifié ici, normalement on passe par les tailles des cellules)
        
        # Correction init hidden
        hidden_states = []
        for cell in self.cells:
            h_layer = torch.zeros(batch, cell.hidden_size).to(x.device)
            hidden_states.append(h_layer)

        outputs = []
        for t in range(seq_len):
            input_t = x[:, t, :]
            for l in range(self.num_layers):
                hidden_states[l] = self.cells[l](input_t, hidden_states[l])
                input_t = hidden_states[l] # L'output de la couche l devient l'input de l+1
            
            outputs.append(self.fc_out(hidden_states[-1]))
        
        return torch.stack(outputs, dim=1).squeeze(-1) # [batch, seq_len]

# ============================================================================
# 2. Données : Série Temporelle (Sinusoïde)
# ============================================================================

def generate_sine_data(num_samples=1000, seq_len=30):
    """Génère des sinusoïdes bruitées pour l'entraînement."""
    x_data = []
    y_data = []
    
    # On prédit la prochaine valeur de la sinusoïde
    # Input: [sin(t), sin(t+1), ..., sin(t+seq_len-1)]
    # Target: [sin(t+1), ..., sin(t+seq_len)] (décalé de 1)
    
    time = np.linspace(0, 20, num_samples + seq_len + 10)
    signal = np.sin(time) + np.random.normal(0, 0.1, len(time)) # Ajout bruit
    
    for i in range(num_samples):
        seq = signal[i : i + seq_len]
        target = signal[i + 1 : i + seq_len + 1]
        x_data.append(seq)
        y_data.append(target)
        
    return np.array(x_data).astype(np.float32), np.array(y_data).astype(np.float32)

# ============================================================================
# 3. Boucle d'Entraînement
# ============================================================================

def train():
    # Hyperparamètres
    INPUT_SIZE = 1
    HIDDEN_SIZE = 16 # LNN est très efficace avec peu de neurones
    NUM_LAYERS = 1
    SEQ_LEN = 30
    EPOCHS = 40
    BATCH_SIZE = 32
    LR = 0.01

    print(f"📊 Préparation des données...")
    X, Y = generate_sine_data(num_samples=2000, seq_len=SEQ_LEN)
    X_tensor = torch.from_numpy(X).unsqueeze(-1) # [N, seq, 1]
    Y_tensor = torch.from_numpy(Y) # [N, seq]
    
    # Train/Test split
    split = 1800
    X_train, X_test = X_tensor[:split], X_tensor[split:]
    Y_train, Y_test = Y_tensor[:split], Y_tensor[split:]
    
    print(f"   Dataset: {X_train.shape[0]} échantillons train, {X_test.shape[0]} test")
    print(f"   Longueur séquence: {SEQ_LEN}")
    print()

    # Modèle
    print(f"🧠 Création du modèle Liquid Neural Network...")
    model = LiquidRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    print(f"   Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Optimiseur & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"🏋️ Démarrage de l'entraînement ({EPOCHS} époques)...")
    start_time = time.time()

    history = {'train': [], 'test': []}

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        optimizer.zero_grad()
        
        # Forward
        preds = model(X_train)
        loss = criterion(preds, Y_train)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        history['train'].append(loss.item())

        # Test (toutes les 5 époques)
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test)
                test_loss = criterion(test_preds, Y_test)
                history['test'].append((epoch, test_loss.item()))
                
                print(f"   Epoch {epoch:>3}/{EPOCHS} | Train Loss: {loss.item():.5f} | Test Loss: {test_loss.item():.5f}")

    duration = time.time() - start_time
    print(f"\n⏱️ Entraînement terminé en {duration:.2f}s")

    # ============================================================================
    # 4. Résultats & Visualisation
    # ============================================================================
    
    print("\n📈 Analyse des résultats...")
    
    # Prédiction sur un échantillon de test
    model.eval()
    with torch.no_grad():
        sample_x = X_test[0:1] # [1, seq, 1]
        sample_y = Y_test[0]   # [seq]
        pred_y = model(sample_x)[0] # [seq]
        
        # Convert to numpy
        x_in = sample_x.squeeze().numpy()
        y_true = sample_y.numpy()
        y_pred = pred_y.numpy()
        
        mse = np.mean((y_true - y_pred)**2)
        print(f"   MSE sur l'échantillon: {mse:.5f}")
        
        # Sauvegarde / Affichage
        if HAS_PLOT:
            plt.figure(figsize=(10, 6))
            plt.plot(y_true, label='Vraie Valeur (Sinusoïde bruitée)', color='blue', marker='o', markersize=4)
            plt.plot(y_pred, label='Prédiction Liquid NN', color='red', linestyle='--', marker='x')
            plt.title("Prédiction de Série Temporelle par Liquid Neural Network")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = Path(__file__).parent.parent / "liquid_nn_result.png"
            plt.savefig(plot_path)
            print(f"   💾 Graphique sauvegardé: {plot_path}")
        else:
            # Fallback CSV
            csv_path = Path(__file__).parent.parent / "liquid_nn_result.csv"
            with open(csv_path, 'w') as f:
                f.write("Vrai,Predit\n")
                for t, p in zip(y_true, y_pred):
                    f.write(f"{t},{p}\n")
            print(f"   💾 Données sauvegardées: {csv_path}")
            print("   (Premières lignes)")
            print(f"   Vrai: {y_true[:5]}")
            print(f"   Pred: {y_pred[:5]}")

    print("\n" + "="*70)
    print(" ✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    print("="*70)

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
