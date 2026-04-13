"""
CUDA Open - Comparaison : Liquid NN vs LSTM vs CNN 1D

Ce script entraîne 3 architectures sur la même tâche de prédiction de série temporelle.
Objectif : Comparer la performance, la vitesse et la taille des modèles.

Modèles testés :
1. Liquid Neural Network (LTC) - Bio-inspiré, dynamique
2. LSTM Classique - Standard industrie
3. CNN 1D - Simple et efficace

Usage:
    python3 training/train_classic_models.py
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
from pathlib import Path
from dataclasses import dataclass

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("⚠️ Matplotlib non trouvé. Résultats sauvegardés en CSV.")

print("\n" + "="*70)
print(" CUDA Open - Comparaison : Liquid vs LSTM vs CNN")
print("="*70 + "\n")

# ============================================================================
# 1. Données Communes
# ============================================================================

def generate_sine_data(num_samples=2000, seq_len=30):
    """Génère des sinusoïdes bruitées pour l'entraînement."""
    time_steps = np.linspace(0, 20, num_samples + seq_len + 10)
    signal = np.sin(time_steps) + np.random.normal(0, 0.1, len(time_steps))
    
    x_data, y_data = [], []
    for i in range(num_samples):
        x_data.append(signal[i : i + seq_len])
        y_data.append(signal[i + 1 : i + seq_len + 1])
        
    return np.array(x_data).astype(np.float32), np.array(y_data).astype(np.float32)

# ============================================================================
# 2. Définitions des Modèles
# ============================================================================

# A. Liquid Neural Network (LTC)
class LTCCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(input_size, hidden_size)
        self.W_tau = nn.Linear(hidden_size, hidden_size)
        self.U_tau = nn.Linear(input_size, hidden_size)
        self.act = nn.Tanh()

    def forward(self, input, hidden):
        state = self.act(self.W(hidden) + self.U(input))
        tau_inv = torch.sigmoid(self.W_tau(hidden) + self.U_tau(input))
        dh = tau_inv * (-hidden + state)
        return hidden + dh

class LiquidRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = LTCCell(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, x):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_size).to(x.device)
        outputs = []
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
            outputs.append(self.fc_out(h))
        return torch.stack(outputs, dim=1).squeeze(-1)

# B. LSTM Classique
class ClassicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out).squeeze(-1)

# C. CNN 1D
class CNN1D(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        # Conv -> Pool -> Conv -> FC
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        
        # Calcul de la taille après pooling: seq_len / 2
        self.fc1 = nn.Linear(32 * (seq_len // 2), seq_len)
        self.fc2 = nn.Linear(seq_len, seq_len)
        
    def forward(self, x):
        # x: [batch, seq, 1] -> [batch, 1, seq]
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ============================================================================
# 3. Boucle d'Entraînement Générique
# ============================================================================

@dataclass
class ModelResult:
    name: str
    params: int
    train_time: float
    train_loss: float
    test_loss: float
    sample_mse: float

def train_model(model, X_train, Y_train, X_test, Y_test, name, epochs=30, lr=0.01):
    print(f"\n🏋️ Entraînement de {name}...")
    print(f"   Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    start = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, Y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test)
                test_loss = criterion(test_preds, Y_test)
                print(f"   Epoch {epoch:>3}/{epochs} | Train: {loss.item():.5f} | Test: {test_loss.item():.5f}")
    
    train_time = time.time() - start
    
    # Test sur un échantillon
    model.eval()
    with torch.no_grad():
        sample_x = X_test[0:1]
        sample_y = Y_test[0]
        pred_y = model(sample_x)[0]
        sample_mse = float(torch.mean((sample_y - pred_y)**2))
    
    final_test_loss = float(criterion(model(X_test), Y_test))
    
    return ModelResult(
        name=name,
        params=sum(p.numel() for p in model.parameters()),
        train_time=train_time,
        train_loss=float(loss),
        test_loss=final_test_loss,
        sample_mse=sample_mse
    )

# ============================================================================
# 4. Exécution de la Comparaison
# ============================================================================

def main():
    # Données
    SEQ_LEN = 30
    print(f"📊 Préparation des données (SeqLen={SEQ_LEN})...")
    X, Y = generate_sine_data(num_samples=2000, seq_len=SEQ_LEN)
    X_tensor = torch.from_numpy(X).unsqueeze(-1) # [N, seq, 1]
    Y_tensor = torch.from_numpy(Y) # [N, seq]
    
    split = 1800
    X_train, X_test = X_tensor[:split], X_tensor[split:]
    Y_train, Y_test = Y_tensor[:split], Y_tensor[split:]
    print(f"   Dataset: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print()

    results = []

    # 1. Liquid NN
    print("="*70)
    print(" MODÈLE 1: Liquid Neural Network")
    print("="*70)
    liquid_model = LiquidRNN(input_size=1, hidden_size=16)
    results.append(train_model(liquid_model, X_train, Y_train, X_test, Y_test, "Liquid NN", epochs=40, lr=0.01))

    # 2. LSTM Classique
    print("\n" + "="*70)
    print(" MODÈLE 2: LSTM Classique")
    print("="*70)
    lstm_model = ClassicLSTM(input_size=1, hidden_size=16)
    results.append(train_model(lstm_model, X_train, Y_train, X_test, Y_test, "LSTM", epochs=40, lr=0.01))

    # 3. CNN 1D
    print("\n" + "="*70)
    print(" MODÈLE 3: CNN 1D")
    print("="*70)
    cnn_model = CNN1D(seq_len=SEQ_LEN)
    results.append(train_model(cnn_model, X_train, Y_train, X_test, Y_test, "CNN 1D", epochs=40, lr=0.005))

    # ============================================================================
    # 5. Tableau Comparatif Final
    # ============================================================================
    
    print("\n" + "="*70)
    print(" 📊 RÉSULTATS COMPARATIFS")
    print("="*70)
    
    print(f"\n  {'Modèle':<12} {'Params':>7} {'Temps':>7} {'TrainLoss':>10} {'TestLoss':>10} {'MSE':>7}")
    print(f"  {'-'*60}")
    for r in results:
        print(f"  {r.name:<12} {r.params:>7,} {r.train_time:>6.1f}s {r.train_loss:>10.5f} {r.test_loss:>10.5f} {r.sample_mse:>6.4f}")
    
    # Trouver le meilleur
    best = min(results, key=lambda x: x.test_loss)
    fastest = min(results, key=lambda x: x.train_time)
    smallest = min(results, key=lambda x: x.params)
    
    print(f"\n  🏆 Meilleure Précision : {best.name}")
    print(f"  ⚡ Le plus rapide : {fastest.name} ({fastest.train_time:.1f}s)")
    print(f"  🐦 Le plus léger : {smallest.name} ({smallest.params:,} params)")
    
    # Sauvegarde des résultats
    csv_path = Path(__file__).parent.parent / "model_comparison_results.csv"
    with open(csv_path, 'w') as f:
        f.write("Model,Params,Time,TrainLoss,TestLoss,SampleMSE\n")
        for r in results:
            f.write(f"{r.name},{r.params},{r.train_time:.2f},{r.train_loss:.5f},{r.test_loss:.5f},{r.sample_mse:.5f}\n")
    print(f"\n  💾 Résultats sauvegardés: {csv_path}")
    
    print("\n" + "="*70)
    print(" ✅ COMPARAISON TERMINÉE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
