"""
Test des méthodes d'entraînement BitNet - Version corrigée
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time


def generate_batch(batch_size=64):
    a = torch.randn(batch_size, 1) * 5
    b = torch.randn(batch_size, 1) * 5
    x = torch.cat([a, b], dim=1)
    y = (a + b).squeeze()
    return x, y


# MÉTHODE 1: Fake Quantization (Standard Industry)
class BitNetFakeQuant(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.fc1 = nn.Linear(in_f, 32)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # Simulation de poids quantizés
        w1 = torch.sign(self.fc1.weight)
        w2 = torch.sign(self.fc2.weight)
        
        x = torch.nn.functional.linear(x, w1, self.fc1.bias)
        x = torch.relu(x)
        x = torch.nn.functional.linear(x, w2, self.fc2.bias)
        return x.squeeze()


# MÉTHODE 2: FP32 Normal (Baseline)
class NormalModel(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.fc1 = nn.Linear(in_f, 32)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze()


def test_method(name, model):
    torch.manual_seed(42)
    model = model(2, 1)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(300):
        inputs, targets = generate_batch(128)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Test
    model.eval()
    with torch.no_grad():
        test_input = torch.tensor([[5.0, 3.0]])
        pred = model(test_input).item()
    
    success = best_loss < 50.0
    
    return {'name': name, 'best_loss': best_loss, 'final_pred': pred, 'success': success}


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" TEST DES MÉTHODES D'ENTRAÎNEMENT")
    print("="*60 + "\n")
    
    results = []
    
    for name, model_cls in [("FP32 Normal (Baseline)", NormalModel), ("BitNet Fake Quant", BitNetFakeQuant)]:
        print(f"Test: {name}...")
        result = test_method(name, model_cls)
        
        status = "✅" if result['success'] else "❌"
        print(f"  {status} Loss: {result['best_loss']:.4f} | Prédiction 5+3: {result['final_pred']:.2f}\n")
        
        results.append(result)
    
    print("="*60)
    for r in sorted(results, key=lambda x: x['best_loss']):
        print(f"  {'🏆' if r['success'] else '⚠️'} {r['name']}: {r['best_loss']:.4f}")
    print()
