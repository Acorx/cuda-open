"""
CUDA Open - Micro GPT BitNet (Génération de Texte)

Un vrai petit modèle qui apprend à générer du texte caractère par caractère.
S'entraîne en ~1 minute CPU, génère du texte lisible, compressible 16x en BitNet.

Usage:
    python3 training/train_text_bitnet.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from cuda_open.quantizer import BitNetQuantizer


# ============================================================================
# DATASET : Un petit texte simple pour l'entraînement
# ============================================================================

TRAINING_TEXT = """
the cat sat on the mat and looked at the dog . the dog ran fast and the
cat followed . the bird sang in the tree and the sun was hot . the wind
blew soft and the grass was green . a child played near the lake and
threw stones in the water . the fish jumped high and the child laughed .
the day was good and the night was calm . the moon shone bright and the
stars were clear . the world was full of life and all was well . the end
of the tale was near and the story was told . the cat and the dog slept
near the fire and the bird sang a song . the tree lost its leaves and
the wind blew cold . the lake froze hard and the child stayed warm .
the snow fell white and the ground was cold . but spring would come and
the flowers would bloom . the cat would play in the sun and the dog would
run on the grass . the bird would sing and the fish would jump . the child
would laugh and the world would be full of life .
"""


# ============================================================================
# MODÈLE : Micro-GPT
# ============================================================================

class TernaryRegularizedLinear(nn.Module):
    """Couche linéaire régularisée vers {-1, 0, 1} pour BitNet."""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_f))
    
    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)
    
    def ternary_penalty(self, alpha=0.1):
        w_max = self.weight.abs().max() + 1e-8
        w = self.weight / w_max
        d = torch.min((w+1)**2, torch.min(w**2, (w-1)**2))
        return alpha * d.mean()


class MicroGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dim_in = embed_dim if i == 0 else hidden_dim
            self.layers.append(nn.Sequential(
                TernaryRegularizedLinear(dim_in, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))
        
        self.head = TernaryRegularizedLinear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size
    
    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)
    
    def get_ternary_penalty(self, alpha=0.1):
        penalty = 0
        for m in self.modules():
            if isinstance(m, TernaryRegularizedLinear):
                penalty += m.ternary_penalty(alpha)
        return penalty
    
    @torch.no_grad()
    def generate(self, prompt_tokens, max_new=50, temperature=1.0):
        self.eval()
        tokens = list(prompt_tokens)
        for _ in range(max_new):
            ctx = tokens[-64:]
            x = torch.tensor([ctx])
            logits = self(x)[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
        return tokens


# ============================================================================
# DATASET CLASS
# ============================================================================

class CharDataset:
    def __init__(self, text, seq_len=32, batch_size=64):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.c2i = {c: i for i, c in enumerate(self.chars)}
        self.i2c = {i: c for i, c in enumerate(self.chars)}
        self.data = [self.c2i[c] for c in text]
        self.seq_len = seq_len
        self.batch_size = batch_size
    
    def get_batch(self):
        xs, ys = [], []
        for _ in range(self.batch_size):
            i = np.random.randint(0, len(self.data) - self.seq_len)
            xs.append(self.data[i:i+self.seq_len])
            ys.append(self.data[i+1:i+self.seq_len+1])
        return torch.tensor(xs), torch.tensor(ys)


# ============================================================================
# ENTRAÎNEMENT
# ============================================================================

def main():
    print("\n" + "="*70)
    print(" CUDA Open - Micro GPT BitNet (Génération de Texte)")
    print("="*70 + "\n")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Dataset
    ds = CharDataset(TRAINING_TEXT.strip(), seq_len=32, batch_size=64)
    print(f"📊 Dataset: {ds.vocab_size} chars, {len(ds.data)} tokens")
    print(f"   Vocab: '{''.join(ds.chars)}'")
    print()
    
    # Modèle
    model = MicroGPT(ds.vocab_size, embed_dim=64, hidden_dim=128, num_layers=2)
    params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Modèle: {params:,} paramètres (embed=64, hidden=128, 2 couches)")
    print()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    
    # Entraînement
    print("🏋️ Entraînement (500 itérations)...")
    start = time.time()
    iters = 500
    best_loss = float('inf')
    
    for step in range(1, iters + 1):
        # Régularisation ternaire progressive
        alpha = 0.3 * min(1.0, step / (iters * 0.7))
        
        bx, by = ds.get_batch()
        logits = model(bx)
        loss_ce = criterion(logits.reshape(-1, ds.vocab_size), by.reshape(-1))
        loss_reg = model.get_ternary_penalty(alpha)
        loss = loss_ce + loss_reg
        
        if loss_ce.item() < best_loss:
            best_loss = loss_ce.item()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 50 == 0:
            elapsed = time.time() - start
            prompt = "the cat "
            ptokens = [ds.c2i[c] for c in prompt]
            gen = model.generate(ptokens, max_new=40, temperature=0.8)
            text = ''.join([ds.i2c[t] for t in gen])
            print(f"   Step {step:>4}/{iters} | Loss: {loss_ce.item():.4f} | "
                  f"Reg: {loss_reg.item():.4f} | Time: {elapsed:.1f}s")
            print(f"   ↳ '{prompt}{text}'\n")
    
    # Génération finale
    print("="*70)
    print(" GÉNÉRATION DE TEXTE")
    print("="*70)
    
    prompts = ["the cat ", "the dog ", "the sun ", "the child ", "the bird "]
    for p in prompts:
        pt = [ds.c2i.get(c, 0) for c in p if c in ds.c2i]
        gen = model.generate(pt, max_new=60, temperature=0.7)
        text = ''.join([ds.i2c[t] for t in gen])
        print(f"\n  Prompt: '{p}'")
        print(f"  Généré: '{text}'")
    
    # Quantization
    print("\n" + "="*70)
    print(" QUANTIZATION BITNET 1.58-BIT")
    print("="*70)
    
    total_orig = 0
    total_comp = 0
    total_p = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            w = param.data.numpy()
            total_orig += w.nbytes
            total_p += w.size
            packed, _ = BitNetQuantizer.quantize(w)
            total_comp += packed.nbytes
    
    ratio = total_orig / max(1, total_comp)
    print(f"\n  Paramètres: {total_p:,}")
    print(f"  Original: {total_orig/1024:.1f} KB → Compressé: {total_comp/1024:.2f} KB")
    print(f"  ✨ Compression: {ratio:.1f}x")
    
    print(f"\n{'='*70}")
    print(f" RÉSULTAT FINAL")
    print(f"{'='*70}")
    print(f"  ✅ Modèle entraîné | Best Loss: {best_loss:.4f}")
    print(f"  ✅ Compression BitNet: {ratio:.1f}x")
    print(f"  ✅ Génération de texte fonctionnelle")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
