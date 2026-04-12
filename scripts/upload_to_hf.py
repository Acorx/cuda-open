"""
Script pour créer un modèle BitNet de démo et préparer les fichiers pour HuggingFace.
100% numpy, pas de torch, pas de téléchargement lourd.
"""
import numpy as np
import json
from pathlib import Path

def main():
    print("\n" + "="*60)
    print(" CUDA Open - HuggingFace Model Creator")
    print("="*60 + "\n")

    # 1. Création d'un petit modèle "dummy"
    print("1. Création d'un petit modèle de démo...")
    np.random.seed(42)
    
    # Simule un modèle avec quelques poids (comme un mini GPT)
    dummy_weights = {
        'transformer.h.0.attn.c_attn.weight': np.random.randn(256, 768).astype(np.float32),
        'transformer.h.0.attn.c_proj.weight': np.random.randn(768, 256).astype(np.float32),
        'transformer.h.0.mlp.c_fc.weight': np.random.randn(768, 3072).astype(np.float32),
        'transformer.h.0.mlp.c_proj.weight': np.random.randn(3072, 768).astype(np.float32),
        'transformer.h.1.attn.c_attn.weight': np.random.randn(256, 768).astype(np.float32),
        'lm_head.weight': np.random.randn(50257, 768).astype(np.float32),
    }
    
    total_params = sum(w.size for w in dummy_weights.values())
    print(f"   ✓ Modèle créé ({len(dummy_weights)} couches, {total_params:,} paramètres)")

    # 2. Quantization BitNet
    print("\n2. Quantization BitNet 1.58-bit...")
    
    from cuda_open.quantizer import BitNetQuantizer
    
    quantized_data = {}
    original_total = 0
    compressed_total = 0

    for name, weight in dummy_weights.items():
        original_total += weight.nbytes
        packed, scale = BitNetQuantizer.quantize(weight)
        compressed_total += packed.nbytes
        
        quantized_data[name] = {
            'packed': packed.tolist(),
            'scale': float(scale),
            'shape': list(weight.shape)
        }
        print(f"   ✓ {name.split('.')[-1]}: {weight.nbytes/1024:.1f}KB -> {packed.nbytes/1024:.1f}KB")

    ratio = original_total / compressed_total
    print(f"\n   ✨ Compression Totale: {ratio:.1f}x")
    print(f"   Original: {original_total/1024/1024:.2f} MB")
    print(f"   Compressé: {compressed_total/1024/1024:.2f} MB")

    # 3. Sauvegarde des fichiers
    print("\n3. Préparation des fichiers pour HuggingFace...")
    output_dir = Path(__file__).parent.parent / "model-bitnet-demo"
    output_dir.mkdir(exist_ok=True)

    # Sauvegarde les poids quantizés (format numpy pour HF)
    np.savez_compressed(output_dir / "quantized_weights.npz", **quantized_data)
    
    # Config du modèle
    config = {
        "model_type": "bitnet_demo",
        "architectures": ["BitNetForCausalLM"],
        "quantization": "bitnet_1.58",
        "compression_ratio": f"{ratio:.1f}x",
        "layers": 2,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "vocab_size": 50257
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # README pour HuggingFace
    readme = f"""---
license: mit
tags:
- bitnet
- quantization
- cuda-open
- 1.58-bit
- efficient-inference
---

# CUDA Open - BitNet Demo Model

Modèle de démonstration quantized en **BitNet 1.58-bit** via [CUDA Open](https://github.com/Acorx/cuda-open).

## 📊 Stats
- **Compression:** {ratio:.1f}x vs FP32
- **Taille Originale:** {original_total/1024/1024:.2f} MB
- **Taille Compressée:** {compressed_total/1024/1024:.2f} MB
- **Paramètres:** {total_params:,}

## 🚀 Utilisation

```python
pip install cuda-open numpy
```

```python
import numpy as np
from cuda_open.quantizer import BitNetQuantizer

# Charger les poids
data = np.load("quantized_weights.npz", allow_pickle=True)

# Reconstruire un poids
name = "lm_head.weight"
info = data[name].item()
packed = np.array(info['packed'], dtype=np.uint8)
shape = tuple(info['shape'])
scale = info['scale']

# Déquantizer
unpacked = BitNetQuantizer.unpack(packed, int(np.prod(shape)), scale)
weights = unpacked.reshape(shape)
print("Exemple OK")
```

## 📖 Plus d'infos
- Repo GitHub : https://github.com/Acorx/cuda-open
- Documentation : https://github.com/Acorx/cuda-open#readme
"""
    with open(output_dir / "README.md", 'w') as f:
        f.write(readme)

    print(f"   ✓ Fichiers sauvés dans: {output_dir}")
    print(f"   Contenu: {list(output_dir.glob('*'))}")
    
    print("\n✅ MODÈLE PRÊT POUR UPLOAD !")
    print(f"\nPour uploader sur HuggingFace, tape:")
    print(f"   huggingface-cli upload Acorx/cuda-open-bitnet-demo \"{output_dir}\"/")

if __name__ == "__main__":
    main()
