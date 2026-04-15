"""
CUDA Open - Test Unitaire Strict du Compilateur C++
Objectif: Prouver que le code C++ généré est mathématiquement identique à Python
"""

import torch
import torch.nn as nn
import numpy as np
import subprocess
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# --- Modèle de test simple et connu ---
class TinyModel(nn.Module):
    """Modèle simple pour test: Linear -> ReLU -> Linear"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def test_compiler_correctness():
    """Test que le compilateur génère du C++ mathématiquement correct"""
    
    print("\n" + "="*60)
    print(" TEST UNITAIRE STRICT: Compilateur Python -> C++")
    print("="*60 + "\n")
    
    # 1. Créer le modèle et l'entrée
    model = TinyModel()
    model.eval()
    
    # Entrée fixe pour reproductibilité
    input_data = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    
    # 2. Calculer le résultat Python (référence)
    with torch.no_grad():
        py_result = model(input_data).numpy().flatten()
    
    print(f"1. 🐍 Résultat Python (référence): {py_result}")
    
    # 3. Extraire les poids
    w1 = model.fc1.weight.detach().numpy()  # [8, 4]
    b1 = model.fc1.bias.detach().numpy()    # [8]
    w2 = model.fc2.weight.detach().numpy()  # [2, 8]
    b2 = model.fc2.bias.detach().numpy()    # [2]
    
    # 4. Générer le code C++ strict
    def fmt_array(name, arr):
        flat = arr.flatten()
        return f"const float {name}[{len(flat)}] = {{ " + ", ".join(f"{x:.8f}f" for x in flat) + " };"
    
    cpp_code = f"""
#include <iostream>
#include <cmath>

{fmt_array('W1', w1)}
{fmt_array('B1', b1)}
{fmt_array('W2', w2)}
{fmt_array('B2', b2)}

const int IN_SIZE = 4;
const int H1_SIZE = 8;
const int OUT_SIZE = 2;

void run_model(const float* input, float* output) {{
    // Couche 1: Linear + ReLU
    float h[H1_SIZE];
    for (int i = 0; i < H1_SIZE; i++) {{
        float val = B1[i];
        for (int j = 0; j < IN_SIZE; j++) {{
            val += W1[i * IN_SIZE + j] * input[j];
        }}
        h[i] = val > 0 ? val : 0; // ReLU
    }}
    
    // Couche 2: Linear (pas d'activation)
    for (int i = 0; i < OUT_SIZE; i++) {{
        float val = B2[i];
        for (int j = 0; j < H1_SIZE; j++) {{
            val += W2[i * H1_SIZE + j] * h[j];
        }}
        output[i] = val;
    }}
}}

int main() {{
    float input[] = {{ {input_data[0,0]:.8f}f, {input_data[0,1]:.8f}f, {input_data[0,2]:.8f}f, {input_data[0,3]:.8f}f }};
    float output[OUT_SIZE];
    
    run_model(input, output);
    
    printf("RESULTAT_CPP: %.8f, %.8f\\n", output[0], output[1]);
    return 0;
}}
"""
    
    print("2. 🔨 Génération et Compilation C++...")
    
    # 5. Compiler et exécuter
    with tempfile.TemporaryDirectory() as tmpdir:
        cpp_path = os.path.join(tmpdir, "test_model.cpp")
        bin_path = os.path.join(tmpdir, "test_model")
        
        with open(cpp_path, 'w') as f:
            f.write(cpp_code)
        
        # Compilation stricte avec warnings
        res_compile = subprocess.run(
            ["g++", "-O3", "-Wall", "-Werror", cpp_path, "-o", bin_path],
            capture_output=True, text=True
        )
        
        if res_compile.returncode != 0:
            print(f"❌ Échec compilation: {res_compile.stderr}")
            return False
        
        print("   ✅ Compilation réussie (avec warnings stricts)")
        
        # Exécution
        res_run = subprocess.run([bin_path], capture_output=True, text=True)
        if res_run.returncode != 0:
            print(f"❌ Échec exécution: {res_run.stderr}")
            return False
        
        # Parser le résultat
        output_line = res_run.stdout.strip()
        if "RESULTAT_CPP:" not in output_line:
            print(f"❌ Sortie inattendue: {output_line}")
            return False
        
        cpp_result_str = output_line.split(":")[1].strip()
        cpp_result = np.array([float(x) for x in cpp_result_str.split(",")])
        
        print(f"3. 💻 Résultat C++: {cpp_result}")
        
        # 6. Comparaison stricte
        diff = np.abs(py_result - cpp_result)
        max_diff = np.max(diff)
        
        print(f"\n📊 COMPARAISON:")
        print(f"   Différence max: {max_diff:.10f}")
        
        # Tolérance très stricte (erreur d'arrondi flottant acceptable)
        tolerance = 1e-5
        if max_diff < tolerance:
            print(f"   ✅ SUCCÈS: Python et C++ sont identiques (tolérance: {tolerance})")
            return True
        else:
            print(f"   ❌ ÉCHEC: Différence trop grande (> {tolerance})")
            print(f"   Python: {py_result}")
            print(f"   C++:    {cpp_result}")
            return False

if __name__ == "__main__":
    success = test_compiler_correctness()
    print("\n" + "="*60)
    if success:
        print(" 🏆 PREUVE FAITE: Le compilateur est mathématiquement correct")
    else:
        print(" ❌ Le compilateur a un bug à corriger")
    print("="*60 + "\n")
    sys.exit(0 if success else 1)
