"""
CUDA Open Compiler - Execution Engine

Interprète l'OpenIR et exécute le graphe sur CPU (ou via backend CUDA si dispo).
C'est le lien manquant entre la génération de code et l'exécution réelle.
"""

import numpy as np
import torch
from .openir import Module, Op, Value
from ..turbo_quant import turbo_quantize_pipeline
from ..advanced_attention import tri_attention, dflash_attention

class ExecutionEngine:
    """
    Exécute un module OpenIR en évaluant les opérations séquentiellement.
    Gère le cycle de vie des tenseurs.
    """
    def __init__(self, module: Module):
        self.module = module
        self.symbol_table = {} # Map Value -> numpy array

    def run(self, inputs: dict) -> dict:
        """
        Exécute le module avec les entrées données.
        inputs: dict mapping input names (e.g., 'A') to numpy arrays.
        Returns: dict mapping output names to numpy arrays.
        """
        # 1. Initialiser les entrées
        for block in self.module.region.blocks:
            for op in block.operations:
                for operand in op.operands:
                    if operand.name in inputs:
                        self.symbol_table[id(operand)] = inputs[operand.name]

        # 2. Exécuter les opérations dans l'ordre
        for block in self.module.region.blocks:
            for op in block.operations:
                self._execute_op(op)

        # 3. Récupérer les sorties
        outputs = {}
        if block.operations:
            last_op = block.operations[-1]
            for res in last_op.results:
                outputs[res.name] = self.symbol_table[id(res)]
                
        return outputs

    def _get_tensor(self, value: Value) -> np.ndarray:
        """Récupère le numpy array associé à une Value."""
        tid = id(value)
        if tid in self.symbol_table:
            return self.symbol_table[tid]
        raise ValueError(f"Tensor for {value.name} not found in symbol table")

    def _register_tensor(self, value: Value, tensor: np.ndarray):
        """Stocke le résultat dans la table."""
        self.symbol_table[id(value)] = tensor

    def _execute_op(self, op: Op):
        """Dispatch vers l'implémentation réelle de l'opération."""
        if op.name == "open.matmul":
            A = self._get_tensor(op.operands[0])
            B = self._get_tensor(op.operands[1])
            res = A @ B
            self._register_tensor(op.results[0], res)
            
        elif op.name == "open.add":
            A = self._get_tensor(op.operands[0])
            B = self._get_tensor(op.operands[1])
            res = A + B
            self._register_tensor(op.results[0], res)

        elif op.name == "open.cuda.turbo_quant_matmul":
            # C'est ici que la magie opère : on simule le kernel optimisé
            # Dans la réalité, ce serait un appel au binaire CUDA chargé dynamiquement
            A = self._get_tensor(op.operands[0])
            B = self._get_tensor(op.operands[1])
            Bias = self._get_tensor(op.operands[2])
            
            # Simulation du kernel TurboQuant:
            # 1. Quantization de B (les poids)
            # 2. MatMul
            # 3. Add Bias
            # Tout en une seule passe conceptuelle
            
            # Conversion vers torch pour notre pipeline TurboQuant
            B_torch = torch.from_numpy(B)
            
            # Quantization simulée (INT4)
            q_res = turbo_quantize_pipeline(B_torch, bits=4, use_rotation=False)
            
            # Reconstruction des poids déquantizés (Fake Quantization)
            # q_res['q_weights'] est en forme [Out, Groups, GroupSize]
            # On doit le remettre en forme [Out, In]
            B_dequant = (q_res['q_weights'].float() * q_res['scales'])
            
            # On utilise la shape originale de B pour reshaper
            original_shape = B.shape 
            B_fake_quant = B_dequant.view(original_shape).numpy()
            
            # Calcul
            res = A @ B_fake_quant + Bias
            self._register_tensor(op.results[0], res)

        else:
            raise NotImplementedError(f"Op {op.name} not implemented in Execution Engine")
