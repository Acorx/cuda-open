"""
CUDA Open Compiler - JIT Engine (Multi-Target Aware)

Connecte le Python au Compilateur et à l'Exécuteur.
Gère la génération de code pour CUDA, HIP, SYCL et CPU.
"""

import numpy as np
import inspect
from typing import Callable, Dict
from .openir import Module, Region, Block, Op, Value, Type
from .lowering import TurboQuantFusionPass, MemoryPlanningPass, CodeGenPass
from .execution import ExecutionEngine

class JITFunction:
    """Fonction wrapper qui contient le compilé."""
    def __init__(self, func, engine: ExecutionEngine, sources: Dict[str, str]):
        self.func = func
        self.engine = engine
        self.sources = sources # Stocke les codes sources générés
        
    def __call__(self, *args, **kwargs):
        # Exécution via le moteur IR (Fallback CPU pour la démo)
        inputs = {}
        for block in self.engine.module.region.blocks:
            for op in block.operations:
                for operand in op.operands:
                    if operand.defining_op is None:
                        idx = len(inputs)
                        if idx < len(args):
                            inputs[operand.name] = args[idx]
                            
        return self.engine.run(inputs)

def compile_function(func, args):
    """
    Trace et compile la fonction en IR, optimise, et génère le code pour toutes les cibles.
    """
    print(f"🔍 [Tracer] Compilation de {func.__name__}...")
    
    # 1. Création de l'IR (Simulation simplifiée du traçage)
    M, K = args[0].shape
    K_, N = args[1].shape
    has_bias = len(args) == 3
    
    mod = Module(name=func.__name__)
    block = Block(parent=mod.region)
    mod.region.blocks.append(block)
    
    # Nœuds
    A = Value("A", Type("tensor", args[0].shape, "f32"))
    B = Value("B", Type("tensor", args[1].shape, "f32"))
    C_temp = Value("C_temp", Type("tensor", (M, N), "f32"))
    
    matmul_op = Op("open.matmul", operands=[A, B], results=[C_temp])
    C_temp.defining_op = matmul_op
    A.uses.append(matmul_op)
    B.uses.append(matmul_op)
    block.add_op(matmul_op)
    
    if has_bias:
        Bias = Value("Bias", Type("tensor", args[2].shape, "f32"))
        C_final = Value("C", Type("tensor", (M, N), "f32"))
        add_op = Op("open.add", operands=[C_temp, Bias], results=[C_final])
        C_final.defining_op = add_op
        C_temp.uses.append(add_op)
        Bias.uses.append(add_op)
        block.add_op(add_op)
    
    # 2. Affichage IR Initial
    print("📥 IR Initial:")
    mod.dump()
    print()
    
    # 3. Passes d'optimisation
    TurboQuantFusionPass().run(mod)
    MemoryPlanningPass().run(mod)
    
    # 4. Affichage IR Optimisé
    print("📤 IR Optimisé (Après Fusion):")
    mod.dump()
    print()
    
    # 5. Code Gen Multi-Target
    sources = CodeGenPass().run(mod)
    
    print("💻 Codes Sources Générés:")
    for target, code in sources.items():
        print(f"   🎯 {target}: {len(code)} chars")
    print()
    
    # 6. Création du moteur d'exécution
    engine = ExecutionEngine(mod)
    
    print("✅ Compilation terminée (Prêt pour déploiement multi-cibles).")
    
    return JITFunction(func, engine, sources)

class LazyCompiler:
    def __init__(self, func):
        self.func = func
        self.compiled_func = None
    
    def __call__(self, *args, **kwargs):
        if self.compiled_func is None:
            print(f"🚀 [JIT] Première appel: Compilation de {self.func.__name__}...")
            self.compiled_func = compile_function(self.func, args)
        
        return self.compiled_func(*args, **kwargs)

def jit(func: Callable = None):
    if func is None:
        return lambda f: LazyCompiler(f)
    return LazyCompiler(func)
