"""
CUDA Open Compiler - JIT Engine

Le décorateur `@cuda_open.jit` qui connecte le Python au Compilateur.
"""

import numpy as np
from typing import Callable
from .openir import Module, Region, Block, Op, Value, Type
from .lowering import TurboQuantFusionPass, MemoryPlanningPass, CodeGenPass
from .execution import ExecutionEngine

class JITFunction:
    """Fonction wrapper qui contient le compilé."""
    def __init__(self, func, engine: ExecutionEngine, source_code: str):
        self.func = func
        self.engine = engine
        self.source_code = source_code
        
    def __call__(self, *args, **kwargs):
        # Mapping des arguments vers le graphe IR
        inputs = {}
        # On récupère les noms des inputs depuis le module
        for block in self.engine.module.region.blocks:
            for op in block.operations:
                for operand in op.operands:
                    # Si l'operand n'a pas d'opérateur défini, c'est une entrée
                    if operand.defining_op is None:
                        # On mappe par position (très simplifié)
                        idx = len([k for k in inputs.keys()])
                        if idx < len(args):
                            inputs[operand.name] = args[idx]
                            
        results = self.engine.run(inputs)
        
        # Retourner le résultat (supposons une seule sortie pour l'instant)
        if results:
            return list(results.values())[0]
        return None

def compile_function(func, args):
    """
    Compile une fonction Python en CUDA Open Module + ExecutionEngine.
    """
    print(f"🔍 [Tracer] Analyse de {func.__name__}...")
    
    # On suppose ici que la fonction est de type: return (A @ B) + Bias
    # C'est une simplification pour la preuve de concept du compilateur
    M, K = args[0].shape
    K_, N = args[1].shape
    has_bias = len(args) == 3
    
    graph = Module(name=func.__name__)
    block = Block(parent=graph.region)
    graph.region.blocks.append(block)
    
    # Création des nœuds IR
    A = Value("A", Type("tensor", args[0].shape, "f32"))
    B = Value("B", Type("tensor", args[1].shape, "f32"))
    
    # Op MatMul
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
        
    # 1. Affichage IR Initial
    print("📥 IR Initial:")
    graph.dump()
    print()
    
    # 2. Optimisation (Fusion MatMul + Add -> TurboQuant)
    TurboQuantFusionPass().run(graph)
    
    # 3. Affichage IR Optimisé
    print("📤 IR Optimisé (Après Fusion):")
    graph.dump()
    print()
    
    # 4. Génération de code CUDA C++ (Preuve du concept)
    source_code = CodeGenPass().run(graph)
    print("💻 Code Généré:")
    print(source_code)
    print()
    
    # 5. Création du moteur d'exécution (Le "Runtime")
    engine = ExecutionEngine(graph)
    
    return JITFunction(func, engine, source_code)

def jit(func: Callable = None):
    """
    Décorateur pour compiler automatiquement une fonction.
    Usage:
        @cuda_open.jit
        def my_kernel(A, B):
            return A @ B
    """
    def decorator(f):
        class LazyCompiler:
            def __init__(self, func):
                self.func = func
                self.jit_func = None
            
            def __call__(self, *args, **kwargs):
                if self.jit_func is None:
                    print(f"🚀 [JIT] Première appel: Compilation de {self.func.__name__}...")
                    self.jit_func = compile_function(self.func, args)
                return self.jit_func(*args, **kwargs)
        
        return LazyCompiler(f)
    
    if func is None:
        return decorator
    return decorator(func)
