"""
CUDA Open Compiler - JIT Engine

Le décorateur `@cuda_open.jit` qui connecte le Python au Compilateur.
"""

import numpy as np
from typing import Callable
from .openir import Module, Region, Block, Op, Value, Type
from .lowering import TurboQuantFusionPass, MemoryPlanningPass, CodeGenPass

class JITFunction:
    """Fonction wrapper qui contient le compilé."""
    def __init__(self, func, module: Module, source_code: str):
        self.func = func
        self.module = module
        self.source_code = source_code
        
    def __call__(self, *args, **kwargs):
        print("⚡ Exécution du Kernel Compilé...")
        return self.func(*args, **kwargs)

def compile_function(func, example_args):
    """
    Compile une fonction Python en CUDA Open Module.
    """
    print(f"🔍 [Tracer] Analyse de {func.__name__}...")
    
    mod = Module(name=func.__name__)
    block = Block(parent=mod.region)
    mod.region.blocks.append(block)
    
    # Simulation du tracing pour la démo
    A = Value("A", Type("tensor", example_args[0].shape, "f32"))
    B = Value("B", Type("tensor", example_args[1].shape, "f32"))
    Bias = Value("Bias", Type("tensor", example_args[2].shape, "f32"))
    
    # Op MatMul
    C_temp = Value("C_temp", Type("tensor", (example_args[0].shape[0], example_args[1].shape[1]), "f32"))
    matmul_op = Op("open.matmul", operands=[A, B], results=[C_temp])
    block.add_op(matmul_op)
    
    # Op Add
    C_final = Value("C", Type("tensor", (example_args[0].shape[0], example_args[1].shape[1]), "f32"))
    add_op = Op("open.add", operands=[C_temp, Bias], results=[C_final])
    block.add_op(add_op)
    
    # Liaison des uses
    C_temp.uses.append(add_op)
    
    print("📥 IR Initial:")
    mod.dump()
    print()
    
    # Compilation Passes
    TurboQuantFusionPass().run(mod)
    MemoryPlanningPass().run(mod)
    
    print("📤 IR Optimisé:")
    mod.dump()
    print()
    
    # Code Gen
    cuda_source = CodeGenPass().run(mod)
    print("💻 Code Généré:")
    print(cuda_source)
    print()
    
    return JITFunction(func, mod, cuda_source)

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
