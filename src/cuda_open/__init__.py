"""
CUDA Open - Main Entry Point
"""

__version__ = "0.2.0-alpha"
__author__ = "Acorx Team"

def get_version():
    return __version__

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'jit':
        from .compiler import jit
        return jit
    if name == 'quantizer':
        from . import quantizer
        return quantizer
    if name == 'turbo_quant':
        from . import turbo_quant
        return turbo_quant
    if name == 'diffusion_accelerator':
        from . import diffusion_accelerator
        return diffusion_accelerator
    raise AttributeError(f"module 'cuda_open' has no attribute '{name}'")
