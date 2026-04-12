"""
CUDA Open - PyTorch Module

Revolutionary computing framework via neuro-symbolic evolution.

Usage:
    import cuda_open
    model = cuda_open.BitNetForCausalLM.from_pretrained("gpt2")
    model.quantize_to_bitnet()
    output = model.generate("Hello")
"""

__version__ = "0.1.0"
__author__ = "CUDA Open Team"

# Lazy imports to avoid loading heavy dependencies
def __getattr__(name):
    if name == "BitNetLinear":
        from cuda_open.bitnet_linear import BitNetLinear
        return BitNetLinear
    elif name == "StraightThroughEstimator":
        from cuda_open.bitnet_linear import StraightThroughEstimator
        return StraightThroughEstimator
    elif name == "BitNetForCausalLM":
        from cuda_open.bitnet_model import BitNetForCausalLM
        return BitNetForCausalLM
    elif name == "BitNetConfig":
        from cuda_open.bitnet_model import BitNetConfig
        return BitNetConfig
    elif name == "BitNetQuantizer":
        from cuda_open.quantizer import BitNetQuantizer
        return BitNetQuantizer
    elif name == "quantize_model":
        from cuda_open.quantizer import quantize_model
        return quantize_model
    elif name == "dequantize_model":
        from cuda_open.quantizer import dequantize_model
        return dequantize_model
    elif name == "BitNetTrainer":
        from cuda_open.trainer import BitNetTrainer
        return BitNetTrainer
    elif name == "TrainingConfig":
        from cuda_open.trainer import TrainingConfig
        return TrainingConfig
    raise AttributeError(f"module 'cuda_open' has no attribute '{name}'")


__all__ = [
    "BitNetLinear",
    "StraightThroughEstimator",
    "BitNetForCausalLM",
    "BitNetConfig",
    "BitNetQuantizer",
    "quantize_model",
    "dequantize_model",
    "BitNetTrainer",
    "TrainingConfig",
]
