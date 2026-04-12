"""
CUDA Open - BitNet Linear Layer

Linear layer with BitNet 1.58-bit ternary weights.
Uses Straight-Through Estimator (STE) for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-Through Estimator for ternary quantization.
    Forward: quantize to {-1, 0, 1}
    Backward: pass gradient directly (as if identity)
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.sign(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # STE: gradient passes directly through
        return grad_output


class BitNetLinear(nn.Module):
    """
    BitNet 1.58-bit linear layer.
    
    Keeps master weights in FP32 for gradients,
    but forward pass uses ternary weights {-1, 0, 1}.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias
        quantize: Whether to quantize weights during forward
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quantize: bool = True
    ):
        super().__init__()
        
        # Master weights (FP32 for gradients)
        self.master_weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02
        )
        
        # Quantization scale
        self.register_buffer('scale', torch.tensor(1.0))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Configuration
        self.in_features = in_features
        self.out_features = out_features
        self.quantize = quantize
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize and self.training:
            # Ternarize with STE during training
            weight = StraightThroughEstimator.apply(self.master_weight)
            weight = weight * self.scale
        elif self.quantize:
            # Inference: use quantized weights
            weight = torch.sign(self.master_weight) * self.scale
        else:
            # No quantization (for comparison)
            weight = self.master_weight
        
        return F.linear(x, weight, self.bias)
    
    def quantize_weights(self):
        """Quantize weights and compute scale."""
        ternary, scale = self._quantize_tensor(self.master_weight)
        self.scale.data = scale
        self.master_weight.data = ternary * scale
    
    def _quantize_tensor(self, tensor: torch.Tensor) -> tuple:
        """Quantize tensor to ternary."""
        scale = tensor.abs().max()
        if scale < 1e-8:
            scale = torch.tensor(1.0, device=tensor.device)
        
        threshold = 0.2 * scale
        ternary = torch.zeros_like(tensor)
        ternary[tensor > threshold] = 1.0
        ternary[tensor < -threshold] = -1.0
        
        return ternary, scale
    
    def extra_repr(self):
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}, '
                f'quantize={self.quantize}')
