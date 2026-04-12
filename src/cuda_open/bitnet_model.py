"""
CUDA Open - BitNet Model Wrapper

Wrap any HuggingFace transformer model with BitNet quantization.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from cuda_open.quantizer import BitNetQuantizer
from cuda_open.bitnet_linear import BitNetLinear


class BitNetConfig:
    """Configuration for BitNet model."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        quantize: bool = True,
        use_ste: bool = True,
        qat_enabled: bool = False,
        qat_rampup_steps: int = 1000,
        **kwargs
    ):
        self.model_name = model_name
        self.quantize = quantize
        self.use_ste = use_ste
        self.qat_enabled = qat_enabled
        self.qat_rampup_steps = qat_rampup_steps
        self.extra_kwargs = kwargs


class BitNetForCausalLM(nn.Module):
    """
    BitNet wrapper for causal language models.
    
    Usage:
        # Load from pretrained
        model = BitNetForCausalLM.from_pretrained("gpt2")
        
        # Quantize to BitNet
        model.quantize_to_bitnet()
        
        # Generate text
        output = model.generate("Hello", max_length=50)
        
        # Train with QAT
        model.train()
        loss = model(input_ids, labels=labels).loss
        loss.backward()
    """
    
    def __init__(self, model: nn.Module, config: BitNetConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.quantized = False
        self.quantization_stats = None
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> 'BitNetForCausalLM':
        """Load a pretrained model and wrap it with BitNet."""
        config = BitNetConfig(model_name=model_name, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return cls(model, config)
    
    def quantize_to_bitnet(self, verbose: bool = True) -> Dict:
        """Quantize all linear layers to BitNet 1.58-bit."""
        quantizer = BitNetQuantizer()
        self.quantization_stats = quantizer.quantize_model(
            self.model, verbose=verbose
        )
        self.quantized = True
        return self.quantization_stats
    
    def forward(self, **kwargs) -> Any:
        """Forward pass through the model."""
        return self.model(**kwargs)
    
    def generate(
        self,
        prompt: str,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 50,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        if hasattr(self.model, 'device'):
            input_ids = input_ids.to(self.model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                **kwargs
            )
        
        # Decode
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    def get_model_size(self) -> Dict:
        """Get model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        
        if self.quantized and self.quantization_stats:
            compressed_size = self.quantization_stats['compressed_size_bytes']
            original_size = self.quantization_stats['original_size_bytes']
        else:
            original_size = total_params * 4  # FP32
            compressed_size = total_params * 0.25  # Approximate 1.58-bit
        
        return {
            'total_params': total_params,
            'original_size_mb': original_size / 1024 / 1024,
            'compressed_size_mb': compressed_size / 1024 / 1024,
            'compression_ratio': original_size / max(1, compressed_size),
            'quantized': self.quantized
        }
    
    def __getattr__(self, name):
        """Delegate unknown attributes to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
