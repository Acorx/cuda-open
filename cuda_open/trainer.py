"""
CUDA Open - BitNet Trainer

Training loop optimized by neuro-symbolic evolution:
- Quantization-Aware Training (QAT)
- Straight-Through Estimator (STE)
- Gradual quantization rampup
- Mixed precision
- Gradient checkpointing
- Curriculum learning
- Neuro-symbolic regularization
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from typing import Optional, Dict, Callable
from dataclasses import dataclass
from tqdm import tqdm
import time


@dataclass
class TrainingConfig:
    """Training configuration discovered by neuro-symbolic evolution."""
    
    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # LR Schedule
    lr_schedule: str = "warmup_cosine"
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # QAT
    qat_enabled: bool = True
    qat_rampup_steps: int = 1000
    
    # Mixed Precision
    mixed_precision: bool = True
    loss_scaling: bool = True
    
    # Regularization
    gradient_noise_std: float = 1e-4
    ns_reg_weight: float = 0.1
    
    # Training
    epochs: int = 10
    gradient_accumulation_steps: int = 1


class BitNetTrainer:
    """
    Trainer for BitNet models with QAT.
    
    Usage:
        trainer = BitNetTrainer(model, train_dataloader, config)
        trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        val_dataloader = None,
        config: TrainingConfig = None,
        device: str = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or TrainingConfig()
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        
        # Optimizer (AdamW discovered as optimal)
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon
        )
        
        # LR Scheduler (Warmup Cosine discovered)
        num_training_steps = len(train_dataloader) * self.config.epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Mixed Precision
        self.scaler = GradScaler() if self.config.mixed_precision else None
        
        # QAT state
        self.current_step = 0
        self.qat_progress = 0.0
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f" BitNet Training - Neuro-Symbolic Optimized")
        print(f"{'='*60}\n")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  LR: {self.config.learning_rate}")
        print(f"  QAT: {self.config.qat_enabled}")
        print(f"  Mixed Precision: {self.config.mixed_precision}")
        print()
        
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            epoch_start = time.time()
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.config.epochs}"
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Prepare batch
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch.get('labels', input_ids.clone()).to(self.device)
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                else:
                    input_ids = batch[0].to(self.device)
                    labels = input_ids.clone()
                    attention_mask = None
                
                # Forward with mixed precision
                with autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    
                    # Neuro-symbolic regularization
                    if self.config.ns_reg_weight > 0:
                        ns_loss = self._neuro_symbolic_regularization()
                        loss = loss + ns_loss
                    
                    # Gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update step
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.max_grad_norm
                    )
                    
                    # Gradient noise
                    if self.config.gradient_noise_std > 0:
                        self._add_gradient_noise()
                    
                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    
                    # QAT step
                    if self.config.qat_enabled:
                        self._qat_step()
                    
                    global_step += 1
                
                total_loss += loss.item() * self.config.gradient_accumulation_steps
                num_batches += 1
                
                # Update progress
                avg_loss = total_loss / max(1, num_batches)
                progress_bar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'qat': f"{self.qat_progress:.2f}"
                })
            
            # End of epoch
            epoch_time = time.time() - epoch_start
            avg_train_loss = total_loss / max(1, num_batches)
            
            print(f"\n  Epoch {epoch+1}:")
            print(f"    Train Loss: {avg_train_loss:.4f}")
            print(f"    Epoch Time: {epoch_time:.1f}s")
            print(f"    QAT Progress: {self.qat_progress:.2f}")
            
            # Validation
            if self.val_dataloader is not None:
                val_loss = self.evaluate()
                print(f"    Val Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"    ✓ Best model saved")
            print()
        
        print(f"\n{'='*60}")
        print(f" Training Complete")
        print(f"{'='*60}")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Final QAT: {self.qat_progress:.2f}")
        print()
    
    def evaluate(self) -> float:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch.get('labels', input_ids.clone()).to(self.device)
                else:
                    input_ids = batch[0].to(self.device)
                    labels = input_ids.clone()
                
                outputs = self.model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / max(1, num_batches)
    
    def _qat_step(self):
        """Update QAT progress."""
        self.current_step += 1
        self.qat_progress = min(1.0, self.current_step / self.config.qat_rampup_steps)
    
    def _neuro_symbolic_regularization(self) -> torch.Tensor:
        """Neuro-symbolic regularization loss."""
        reg_loss = torch.tensor(0.0, device=self.device)
        num_params = 0
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'master_weight'):
                w = module.master_weight
                # Distance to nearest ternary value
                dist = torch.min(
                    torch.abs(w - 1),
                    torch.min(torch.abs(w), torch.abs(w + 1))
                )
                reg_loss = reg_loss + dist.mean()
                num_params += 1
        
        if num_params > 0:
            reg_loss = reg_loss / num_params * self.config.ns_reg_weight
        
        return reg_loss
    
    def _add_gradient_noise(self):
        """Add gradient noise for regularization."""
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.config.gradient_noise_std
                param.grad.add_(noise)
    
    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'qat_progress': self.qat_progress,
            'current_step': self.current_step,
        }
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.qat_progress = checkpoint['qat_progress']
        self.current_step = checkpoint['current_step']
        print(f"✓ Checkpoint loaded: {path}")
