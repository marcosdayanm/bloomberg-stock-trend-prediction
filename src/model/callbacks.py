"""Custom callbacks for training monitoring and anti-overfitting."""

import lightning as L
from lightning.pytorch.callbacks import Callback
import torch


class OverfittingMonitor(Callback):
    """
    Monitor train/val accuracy gap and prevent severe overfitting.
    
    Features:
    - Logs train/val accuracy gap
    - Warns when gap exceeds threshold
    - Can stop training if gap is too large
    """
    
    def __init__(self, max_gap: float = 0.10, stop_on_large_gap: bool = False):
        """
        Args:
            max_gap: Maximum allowed gap between train and val accuracy (default 10%)
            stop_on_large_gap: Whether to stop training when gap exceeds threshold
        """
        super().__init__()
        self.max_gap = max_gap
        self.stop_on_large_gap = stop_on_large_gap
        self.best_val_acc = 0.0
        
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Check for overfitting after each validation epoch."""
        # Get metrics
        train_acc = trainer.callback_metrics.get("train_acc", 0.0)
        val_acc = trainer.callback_metrics.get("val_acc", 0.0)
        
        if train_acc > 0 and val_acc > 0:
            gap = train_acc - val_acc
            
            # Log the gap
            pl_module.log("accuracy_gap", gap, prog_bar=True)
            
            # Update best val acc
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
            
            # Warning if gap is large
            if gap > self.max_gap:
                print(f"\nWARNING: Large train/val gap detected!")
                print(f"   Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Gap: {gap:.4f}")
                print(f"   Model may be overfitting. Consider:")
                print(f"   - Increasing dropout")
                print(f"   - Reducing model capacity")
                print(f"   - Adding more regularization\n")
                
                if self.stop_on_large_gap:
                    trainer.should_stop = True
                    print("   Training stopped due to excessive overfitting.")


class GradientNormMonitor(Callback):
    """Monitor gradient norms to detect training instabilities."""
    
    def on_after_backward(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Log gradient norms after backward pass."""
        # Calculate gradient norm
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Log gradient norm every 50 steps
        if trainer.global_step % 50 == 0:
            pl_module.log("grad_norm", total_norm, prog_bar=False)


class AdaptiveLRCallback(Callback):
    """
    Adaptively adjust learning rate based on train/val gap.
    Reduces LR when overfitting is detected.
    """
    
    def __init__(self, gap_threshold: float = 0.08, lr_reduction: float = 0.5):
        """
        Args:
            gap_threshold: Trigger LR reduction when gap exceeds this (default 8%)
            lr_reduction: Multiply LR by this factor when triggered
        """
        super().__init__()
        self.gap_threshold = gap_threshold
        self.lr_reduction = lr_reduction
        self.triggered = False
        
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Adjust LR if overfitting detected."""
        train_acc = trainer.callback_metrics.get("train_acc", 0.0)
        val_acc = trainer.callback_metrics.get("val_acc", 0.0)
        
        if train_acc > 0 and val_acc > 0:
            gap = train_acc - val_acc
            
            # If gap too large and not yet triggered this epoch
            if gap > self.gap_threshold and not self.triggered:
                # Reduce learning rate
                for optimizer in trainer.optimizers:
                    for param_group in optimizer.param_groups:
                        old_lr = param_group['lr']
                        new_lr = old_lr * self.lr_reduction
                        param_group['lr'] = new_lr
                        
                        print(f"\nOverfitting detected! Reducing LR: {old_lr:.6f} → {new_lr:.6f}")
                        print(f"   Gap: {gap:.4f} (threshold: {self.gap_threshold})\n")
                
                self.triggered = True
            elif gap <= self.gap_threshold:
                self.triggered = False


class BestMetricsPrinter(Callback):
    """Print best metrics at the end of training."""
    
    def __init__(self):
        super().__init__()
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Track best metrics."""
        val_acc = trainer.callback_metrics.get("val_acc", 0.0)
        val_f1 = trainer.callback_metrics.get("val_f1", 0.0)
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_val_f1 = val_f1
            self.best_epoch = trainer.current_epoch
    
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Print summary at end."""
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
        print(f"Best Validation F1 Score: {self.best_val_f1:.4f}")
        print("="*80 + "\n")
