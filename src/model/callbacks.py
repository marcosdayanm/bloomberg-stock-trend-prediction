"""Custom callbacks for model training."""

import lightning as L
from lightning.pytorch.callbacks import Callback
import torch


class OverfittingDetector(Callback):
    """
    Reduce learning rate when train and validation accuracy diverge too much.
    This helps prevent overfitting by reducing LR when the model starts memorizing.
    """
    
    def __init__(
        self,
        monitor_train: str = "train_acc",
        monitor_val: str = "val_acc",
        threshold: float = 0.08,
        factor: float = 0.5,
        patience: int = 3,
        min_lr: float = .0000007,
        verbose: bool = True
    ):
        super().__init__()
        self.monitor_train = monitor_train
        self.monitor_val = monitor_val
        self.threshold = threshold
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.wait = 0
        self.best_gap = 0
        
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Check gap between train and val metrics."""
        train_metric = trainer.callback_metrics.get(self.monitor_train)
        val_metric = trainer.callback_metrics.get(self.monitor_val)
        
        if train_metric is None or val_metric is None:
            return
        
        gap = float(train_metric - val_metric)
        
        if gap > self.threshold:
            self.wait += 1
            
            if self.wait >= self.patience:
                # Reduce learning rate for all optimizers
                for optimizer in trainer.optimizers:
                    for param_group in optimizer.param_groups:
                        old_lr = param_group['lr']
                        new_lr = max(old_lr * self.factor, self.min_lr)
                        param_group['lr'] = new_lr
                        
                        if self.verbose and new_lr > self.min_lr:
                            print(f"\n{'='*80}")
                            print(f"OVERFITTING DETECTED (gap={gap:.4f} > threshold={self.threshold})")
                            print(f"Reducing learning rate: {old_lr:.2e} -> {new_lr:.2e}")
                            print(f"{'='*80}\n")
                
                self.wait = 0
        else:
            self.wait = 0
