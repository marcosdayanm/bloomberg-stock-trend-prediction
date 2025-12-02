"""Training script for CNN-BiLSTM stock prediction model."""

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from lightning.pytorch.loggers import TensorBoardLogger

from src.model.config import ModelConfig
from src.model.data_module import StockDataModule
from src.model.model import CNNBiLSTMModel
from src.model.utils import get_most_optimal_device
from src.model.callbacks import (
    OverfittingMonitor,
    GradientNormMonitor,
    AdaptiveLRCallback,
    BestMetricsPrinter
)


def train():
    """Train the CNN-BiLSTM model."""
    # Configuration
    config = ModelConfig()
    
    # Detect optimal device
    device, accelerator, devices, pin_memory = get_most_optimal_device()
    print(f"\n{'='*80}")
    print(f"DEVICE CONFIGURATION")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Accelerator: {accelerator}")
    print(f"Devices: {devices}")
    print(f"Pin Memory: {pin_memory}")
    print(f"{'='*80}\n")
    
    # Data module
    data_module = StockDataModule(config, pin_memory=pin_memory)
    data_module.prepare_data()
    
    # Model
    model = CNNBiLSTMModel(config)
    
    # Callbacks
    best_checkpoint = ModelCheckpoint(
        dirpath=config.model_dir / "best",
        filename=f"{{val_acc:.4f}}_{{val_loss:.4f}}_{{train_acc:.4f}}_{{epoch:02d}}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,  # Only keep the absolute best
        save_last=False,
        verbose=True,
        every_n_epochs=1,
        save_on_train_epoch_end=False
    )
    
    milestone_checkpoint = ModelCheckpoint(
        dirpath=config.model_dir / "checkpoints",
        filename=f"{{val_acc:.4f}}_{{val_loss:.4f}}_{{train_acc:.4f}}_{{epoch:02d}}",
        monitor="val_acc",
        mode="max",
        save_top_k=-1,  # Keep all milestones
        every_n_epochs=10,
        verbose=False
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=config.early_stopping_patience,
        mode="min",
        verbose=True,
        min_delta=0.001
    )
    
    # Anti-overfitting callbacks
    overfitting_monitor = OverfittingMonitor(
        max_gap=config.overfitting_warning_threshold,
        stop_on_large_gap=False
    )
    
    adaptive_lr = AdaptiveLRCallback(
        gap_threshold=config.overfitting_lr_threshold,
        lr_reduction=0.5
    )
    
    gradient_monitor = GradientNormMonitor()
    metrics_printer = BestMetricsPrinter()
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name="cnn_bilstm",
        version=None
    )
    
    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator=accelerator,
        devices=devices if devices is not None else "auto",
        precision="32" if accelerator == "mps" else "16-mixed",
        callbacks=[
            best_checkpoint,
            milestone_checkpoint,
            early_stop_callback,
            overfitting_monitor,  # Monitor train/val gap
            adaptive_lr,  # Reduce LR if overfitting
            gradient_monitor,  # Monitor gradient norms
            metrics_printer,  # Print best metrics at end
            lr_monitor,
            RichProgressBar()
        ],
        logger=logger,
        log_every_n_steps=10,
        deterministic=False,
        gradient_clip_val=config.gradient_clip_val,  # Use config value
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train
    print("\n" + "="*80)
    print("TRAINING CNN-BiLSTM MODEL")
    print("="*80 + "\n")
    
    trainer.fit(model, data_module)
    
    # Test (fix para PyTorch 2.6 weights_only issue)
    print("\n" + "="*80)
    print("TESTING MODEL")
    print("="*80 + "\n")
    
    # Cargar checkpoint con weights_only=False (seguro porque lo generamos nosotros)
    import torch
    torch.serialization.add_safe_globals([type(config.model_dir)])
    
    trainer.test(model, data_module, ckpt_path="best")
    
    # Local predictions
    print("\n" + "="*80)
    print("LOCAL TEST PREDICTIONS")
    print("="*80 + "\n")
    
    # Añadir Path a safe globals también para predict
    from pathlib import PosixPath, WindowsPath
    torch.serialization.add_safe_globals([PosixPath, WindowsPath])
    
    predictions = trainer.predict(model, data_module, ckpt_path="best")
    
    # Print predictions
    print("\nLocal Test Results:")
    print("-" * 60)
    for i, pred in enumerate(predictions):  # type: ignore
        pred_class = pred["predictions"].item()  # type: ignore
        true_class = pred["ground_truth"].item()  # type: ignore
        probs = pred["probabilities"].squeeze()  # type: ignore
        
        print(f"\nSample {i+1}:")
        print(f"  Predicted: Class {pred_class} (confidence: {probs[pred_class]:.4f})")
        print(f"  Ground Truth: Class {true_class}")
        print(f"  Correct: {'✓' if pred_class == true_class else '✗'}")
    
    print("\n" + "="*80)
    print(f"Training complete! Model saved to: {config.model_dir}")
    print(f"TensorBoard logs saved to: {config.log_dir}")
    print("\nView training logs with:")
    print(f"  tensorboard --logdir {config.log_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    L.seed_everything(42)
    
    train()
