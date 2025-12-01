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
    # Best model checkpoint (only saves when accuracy > 40% and improves)
    best_checkpoint = ModelCheckpoint(
        dirpath=config.model_dir,
        filename="best-{epoch:02d}-acc{val_acc:.4f}-loss{val_loss:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,  # Only keep the absolute best
        save_last=False,
        verbose=True,
        every_n_epochs=1,  # Check every epoch but only save if improves
        save_on_train_epoch_end=False  # Save after validation
    )
    
    # Milestone checkpoints (saves every 10 epochs if acc > 40%)
    milestone_checkpoint = ModelCheckpoint(
        dirpath=config.model_dir / "milestones",
        filename="milestone-epoch{epoch:02d}-acc{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=-1,  # Keep all milestones
        every_n_epochs=10,  # Save every 10 epochs
        verbose=False
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",  # Mejor usar loss (más estable que accuracy)
        patience=config.early_stopping_patience,
        mode="min",
        verbose=True,
        min_delta=0.001  # Mejora mínima: 0.1% (era 1% - demasiado agresivo)
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name="cnn_bilstm",
        version=None
    )
    
    # Trainer with auto-detected optimal device
    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator=accelerator,  # Auto-detected (CUDA/MPS/CPU)
        devices=devices if devices is not None else "auto",
        precision="32" if accelerator == "mps" else "16-mixed",  # MPS uses FP32, CUDA uses FP16
        callbacks=[
            best_checkpoint,
            milestone_checkpoint,
            early_stop_callback,
            lr_monitor,
            RichProgressBar()
        ],
        logger=logger,
        log_every_n_steps=10,
        deterministic=False,  # MPS doesn't support full determinism
        gradient_clip_val=1.0,
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
    # Set seed for reproducibility
    L.seed_everything(42)
    
    train()
