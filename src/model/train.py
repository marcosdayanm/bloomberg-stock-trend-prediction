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
from src.model.callbacks import OverfittingDetector
import torch
from pathlib import PosixPath, WindowsPath


def train():
    """Train the CNN-BiLSTM model."""
    # Configuration
    config = ModelConfig()
    
    device, accelerator, devices, pin_memory = get_most_optimal_device()
    precision = "32" if accelerator == "mps" else "16-mixed"
    if not devices:
        devices = "auto"

    print(f"\n{'='*80}")
    print(f"DEVICE CONFIGURATION")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Accelerator: {accelerator}")
    print(f"Devices: {devices}")
    print(f"Pin Memory: {pin_memory}")
    print(f"Precision: {precision}")
    print(f"{'='*80}\n")
    
    # Data module
    data_module = StockDataModule(config, pin_memory=pin_memory)
    data_module.prepare_data()
    
    # Model
    model = CNNBiLSTMModel(config)
    
    # Callbacks
    # Best model checkpoint (adapt to task type)
    if config.task_type == 'classification':
        best_checkpoint = ModelCheckpoint(
            dirpath=config.model_dir,
            filename="best-{epoch:02d}-acc{val_acc:.4f}-loss{val_loss:.4f}",
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            save_last=False,
            verbose=True,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        )
        
        milestone_checkpoint = ModelCheckpoint(
            dirpath=config.model_dir / "milestones",
            filename="milestone-epoch{epoch:02d}-acc{val_acc:.4f}",
            monitor="val_acc",
            mode="max",
            save_top_k=-1,
            every_n_epochs=10,
            verbose=False,
        )
    else:
        best_checkpoint = ModelCheckpoint(
            dirpath=config.model_dir,
            filename="best-{epoch:02d}-mae{val_mae:.4f}-loss{val_loss:.4f}",
            monitor="val_mae",
            mode="min",
            save_top_k=1,
            save_last=False,
            verbose=True,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        )
        
        milestone_checkpoint = ModelCheckpoint(
            dirpath=config.model_dir / "milestones",
            filename="milestone-epoch{epoch:02d}-mae{val_mae:.4f}",
            monitor="val_mae",
            mode="min",
            save_top_k=-1,
            every_n_epochs=10,
            verbose=False,
        )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=config.early_stopping_patience,
        mode="min",
        verbose=True,
        min_delta=0.001,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    overfitting_detector = OverfittingDetector(
        monitor_train="train_acc",
        monitor_val="val_acc",
        threshold=0.08,
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name=f"model_{config.task_type}",
        version=None
    )
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=[
            best_checkpoint,
            milestone_checkpoint,
            early_stop_callback,
            overfitting_detector,
            lr_monitor,
            RichProgressBar()
        ],
        logger=logger,
        log_every_n_steps=10,
        deterministic=False,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80 + "\n")
    
    trainer.fit(model, data_module)
    
    print("\n" + "="*80)
    print("TESTING MODEL")
    print("="*80 + "\n")
    
    torch.serialization.add_safe_globals([type(config.model_dir)])
    
    trainer.test(model, data_module, ckpt_path="best")
    
    # Local predictions
    print("\n" + "="*80)
    print("LOCAL PREDICTIONS")
    print("="*80 + "\n")
    
    torch.serialization.add_safe_globals([PosixPath, WindowsPath])
    
    predictions = trainer.predict(model, data_module, ckpt_path="best")
    
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
    print(f"Training complete. Model saved at: {config.model_dir}")
    print(f"TensorBoard logs saved at: {config.log_dir}")
    print("\nView training logs with:")
    print(f"  tensorboard --logdir {config.log_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Set seed for reproducibility
    L.seed_everything(42)
    
    train()
