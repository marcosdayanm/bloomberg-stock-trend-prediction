"""Training script for CNN-BiLSTM stock prediction model."""

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

from src.model.config import ModelConfig
from src.model.data_module import StockDataModule
from src.model.model import CNNBiLSTMModel
from src.model.utils import get_most_optimal_device


def plot_training_history(log_dir: Path, output_dir: Path):
    """
    Plot training history from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        output_dir: Directory to save plots
    """
    # Find the latest version directory
    version_dirs = sorted(log_dir.glob("cnn_bilstm/version_*"))
    if not version_dirs:
        print("WARNING: No TensorBoard logs found. Skipping plots.")
        return
    
    latest_version = version_dirs[-1]
    event_file = list(latest_version.glob("events.out.tfevents.*"))
    
    if not event_file:
        print("WARNING: No event files found. Skipping plots.")
        return
    
    print(f"\nGenerating training plots from {latest_version.name}...")
    
    # Load TensorBoard logs
    ea = event_accumulator.EventAccumulator(str(latest_version))
    ea.Reload()
    
    # Extract metrics
    metrics_data = {}
    
    # Available scalar tags
    tags = ea.Tags()['scalars']
    
    for tag in tags:
        events = ea.Scalars(tag)
        metrics_data[tag] = pd.DataFrame([
            {'step': e.step, 'value': e.value, 'epoch': e.step}
            for e in events
        ])
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Accuracy (Train vs Val vs Test)
    plt.figure(figsize=(12, 6))
    
    if 'train_acc' in metrics_data and 'val_acc' in metrics_data:
        plt.plot(metrics_data['train_acc']['epoch'], 
                metrics_data['train_acc']['value'], 
                label='Train Accuracy', linewidth=2, color='#2ecc71')
        plt.plot(metrics_data['val_acc']['epoch'], 
                metrics_data['val_acc']['value'], 
                label='Validation Accuracy', linewidth=2, color='#3498db')
        
        # Add test accuracy as horizontal line if available
        if 'test_acc' in metrics_data:
            test_acc = metrics_data['test_acc']['value'].iloc[0]
            plt.axhline(y=test_acc, color='#e74c3c', linestyle='--', 
                       linewidth=2, label=f'Test Accuracy ({test_acc:.4f})')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Accuracy Over Training', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_history.png', dpi=150)
    print(f"  SAVED: {output_dir / 'accuracy_history.png'}")
    plt.close()
    
    # Plot 2: Loss (Train vs Val vs Test)
    plt.figure(figsize=(12, 6))
    
    if 'train_loss' in metrics_data and 'val_loss' in metrics_data:
        plt.plot(metrics_data['train_loss']['epoch'], 
                metrics_data['train_loss']['value'], 
                label='Train Loss', linewidth=2, color='#2ecc71')
        plt.plot(metrics_data['val_loss']['epoch'], 
                metrics_data['val_loss']['value'], 
                label='Validation Loss', linewidth=2, color='#3498db')
        
        # Add test loss as horizontal line if available
        if 'test_loss' in metrics_data:
            test_loss = metrics_data['test_loss']['value'].iloc[0]
            plt.axhline(y=test_loss, color='#e74c3c', linestyle='--', 
                       linewidth=2, label=f'Test Loss ({test_loss:.4f})')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Model Loss Over Training', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_history.png', dpi=150)
    print(f"  SAVED: {output_dir / 'loss_history.png'}")
    plt.close()
    
    # Plot 3: F1 Score (Val vs Test)
    if 'val_f1' in metrics_data:
        plt.figure(figsize=(12, 6))
        
        plt.plot(metrics_data['val_f1']['epoch'], 
                metrics_data['val_f1']['value'], 
                label='Validation F1', linewidth=2, color='#9b59b6')
        
        # Add test F1 as horizontal line if available
        if 'test_f1' in metrics_data:
            test_f1 = metrics_data['test_f1']['value'].iloc[0]
            plt.axhline(y=test_f1, color='#e74c3c', linestyle='--', 
                       linewidth=2, label=f'Test F1 ({test_f1:.4f})')
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('F1 Score Over Training', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'f1_history.png', dpi=150)
        print(f"  SAVED: {output_dir / 'f1_history.png'}")
        plt.close()
    
    # Plot 4: Combined Overview (2x2 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Accuracy
    if 'train_acc' in metrics_data and 'val_acc' in metrics_data:
        axes[0, 0].plot(metrics_data['train_acc']['epoch'], 
                       metrics_data['train_acc']['value'], 
                       label='Train', linewidth=2, color='#2ecc71')
        axes[0, 0].plot(metrics_data['val_acc']['epoch'], 
                       metrics_data['val_acc']['value'], 
                       label='Validation', linewidth=2, color='#3498db')
        if 'test_acc' in metrics_data:
            test_acc = metrics_data['test_acc']['value'].iloc[0]
            axes[0, 0].axhline(y=test_acc, color='#e74c3c', linestyle='--', 
                             linewidth=2, label=f'Test ({test_acc:.4f})')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: Loss
    if 'train_loss' in metrics_data and 'val_loss' in metrics_data:
        axes[0, 1].plot(metrics_data['train_loss']['epoch'], 
                       metrics_data['train_loss']['value'], 
                       label='Train', linewidth=2, color='#2ecc71')
        axes[0, 1].plot(metrics_data['val_loss']['epoch'], 
                       metrics_data['val_loss']['value'], 
                       label='Validation', linewidth=2, color='#3498db')
        if 'test_loss' in metrics_data:
            test_loss = metrics_data['test_loss']['value'].iloc[0]
            axes[0, 1].axhline(y=test_loss, color='#e74c3c', linestyle='--', 
                             linewidth=2, label=f'Test ({test_loss:.4f})')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Subplot 3: F1 Score
    if 'val_f1' in metrics_data:
        axes[1, 0].plot(metrics_data['val_f1']['epoch'], 
                       metrics_data['val_f1']['value'], 
                       label='Validation F1', linewidth=2, color='#9b59b6')
        if 'test_f1' in metrics_data:
            test_f1 = metrics_data['test_f1']['value'].iloc[0]
            axes[1, 0].axhline(y=test_f1, color='#e74c3c', linestyle='--', 
                             linewidth=2, label=f'Test ({test_f1:.4f})')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 4: Learning Rate
    if 'lr-AdamW' in metrics_data:
        axes[1, 1].plot(metrics_data['lr-AdamW']['epoch'], 
                       metrics_data['lr-AdamW']['value'], 
                       label='Learning Rate', linewidth=2, color='#f39c12')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_overview.png', dpi=150)
    print(f"  SAVED: {output_dir / 'training_overview.png'}")
    plt.close()
    
    print(f"\nSUCCESS: Training plots saved to: {output_dir}/")
    print(f"   - accuracy_history.png")
    print(f"   - loss_history.png")
    print(f"   - f1_history.png")
    print(f"   - training_overview.png")


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
        monitor="val_loss",
        patience=config.early_stopping_patience,
        mode="min",
        verbose=True,
        min_delta=0.0001  # Reducido para detectar mejoras más pequeñas
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
        gradient_clip_val=1.0,  # Clip gradients para estabilidad
        accumulate_grad_batches=config.accumulate_grad_batches,  # Batch efectivo = 16*4 = 64
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
        print(f"  Correct: {'YES' if pred_class == true_class else 'NO'}")
    
    print("\n" + "="*80)
    print(f"Training complete! Model saved to: {config.model_dir}")
    print(f"TensorBoard logs saved to: {config.log_dir}")
    print("\nView training logs with:")
    print(f"  tensorboard --logdir {config.log_dir}")
    print("="*80 + "\n")
    
    # Generate training plots
    plot_output_dir = config.model_dir / "training_plots"
    plot_training_history(config.log_dir, plot_output_dir)


if __name__ == "__main__":
    # Set seed for reproducibility
    L.seed_everything(42)
    
    train()
