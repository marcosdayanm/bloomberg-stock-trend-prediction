"""
Continue training from pre-trained model checkpoint.

This script loads the best v2.0 model (79.5% test accuracy) and continues
training for additional epochs with early stopping to prevent overfitting.

Author: Miguel Noriega Bedolla
Date: December 2025
"""

import torch
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


def continue_training_from_checkpoint(
    checkpoint_path: str = "pretrained_models/best_model_v2.0.ckpt",
    additional_epochs: int = 300,
    early_stopping_patience: int = 15,
    learning_rate: float = 0.0001,  # Slightly lower for fine-tuning
):
    """
    Continue training from a pre-trained checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint to resume from
        additional_epochs: Number of additional epochs to train
        early_stopping_patience: Epochs to wait before stopping if no improvement
        learning_rate: Learning rate for continued training
    """
    print("=" * 80)
    print("CONTINUING TRAINING FROM PRE-TRAINED MODEL")
    print("=" * 80)
    
    # Load checkpoint manually to extract epoch info
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    previous_epoch = ckpt['epoch']
    previous_step = ckpt['global_step']
    
    print(f"\nCheckpoint Information:")
    print(f"  Previous training: {previous_epoch} epochs")
    print(f"  Global step: {previous_step}")
    print(f"  Additional epochs: {additional_epochs}")
    print(f"  Total target epochs: {previous_epoch + additional_epochs}")
    print(f"  Early stopping patience: {early_stopping_patience}")
    print(f"  Learning rate: {learning_rate}")
    
    # Update config for continued training
    config = ModelConfig()
    config.max_epochs = previous_epoch + additional_epochs  # Total epochs
    config.learning_rate = learning_rate
    config.early_stopping_patience = early_stopping_patience
    
    print(f"\nUpdated Configuration:")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Early stopping patience: {config.early_stopping_patience}")
    
    # Load model from checkpoint
    print(f"\nLoading model from: {checkpoint_path}")
    model = CNNBiLSTMModel(config)
    model.load_state_dict(ckpt['state_dict'])
    
    print(f"  Model loaded successfully!")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup data module
    print(f"\nSetting up data module...")
    data_module = StockDataModule(config)
    data_module.setup('fit')
    
    print(f"  Train samples: {len(data_module.train_dataset)}")
    print(f"  Val samples: {len(data_module.val_dataset)}")
    
    # Setup test data separately
    data_module.setup('test')
    print(f"  Test samples: {len(data_module.test_dataset)}")
    
    # Setup callbacks
    print(f"\nConfiguring callbacks...")
    
    # Best model checkpoint (monitors validation accuracy)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=config.model_dir,
        filename='continued-best-epoch={epoch:02d}-acc{val_acc:.4f}-loss{val_loss:.4f}',
        save_top_k=1,
        mode='max',
        verbose=True
    )
    
    # Milestone checkpoints (save every 20 epochs)
    milestone_callback = ModelCheckpoint(
        every_n_epochs=20,
        dirpath=config.model_dir / "milestones",
        filename='continued-milestone-epoch{epoch:02d}-acc{val_acc:.4f}',
        verbose=True
    )
    
    # Early stopping (patience = 15 epochs)
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        mode='min',
        verbose=True,
        min_delta=0.001  # Minimum change to qualify as improvement
    )
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Progress bar
    progress_bar = RichProgressBar()
    
    callbacks = [
        checkpoint_callback,
        milestone_callback,
        early_stop_callback,
        lr_monitor,
        progress_bar
    ]
    
    print(f"  Callbacks configured:")
    print(f"    - ModelCheckpoint (best val_acc)")
    print(f"    - ModelCheckpoint (milestones every 20 epochs)")
    print(f"    - EarlyStopping (patience={early_stopping_patience})")
    print(f"    - LearningRateMonitor")
    print(f"    - RichProgressBar")
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name='cnn_bilstm',
        version='continued_training'
    )
    
    print(f"\nTensorBoard logs: {config.log_dir}/cnn_bilstm/continued_training")
    
    # Get optimal device
    device_info = get_most_optimal_device()
    # Extract string accelerator name
    if isinstance(device_info, tuple):
        device = device_info[1]  # Get 'mps', 'cuda', or 'cpu' string
    else:
        device = device_info
    
    print(f"\nUsing device: {device}")
    
    # Create trainer
    print(f"\nCreating trainer...")
    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator=device,
        devices=1,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        precision='32-true' if device == 'mps' else '16-mixed',
        deterministic=False,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    print(f"  Trainer configured:")
    print(f"    - Max epochs: {config.max_epochs}")
    print(f"    - Accelerator: {device}")
    print(f"    - Gradient clip: {config.gradient_clip_val}")
    print(f"    - Gradient accumulation: {config.accumulate_grad_batches}")
    print(f"    - Precision: {'32-true' if device == 'mps' else '16-mixed'}")
    
    # Start training
    print("\n" + "=" * 80)
    print("STARTING CONTINUED TRAINING")
    print("=" * 80)
    print(f"\nTraining will resume from epoch {previous_epoch + 1}")
    print(f"Target: {additional_epochs} additional epochs")
    print(f"Early stopping: Will stop if no improvement for {early_stopping_patience} epochs")
    print(f"\nMonitoring validation loss and accuracy...")
    print("=" * 80 + "\n")
    
    # Train model
    try:
        trainer.fit(model, datamodule=data_module)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Print training summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    
    # Print final metrics
    if trainer.callback_metrics:
        print(f"\nFinal metrics:")
        for key, value in trainer.callback_metrics.items():
            if 'val' in key or 'test' in key:
                print(f"  {key}: {value:.4f}")
    
    print(f"\nBest checkpoint saved to:")
    print(f"  {checkpoint_callback.best_model_path}")
    print(f"\nBest validation accuracy: {checkpoint_callback.best_model_score:.4f}")
    
    # Test on best model
    print("\n" + "=" * 80)
    print("TESTING BEST MODEL")
    print("=" * 80)
    
    if checkpoint_callback.best_model_path:
        print(f"\nLoading best checkpoint: {checkpoint_callback.best_model_path}")
        best_model = CNNBiLSTMModel.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            config=config
        )
        
        test_results = trainer.test(best_model, datamodule=data_module)
        
        print(f"\nTest Results:")
        print(f"  Test Accuracy: {test_results[0]['test_acc']:.4f} ({test_results[0]['test_acc']*100:.2f}%)")
        print(f"  Test Loss: {test_results[0]['test_loss']:.4f}")
        print(f"  Test F1: {test_results[0]['test_f1']:.4f}")
    
    print("\n" + "=" * 80)
    print("TRAINING SESSION COMPLETE")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Check TensorBoard: tensorboard --logdir={config.log_dir}")
    print(f"  2. Review best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"  3. Compare with previous best (79.5% test acc)")
    print(f"  4. Update pretrained_models/ if improved")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Continue training with default settings
    continue_training_from_checkpoint(
        checkpoint_path="pretrained_models/best_model_v2.0.ckpt",
        additional_epochs=300,
        early_stopping_patience=15,
        learning_rate=0.0001  # Slightly lower than original 0.0002
    )
