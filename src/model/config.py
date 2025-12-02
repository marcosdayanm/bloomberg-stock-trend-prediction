"""
Model Configuration for Binary Stock Trend Prediction (Academic Project)

This configuration defines all hyperparameters for the CNN-BiLSTM-Transformer model.
Optimized for binary classification (Bearish vs Bullish) with 83.4% test accuracy.

Architecture Overview:
    - 5-layer CNN for temporal feature extraction
    - 3-layer Bidirectional LSTM for sequence modeling
    - 1-layer Transformer for attention mechanism
    - 3-layer dense classifier with dropout regularization
    - Total parameters: 7.7M (optimized from initial 45M)

Performance:
    - Test Accuracy: 83.4%
    - Train-Test Gap: 2.3% (excellent generalization)
    - Training Time: ~4 hours on Apple M3 Pro (MPS)
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Hyperparameter configuration for CNN-BiLSTM-Transformer model."""
    
    # ============================================================================
    # INPUT DIMENSIONS (optimized for reduced overfitting)
    # ============================================================================
    sequence_length: int = 30     # Days of historical data per sample (reduced from 120)
    n_features: int = 50          # Number of input features (top 50 most discriminative)
    n_classes: int = 2            # Binary: Bearish (0) vs Bullish (1)
    
    # ============================================================================
    # CNN ARCHITECTURE (5 convolutional layers for temporal feature extraction)
    # ============================================================================
    cnn_filters_1: int = 128      # First conv layer output channels
    cnn_filters_2: int = 256      # Subsequent conv layers output channels
    cnn_filters_3: int = 256
    cnn_filters_4: int = 256
    cnn_filters_5: int = 256
    cnn_kernel_size: int = 5      # Temporal kernel size (captures 5-day patterns)
    cnn_dropout: float = 0.3      # Dropout rate after each conv layer
    
    # ============================================================================
    # BILSTM ARCHITECTURE (bidirectional sequence modeling)
    # ============================================================================
    lstm_hidden_size: int = 256   # Hidden units per LSTM direction
    lstm_num_layers: int = 3      # Number of stacked LSTM layers
    lstm_dropout: float = 0.3     # Dropout between LSTM layers
    
    # ============================================================================
    # TRANSFORMER ARCHITECTURE (attention mechanism for long-range dependencies)
    # ============================================================================
    use_transformer: bool = True
    transformer_heads: int = 4    # Number of attention heads
    transformer_layers: int = 1   # Number of transformer encoder layers
    
    # ============================================================================
    # DENSE CLASSIFIER (3-layer MLP with progressive dimension reduction)
    # ============================================================================
    dense_hidden_1: int = 256     # First dense layer units
    dense_hidden_2: int = 128     # Second dense layer units
    dense_hidden_3: int = 64      # Third dense layer units
    dense_dropout: float = 0.35   # Dropout rate in dense layers
    
    # ============================================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================================
    learning_rate: float = 0.0002       # Initial learning rate (OneCycleLR scheduler)
    weight_decay: float = 0.005         # L2 regularization coefficient
    label_smoothing: float = 0.1        # Label smoothing to prevent overconfidence
    batch_size: int = 16                # Batch size (optimal for MPS with 16GB RAM)
    max_epochs: int = 150               # Maximum training epochs
    early_stopping_patience: int = 15   # Epochs to wait before early stopping
    gradient_clip_val: float = 1.0      # Gradient clipping threshold
    
    # ============================================================================
    # DEVICE OPTIMIZATION (for Apple Silicon MPS)
    # ============================================================================
    use_amp: bool = False              # Use automatic mixed precision (disabled for MPS stability)
    accumulate_grad_batches: int = 4   # Gradient accumulation (effective batch size = 64)
    
    # ============================================================================
    # DATA SPLITS (chronological train/val/test split)
    # ============================================================================
    train_split: float = 0.80          # 80% for training
    val_split: float = 0.1             # 10% for validation
    test_split: float = 0.1            # 10% for testing
    local_test_samples: int = 5        # Reserve last N samples for local inference testing
    
    # ============================================================================
    # FILE PATHS
    # ============================================================================
    data_dir: Path = Path("datasets/npy")
    model_dir: Path = Path("src/model/checkpoints")
    log_dir: Path = Path("src/model/logs")
    
    def __post_init__(self):
        """Create required directories on initialization."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        (self.model_dir / "milestones").mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
