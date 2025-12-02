"""Configuration for CNN-BiLSTM stock prediction model."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Input dimensions
    sequence_length: int = 60
    n_features: int = 172
    n_classes: int = 2 
    
    # CNN layers
    cnn_filters_1: int = 32
    cnn_filters_2: int = 64
    cnn_filters_3: int = 64
    cnn_kernel_size: int = 3
    cnn_dropout: float = 0.55  # 0.6 -0.5
    
    # BiLSTM layers  
    lstm_hidden_size: int = 72  #64 - 80
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.65  # 0.6 - 0.7
    
    # Dense layers
    dense_hidden_1: int = 144  # 128 - 160
    dense_hidden_2: int = 72   # 64 - 80
    dense_dropout: float = 0.55  # 0.5 - 0.6
    
    # Training
    learning_rate: float = 0.0007 
    weight_decay: float = 0.0018   # 0.0015 - 0.002
    batch_size: int = 112          # 96 - 128
    max_epochs: int = 200
    early_stopping_patience: int = 30
    gradient_clip_val: float = 0.5
    
    # Data splits (percentages) 
    train_split: float = 0.8
    val_split: float = .1 #.09-.1
    test_split: float = 0.1 #.09-.1
    local_test_samples: int = 5  # Last samples for local testing (excluded from splits)
    
    # Data augmentation
    noise_std: float = 0.015  # .01-.015
    
    # Anti-overfitting monitoring thresholds
    overfitting_lr_threshold: float = 0.07 
    overfitting_warning_threshold: float = 0.10
    
    # Paths
    data_dir: Path = Path("datasets/npy")
    model_dir: Path = Path("src/model/checkpoints")
    log_dir: Path = Path("src/model/logs")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        (self.model_dir / "milestones").mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
