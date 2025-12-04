from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    task_type: str = 'classification'  # 'classification' | 'regression'
    
    # Input dimensions
    sequence_length: int = 30
    n_features: int = 50
    n_classes: int = 2  # Only used for classification
    
    # CNN layers
    cnn_filters_1: int = 128
    cnn_filters_2: int = 256
    cnn_filters_3: int = 256
    cnn_filters_4: int = 256
    cnn_filters_5: int = 256
    cnn_kernel_size: int = 5
    cnn_dropout: float = 0.3
    
    # BiLSTM Layers
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 3
    lstm_dropout: float = 0.3
    
    # Transformer
    use_transformer: bool = True
    transformer_heads: int = 4
    transformer_layers: int = 1
    
    # Dense layers
    dense_hidden_1: int = 256
    dense_hidden_2: int = 128
    dense_hidden_3: int = 64
    dense_dropout: float = 0.35
    
    # Training
    learning_rate: float = 0.0001
    weight_decay: float = 0.005
    batch_size: int = 16
    max_epochs: int = 443
    early_stopping_patience: int = 15
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 4
    use_amp: bool = False
    label_smoothing: float = 0.1
    
    # Data splits percentages
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    local_test_samples: int = 5
    
    # Paths (will be set in __post_init__ based on task_type)
    data_dir: Path = Path("datasets/npy")
    model_dir: Path | None = None
    log_dir: Path | None = None
    
    def __post_init__(self):
        """Create task-specific directories."""
        # Set paths based on task type
        base_checkpoint_dir = Path("src/model/checkpoints")
        base_log_dir = Path("src/model/logs")
        
        self.model_dir = base_checkpoint_dir / self.task_type
        self.log_dir = base_log_dir / self.task_type
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        (self.model_dir / "milestones").mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Task-specific directories:")
        print(f"   Checkpoints: {self.model_dir}")
        print(f"   Logs: {self.log_dir}")
