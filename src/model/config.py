"""Configuration for CNN-BiLSTM stock prediction model."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Input dimensions
    sequence_length: int = 60
    n_features: int = 237
    n_classes: int = 12
    
    # CNN layers
    cnn_filters_1: int = 32
    cnn_filters_2: int = 64
    cnn_kernel_size: int = 5
    cnn_dropout: float = 0.5
    
    # BiLSTM layer - Simplificar modelo
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.5  
    
    # Dense layers - Simplificar
    dense_hidden: int = 64
    dense_dropout: float = 0.5 
    
    # Training - Configuración anti-overfitting
    learning_rate: float = .000001 
    weight_decay: float = .0001  
    batch_size: int = 128
    max_epochs: int = 150
    early_stopping_patience: int = 20  # Menos paciencia
    
    # Data splits (percentages) 
    train_split: float = 0.80
    val_split: float = 0.1 
    test_split: float = 0.1 
    local_test_samples: int = 5  # Last samples for local testing (excluded from splits)
    
    # Paths
    data_dir: Path = Path("datasets/npy")
    model_dir: Path = Path("src/model/checkpoints")
    log_dir: Path = Path("src/model/logs")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        (self.model_dir / "milestones").mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
