"""Configuration for CNN-BiLSTM stock prediction model."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Input dimensions - DATASET OPTIMIZADO (solo top 50 features discriminativas)
    sequence_length: int = 30     # REDUCIDO de 120 a 30 días - menos overfitting
    n_features: int = 50          # REDUCIDO de 224 a 50 - solo features importantes
    n_classes: int = 2            # BINARIO: Bajista vs Alcista (PERFECTAMENTE BALANCEADO 50-50)
    
    # CNN layers - OPTIMIZADO para 50 features (menos filtros)
    cnn_filters_1: int = 128     # Reducido de 256
    cnn_filters_2: int = 256     # Reducido de 512
    cnn_filters_3: int = 256     # Reducido de 512
    cnn_filters_4: int = 256     # Reducido de 512
    cnn_filters_5: int = 256     # Reducido de 512
    cnn_kernel_size: int = 5
    cnn_dropout: float = 0.3     # Balanceado: permite aprendizaje sin overfitting extremo
    
    # BiLSTM layer - REDUCIDO para evitar overfitting (45M→5M params)
    lstm_hidden_size: int = 256  # REDUCIDO de 512 a 256
    lstm_num_layers: int = 3     # REDUCIDO de 4 a 3 capas
    lstm_dropout: float = 0.3    # Balanceado para permitir aprendizaje
    
    # Transformer Encoder - REDUCIDO para evitar overfitting
    use_transformer: bool = True
    transformer_heads: int = 4   # REDUCIDO de 8 a 4
    transformer_layers: int = 1  # REDUCIDO de 2 a 1 capa
    
    # Dense layers - REDUCIDO para evitar overfitting
    dense_hidden_1: int = 256    # REDUCIDO de 512 a 256
    dense_hidden_2: int = 128    # REDUCIDO de 256 a 128
    dense_hidden_3: int = 64     # REDUCIDO de 128 a 64
    dense_dropout: float = 0.35  # Balanceado: permite aprendizaje
    
    # Training - OPTIMIZADO anti-overfitting
    learning_rate: float = 0.0002  # AUMENTADO de 0.0001 para aprendizaje más rápido
    weight_decay: float = 0.005    # REDUCIDO de 0.01 a 0.005 para menos restricción
    label_smoothing: float = 0.1   # NUEVO: evita overconfidence
    batch_size: int = 16          # Batch óptimo para MPS con modelo grande
    max_epochs: int = 150         # REDUCIDO de 300 a 150 para entrenamiento más rápido
    early_stopping_patience: int = 15  # REDUCIDO de 40 a 15 para detectar overfitting temprano
    gradient_clip_val: float = 1.0  # Gradient clipping estándar
    
    # MPS Optimizations
    use_amp: bool = False         # MPS usa FP32 (más estable que mixed precision)
    accumulate_grad_batches: int = 4  # Gradient accumulation para batch efectivo de 64
    
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
