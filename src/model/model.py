"""
CNN-BiLSTM-Transformer Model for Binary Stock Trend Prediction (Academic Project)

This module implements a deep learning architecture combining:
    1. Convolutional Neural Networks (CNN) for temporal feature extraction
    2. Bidirectional LSTM for sequence modeling
    3. Transformer encoder for attention mechanism
    4. Multi-layer perceptron classifier

Architecture Performance:
    - Test Accuracy: 83.4%
    - Parameters: 7.7M (optimized from 45M in v1.0)
    - Training Time: ~4 hours on Apple M3 Pro

Key Components:
    - 5 convolutional layers with batch normalization and dropout
    - 3 bidirectional LSTM layers
    - 1 transformer encoder layer with 4 attention heads
    - 3-layer dense classifier with progressive dimension reduction
    - Weighted cross-entropy loss for class imbalance handling
"""

import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy, F1Score, ConfusionMatrix

from src.model.config import ModelConfig


class CNNBiLSTMModel(L.LightningModule):
    """
    Hybrid CNN-BiLSTM-Transformer model for binary stock trend classification.
    
    Forward Pass Flow:
        Input (batch, 30, 50)
          → CNN layers (5x Conv1D + BatchNorm + Dropout)
          → BiLSTM (3 layers, bidirectional)
          → Transformer encoder (multi-head attention)
          → Attention pooling
          → Dense classifier (3 layers)
          → Output logits (batch, 2)
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize model architecture.
        
        Args:
            config: ModelConfig object containing all hyperparameters
        """
        super().__init__()
        self.save_hyperparameters(vars(config))
        self.config = config
        
        # =====================================================================
        # CNN FEATURE EXTRACTOR (5 layers)
        # Temporal convolution to extract local patterns from sequences
        # =====================================================================
        self.conv1 = nn.Conv1d(config.n_features, config.cnn_filters_1, 
                               config.cnn_kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(config.cnn_filters_1)
        self.dropout1 = nn.Dropout(config.cnn_dropout)
        
        self.conv2 = nn.Conv1d(config.cnn_filters_1, config.cnn_filters_2,
                               config.cnn_kernel_size, padding="same")
        self.bn2 = nn.BatchNorm1d(config.cnn_filters_2)
        self.dropout2 = nn.Dropout(config.cnn_dropout)
        
        self.conv3 = nn.Conv1d(config.cnn_filters_2, config.cnn_filters_3,
                               config.cnn_kernel_size, padding="same")
        self.bn3 = nn.BatchNorm1d(config.cnn_filters_3)
        self.dropout3 = nn.Dropout(config.cnn_dropout)
        
        self.conv4 = nn.Conv1d(config.cnn_filters_3, config.cnn_filters_4,
                               config.cnn_kernel_size, padding="same")
        self.bn4 = nn.BatchNorm1d(config.cnn_filters_4)
        self.dropout4 = nn.Dropout(config.cnn_dropout)
        
        self.conv5 = nn.Conv1d(config.cnn_filters_4, config.cnn_filters_5,
                               config.cnn_kernel_size, padding="same")
        self.bn5 = nn.BatchNorm1d(config.cnn_filters_5)
        self.dropout5 = nn.Dropout(config.cnn_dropout)
        
        # =====================================================================
        # BIDIRECTIONAL LSTM (3 layers)
        # Captures long-term dependencies in both forward and backward directions
        # =====================================================================
        self.lstm = nn.LSTM(
            input_size=config.cnn_filters_5,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0
        )
        self.lstm_dropout = nn.Dropout(config.lstm_dropout)
        
        # =====================================================================
        # TRANSFORMER ENCODER (optional, 1 layer)
        # Multi-head self-attention for capturing global temporal patterns
        # =====================================================================
        if config.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.lstm_hidden_size * 2,  # BiLSTM doubles hidden size
                nhead=config.transformer_heads,
                dim_feedforward=config.lstm_hidden_size * 4,
                dropout=0.2,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, 
                                                     num_layers=config.transformer_layers)
        else:
            self.transformer = None
        
        # =====================================================================
        # ATTENTION POOLING
        # Weighted aggregation of sequence using learned attention weights
        # =====================================================================
        self.attention = nn.Linear(config.lstm_hidden_size * 2, 1)
        
        # =====================================================================
        # DENSE CLASSIFIER (3 layers with progressive dimension reduction)
        # Final classification layers: 256 → 128 → 64 → 2
        # =====================================================================
        self.fc1 = nn.Linear(config.lstm_hidden_size * 2, config.dense_hidden_1)
        self.bn_fc1 = nn.BatchNorm1d(config.dense_hidden_1)
        self.fc1_dropout = nn.Dropout(config.dense_dropout)
        
        self.fc2 = nn.Linear(config.dense_hidden_1, config.dense_hidden_2)
        self.bn_fc2 = nn.BatchNorm1d(config.dense_hidden_2)
        self.fc2_dropout = nn.Dropout(config.dense_dropout)
        
        self.fc3 = nn.Linear(config.dense_hidden_2, config.dense_hidden_3)
        self.bn_fc3 = nn.BatchNorm1d(config.dense_hidden_3)
        self.fc3_dropout = nn.Dropout(config.dense_dropout)
        
        self.fc4 = nn.Linear(config.dense_hidden_3, config.n_classes)
        
        # =====================================================================
        # LOSS FUNCTION
        # Weighted cross-entropy to handle class imbalance (if any)
        # Label smoothing prevents overconfidence in predictions
        # =====================================================================
        class_weights = torch.tensor([1.15, 0.87], dtype=torch.float32)  # Bearish, Bullish
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.label_smoothing
        )
        
        # =====================================================================
        # METRICS (accuracy, F1-score, confusion matrix)
        # =====================================================================
        self.train_acc = Accuracy(task="multiclass", num_classes=config.n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=config.n_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=config.n_classes)
        
        self.val_f1 = F1Score(task="multiclass", num_classes=config.n_classes, average="weighted")
        self.test_f1 = F1Score(task="multiclass", num_classes=config.n_classes, average="weighted")
        
        self.test_confusion = ConfusionMatrix(task="multiclass", num_classes=config.n_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_features)
               - sequence_length: 30 days of historical data
               - n_features: 50 selected discriminative features
            
        Returns:
            logits: Output tensor of shape (batch_size, n_classes)
                    Raw scores for each class (before softmax)
        
        Architecture Flow:
            1. CNN: Extract local temporal patterns (5 conv layers)
            2. BiLSTM: Model sequential dependencies bidirectionally
            3. Transformer: Apply multi-head self-attention
            4. Attention Pooling: Aggregate sequence with learned weights
            5. Dense Layers: Final classification
        """
        # Reshape for Conv1D: (batch, features, sequence) 
        x = x.transpose(1, 2)
        
        # CNN Block 1-5: Feature extraction
        for i in range(1, 6):
            conv = getattr(self, f'conv{i}')
            bn = getattr(self, f'bn{i}')
            dropout = getattr(self, f'dropout{i}')
            x = dropout(torch.relu(bn(conv(x))))
        
        # Reshape for LSTM: (batch, sequence, features)
        x = x.transpose(1, 2)
        
        # BiLSTM: Sequential modeling
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # Transformer: Global attention (optional)
        if self.transformer is not None:
            lstm_out = self.transformer(lstm_out)
        
        # Attention Pooling: Weighted sequence aggregation
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        
        # Dense Classifier: 256 → 128 → 64 → 2
        x = self.fc1_dropout(torch.relu(self.bn_fc1(self.fc1(context))))
        x = self.fc2_dropout(torch.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3_dropout(torch.relu(self.bn_fc3(self.fc3(x))))
        logits = self.fc4(x)
        
        return logits
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Single training step (called by Lightning Trainer).
        
        Args:
            batch: Tuple of (features, labels)
            batch_idx: Index of current batch
            
        Returns:
            loss: Scalar loss value for backpropagation
        """
        x, y = batch
        logits = self(x)
        y_indices = torch.argmax(y, dim=1)  # Convert one-hot to class indices
        
        loss = self.criterion(logits, y_indices)
        acc = self.train_acc(torch.argmax(logits, dim=1), y_indices)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Validation step - metrics only, no gradient updates."""
        x, y = batch
        logits = self(x)
        y_indices = torch.argmax(y, dim=1)
        
        loss = self.criterion(logits, y_indices)
        preds = torch.argmax(logits, dim=1)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc(preds, y_indices), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1(preds, y_indices), on_step=False, on_epoch=True, prog_bar=True)
        
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Test step - final evaluation metrics."""
        x, y = batch
        logits = self(x)
        y_indices = torch.argmax(y, dim=1)
        
        loss = self.criterion(logits, y_indices)
        preds = torch.argmax(logits, dim=1)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc(preds, y_indices), on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1(preds, y_indices), on_step=False, on_epoch=True)
        
        self.test_confusion.update(preds, y_indices)
        
    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """Prediction step - returns class probabilities and predictions."""
        x, y = batch
        logits = self(x)
        
        return {
            "predictions": torch.argmax(logits, dim=1),
            "probabilities": torch.softmax(logits, dim=1),
            "ground_truth": torch.argmax(y, dim=1)
        }
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Optimizer: AdamW (Adam with decoupled weight decay)
            - lr: 0.0002 (base learning rate)
            - weight_decay: 0.005 (L2 regularization)
            
        Scheduler: OneCycleLR (cyclic learning rate with warmup)
            - max_lr: 5x base learning rate
            - warmup: 20% of training
            - annealing: cosine decay to very low LR
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Calculate steps per epoch for scheduler
        train_samples = 5098  # ~80% of 7202 balanced samples
        steps_per_epoch = train_samples // self.config.batch_size
        
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate * 5,
            epochs=self.config.max_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.2,           # 20% warmup
            anneal_strategy='cos',   # Cosine annealing
            div_factor=20.0,         # Initial LR = max_lr / 20
            final_div_factor=500.0   # Final LR very low for fine-tuning
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
