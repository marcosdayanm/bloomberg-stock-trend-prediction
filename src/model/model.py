"""CNN-BiLSTM model for stock prediction using PyTorch Lightning."""

import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy, F1Score, ConfusionMatrix

from src.model.config import ModelConfig


class CNNBiLSTMModel(L.LightningModule):
    """CNN-1D + BiLSTM model with attention for stock trend prediction."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.save_hyperparameters(vars(config))
        self.config = config
        
        # CNN layers
        self.conv1 = nn.Conv1d(
            in_channels=config.n_features,
            out_channels=config.cnn_filters_1,
            kernel_size=config.cnn_kernel_size,
            padding="same"
        )
        self.bn1 = nn.BatchNorm1d(config.cnn_filters_1)
        self.dropout1 = nn.Dropout(config.cnn_dropout)
        
        self.conv2 = nn.Conv1d(
            in_channels=config.cnn_filters_1,
            out_channels=config.cnn_filters_2,
            kernel_size=config.cnn_kernel_size,
            padding="same"
        )
        self.bn2 = nn.BatchNorm1d(config.cnn_filters_2)
        self.dropout2 = nn.Dropout(config.cnn_dropout)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=config.cnn_filters_2,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0
        )
        self.lstm_dropout = nn.Dropout(config.lstm_dropout)
        
        # Attention mechanism
        self.attention = nn.Linear(config.lstm_hidden_size * 2, 1)
        
        # Dense layers con BatchNorm
        self.fc1 = nn.Linear(config.lstm_hidden_size * 2, config.dense_hidden)
        self.bn_fc1 = nn.BatchNorm1d(config.dense_hidden)  # BatchNorm para estabilizar
        self.fc1_dropout = nn.Dropout(config.dense_dropout)
        
        self.fc2 = nn.Linear(config.dense_hidden, config.n_classes)
        
        # Loss function con class weights para balancear dataset
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing anti-overfitting
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=config.n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=config.n_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=config.n_classes)
        
        self.val_f1 = F1Score(task="multiclass", num_classes=config.n_classes, average="weighted")
        self.test_f1 = F1Score(task="multiclass", num_classes=config.n_classes, average="weighted")
        
        self.test_confusion = ConfusionMatrix(task="multiclass", num_classes=config.n_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_length, n_features)
            
        Returns:
            Logits (batch_size, n_classes)
        """
        batch_size = x.size(0)
        
        # CNN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, n_features, seq_len)
        
        # CNN block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        # CNN block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        # BiLSTM expects (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, seq_len, cnn_filters_2)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden*2)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # Attention mechanism
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), dim=1
        )  # (batch, seq_len)
        
        # Apply attention
        context = torch.sum(
            lstm_out * attention_weights.unsqueeze(-1), dim=1
        )  # (batch, lstm_hidden*2)
        
        # Dense layers con BatchNorm
        x = self.fc1(context)
        x = self.bn_fc1(x)  # BatchNorm antes de activación
        x = torch.relu(x)
        x = self.fc1_dropout(x)
        
        logits = self.fc2(x)  # (batch, n_classes)
        
        return logits
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch
        logits = self(x)
        
        # Convert one-hot to class indices
        y_indices = torch.argmax(y, dim=1)
        
        loss = self.criterion(logits, y_indices)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y_indices)
        
        # Logging
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Validation step."""
        x, y = batch
        logits = self(x)
        
        y_indices = torch.argmax(y, dim=1)
        loss = self.criterion(logits, y_indices)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y_indices)
        f1 = self.val_f1(preds, y_indices)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Test step."""
        x, y = batch
        logits = self(x)
        
        y_indices = torch.argmax(y, dim=1)
        loss = self.criterion(logits, y_indices)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y_indices)
        f1 = self.test_f1(preds, y_indices)
        self.test_confusion.update(preds, y_indices)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        self.log("test_f1", f1, on_step=False, on_epoch=True)
        
    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """Prediction step."""
        x, y = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        y_true = torch.argmax(y, dim=1)
        
        return {
            "predictions": preds,
            "probabilities": probs,
            "ground_truth": y_true
        }
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine Annealing con Warmup - mejor que ReduceLROnPlateau
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,      # Restart cada 10 épocas
            T_mult=2,    # Duplicar periodo cada restart
            eta_min=1e-6 # LR mínimo
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
