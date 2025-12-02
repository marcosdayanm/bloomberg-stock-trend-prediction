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
        # in_channels usa n_features del config
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
        
        # Tercera capa CNN para más profundidad
        self.conv3 = nn.Conv1d(
            in_channels=config.cnn_filters_2,
            out_channels=config.cnn_filters_3,
            kernel_size=config.cnn_kernel_size,
            padding="same"
        )
        self.bn3 = nn.BatchNorm1d(config.cnn_filters_3)
        self.dropout3 = nn.Dropout(config.cnn_dropout)
        
        # Cuarta capa CNN (NUEVA)
        self.conv4 = nn.Conv1d(
            in_channels=config.cnn_filters_3,
            out_channels=config.cnn_filters_4,
            kernel_size=config.cnn_kernel_size,
            padding="same"
        )
        self.bn4 = nn.BatchNorm1d(config.cnn_filters_4)
        self.dropout4 = nn.Dropout(config.cnn_dropout)
        
        # Quinta capa CNN (NUEVA)
        self.conv5 = nn.Conv1d(
            in_channels=config.cnn_filters_4,
            out_channels=config.cnn_filters_5,
            kernel_size=config.cnn_kernel_size,
            padding="same"
        )
        self.bn5 = nn.BatchNorm1d(config.cnn_filters_5)
        self.dropout5 = nn.Dropout(config.cnn_dropout)
        
        # BiLSTM layer (ahora recibe cnn_filters_5)
        self.lstm = nn.LSTM(
            input_size=config.cnn_filters_5,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0
        )
        self.lstm_dropout = nn.Dropout(config.lstm_dropout)
        
        # Transformer Encoder (NUEVO) - Para capturar patrones de largo plazo
        if config.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.lstm_hidden_size * 2,
                nhead=config.transformer_heads,
                dim_feedforward=config.lstm_hidden_size * 4,
                dropout=0.2,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.transformer_layers
            )
        else:
            self.transformer = None
        
        # Attention mechanism
        self.attention = nn.Linear(config.lstm_hidden_size * 2, 1)
        
        # Dense layers - Clasificador de 2 capas
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
        
        # Loss function para clasificación binaria
        # Distribución: Bajista (43.5%), Alcista (56.5%)
        # Peso inverso para balancear
        class_weights = torch.tensor([
            1.15,   # Clase 0 (Bajista) - aumentar peso (minoritaria)
            0.87    # Clase 1 (Alcista) - reducir peso (mayoritaria)
        ], dtype=torch.float32)
        
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.1  # AUMENTADO de 0.05 a 0.1 para evitar overconfidence
        )
        
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
        
        # CNN block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        # CNN block 4 (NUEVO)
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.dropout4(x)
        
        # CNN block 5 (NUEVO)
        x = self.conv5(x)
        x = self.bn5(x)
        x = torch.relu(x)
        x = self.dropout5(x)
        
        # BiLSTM expects (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, seq_len, cnn_filters_5)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden*2)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # Transformer (NUEVO) - Para patrones temporales complejos
        if self.transformer is not None:
            lstm_out = self.transformer(lstm_out)
        
        # Attention mechanism
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), dim=1
        )  # (batch, seq_len)
        
        # Apply attention
        context = torch.sum(
            lstm_out * attention_weights.unsqueeze(-1), dim=1
        )  # (batch, lstm_hidden*2)
        
        # Dense layers con BatchNorm - Clasificador profundo
        x = self.fc1(context)
        x = self.bn_fc1(x)
        x = torch.relu(x)
        x = self.fc1_dropout(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = torch.relu(x)
        x = self.fc2_dropout(x)
        
        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = torch.relu(x)
        x = self.fc3_dropout(x)
        
        logits = self.fc4(x)  # (batch, n_classes)
        
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
        
        # OneCycleLR - Mejor que CosineAnnealing para convergencia rápida
        from torch.optim.lr_scheduler import OneCycleLR
        
        # Calcular steps por época (aprox) - usando nuevo batch size
        train_samples = 5098  # Dataset binario: 6373 * 0.8 = 5098 muestras
        steps_per_epoch = train_samples // self.config.batch_size
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate * 5,  # Pico 5x el LR base (antes 10x)
            epochs=self.config.max_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.2,  # 20% warmup (más rápido)
            anneal_strategy='cos',
            div_factor=20.0,  # LR inicial = max_lr / 20
            final_div_factor=500.0  # LR final muy bajo para fine-tuning
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update every step, not epoch
                "frequency": 1
            }
        }
