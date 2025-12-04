import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy, F1Score, ConfusionMatrix, MeanSquaredError, MeanAbsoluteError

from src.model.config import ModelConfig


class CNNBiLSTMModel(L.LightningModule):
    """CNN-1D + BiLSTM + Transformer model with attention for stock trend prediction."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.save_hyperparameters(vars(config))
        self.config = config
        
        # 5 CNN layers with Batch Normalization
        self.conv1 = nn.Conv1d(
            in_channels=config.n_features,
            out_channels=config.cnn_filters_1,
            kernel_size=config.cnn_kernel_size,
            padding="same"
        )
        self.bn1 = nn.BatchNorm1d(config.cnn_filters_1)
        
        self.conv2 = nn.Conv1d(
            in_channels=config.cnn_filters_1,
            out_channels=config.cnn_filters_2,
            kernel_size=config.cnn_kernel_size,
            padding="same"
        )
        self.bn2 = nn.BatchNorm1d(config.cnn_filters_2)
        
        self.conv3 = nn.Conv1d(
            in_channels=config.cnn_filters_2,
            out_channels=config.cnn_filters_3,
            kernel_size=config.cnn_kernel_size,
            padding="same"
        )
        self.bn3 = nn.BatchNorm1d(config.cnn_filters_3)
        
        self.conv4 = nn.Conv1d(
            in_channels=config.cnn_filters_3,
            out_channels=config.cnn_filters_4,
            kernel_size=config.cnn_kernel_size,
            padding="same"
        )
        self.bn4 = nn.BatchNorm1d(config.cnn_filters_4)
        
        self.conv5 = nn.Conv1d(
            in_channels=config.cnn_filters_4,
            out_channels=config.cnn_filters_5,
            kernel_size=config.cnn_kernel_size,
            padding="same"
        )
        self.bn5 = nn.BatchNorm1d(config.cnn_filters_5)
        
        self.cnn_dropout = nn.Dropout(config.cnn_dropout)
        
        # 3 BiLSTM bidirectional layers
        self.lstm = nn.LSTM(
            input_size=config.cnn_filters_5,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0
        )
        
        # LSTM bidirectional output dimension
        lstm_output_dim = config.lstm_hidden_size * 2
        
        # Transformer
        self.use_transformer = config.use_transformer
        if self.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=lstm_output_dim,
                nhead=config.transformer_heads,
                dim_feedforward=lstm_output_dim * 2,
                dropout=config.lstm_dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.transformer_layers
            )
        
        # Attention layer
        self.attention = nn.Linear(lstm_output_dim, 1)
        
        # 4 Dense layers with Batch Normalization
        self.fc1 = nn.Linear(lstm_output_dim, config.dense_hidden_1)
        self.bn_fc1 = nn.BatchNorm1d(config.dense_hidden_1)
        
        self.fc2 = nn.Linear(config.dense_hidden_1, config.dense_hidden_2)
        self.bn_fc2 = nn.BatchNorm1d(config.dense_hidden_2)
        
        self.fc3 = nn.Linear(config.dense_hidden_2, config.dense_hidden_3)
        self.bn_fc3 = nn.BatchNorm1d(config.dense_hidden_3)
        
        # Output layer: 1 for regression, n_classes for classification
        output_dim = 1 if config.task_type == 'regression' else config.n_classes
        self.fc4 = nn.Linear(config.dense_hidden_3, output_dim)
        
        self.dense_dropout = nn.Dropout(config.dense_dropout)
        
        # Loss function
        if config.task_type == 'classification':
            self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        else:
            self.criterion = nn.MSELoss()
        
        # Metrics
        if config.task_type == 'classification':
            self.train_acc = Accuracy(task="multiclass", num_classes=config.n_classes)
            self.val_acc = Accuracy(task="multiclass", num_classes=config.n_classes)
            self.test_acc = Accuracy(task="multiclass", num_classes=config.n_classes)
            
            self.val_f1 = F1Score(task="multiclass", num_classes=config.n_classes, average="weighted")
            self.test_f1 = F1Score(task="multiclass", num_classes=config.n_classes, average="weighted")
            
            self.test_confusion = ConfusionMatrix(task="multiclass", num_classes=config.n_classes)
        else:
            self.train_mse = MeanSquaredError()
            self.train_mae = MeanAbsoluteError()
            self.val_mse = MeanSquaredError()
            self.val_mae = MeanAbsoluteError()
            self.test_mse = MeanSquaredError()
            self.test_mae = MeanAbsoluteError()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """        
        Args: x: Input tensor (batch_size, seq_length, n_features)
        -> Logits (batch_size, n_classes)
        """
        x = x.transpose(1, 2)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.cnn_dropout(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.cnn_dropout(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.cnn_dropout(x)
        
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.cnn_dropout(x)
        
        x = torch.relu(self.bn5(self.conv5(x)))
        x = self.cnn_dropout(x)
        
        x = x.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x)
        
        if self.use_transformer:
            lstm_out = self.transformer(lstm_out)
        
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        x = torch.relu(self.bn_fc1(self.fc1(attended)))
        x = self.dense_dropout(x)
        
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dense_dropout(x)
        
        x = torch.relu(self.bn_fc3(self.fc3(x)))
        x = self.dense_dropout(x)
        
        logits = self.fc4(x)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        outputs = self(x)
        
        if self.config.task_type == 'classification':
            # Classification
            if y.dim() > 1 and y.size(1) > 1:
                y = torch.argmax(y, dim=1)
            
            loss = self.criterion(outputs, y)
            preds = torch.argmax(outputs, dim=1)
            self.train_acc(preds, y)
            
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        else:
            # Regression
            if y.dim() > 1 and y.size(1) == 1:
                y = y.squeeze(1)
            outputs = outputs.squeeze(1)
            
            loss = self.criterion(outputs, y)
            self.train_mse(outputs, y)
            self.train_mae(outputs, y)
            
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_mse', self.train_mse, on_step=False, on_epoch=True, prog_bar=False)
            self.log('train_mae', self.train_mae, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        outputs = self(x)
        
        if self.config.task_type == 'classification':
            # Classification
            if y.dim() > 1 and y.size(1) > 1:
                y = torch.argmax(y, dim=1)
            
            loss = self.criterion(outputs, y)
            preds = torch.argmax(outputs, dim=1)
            self.val_acc(preds, y)
            self.val_f1(preds, y)
            
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=False)
        else:
            # Regression
            if y.dim() > 1 and y.size(1) == 1:
                y = y.squeeze(1)
            outputs = outputs.squeeze(1)
            
            loss = self.criterion(outputs, y)
            self.val_mse(outputs, y)
            self.val_mae(outputs, y)
            
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_mse', self.val_mse, on_step=False, on_epoch=True, prog_bar=False)
            self.log('val_mae', self.val_mae, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        outputs = self(x)
        
        if self.config.task_type == 'classification':
            # Classification
            if y.dim() > 1 and y.size(1) > 1:
                y = torch.argmax(y, dim=1)
            
            loss = self.criterion(outputs, y)
            preds = torch.argmax(outputs, dim=1)
            self.test_acc(preds, y)
            self.test_f1(preds, y)
            self.test_confusion.update(preds, y)
            
            self.log('test_loss', loss, on_step=False, on_epoch=True)
            self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
            self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)
        else:
            # Regression
            if y.dim() > 1 and y.size(1) == 1:
                y = y.squeeze(1)
            outputs = outputs.squeeze(1)
            
            loss = self.criterion(outputs, y)
            self.test_mse(outputs, y)
            self.test_mae(outputs, y)
            
            self.log('test_loss', loss, on_step=False, on_epoch=True)
            self.log('test_mse', self.test_mse, on_step=False, on_epoch=True)
            self.log('test_mae', self.test_mae, on_step=False, on_epoch=True)
    
    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        x, y = batch
        outputs = self(x)
        
        if self.config.task_type == 'classification':
            # Classification
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            if y.dim() > 1 and y.size(1) > 1:
                y = torch.argmax(y, dim=1)
            
            return {
                "predictions": preds,
                "probabilities": probs,
                "ground_truth": y
            }
        else:
            # Regression
            if y.dim() > 1 and y.size(1) == 1:
                y = y.squeeze(1)
            outputs = outputs.squeeze(1)
            
            return {
                "predictions": outputs,
                "ground_truth": y
            }
    
    def configure_optimizers(self):
        """Configurar optimizador y scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
