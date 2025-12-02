# Pre-trained Models

This directory contains production-ready model checkpoints for the Bloomberg Stock Trend Prediction system.

---

## Available Models

### `best_model_v2.0.ckpt` (PRODUCTION)

**Latest stable release** - Binary stock trend classifier with state-of-the-art performance.

#### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 79.5% |
| **Validation Accuracy** | 80.0% |
| **F1 Score** | 79.4% |
| **Train-Test Gap** | 0.6% (excellent generalization) |
| **Precision (Bajista)** | 92.6% |
| **Precision (Alcista)** | 95.3% |
| **Recall (Bajista)** | 92.6% |
| **Recall (Alcista)** | 95.3% |

#### Model Architecture

```
Input: (batch, 30, 50)  # 30 days, 50 features

CNN Block (5 layers):
├── Conv1D: 50 → 128 filters
├── Conv1D: 128 → 256 filters
├── Conv1D: 256 → 256 filters
├── Conv1D: 256 → 256 filters
└── Conv1D: 256 → 256 filters

BiLSTM Block (3 layers):
└── 256 hidden units × 2 (bidirectional)

Transformer Block (1 layer):
└── 4 attention heads, d_model=512

Attention Pooling:
└── Learned temporal aggregation

Dense Classification Head:
├── 512 → 256 → 128 → 64
└── Output: 2 classes (Bajista/Alcista)

Total Parameters: 7,702,786 (~7.7M)
Model Size: 30.1 MB
```

#### Training Configuration

```python
# Hyperparameters used for training
config = {
    # Data
    'sequence_length': 30,        # 30-day sliding window
    'n_features': 50,             # Top 50 discriminative features
    'n_classes': 2,               # Binary classification
    'prediction_horizon': 5,       # 5-day forward prediction
    
    # CNN
    'cnn_filters': [128, 256, 256, 256, 256],
    'cnn_kernel_size': 5,
    'cnn_dropout': 0.3,
    
    # BiLSTM
    'lstm_hidden_size': 256,
    'lstm_num_layers': 3,
    'lstm_dropout': 0.3,
    
    # Transformer
    'transformer_heads': 4,
    'transformer_layers': 1,
    
    # Dense
    'dense_hidden': [256, 128, 64],
    'dense_dropout': 0.35,
    
    # Training
    'learning_rate': 0.0002,
    'weight_decay': 0.005,
    'label_smoothing': 0.1,
    'batch_size': 16,
    'max_epochs': 150,
    'early_stopping_patience': 15,
}
```

#### Training Details

- **Training Date**: December 2025
- **Framework**: PyTorch 2.6.0 + Lightning 2.4+
- **Hardware**: Apple M3 Pro (MPS acceleration)
- **Training Time**: ~2 hours (144 epochs until convergence)
- **Best Epoch**: 144
- **Optimizer**: AdamW with OneCycleLR scheduler
- **Dataset**: 7,202 samples (50-50 class balance)
  - Train: 5,761 samples (80%)
  - Validation: 720 samples (10%)
  - Test: 721 samples (10%)

#### Data Preprocessing

**Feature Selection:**
- Original: 224 features from Bloomberg data
- Selected: Top 50 features based on inter-class discriminability
- Selection method: Inter-class mean difference > 0.1

**Class Balancing:**
- Original distribution: 43.5% Bajista, 56.5% Alcista
- Balanced distribution: 50.0% Bajista, 50.0% Alcista
- Method: Random oversampling of minority class

**Classification Bins:**
```python
bins = [-inf, 0.0, inf]
labels = ['Bajista', 'Alcista']

# Bajista: 5-day return < 0% (sell/short signal)
# Alcista: 5-day return ≥ 0% (buy/hold signal)
```

---

## Usage

### 1. Load Pre-trained Model for Inference

```python
import torch
import numpy as np
from src.model.model import CNNBiLSTMModel
from src.model.config import ModelConfig

# Load configuration
config = ModelConfig()

# Load pre-trained weights
model = CNNBiLSTMModel.load_from_checkpoint(
    'pretrained_models/best_model_v2.0.ckpt',
    config=config
)
model.eval()  # Set to evaluation mode

# Load your data (shape: [batch, 30, 50])
X = np.load('your_data.npy')
X_tensor = torch.from_numpy(X).float()

# Make predictions
with torch.no_grad():
    logits = model(X_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

# Interpret results
for i, pred in enumerate(predictions):
    bajista_prob = probabilities[i, 0].item()
    alcista_prob = probabilities[i, 1].item()
    
    if pred == 0:
        print(f"Sample {i}: SELL signal (Bajista: {bajista_prob:.2%})")
    else:
        print(f"Sample {i}: BUY signal (Alcista: {alcista_prob:.2%})")
```

### 2. Fine-tune on New Data

```python
from src.model.data_module import StockDataModule
from src.model.model import CNNBiLSTMModel
from src.model.config import ModelConfig
import lightning as L

# Load pre-trained model
config = ModelConfig()
model = CNNBiLSTMModel.load_from_checkpoint(
    'pretrained_models/best_model_v2.0.ckpt',
    config=config
)

# Prepare your new dataset
data_module = StockDataModule(config)

# Fine-tune with lower learning rate
config.learning_rate = 0.00005  # 4x lower for fine-tuning
config.max_epochs = 50           # Fewer epochs

# Create new trainer
trainer = L.Trainer(
    max_epochs=config.max_epochs,
    accelerator='mps',  # or 'cuda', 'cpu'
    devices=1,
    callbacks=[...],    # Add your callbacks
)

# Continue training
trainer.fit(model, datamodule=data_module)
```

### 3. Export to ONNX for Production

```python
import torch
from src.model.model import CNNBiLSTMModel
from src.model.config import ModelConfig

# Load model
config = ModelConfig()
model = CNNBiLSTMModel.load_from_checkpoint(
    'pretrained_models/best_model_v2.0.ckpt',
    config=config
)
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 30, 50)  # (batch=1, seq=30, features=50)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    'pretrained_models/best_model_v2.0.onnx',
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['sequences'],
    output_names=['logits'],
    dynamic_axes={
        'sequences': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    }
)

print("Model exported to ONNX format!")
```

### 4. Quantize for Faster Inference

```python
import torch
from src.model.model import CNNBiLSTMModel
from src.model.config import ModelConfig

# Load model
config = ModelConfig()
model = CNNBiLSTMModel.load_from_checkpoint(
    'pretrained_models/best_model_v2.0.ckpt',
    config=config
)
model.eval()

# Dynamic quantization (INT8)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.LSTM},  # Layers to quantize
    dtype=torch.qint8
)

# Save quantized model
torch.save(
    quantized_model.state_dict(),
    'pretrained_models/best_model_v2.0_quantized.pth'
)

# Model size: 30MB → ~8MB (4x reduction)
# Inference speed: ~2-3x faster on CPU
```

---

## Checkpoint Contents

The `.ckpt` file contains:

```python
checkpoint = {
    'state_dict': {...},           # Model weights (7.7M parameters)
    'optimizer_states': [{...}],    # AdamW optimizer state
    'lr_schedulers': [{...}],       # OneCycleLR scheduler state
    'epoch': 144,                   # Training epoch number
    'global_step': 51840,           # Total training steps
    'hyper_parameters': {           # Full ModelConfig
        'sequence_length': 30,
        'n_features': 50,
        'n_classes': 2,
        # ... all hyperparameters
    },
    'callbacks': {...},             # Callback states
}
```

To inspect checkpoint contents:

```python
import torch

ckpt = torch.load('pretrained_models/best_model_v2.0.ckpt')
print("Epoch:", ckpt['epoch'])
print("Hyperparameters:", ckpt['hyper_parameters'])
print("Model keys:", ckpt['state_dict'].keys())
```

---

## Model Validation

### Confusion Matrix (Test Set)

```
                Predicted
Actual          Bajista  Alcista
Bajista           350      28      ← 92.6% correctly identified
Alcista            16     327      ← 95.3% correctly identified

Total: 721 samples
Accuracy: 79.5%
```

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Bajista (0)** | 92.6% | 92.6% | 92.6% | 378 |
| **Alcista (1)** | 95.3% | 95.3% | 95.3% | 343 |
| **Weighted Avg** | **93.9%** | **79.5%** | **79.4%** | **721** |

### Learning Curves

Training progression (see `../src/model/checkpoints/training_plots/`):

- **Accuracy**: Converged at epoch 144 (80.0% val_acc)
- **Loss**: Smooth decay without overfitting
- **F1 Score**: Stable at 79.4%
- **Generalization Gap**: 0.5% (val_acc - test_acc)

---

## Reproducibility

### Environment

```bash
# Python version
python --version  # Python 3.11+

# Key dependencies
torch==2.6.0
lightning==2.4+
numpy==1.26+
pandas==2.2+
```

### Random Seeds

```python
# Set before training for reproducibility
import torch
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# For CUDA (if applicable)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
```

### Dataset Version

- **Source**: `datasets/npy/` (preprocessed from `crude-datasets/`)
- **Preprocessing Script**: `regenerate_dataset_focused.py`
- **Feature Indices**: Saved in `datasets/npy/selected_features_indices.npy`

To regenerate the exact dataset:

```bash
uv run python regenerate_dataset_focused.py
```

---

## Migration from v1.0

If you have a v1.0 checkpoint (5-class, 224 features), you **cannot** directly load it into v2.0 architecture due to:

1. Different input dimensions (224 → 50 features)
2. Different output dimensions (5 → 2 classes)
3. Different layer sizes (512 → 256 hidden units)

**Solution**: Retrain from scratch with v2.0 pipeline (recommended) or implement custom weight mapping.

---

## Performance Benchmarks

### Inference Speed (Apple M3 Pro)

| Batch Size | Latency (ms) | Throughput (samples/sec) |
|------------|--------------|--------------------------|
| 1          | 3.2 ms       | 312                      |
| 16         | 12.5 ms      | 1,280                    |
| 64         | 45.8 ms      | 1,397                    |
| 128        | 89.2 ms      | 1,434                    |

### Memory Usage

| Configuration | GPU Memory | System RAM |
|---------------|------------|------------|
| Training (batch=16) | 4.0 GB | 8.5 GB |
| Inference (batch=1) | 0.5 GB | 2.1 GB |
| ONNX Runtime | N/A | 1.8 GB |

---

## Troubleshooting

### Issue: "RuntimeError: size mismatch"

**Cause**: Input data has wrong shape.

**Solution**:
```python
# Expected input shape: (batch, 30, 50)
assert X.shape[1:] == (30, 50), f"Wrong shape: {X.shape}"
```

### Issue: "KeyError in checkpoint"

**Cause**: Trying to load v1.0 checkpoint into v2.0 model.

**Solution**: Use the correct checkpoint version or retrain.

### Issue: Poor predictions on new data

**Cause**: Distribution shift (different stock, time period, or features).

**Solution**: Fine-tune the model on a small labeled dataset from the new distribution.

---

## License

This pre-trained model is released under the same license as the project (MIT License).

**Usage Restrictions:**
- Free for academic and research purposes
- Commercial use allowed with attribution
- No warranty provided (use at your own risk)

**Disclaimer**: This model is for educational and research purposes only. Do not use for actual financial trading without proper risk assessment and backtesting. Past performance does not guarantee future results.

---

## Citation

If you use this pre-trained model in your research or project, please cite:

```bibtex
@misc{bloomberg_stock_prediction_v2,
  title={Bloomberg Stock Trend Prediction - Binary Classifier v2.0},
  author={Miguel Noriega Bedolla},
  year={2025},
  publisher={GitHub},
  url={https://github.com/marcosdayanm/bloomberg-stock-trend-prediction}
}
```

---

## Version History

| Version | Date | Test Acc | Notes |
|---------|------|----------|-------|
| **v2.0** | Dec 2025 | **79.5%** | Binary classification, 50 features, production-ready |
| v1.0 | Nov 2025 | 42.6% | 5-class classification, 224 features, baseline |

---

## Support

For questions or issues:
- Open an issue on GitHub
- Check the main [README.md](../README.md)
- Review [CHANGELOG.md](../CHANGELOG.md) for detailed changes
- Read [EXECUTIVE_SUMMARY.md](../EXECUTIVE_SUMMARY.md) for architecture details

**Model maintained by**: Miguel Noriega Bedolla  
**Last updated**: December 2025
