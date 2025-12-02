# Quick Start: Using Pre-trained Model

## 1. Load and Test (30 seconds)

```bash
# Run the demo script
uv run python load_pretrained_model.py
```

**Expected output:**
- Model loaded successfully
- Test accuracy: 79.5%
- Example predictions with confidence scores

---

## 2. Use in Your Code (Copy-Paste Ready)

```python
import torch
import numpy as np
from src.model.model import CNNBiLSTMModel
from src.model.config import ModelConfig

# Load model
config = ModelConfig()
model = CNNBiLSTMModel.load_from_checkpoint(
    'pretrained_models/best_model_v2.0.ckpt',
    config=config,
    map_location='cpu'  # or 'mps', 'cuda'
)
model.eval()

# Your data: shape (batch, 30 days, 50 features)
X = np.load('your_preprocessed_data.npy')
X_tensor = torch.from_numpy(X).float()

# Predict
with torch.no_grad():
    logits = model(X_tensor)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

# Results
for i in range(len(preds)):
    signal = "SELL" if preds[i] == 0 else "BUY"
    confidence = probs[i, preds[i]].item()
    print(f"{signal} ({confidence:.1%} confident)")
```

---

## 3. Requirements

**Input data must be:**
- Shape: `(batch_size, 30, 50)`
- 30 days of historical data
- 50 features (same as training - see `datasets/npy/selected_features_indices.npy`)
- Normalized/standardized like training data

**See:** `regenerate_dataset_focused.py` for preprocessing pipeline

---

## 4. Fine-tune on Your Data

```python
# Load pre-trained weights
model = CNNBiLSTMModel.load_from_checkpoint(
    'pretrained_models/best_model_v2.0.ckpt',
    config=config
)

# Lower learning rate for fine-tuning
config.learning_rate = 0.00005
config.max_epochs = 50

# Train on your data
trainer = L.Trainer(max_epochs=50, accelerator='mps')
trainer.fit(model, your_datamodule)
```

---

## 5. Model Details

| Property | Value |
|----------|-------|
| Test Accuracy | 79.5% |
| Model Size | 88 MB |
| Parameters | 7.7M |
| Input Shape | (batch, 30, 50) |
| Output | 2 classes (Bajista/Alcista) |
| Inference Time | ~3ms per sample (M3 Pro) |

**See `pretrained_models/README.md` for full documentation.**

---

## 6. Troubleshooting

**Error: "Shape mismatch"**
- Check input shape is exactly `(batch, 30, 50)`

**Error: "Module not found"**
- Run from project root directory
- Ensure all dependencies installed: `uv sync`

**Poor predictions on new data**
- Check if features match training distribution
- Consider fine-tuning on your specific dataset

---

**For more details:**
- Full documentation: `pretrained_models/README.md`
- Training details: `CHANGELOG.md`
- Architecture analysis: `EXECUTIVE_SUMMARY.md`
