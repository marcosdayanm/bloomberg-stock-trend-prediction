# bloomberg-stock-trend-prediction

Stock prediction model using Bloomberg data and macroeconomic indicators. Predicts MSFT stock movements (UP/DOWN) 10 days ahead using deep learning.

## Installation

```bash
# Install uv (macOS)
brew install uv

# Install dependencies
uv sync
```

## Model Architecture

Hybrid deep learning model combining:

- **CNN layers** - Extract temporal patterns from 120-day sequences
- **Transformer encoder** - Capture long-range dependencies with multi-head attention
- **BiLSTM** - Model bidirectional temporal dynamics
- **Classification head** - Binary prediction (UP/DOWN)

## Dataset

The model uses 225 features including:

- Stock prices (MSFT, QQQ, AMZN, SPY, etc.)
- Technical indicators (moving averages, volatility, momentum)
- Macroeconomic data (VIX, GDP, inflation, Fed rates)
- Quarterly fundamentals (revenue, EPS, cash flow)

**Input shape:** `(batch, 120, 225)` - 120 days × 225 features  
**Output:** Binary classification (DOWN/UP)

## Training

```bash
uv run python -m src.model.train
```

Training uses:

- Mixed precision (FP16) for faster training
- Early stopping and learning rate scheduling
- PyTorch Lightning for experiment tracking

Monitor with TensorBoard:

```bash
tensorboard --logdir src/model/logs
```

## Predictions

### Interactive Dashboard

```bash
uv run streamlit run predict_dashboard.py
```

Opens web interface at http://localhost:8501 with:

- Live predictions with real-time animation
- Confusion matrix
- Accuracy by class
- Adjustable speed (0-1 sec)

### Command Line

```bash
uv run python -m src.model.predict
```

## Results

The model achieves ~83% accuracy on test data for binary UP/DOWN classification.
