# bloomberg-stock-trend-prediction

Stock trend prediction AI model powered by Bloomberg data and macroeconomic indicators. Predicts future stock price movements using CNN-1D + BiLSTM architecture with 60-day historical sequences.

## Quick Start

### Installation

#### Option A: uv package manager (recommended)

```bash
# Install uv via Homebrew (macOS)
brew install uv

# Sync environment
uv sync
```

#### Option B: pip

```bash
pip install -r requirements.txt
```

---

## Dataset Preprocessing

### Complete Pipeline (Recommended)

Run the entire preprocessing workflow with a single command:

```bash
uv run python -m src.preprocessing.pipeline
```

This executes all preprocessing steps:

1. Parse Bloomberg Excel → CSV files
2. Combine datasets with feature prefixes
3. Analyze feature frequencies (optional)
4. Build time series sequences (60 days lookback, 10 days forward)
5. Generate analysis plots (price, volatility, macro indicators, etc.)
6. Save final dataset as `.npy` arrays

**Output:**

- `datasets/npy/msft_10day_prediction_X.npy` - Input sequences (6433, 60, 237)
- `datasets/npy/msft_10day_prediction_y.npy` - One-hot labels (6433, 12)
- `datasets/analysis/*.png` - 19 analysis plots

---

## Preprocessing Modules

### 1. `parse_excel_dataset.py` - Bloomberg Excel Parser

**Purpose:** Parse Bloomberg Excel files into clean CSV datasets

**Classes:**

- `BloombergExcelParser` - Extracts time series from Excel sheets
- `DatasetCombiner` - Merges CSV files with prefixed columns

**Usage:**

```bash
uv run python -m src.preprocessing.parse_excel_dataset
```

**Output:** Individual CSV files in `crude-datasets/` (e.g., `MSFT_CRUDE.csv`, `INDICATORS_CRUDE.csv`)

---

### 2. `csv_feature_freq_analyzer.py` - Feature Quality Analyzer

**Purpose:** Analyze feature frequencies and data quality metrics

**Class:**

- `FeatureFrequencyAnalyzer` - Detects daily/weekly/monthly/quarterly patterns, calculates null percentages

**Usage:**

```bash
uv run python -m src.preprocessing.csv_feature_freq_analyzer
```

**Output:** Terminal tables + optional CSV reports in `datasets/analysis/`

---

### 3. `stock_dataset_builder.py` - Time Series Dataset Builder

**Purpose:** Build training-ready datasets with feature engineering

**Class:**

- `StockDatasetBuilder` - Complete pipeline for sequence creation

**Features:**

- NYSE trading calendar filtering
- Technical indicators (returns, momentum, volatility, ROC)
- Macro ratio derivatives (unemployment change, inflation acceleration)
- Forward-fill for quarterly/monthly data propagation
- Z-score normalization
- One-hot encoded labels with configurable return bins

**Usage:**

```bash
uv run python -m src.preprocessing.stock_dataset_builder
```

---

### 4. `dataset_analyzer.py` - Statistical Analysis & Visualization

**Purpose:** Generate comprehensive dataset analysis and plots

**Class:**

- `DatasetAnalyzer` - Statistical analysis and visualization tools

**Analysis includes:**

- Price fluctuation (returns, volatility, Sharpe ratio, skewness)
- Label distribution across return bins
- Feature correlations with target price
- Macroeconomic indicator comparisons
- Quarterly fundamental trends

**Plots generated:**

- Price history
- Daily/forward returns distributions
- Rolling volatility
- Label distribution (bar chart)
- Cumulative returns
- Volume trends
- Macro indicators vs price (6 plots)
- Quarterly fundamentals vs price (6 plots)

---

### 5. `pipeline.py` - Complete Preprocessing Orchestration

**Purpose:** End-to-end preprocessing automation in a single function

**Function:**

- `run_preprocessing()` - Executes all preprocessing steps sequentially

**Configuration:**

```python
X, y = run_preprocessing(
    target_ticker="MSFT",
    sequence_length=60,    # Days of historical data
    horizon=10,            # Days ahead to predict
    return_bins=[...],     # Custom return bins
    skip_excel_parsing=True,  # Skip if CSVs already exist
    analyze_features=False    # Skip feature frequency analysis
)
```

**Simple execution:**

```bash
uv run python -m src.preprocessing.pipeline
```

The pipeline automatically:

1. Parses Excel files (if needed)
2. Combines CSVs with feature prefixes
3. Analyzes feature frequencies (optional)
4. Builds time series sequences
5. Generates 19 analysis plots
6. Saves dataset as `.npy` files

---

## Dataset Specifications

**Input (X):**

- Shape: `(# trading day samples, # hist. sequences, # features)`
- N training day samples
- N-day historical sequences
- N features (prices, technical indicators, macro factors, fundamentals)

**Output (y):**

- Shape: `(# N-day historical sequences, # one hot encoded label)`
- One-hot encoded labels
- N return bins:

**Features include:**

- Stock prices (MSFT, AAPL, GOOG, AMZN, NVDA, IBM, QQQ, SPY)
- Technical indicators (returns, momentum, volatility, moving averages)
- Macroeconomic indicators (VIX, unemployment, GDP, inflation, Fed rates, DXY)
- Quarterly fundamentals (revenue, EPS, EBITDA, cash flow, margins)
- Time features (day of week, month, quarter)

---

## Next Steps

After preprocessing, the dataset is ready for model training:

1. Load arrays: `X = np.load('datasets/npy/msft_10day_prediction_X.npy')`
2. Build CNN-1D + BiLSTM model
3. Train with appropriate loss function (categorical crossentropy)
4. Evaluate on validation set

---

## Development

```bash
# Add new dependency
uv add package_name

# Remove dependency
uv remove package_name

# Generate requirements.txt
uv pip freeze > requirements.txt
```
