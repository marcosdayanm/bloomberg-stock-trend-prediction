# bloomberg-stock-trend-prediction

Repo that stores a Stock Trend prediction AI Model powered with Bloomberg data and indicators

## Getting started

### A. [Option A] Install uv package manager

#### 1. Install `uv` via Homebrew (macOS)

```bash
brew install uv
```

#### 2. Add/install a package (updates pyproject.toml + uv.lock)

```bash
uv add package_name
```

#### 3. Remove a package (updates pyproject.toml + uv.lock)

```bash
uv remove package_name
```

#### 4. Sync environment from uv.lock

```bash
uv sync
```

#### 5. Freeze installed packages (pip-compatible)

```bash
uv pip freeze > requirements.txt
```

#### 6. Run any script (as module):

```bash
uv run python -m [route.from.root]
```

### B. [Option B] Install dependencies via pip directly

Run the following command:

```bash
pip install -r requirements.txt
```
---

### C. Initialize dataset

#### 1. Load crude dataset

Create the `crude-datasets` directory at the project root level, and add the `.xlsx` dataset provided file into it

#### 2. Split dataset into several .csv files

Run the following command

```bash
uv run python -m src.preprocessing.parse_excel_dataset
```

#### 3. (Optional) Analyze feature frequency of each dataset

This step was very important for understanding each feature frequency and for defining steps to take in feature engineering step

Run the following command

```bash
uv run python -m src.preprocessing.csv_feature_freq_analyzer
```
