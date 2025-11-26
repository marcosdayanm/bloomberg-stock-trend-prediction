# bloomberg-stock-trend-prediction

Repo that stores a Stock Trend prediction AI Model powered with Bloomberg data and indicators

# Getting started

## (Recommended) Install uv package manager

### 1. Install `uv` via Homebrew (macOS)

```bash
brew install uv
```

### 2. Add/install a package (updates pyproject.toml + uv.lock)

```bash
uv add package_name
```

## 3. Remove a package (updates pyproject.toml + uv.lock)

```bash
uv remove package_name
```

## 4. Sync environment from uv.lock

```bash
uv sync
```

## 5. Freeze installed packages (pip-compatible)

```bash
uv pip freeze > requirements.txt
```


## 6. Run any script (as module):

```bash
uv run python -m [route.from.root]
```
