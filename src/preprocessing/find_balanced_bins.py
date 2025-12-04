"""Find balanced return bins using quantiles."""

import numpy as np
import pandas as pd
from pathlib import Path

from src.constants import CRUDE_DATASETS_DIR
from src.preprocessing.stock_dataset_builder import StockDatasetBuilder


def find_balanced_bins(
    target_ticker: str = "MSFT",
    horizon: int = 10,
    n_bins: int = 13,
    start_date: str = "2000-01-03",
    end_date: str = "2025-11-20"
) -> list[float]:
    print(f"Finding balanced bins for {target_ticker} ({horizon}-day returns)...\n")
    
    # Load data
    csv_path = CRUDE_DATASETS_DIR / "CRUDE.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run pipeline first to create {csv_path}")
    
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.loc[start_date:end_date]
    
    # Calculate forward returns
    target_col = f"{target_ticker}_PX_LAST"
    if target_col not in df.columns:
        raise ValueError(f"Column {target_col} not found")
    
    forward_returns = (
        df[target_col].shift(-horizon) / df[target_col] - 1
    ) * 100  # Convert to percentage
    
    # Remove NaN values
    forward_returns = forward_returns.dropna()
    
    print(f"Total samples: {len(forward_returns)}")
    print(f"Return range: [{forward_returns.min():.2f}%, {forward_returns.max():.2f}%]")
    print(f"Mean return: {forward_returns.mean():.2f}%")
    print(f"Median return: {forward_returns.median():.2f}%\n")
    
    # Method 1: Equal-frequency bins (quantiles)
    print("=" * 80)
    print("METHOD 1: Quantile-based (equal frequency per bin)")
    print("=" * 80)
    
    # Create quantiles (n_bins-1 edges, excluding -inf and inf)
    n_quantiles = n_bins - 1
    quantiles = np.linspace(0, 1, n_quantiles + 1)
    bin_edges = forward_returns.quantile(quantiles).values
    
    # Add -inf and inf
    quantile_bins = [-np.inf] + bin_edges[1:-1].tolist() + [np.inf]
    
    print(f"\nQuantile bins ({n_bins} bins):")
    print([f"{x:.2f}" if not np.isinf(x) else ("inf" if x > 0 else "-inf") for x in quantile_bins])
    
    # Test distribution
    labels = pd.cut(forward_returns, bins=quantile_bins, labels=False)
    dist = labels.value_counts().sort_index()
    
    print("\nDistribution:")
    for i, count in dist.items():
        pct = count / len(labels) * 100
        bin_start = quantile_bins[int(i)]
        bin_end = quantile_bins[int(i) + 1]
        print(f"  Bin {int(i):2d}: {count:5d} samples ({pct:5.2f}%) | "
              f"[{bin_start:>7.2f}%, {bin_end:>7.2f}%]" if not np.isinf(bin_start) else 
              f"  Bin {int(i):2d}: {count:5d} samples ({pct:5.2f}%) | "
              f"[{'-inf':>7s}%, {bin_end:>7.2f}%]" if np.isinf(bin_start) and bin_start < 0 else
              f"  Bin {int(i):2d}: {count:5d} samples ({pct:5.2f}%) | "
              f"[{bin_start:>7.2f}%, {'inf':>7s}%]")
    
    # Method 2: Symmetric bins around 0
    print("\n" + "=" * 80)
    print("METHOD 2: Symmetric around 0")
    print("=" * 80)
    
    # Find symmetric percentiles
    abs_returns = forward_returns.abs()
    n_symmetric = (n_bins - 1) // 2  # Half bins on each side
    
    symmetric_quantiles = np.linspace(0, 1, n_symmetric + 1)[1:]  # Exclude 0
    symmetric_edges = abs_returns.quantile(symmetric_quantiles).values
    
    # Create symmetric bins
    symmetric_bins = (
        [-np.inf] + 
        [-x for x in symmetric_edges[::-1]] +  # Negative side
        [0] +
        symmetric_edges.tolist() +  # Positive side
        [np.inf]
    )
    
    print(f"\nSymmetric bins ({len(symmetric_bins) - 1} bins):")
    print([f"{x:.2f}" if not np.isinf(x) else ("inf" if x > 0 else "-inf") for x in symmetric_bins])
    
    # Test distribution
    labels_sym = pd.cut(forward_returns, bins=symmetric_bins, labels=False)
    dist_sym = labels_sym.value_counts().sort_index()
    
    print("\nDistribution:")
    for i, count in dist_sym.items():
        pct = count / len(labels_sym) * 100
        bin_start = symmetric_bins[int(i)]
        bin_end = symmetric_bins[int(i) + 1]
        print(f"  Bin {int(i):2d}: {count:5d} samples ({pct:5.2f}%) | "
              f"[{bin_start:>7.2f}%, {bin_end:>7.2f}%]" if not np.isinf(bin_start) and not np.isinf(bin_end) else
              f"  Bin {int(i):2d}: {count:5d} samples ({pct:5.2f}%) | "
              f"[{'-inf':>7s}%, {bin_end:>7.2f}%]" if np.isinf(bin_start) and bin_start < 0 else
              f"  Bin {int(i):2d}: {count:5d} samples ({pct:5.2f}%) | "
              f"[{bin_start:>7.2f}%, {'inf':>7s}%]")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    print("\nFor neural networks, quantile-based bins (Method 1) are usually better")
    print("because they ensure equal training samples per class.")
    print("\nCopy-paste for pipeline.py:")
    print(f"return_bins = {quantile_bins}")
    
    return quantile_bins


if __name__ == "__main__":
    # Find balanced bins
    balanced_bins = find_balanced_bins(
        target_ticker="MSFT",
        horizon=10,
        n_bins=13,  # Same as current setup
        start_date="2000-01-03",
        end_date="2025-11-20"
    )
