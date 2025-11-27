import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

from src.constants import CRUDE_DATASETS_DIR


@dataclass
class FeatureMetrics:
    """Metrics for a single feature in a time series dataset."""
    feature_name: str # Name of the feature/column
    start_date: str # Start date of the time series
    end_date: str # End date of the time series
    frequency: Literal["daily", "weekly", "monthly", "quarterly", "annual", "irregular"] # Inferred frequency
    total_points: int # Total non-null data points
    null_count: int # Count of null values within the active date range
    null_percentage: float # Percentage of null values within the active date range
    coverage_days: int # Number of days covered by the feature
    avg_days_between: float # Average days between observations

def _infer_frequency(dates: pd.Series) -> tuple[Literal["daily", "weekly", "monthly", "quarterly", "annual", "irregular"], float]:
    """Infer the frequency of a time series based on average days between observations."""
    if len(dates) < 2:
        return "irregular", 0.0
    
    dates = pd.to_datetime(dates).sort_values()
    deltas = dates.diff().dt.days.dropna()  # type: ignore
    
    if len(deltas) == 0:
        return "irregular", 0.0
    
    avg_delta = deltas.median()
    
    # Frequency thresholds
    if avg_delta <= 3:
        return "daily", avg_delta
    elif 4 <= avg_delta <= 10:
        return "weekly", avg_delta
    elif 20 <= avg_delta <= 40:
        return "monthly", avg_delta
    elif 80 <= avg_delta <= 100:
        return "quarterly", avg_delta
    elif 350 <= avg_delta <= 380:
        return "annual", avg_delta
    else:
        return "irregular", avg_delta


def analyze_feature(df: pd.DataFrame, feature_name: str, date_col: str = "date") -> FeatureMetrics:
    """Analyze a single feature column in the context of time series stock data."""
    feature_data = df[[date_col, feature_name]].copy()
    feature_data = feature_data[feature_data[feature_name].notna()]
    
    if len(feature_data) == 0:
        return FeatureMetrics(
            feature_name=feature_name,
            start_date="N/A",
            end_date="N/A",
            frequency="irregular",
            total_points=0,
            null_count=0,
            null_percentage=100.0,
            coverage_days=0,
            avg_days_between=0.0
        )
    
    start_date = feature_data[date_col].min()
    end_date = feature_data[date_col].max()
    total_points = len(feature_data)
    coverage_days = (end_date - start_date).days
    
    frequency, avg_days = _infer_frequency(feature_data[date_col])
    
    # Calculate expected data points based on frequency
    if avg_days > 0:
        expected_points = int(coverage_days / avg_days) + 1
    else:
        expected_points = total_points

    null_count = max(0, expected_points - total_points)
    null_percentage = (null_count / expected_points) * 100 if expected_points > 0 else 0.0
    
    return FeatureMetrics(
        feature_name=feature_name,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        frequency=frequency,
        total_points=total_points,
        null_count=null_count,
        null_percentage=round(null_percentage, 2),
        coverage_days=coverage_days,
        avg_days_between=round(avg_days, 2)
    )


def analyze_csv_features(csv_path: Path, date_col: str = "date") -> list[FeatureMetrics]:
    """Analyze all features in a CSV file with stock market time series data."""
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df.sort_values(date_col)
    
    # Get all columns except the date column
    feature_cols = [col for col in df.columns if col != date_col]
    
    metrics: list[FeatureMetrics] = []
    for feature in feature_cols:
        metric = analyze_feature(df, feature, date_col)
        metrics.append(metric)
    
    return metrics


def print_analysis_table(metrics: list[FeatureMetrics], title: str = "Feature Analysis") -> None:
    """Print analysis results in a formatted table to the terminal."""
    print(f"{'='*120}")
    print(f"{title:^120}")
    print(f"{'='*120}")
    print(f"{'Feature':<30} {'Start Date':<12} {'End Date':<12} {'Freq':<12} {'Points':<8} {'Null Cnt':<10} {'Null %':<8} {'Days':<8} {'Avg Δ':<8}")
    print(f"{'-'*120}")
    
    for m in metrics:
        print(f"{m.feature_name:<30} {m.start_date:<12} {m.end_date:<12} {m.frequency:<12} "
              f"{m.total_points:<8}  {m.null_count:<10} {m.null_percentage:<8.2f} {m.coverage_days:<8} {m.avg_days_between:<8.2f}")
    
    print(f"{'='*120}\n")


def save_analysis_to_csv(metrics: list[FeatureMetrics], output_path: Path) -> None:
    """Save analysis results to a CSV file."""
    data = [
        {
            "feature_name": m.feature_name,
            "start_date": m.start_date,
            "end_date": m.end_date,
            "frequency": m.frequency,
            "total_points": m.total_points,
            "null_count": m.null_count,
            "null_percentage": m.null_percentage,
            "coverage_days": m.coverage_days,
            "avg_days_between": m.avg_days_between
        }
        for m in metrics
    ]
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Analysis saved to: {output_path}")


def analyze_dataset_feature_frequencies(csv_paths: list[Path], output_dir: Path | None = None) -> None:
    """Analyze multiple CSV files and display results in terminal."""
    for csv_path in csv_paths:
        metrics = analyze_csv_features(csv_path)
        print_analysis_table(metrics, title=f"Analysis: {csv_path.name}")
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{csv_path.stem}_analysis.csv"
            save_analysis_to_csv(metrics, output_path)


if __name__ == "__main__":
    # Analyze all CSV files in crude datasets
    csv_files = list(CRUDE_DATASETS_DIR.glob("*.csv"))
    csv_files = [f for f in csv_files if f.name != "CRUDE.csv"]  # Exclude combined file
    
    analyze_dataset_feature_frequencies(csv_files)
