import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

from src.constants import CRUDE_DATASETS_DIR


@dataclass
class FeatureMetrics:
    """Metrics for a single feature in a time series dataset."""
    feature_name: str
    start_date: str
    end_date: str
    frequency: Literal["daily", "weekly", "monthly", "quarterly", "annual", "irregular"]
    total_points: int
    null_count: int
    null_percentage: float
    coverage_days: int
    avg_days_between: float


class FeatureFrequencyAnalyzer:
    """Analyze feature frequencies and quality metrics in time series datasets."""
    
    def __init__(self, date_col: str = "date"):
        """
        Initialize the analyzer.
        
        Args:
            date_col: Name of the date column in CSV files
        """
        self.date_col = date_col
        self.metrics: dict[str, list[FeatureMetrics]] = {}
    
    @staticmethod
    def infer_frequency(dates: pd.Series) -> tuple[Literal["daily", "weekly", "monthly", "quarterly", "annual", "irregular"], float]:
        """Infer the frequency of a time series based on average days between observations."""
        if len(dates) < 2:
            return "irregular", 0.0
        
        dates = pd.to_datetime(dates).sort_values()
        deltas = dates.diff().dt.days.dropna()  # type: ignore
        
        if len(deltas) == 0:
            return "irregular", 0.0
        
        avg_delta = deltas.median()
        
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
    
    def analyze_feature(self, df: pd.DataFrame, feature_name: str) -> FeatureMetrics:
        """Analyze a single feature column in the context of time series stock data."""
        feature_data = df[[self.date_col, feature_name]].copy()
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
        
        start_date = feature_data[self.date_col].min()
        end_date = feature_data[self.date_col].max()
        total_points = len(feature_data)
        coverage_days = (end_date - start_date).days
        
        frequency, avg_days = self.infer_frequency(feature_data[self.date_col])
        
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
    
    def analyze_csv(self, csv_path: Path) -> list[FeatureMetrics]:
        """Analyze all features in a CSV file with stock market time series data."""
        df = pd.read_csv(csv_path, parse_dates=[self.date_col])
        df = df.sort_values(self.date_col)
        
        feature_cols = [col for col in df.columns if col != self.date_col]
        
        metrics = []
        for feature in feature_cols:
            metric = self.analyze_feature(df, feature)
            metrics.append(metric)
        
        self.metrics[csv_path.stem] = metrics
        return metrics
    
    def analyze_multiple(self, csv_paths: list[Path]) -> dict[str, list[FeatureMetrics]]:
        """Analyze multiple CSV files."""
        for csv_path in csv_paths:
            self.analyze_csv(csv_path)
        return self.metrics
    
    @staticmethod
    def print_table(metrics: list[FeatureMetrics], title: str = "Feature Analysis") -> None:
        """Print analysis results in a formatted table."""
        print(f"{'='*120}")
        print(f"{title:^120}")
        print(f"{'='*120}")
        print(f"{'Feature':<30} {'Start Date':<12} {'End Date':<12} {'Freq':<12} {'Points':<8} {'Null Cnt':<10} {'Null %':<8} {'Days':<8} {'Avg Δ':<8}")
        print(f"{'-'*120}")
        
        for m in metrics:
            print(f"{m.feature_name:<30} {m.start_date:<12} {m.end_date:<12} {m.frequency:<12} "
                  f"{m.total_points:<8}  {m.null_count:<10} {m.null_percentage:<8.2f} {m.coverage_days:<8} {m.avg_days_between:<8.2f}")
        
        print(f"{'='*120}\n")
    
    def save_to_csv(self, metrics: list[FeatureMetrics], output_path: Path) -> None:
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
    
    def run_analysis(self, csv_paths: list[Path], output_dir: Path | None = None, print_results: bool = True) -> None:
        """Run complete analysis pipeline."""
        for csv_path in csv_paths:
            metrics = self.analyze_csv(csv_path)
            
            if print_results:
                self.print_table(metrics, title=f"Analysis: {csv_path.name}")
            
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{csv_path.stem}_analysis.csv"
                self.save_to_csv(metrics, output_path)


if __name__ == "__main__":
    csv_files = list(CRUDE_DATASETS_DIR.glob("*.csv"))
    csv_files = [f for f in csv_files if f.name != "CRUDE.csv"]
    
    analyzer = FeatureFrequencyAnalyzer(date_col="date")
    analyzer.run_analysis(csv_files)
