"""Complete preprocessing pipeline: Excel → CSV → Dataset."""

import numpy as np
from pathlib import Path

from src.constants import CRUDE_DATASETS_DIR, DATASETS_DIR
from src.preprocessing.parse_excel_dataset import BloombergExcelParser, DatasetCombiner
from src.preprocessing.csv_feature_freq_analyzer import FeatureFrequencyAnalyzer
from src.preprocessing.stock_dataset_builder import StockDatasetBuilder


def run_preprocessing(
    target_ticker: str = "MSFT",
    sequence_length: int = 60,
    horizon: int = 10,
    task_type: str = 'classification',  # 'classification' | 'regression'
    return_bins: list[float] | None = None,
    start_date: str = "2000-01-03",
    end_date: str = "2025-11-20",
    skip_excel_parsing: bool = True,
    analyze_features: bool = False,
    analyze_data: bool = True,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run complete preprocessing pipeline.
    """
    if return_bins is None:
        if task_type == 'classification':
            return_bins = [-np.inf, 0, np.inf]
        else:
            return_bins = None  # No bins for regression
    
    print(f"\n{'='*80}")
    print(f"PREPROCESSING: {target_ticker} | {task_type.upper()} | {sequence_length}d seq | {horizon}d horizon")
    print(f"{'='*80}\n")
    
    # Parse Excel → CSV
    if not skip_excel_parsing:
        print("1. Parsing Excel...")
        combiner = DatasetCombiner()
        excel = combiner.get_latest_excel()
        parser = BloombergExcelParser(header_row=4, search_start_col=3)
        parser.process_workbook(excel, CRUDE_DATASETS_DIR, ignore_sheets={"META", "GOOG"})
        combiner.combine_csvs()
        print("CSV files created\n")
    
    # Analyze features
    if analyze_features:
        print("2. Analyzing features...")
        analyzer = FeatureFrequencyAnalyzer()
        csv_files = [f for f in CRUDE_DATASETS_DIR.glob("*.csv") if f.name != "CRUDE.csv"]
        analyzer.run_analysis(csv_files, DATASETS_DIR / "analysis", print_results=True)
        print("Analysis complete\n")
    
    # Build dataset
    print("3. Building dataset...")
    builder = StockDatasetBuilder(
        sequence_length=sequence_length,
        horizon=horizon,
        task_type=task_type,
        return_bins=return_bins,
        start_date=start_date,
        end_date=end_date,
        feature_columns=feature_columns
    )
    X, y = builder.build(target_ticker, visualize=False, analyze=analyze_data)
    builder.save_dataset(f"{target_ticker.lower()}_{horizon}day_prediction")
    print("Dataset building complete\n")
    
    return X, y


if __name__ == "__main__":
    TASK_TYPE = 'regression'  # regression | classification
    
    # Binary classification bins (only used if task_type='classification')
    binary_bins = [-np.inf, 0, np.inf]
    start_date = "2000-01-03"
    end_date = "2025-11-20"

    X, y = run_preprocessing(
        target_ticker="MSFT",
        sequence_length=30,
        horizon=10,
        task_type=TASK_TYPE,
        return_bins=binary_bins if TASK_TYPE == 'classification' else None,
        start_date=start_date,
        end_date=end_date,
        skip_excel_parsing=True,
        analyze_features=False,
        analyze_data=True,
    )
