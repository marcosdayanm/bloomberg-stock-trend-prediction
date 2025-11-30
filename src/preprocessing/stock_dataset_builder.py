import pandas as pd
import numpy as np
from pathlib import Path
import pandas_market_calendars as mcal

from src.constants import CRUDE_DATASETS_DIR, DATASETS_DIR
from src.preprocessing.dataset_analyzer import DatasetAnalyzer


class StockDatasetBuilder:
    """Pipeline for creating stock prediction datasets with time series features."""
    
    def __init__(
        self,
        sequence_length: int = 60,
        horizon: int = 10,
        return_bins: list[float] = [-np.inf, -2, -1, 0, 1, 2, np.inf],
        start_date: str = "2000-01-03",
        end_date: str = "2025-11-20"
    ):
        """
        Initialize the dataset builder."""
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.return_bins = return_bins
        self.start_date = start_date
        self.end_date = end_date
        
        self.df: pd.DataFrame | None = None
        self.feature_cols: list[str] = []
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
    
    @staticmethod
    def get_us_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
        """Get all US trading days between start and end date."""
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        return schedule.index  # type: ignore
    
    def load_and_merge_datasets(self) -> pd.DataFrame:
        """Load all crude datasets and merge them on trading days."""
        all_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        df = pd.DataFrame({'date': all_dates})
        
        csv_files = list(CRUDE_DATASETS_DIR.glob("*.csv"))
        csv_files = [f for f in csv_files if f.name != "CRUDE.csv"]
        
        for csv_path in csv_files:
            data = pd.read_csv(csv_path, parse_dates=['date'])
            prefix = csv_path.stem
            data = data.rename(columns={col: f"{prefix}_{col}" for col in data.columns if col != 'date'})
            df = df.merge(data, on='date', how='left')
        
        df = df.sort_values('date').ffill()
        
        trading_days = self.get_us_trading_days(self.start_date, self.end_date)
        df = df[df['date'].isin(trading_days)].reset_index(drop=True)
        
        # Clean column names: remove CRUDE from all feature names
        df.columns = [col.replace('_CRUDE', '').replace('CRUDE_', '') for col in df.columns]
        
        return df
    
    @staticmethod
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features for better temporal understanding."""
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        return df
    
    @staticmethod
    def add_technical_features(df: pd.DataFrame, price_cols: list[str]) -> pd.DataFrame:
        """Add technical indicators derived from price data."""
        for col in price_cols:
            if col not in df.columns:
                continue
            
            df[f'{col}_return_1d'] = df[col].pct_change(1)
            df[f'{col}_return_5d'] = df[col].pct_change(5)
            df[f'{col}_return_20d'] = df[col].pct_change(20)
            
            df[f'{col}_momentum_5d'] = df[col] - df[col].shift(5)
            df[f'{col}_momentum_20d'] = df[col] - df[col].shift(20)
            
            returns = df[col].pct_change()
            df[f'{col}_vol_5d'] = returns.rolling(5).std()
            df[f'{col}_vol_20d'] = returns.rolling(20).std()
            
            df[f'{col}_roc_10d'] = ((df[col] - df[col].shift(10)) / df[col].shift(10)) * 100
        
        return df
    
    @staticmethod
    def add_macro_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """Add derived macro indicators."""
        if 'INDICATORS_USURTOT Index' in df.columns:
            df['unemployment_change'] = df['INDICATORS_USURTOT Index'].diff()
        
        if 'INDICATORS_CPI YOY Index' in df.columns:
            df['inflation_acceleration'] = df['INDICATORS_CPI YOY Index'].diff()
        
        return df
    
    def create_labels(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create classification labels based on future return buckets."""
        future_return = df[target_col].pct_change(self.horizon).shift(-self.horizon) * 100
        df['label'] = pd.cut(future_return, bins=self.return_bins, labels=False)
        
        label_dummies = pd.get_dummies(df['label'], prefix='label')
        df = pd.concat([df, label_dummies], axis=1)
        
        return df
    
    def create_sequences(self, df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Create sequences for time series modeling."""
        df = df.copy()
        df = self.create_labels(df, target_col)
        
        exclude_cols = ['date', 'label'] + [col for col in df.columns if col.startswith('label_')]
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]
        
        label_cols = [col for col in df.columns if col.startswith('label_')]
        df_clean = df[['date'] + feature_cols + ['label'] + label_cols].copy()
        
        df_clean = df_clean.dropna(subset=['label'] + label_cols)
        df_clean[feature_cols] = df_clean[feature_cols].ffill().fillna(0)
        
        print(f"  Rows after cleaning: {len(df_clean)} (started with {len(df)})")
        
        feature_data = df_clean[feature_cols].values
        mean = feature_data.mean(axis=0)
        std = feature_data.std(axis=0)
        std[std == 0] = 1
        feature_data_norm = (feature_data - mean) / std
        
        X, y = [], []
        for i in range(len(df_clean) - self.sequence_length - self.horizon + 1):
            X.append(feature_data_norm[i:i + self.sequence_length])
            y.append(df_clean[label_cols].iloc[i + self.sequence_length + self.horizon - 1].values)
        
        return np.array(X), np.array(y), feature_cols
    
    @staticmethod
    def visualize_sample(X: np.ndarray, y: np.ndarray, feature_cols: list[str], sample_idx: int = 0) -> None:
        """Visualize a sample from the dataset."""
        print(f"\n{'='*100}")
        print(f"DATASET SAMPLE VISUALIZATION - Sample #{sample_idx}")
        print(f"{'='*100}")
        
        print(f"\nINPUT SEQUENCE (X):")
        print(f"  Shape: {X[sample_idx].shape} -> (sequence_length={X.shape[1]}, features={X.shape[2]})")
        print(f"\n  First 5 timesteps and first 5 features (normalized):")
        sample_df = pd.DataFrame(
            X[sample_idx][:5, :5],
            columns=feature_cols[:5],
            index=[f"Day {i}" for i in range(5)]
        )
        print(sample_df.to_string())
        
        print(f"\nLABEL (y):")
        print(f"  Shape: {y[sample_idx].shape} -> (n_classes={y.shape[1]})")
        print(f"  One-hot encoded: {y[sample_idx]}")
        print(f"  Predicted class: {np.argmax(y[sample_idx])}")
        print(f"{'='*100}\n")
    
    def save_dataset(self, output_name: str, save_csv: bool = False) -> None:
        """Save dataset to disk."""
        if self.X is None or self.y is None:
            raise ValueError("No dataset to save. Run build() first.")
        
        output_dir = DATASETS_DIR / "npy"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / f"{output_name}_X.npy", self.X)
        np.save(output_dir / f"{output_name}_y.npy", self.y)
        
        print(f"\nDataset saved:")
        print(f"  X shape: {self.X.shape} -> (samples, sequence_length, features)")
        print(f"  y shape: {self.y.shape} -> (samples, classes)")
        print(f"  Location: {output_dir}")
        
        # Save feature names
        feature_list_path = output_dir / f"{output_name}_features.txt"
        with open(feature_list_path, 'w') as f:
            for i, feat in enumerate(self.feature_cols):
                f.write(f"{i}: {feat}\n")
        print(f"  Feature names saved to: {feature_list_path}")
        
        # Optionally save full DataFrame
        if save_csv and self.df is not None:
            csv_dir = DATASETS_DIR / "csv"
            csv_dir.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(csv_dir / f"{output_name}_full_data.csv", index=False)
            print(f"  Full DataFrame saved to: {csv_dir / f'{output_name}_full_data.csv'}")
    
    def analyze(self, target_ticker: str, plot: bool = True) -> None:
        """
        Analyze the processed dataset before creating sequences.
        
        Args:
            target_ticker: Ticker symbol to analyze
            plot: Whether to generate distribution plots
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Run build() first or load data manually.")
        
        analyzer = DatasetAnalyzer(self.df, target_ticker)
        analyzer.print_summary(self.return_bins, self.horizon)
        
        if plot:
            plot_dir = DATASETS_DIR / "analysis"
            analyzer.plot_distributions(plot_dir, self.horizon)
            analyzer.plot_label_distribution(self.return_bins, self.horizon, plot_dir)
            analyzer.plot_macro_indicators(plot_dir)
            analyzer.plot_quarterly_fundamentals(plot_dir)
    
    def build(self, target_ticker: str, visualize: bool = False, analyze: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the complete dataset pipeline.
        
        Args:
            target_ticker: Ticker symbol to predict (e.g., 'MSFT', 'AAPL', 'GOOG')
            visualize: Whether to show sample visualizations
            analyze: Whether to run statistical analysis on the dataset
        
        Returns:
            X, y arrays
        """
        print("Loading and merging datasets...")
        self.df = self.load_and_merge_datasets()
        
        print(f"Trading days: {len(self.df)}")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        
        print("\nAdding time features...")
        self.df = self.add_time_features(self.df)
        
        print("Adding technical features...")
        price_cols = [col for col in self.df.columns if '_PX_LAST' in col or '_PX_OPEN' in col]
        self.df = self.add_technical_features(self.df, price_cols)
        
        print("Adding macro ratios...")
        self.df = self.add_macro_ratios(self.df)
        
        if analyze:
            self.analyze(target_ticker, plot=True)
        
        print("\nCreating sequences...")
        target_col = f"{target_ticker}_PX_LAST"
        self.X, self.y, self.feature_cols = self.create_sequences(self.df, target_col)
        
        print(f"\nDataset created:")
        print(f"  Input shape: {self.X.shape}")
        print(f"  Output shape: {self.y.shape}")
        print(f"  Class distribution: {self.y.sum(axis=0)}")
        print(f"  Number of features: {len(self.feature_cols)}")
        
        if visualize:
            self.visualize_sample(self.X, self.y, self.feature_cols, sample_idx=0)
            self.visualize_sample(self.X, self.y, self.feature_cols, sample_idx=1)
        
        return self.X, self.y


if __name__ == "__main__":
    return_bins = [-np.inf, -6.1, -3.8, -2.2, -1, -0.2, 0.64, 1.4, 2.3, 3.3, 4.6, 6.7, np.inf]
    
    builder = StockDatasetBuilder(
        sequence_length=60,
        horizon=10,
        return_bins=return_bins,
        start_date="2000-01-03",
        end_date="2025-11-20"
    )
    
    X, y = builder.build(target_ticker="MSFT", visualize=False, analyze=True)
    builder.save_dataset(output_name="msft_10day_prediction", save_csv=False)
    
    print("\nDataset ready for training!")
