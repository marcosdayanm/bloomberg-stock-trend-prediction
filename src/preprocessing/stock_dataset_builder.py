import pandas as pd
import numpy as np
from pathlib import Path
import pandas_market_calendars as mcal
from fnmatch import fnmatch
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import RandomOverSampler

from src.constants import CRUDE_DATASETS_DIR, DATASETS_DIR
from src.preprocessing.dataset_analyzer import DatasetAnalyzer


class StockDatasetBuilder:
    """Pipeline for creating stock prediction datasets with time series features."""
    
    def __init__(
        self,
        sequence_length: int = 60,
        horizon: int = 10,
        task_type: str = 'classification',
        return_bins: list[float] | None = None,
        start_date: str = "2000-01-03",
        end_date: str = "2025-11-20",
        feature_columns: list[str] | None = None,
    ):
        """Initialize the dataset builder.
        
        Args:
            task_type: 'classification' or 'regression'
            return_bins: Bins for classification (ignored for regression)
        """
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.task_type = task_type
        self.return_bins = return_bins if return_bins else [-np.inf, -2, -1, 0, 1, 2, np.inf]
        self.start_date = start_date
        self.end_date = end_date
        self.feature_columns = feature_columns
        
        self.df: pd.DataFrame | None = None
        self.feature_cols: list[str] = []
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.removed_features: dict[str, list[str]] = {}
    
    def _filter_by_nulls(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Remove features with >threshold null values."""
        null_pct = df.isnull().sum() / len(df)
        cols_to_drop = null_pct[null_pct > threshold].index.tolist()
        cols_to_drop = [c for c in cols_to_drop if c != 'date']
        
        if cols_to_drop:
            self.removed_features['nulls'] = cols_to_drop
            print(f"  Removed {len(cols_to_drop)} features with >{threshold*100:.0f}% nulls")
            df = df.drop(columns=cols_to_drop)
        return df
    
    def _filter_by_variance(self, df: pd.DataFrame, threshold: float = 0.001) -> pd.DataFrame:
        """Remove features with variance <threshold."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'date']
        
        variances = df[numeric_cols].var()
        cols_to_drop = variances[variances < threshold].index.tolist()
        
        if cols_to_drop:
            self.removed_features['variance'] = cols_to_drop
            print(f"  Removed {len(cols_to_drop)} features with variance <{threshold}")
            df = df.drop(columns=cols_to_drop)
        return df
    
    def _filter_by_correlation(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features (>threshold)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'date']
        
        corr_matrix = df[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        cols_to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
        
        if cols_to_drop:
            self.removed_features['correlation'] = cols_to_drop
            print(f"  Removed {len(cols_to_drop)} features with correlation >{threshold}")
            df = df.drop(columns=cols_to_drop)
        return df
    
    def _select_top_features(self, X: np.ndarray, y: np.ndarray, feature_cols: list[str], top_n: int = 50) -> tuple[np.ndarray, list[str]]:
        """Select top N features by mutual information."""
        print(f"\nSelecting top {top_n} features...")
        
        # Flatten sequences for feature selection
        X_flat = X.reshape(X.shape[0], -1)
        
        # Convert y to labels for mutual information
        if y.ndim > 1 and y.shape[1] > 1:
            y_labels = np.argmax(y, axis=1)  # Classification
        else:
            # Regression: bin continuous values for mutual information
            y_labels = pd.cut(y.ravel(), bins=10, labels=False)
        
        # Calculate importance per timestep
        n_timesteps = X.shape[1]
        feature_importance = np.zeros(len(feature_cols))
        
        for t in range(n_timesteps):
            start_idx = t * len(feature_cols)
            end_idx = start_idx + len(feature_cols)
            X_timestep = X_flat[:, start_idx:end_idx]
            
            # Mutual information
            mi_scores = mutual_info_classif(X_timestep, y_labels, random_state=42)
            feature_importance += mi_scores
        
        # Average importance across timesteps
        feature_importance /= n_timesteps
        
        # Select top N
        top_indices = np.argsort(feature_importance)[-top_n:]
        top_features = [feature_cols[i] for i in top_indices]
        
        X_selected = X[:, :, top_indices]
        
        print(f"  Selected features: {len(top_features)}")
        print(f"  Shape: {X.shape} → {X_selected.shape}")
        
        # Show top 10 most important
        sorted_idx = np.argsort(feature_importance)[-10:][::-1]
        print("\n  Top 10 most important features:")
        for i, idx in enumerate(sorted_idx, 1):
            print(f"    {i}. {feature_cols[idx]}: {feature_importance[idx]:.4f}")
        
        return X_selected, top_features
    
    @staticmethod
    def _apply_random_oversampling(X: np.ndarray, y: np.ndarray, task_type: str = 'classification') -> tuple[np.ndarray, np.ndarray]:
        """Apply random oversampling to balance classes (classification only)."""
        if task_type == 'regression':
            print("\nSkipping oversampling (regression task)...")
            return X, y
        
        print("\nApplying random oversampling...")
        print(f"  Original class distribution: {y.sum(axis=0).astype(int)}")
        
        # Reshape for oversampling
        n_samples, n_timesteps, n_features = X.shape
        X_flat = X.reshape(n_samples, -1)
        
        # Convert one-hot to labels for oversampling
        y_labels = np.argmax(y, axis=1)
        
        # Oversample
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_labels_resampled = ros.fit_resample(X_flat, y_labels)
        
        # Convert back to one-hot
        n_classes = y.shape[1]
        y_resampled = np.zeros((len(y_labels_resampled), n_classes))
        y_resampled[np.arange(len(y_labels_resampled)), y_labels_resampled] = 1
        
        # Reshape back
        X_resampled = X_resampled.reshape(-1, n_timesteps, n_features)
        
        print(f"  New class distribution: {y_resampled.sum(axis=0).astype(int)}")
        print(f"  Samples: {n_samples} → {X_resampled.shape[0]}")
        
        return X_resampled, y_resampled
    
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
        """Create labels for classification or regression."""
        future_return = df[target_col].pct_change(self.horizon).shift(-self.horizon) * 100
        
        if self.task_type == 'classification':
            # Classification: bins
            df['label'] = pd.cut(future_return, bins=self.return_bins, labels=False)
            label_dummies = pd.get_dummies(df['label'], prefix='label')
            df = pd.concat([df, label_dummies], axis=1)
        else:
            # Regression: continuous values
            df['label'] = future_return
        
        return df
    
    def create_sequences(self, df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Create sequences for time series modeling."""
        df = df.copy()
        df = self.create_labels(df, target_col)
        
        exclude_cols = ['date', 'label'] + [col for col in df.columns if col.startswith('label_')]
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]
        
        # Apply feature selection if specified
        if self.feature_columns is not None:
            selected_features = []
            for pattern in self.feature_columns:
                matched = [col for col in feature_cols if fnmatch(col, pattern)]
                selected_features.extend(matched)
            
            # Remove duplicates while preserving order
            feature_cols = list(dict.fromkeys(selected_features))
            
            if len(feature_cols) == 0:
                raise ValueError(
                    f"No features matched the patterns: {self.feature_columns}\n"
                    f"Available features: {df.columns.tolist()[:20]}..."
                )
            
            print(f"  Feature selection applied: {len(feature_cols)} features selected from patterns {self.feature_columns}")
        
        if self.task_type == 'classification':
            label_cols = [col for col in df.columns if col.startswith('label_')]
            df_clean = df[['date'] + feature_cols + ['label'] + label_cols].copy()
            df_clean = df_clean.dropna(subset=['label'] + label_cols)
        else:
            # Regression: only one label column
            label_cols = ['label']
            df_clean = df[['date'] + feature_cols + ['label']].copy()
            df_clean = df_clean.dropna(subset=['label'])
        
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
            if self.task_type == 'classification':
                y.append(df_clean[label_cols].iloc[i + self.sequence_length + self.horizon - 1].values)
            else:
                # Regression: single value
                y.append(df_clean['label'].iloc[i + self.sequence_length + self.horizon - 1])
        
        X = np.array(X)
        y = np.array(y)
        if self.task_type == 'regression':
            y = y.reshape(-1, 1)  # Shape: (samples, 1)
        
        return X, y, feature_cols
    
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
        
        output_dir = DATASETS_DIR / "npy" / self.task_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / f"{output_name}_X.npy", self.X)
        np.save(output_dir / f"{output_name}_y.npy", self.y)
        
        print(f"\nDataset saved:")
        print(f"  X shape: {self.X.shape} -> (samples, sequence_length, features)")
        print(f"  y shape: {self.y.shape} -> (samples, {'classes' if self.task_type == 'classification' else 'values'})")
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
        
        # Protect target column before filtering
        target_col = f"{target_ticker}_PX_LAST"
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        target_data = self.df[[target_col]].copy()
        self.df = self.df.drop(columns=[target_col])
        
        print("\nApplying automatic filters...")
        initial_cols = len([c for c in self.df.columns if c != 'date'])
        self.df = self._filter_by_nulls(self.df, threshold=0.5)
        self.df = self._filter_by_variance(self.df, threshold=0.001)
        self.df = self._filter_by_correlation(self.df, threshold=0.95)
        final_cols = len([c for c in self.df.columns if c != 'date'])
        print(f"  Features: {initial_cols} → {final_cols} (removed {initial_cols - final_cols})")
        
        # Restore target column
        self.df = pd.concat([self.df, target_data], axis=1)
        
        if analyze:
            self.analyze(target_ticker, plot=True)
        
        print("\nCreating sequences...")
        self.X, self.y, self.feature_cols = self.create_sequences(self.df, target_col)
        
        print(f"\nDataset created:")
        print(f"  Input shape: {self.X.shape}")
        print(f"  Output shape: {self.y.shape}")
        if self.task_type == 'classification':
            print(f"  Class distribution: {self.y.sum(axis=0).astype(int)}")
        else:
            print(f"  Target range: [{self.y.min():.2f}, {self.y.max():.2f}]")
            print(f"  Target mean: {self.y.mean():.2f} ± {self.y.std():.2f}")
        print(f"  Number of features: {len(self.feature_cols)}")
        
        # Feature selection: Top 50
        self.X, self.feature_cols = self._select_top_features(self.X, self.y, self.feature_cols, top_n=50)
        
        # Apply oversampling (classification only)
        self.X, self.y = self._apply_random_oversampling(self.X, self.y, task_type=self.task_type)
        
        print(f"\nFinal dataset:")
        print(f"  Input shape: {self.X.shape}")
        print(f"  Output shape: {self.y.shape}")
        if self.task_type == 'classification':
            print(f"  Class distribution: {self.y.sum(axis=0).astype(int)}")
        else:
            print(f"  Target range: [{self.y.min():.2f}, {self.y.max():.2f}]")
            print(f"  Target mean: {self.y.mean():.2f} ± {self.y.std():.2f}")
        print(f"  Number of features: {len(self.feature_cols)}")
        
        if visualize:
            self.visualize_sample(self.X, self.y, self.feature_cols, sample_idx=0)
            self.visualize_sample(self.X, self.y, self.feature_cols, sample_idx=1)
        
        return self.X, self.y


if __name__ == "__main__":
    # Choose task: 'classification' or 'regression'
    TASK_TYPE = 'regression'
    
    # Binary classification: DOWN vs UP (only for classification)
    return_bins = [-np.inf, 0, np.inf] if TASK_TYPE == 'classification' else None
    
    builder = StockDatasetBuilder(
        sequence_length=30,
        horizon=10,
        task_type=TASK_TYPE,
        return_bins=return_bins,
        start_date="2000-01-03",
        end_date="2025-11-20"
    )
    
    X, y = builder.build(target_ticker="MSFT", visualize=False, analyze=True)
    builder.save_dataset(output_name="msft_10day_prediction", save_csv=False)
    
    print("\nDataset ready for training!")
