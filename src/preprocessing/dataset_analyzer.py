import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal

from src.constants import CRUDE_DATASETS_DIR, DATASETS_DIR


class DatasetAnalyzer:
    """Analyze stock datasets to understand distributions, correlations, and patterns."""
    
    def __init__(self, df: pd.DataFrame, target_ticker: str):
        """
        Initialize analyzer with processed dataset.
        
        Args:
            df: DataFrame with features and date column
            target_ticker: Target stock ticker (e.g., 'MSFT')
        """
        self.df = df.copy()
        self.target_ticker = target_ticker
        self.target_col = f"{target_ticker}_PX_LAST"
        
    def analyze_price_fluctuation(self, window: int = 20) -> dict[str, float]:
        """Analyze price volatility and fluctuation patterns."""
        returns = self.df[self.target_col].pct_change() * 100
        
        stats = {
            "min_return": returns.min(),
            "max_return": returns.max(),
            "mean_return": returns.mean(),
            "std_return": returns.std(),
        }
        
        rolling_vol = returns.rolling(window).std()
        stats["avg_volatility"] = rolling_vol.mean()
        stats["max_volatility"] = rolling_vol.max()
        
        return stats
    
    def analyze_label_distribution(self, return_bins: list[float], horizon: int = 10) -> pd.DataFrame:
        """Analyze the distribution of return labels."""
        future_return = self.df[self.target_col].pct_change(horizon).shift(-horizon) * 100
        labels = pd.cut(future_return, bins=return_bins, labels=False)
        
        distribution = labels.value_counts().sort_index()
        percentages = (distribution / distribution.sum() * 100).round(2)
        
        result = pd.DataFrame({
            'label': distribution.index,
            'count': distribution.values,
            'percentage': percentages.values,
            'bin_range': [f"{return_bins[int(i)]:.1f}% to {return_bins[int(i)+1]:.1f}%" 
                         for i in distribution.index]
        })
        
        return result
    
    def analyze_feature_distributions(self, features: list[str] | None = None, 
                                     top_n: int = 10) -> pd.DataFrame:
        """Analyze statistical distributions of key features."""
        if features is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            exclude = ['date', 'label'] + [col for col in numeric_cols if col.startswith('label_')]
            features = [col for col in numeric_cols if col not in exclude][:top_n]
        
        stats = []
        for feat in features:
            if feat not in self.df.columns:
                continue
                
            data = self.df[feat].dropna()
            stats.append({
                'feature': feat,
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'q25': data.quantile(0.25),
                'median': data.median(),
                'q75': data.quantile(0.75),
                'max': data.max(),
                'skew': data.skew(),
                'nulls': self.df[feat].isna().sum()
            })
        
        return pd.DataFrame(stats)
    
    def analyze_correlations(self, features: list[str] | None = None,
                            threshold: float = 0.5) -> pd.DataFrame:
        """Find features with strong correlation to target price."""
        if features is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            features = [col for col in numeric_cols if col != self.target_col 
                       and not col.startswith('label_') and col != 'label']
        
        correlations = []
        target_data = self.df[self.target_col]
        
        for feat in features:
            if feat not in self.df.columns:
                continue
            
            corr = self.df[feat].corr(target_data)
            if abs(corr) >= threshold and self.target_ticker.lower() not in feat.lower():
                correlations.append({
                    'feature': feat,
                    'correlation': abs(corr),
                })
        
        result = pd.DataFrame(correlations)
        if len(result) > 0:
            result = result.sort_values('correlation', ascending=False)
        
        return result
    
    def print_summary(self, return_bins: list[float], horizon: int = 10) -> None:
        """Print comprehensive analysis summary."""
        print(f"\n{'='*100}")
        print(f"DATASET ANALYSIS: {self.target_ticker}")
        print(f"{'='*100}")
        
        # Basic info
        print(f"\nDataset Overview:")
        print(f"  Total rows: {len(self.df):,}")
        print(f"  Date range: {self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}")
        print(f"  Total features: {len(self.df.columns)}")
        
        # Price fluctuation
        print(f"\nPrice Fluctuation Analysis ({self.target_ticker}):")
        fluct_stats = self.analyze_price_fluctuation()
        for key, value in fluct_stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
        
        # Label distribution
        print(f"\nLabel Distribution (One-Hot Encoded, {horizon}-day horizon):")
        label_dist = self.analyze_label_distribution(return_bins, horizon)
        print(label_dist.to_string(index=False))
        
        # Feature distributions
        print(f"\nTop Feature Statistics:")
        feat_stats = self.analyze_feature_distributions(top_n=8)
        print(feat_stats.to_string(index=False))
        
        # Correlations
        correlation_threshold = 0.3
        print(f"\nStrong Correlations with {self.target_col}:")
        corr_df = self.analyze_correlations(threshold=correlation_threshold)
        if len(corr_df) > 0:
            print(corr_df.head(50).to_string(index=False))
        else:
            print(f"  No features with correlation >= {correlation_threshold} found")
        
        print(f"\n{'='*100}\n")
    
    def plot_distributions(self, output_dir: Path | None = None, horizon: int = 10) -> None:
        """Generate individual distribution plots for key features."""
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Price over time
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.df['date'], self.df[self.target_col], linewidth=0.8, color='steelblue')
        ax.set_title(f'{self.target_ticker} Price Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / f'{self.target_ticker}_price_history.png', dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        
        # Daily returns distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        returns = self.df[self.target_col].pct_change() * 100
        ax.hist(returns.dropna(), bins=60, edgecolor='black', alpha=0.7, color='teal')
        ax.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Return (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / f'{self.target_ticker}_daily_returns.png', dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        
        # Rolling volatility
        fig, ax = plt.subplots(figsize=(12, 6))
        rolling_vol = returns.rolling(20).std()
        ax.plot(self.df['date'], rolling_vol, linewidth=0.8, color='orange')
        ax.set_title('20-Day Rolling Volatility', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Volatility (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / f'{self.target_ticker}_volatility.png', dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        
        # N-day forward returns distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        forward_returns = self.df[self.target_col].pct_change(horizon).shift(-horizon) * 100
        ax.hist(forward_returns.dropna(), bins=60, edgecolor='black', alpha=0.7, color='crimson')
        ax.set_title(f'{horizon}-Day Forward Returns Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Return (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.axvline(0, color='blue', linestyle='--', linewidth=2, label='Zero Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / f'{self.target_ticker}_{horizon}day_forward_returns.png', dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        
        # Volume over time
        volume_cols = [col for col in self.df.columns if 'VOLUME' in col.upper() and self.target_ticker in col]
        if volume_cols:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(self.df['date'], self.df[volume_cols[0]], linewidth=0.5, color='green', alpha=0.7)
            ax.set_title(f'{self.target_ticker} Trading Volume', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Volume', fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if output_dir:
                plt.savefig(output_dir / f'{self.target_ticker}_volume.png', dpi=150, bbox_inches='tight')
            else:
                plt.show()
            plt.close()
        
        # Cumulative returns
        fig, ax = plt.subplots(figsize=(12, 6))
        cumulative_returns = (1 + returns / 100).cumprod()
        ax.plot(self.df['date'], cumulative_returns, linewidth=0.8, color='purple')
        ax.set_title(f'{self.target_ticker} Cumulative Returns', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return Factor', fontsize=12)
        ax.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / f'{self.target_ticker}_cumulative_returns.png', dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        
        if output_dir:
            print(f"\nPlots saved to: {output_dir}/")
            print(f"   - {self.target_ticker}_price_history.png")
            print(f"   - {self.target_ticker}_daily_returns.png")
            print(f"   - {self.target_ticker}_volatility.png")
            print(f"   - {self.target_ticker}_{horizon}day_forward_returns.png")
            print(f"   - {self.target_ticker}_label_distribution.png")
            if volume_cols:
                print(f"   - {self.target_ticker}_volume.png")
            print(f"   - {self.target_ticker}_cumulative_returns.png")
    
    def plot_label_distribution(self, return_bins: list[float], horizon: int = 10, output_dir: Path | None = None) -> None:
        """Plot the distribution of prediction labels as a bar chart."""
        label_dist = self.analyze_label_distribution(return_bins, horizon)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(label_dist)))  # type: ignore
        bars = ax.bar(label_dist['label'], label_dist['count'], color=colors, edgecolor='black', linewidth=1.2)
        
        ax.set_title(f'{self.target_ticker} Label Distribution ({horizon}-Day Horizon)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Label Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xticks(label_dist['label'])
        ax.set_xticklabels([f"L{int(l)}" for l in label_dist['label']], fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, row) in enumerate(zip(bars, label_dist.itertuples())):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                   f'{row.count}\n({row.percentage}%)',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2., -80,
                   row.bin_range.replace(' to ', '\nto\n'),  # type: ignore
                   ha='center', va='top', fontsize=7, rotation=0)
        
        plt.tight_layout()
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / f'{self.target_ticker}_label_distribution.png', dpi=150, bbox_inches='tight')
            print(f"   - {self.target_ticker}_label_distribution.png")
        else:
            plt.show()
        
        plt.close()
    
    def plot_macro_indicators(self, output_dir: Path | None = None) -> None:
        """Plot macroeconomic indicators vs stock price."""
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        macro_indicators = {
            'INDICATORS_CPI YOY Index': 'Inflation Rate (CPI YoY)',
            'INDICATORS_USURTOT Index': 'Unemployment Rate',
            'INDICATORS_FDTR Index': 'Federal Funds Rate',
            'INDICATORS_GDP CURY Index': 'GDP Growth (YoY)',
            'INDICATORS_CONSSENT Index': 'Consumer Sentiment',
            'INDICATORS_DXY Curncy': 'US Dollar Index',
        }
        
        for col, label in macro_indicators.items():
            if col not in self.df.columns:
                continue
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            
            # Price on top
            ax1.plot(self.df['date'], self.df[self.target_col], linewidth=0.8, color='steelblue', label=f'{self.target_ticker} Price')
            ax1.set_ylabel('Price (USD)', fontsize=11, fontweight='bold')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f'{self.target_ticker} Price vs {label}', fontsize=14, fontweight='bold')
            
            # Indicator on bottom
            color = 'crimson'
            ax2.plot(self.df['date'], self.df[col], linewidth=0.8, color=color)
            ax2.set_xlabel('Date', fontsize=11)
            ax2.set_ylabel(label, fontsize=11, fontweight='bold', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_dir:
                safe_name = label.replace(' ', '_').replace('(', '').replace(')', '')
                plt.savefig(output_dir / f'{self.target_ticker}_vs_{safe_name}.png', dpi=150, bbox_inches='tight')
                print(f"   - {self.target_ticker}_vs_{safe_name}.png")
            else:
                plt.show()
            
            plt.close()
        
        if output_dir:
            print(f"\nMacro indicator plots saved to: {output_dir}/")
    
    def plot_quarterly_fundamentals(self, output_dir: Path | None = None) -> None:
        """Plot quarterly fundamentals vs stock price."""
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        qtr_metrics = {
            'MSFT_QTR_SALES_REV_TURN': 'Revenue',
            'MSFT_QTR_NET_INCOME': 'Net Income',
            'MSFT_QTR_IS_EPS': 'Earnings Per Share',
            'MSFT_QTR_EBITDA': 'EBITDA',
            'MSFT_QTR_CF_FREE_CASH_FLOW': 'Free Cash Flow',
            'MSFT_QTR_HISTORICAL_MARKET_CAP': 'Market Cap',
        }
        
        for col, label in qtr_metrics.items():
            if col not in self.df.columns:
                continue
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            
            # Price on top
            ax1.plot(self.df['date'], self.df[self.target_col], linewidth=0.8, color='steelblue', label=f'{self.target_ticker} Price')
            ax1.set_xlabel('Date', fontsize=11)
            ax1.set_ylabel('Price (USD)', fontsize=11, fontweight='bold')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f'{self.target_ticker} Price vs {label}', fontsize=14, fontweight='bold')
            
            # Fundamental on bottom
            color = 'darkgreen'
            qtr_data = self.df[['date', col]].dropna()
            ax2.plot(qtr_data['date'], qtr_data[col], linewidth=1.5, color=color, marker='o', markersize=3)
            ax2.set_xlabel('Date', fontsize=11)
            ax2.set_ylabel(label, fontsize=11, fontweight='bold', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_dir:
                safe_name = label.replace(' ', '_')
                plt.savefig(output_dir / f'{self.target_ticker}_vs_{safe_name}.png', dpi=150, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
        
        if output_dir:
            print(f"Quarterly fundamental plots saved to: {output_dir}/")


def analyze_crude_dataset(csv_path: Path, ticker: str) -> None:
    """Analyze a single crude CSV dataset."""
    df = pd.read_csv(csv_path, parse_dates=['date'])
    
    analyzer = DatasetAnalyzer(df, ticker)
    
    print(f"\n{'='*100}")
    print(f"DATASET ANALYSIS: {csv_path.name}")
    print(f"{'='*100}")
    
    print(f"\nBasic Info:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Feature list
    print(f"\nFeatures:")
    for i, col in enumerate(df.columns):
        if col != 'date':
            nulls = df[col].isna().sum()
            null_pct = (nulls / len(df)) * 100
            print(f"  {i}. {col:<50} | Nulls: {nulls:>6} ({null_pct:>5.2f}%)")
    
    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    msft_csv = CRUDE_DATASETS_DIR / "MSFT_CRUDE.csv"
    analyze_crude_dataset(msft_csv, "MSFT")
