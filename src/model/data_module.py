"""Lightning DataModule for stock prediction."""

import numpy as np
import lightning as L
from pathlib import Path
from torch.utils.data import DataLoader

from src.model.dataset import StockDataset
from src.model.config import ModelConfig


class StockDataModule(L.LightningDataModule):
    """DataModule for stock prediction with train/val/test/local splits."""
    
    def __init__(self, config: ModelConfig, pin_memory: bool = False):
        super().__init__()
        self.config = config
        self.data_dir = config.data_dir
        self.pin_memory = pin_memory
        
        # Data arrays
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        
        # Datasets
        self.train_dataset: StockDataset | None = None
        self.val_dataset: StockDataset | None = None
        self.test_dataset: StockDataset | None = None
        self.local_test_dataset: StockDataset | None = None
        
    def prepare_data(self):
        """Load and split data into train/val/test/local test sets."""
        # Load full dataset
        X = np.load(self.data_dir / "msft_10day_prediction_X.npy")
        y = np.load(self.data_dir / "msft_10day_prediction_y.npy")
        
        print(f"\nDataset loaded: X {X.shape}, y {y.shape}")
        
        # Reserve last samples for local testing
        n_local = self.config.local_test_samples
        X_local = X[-n_local:]
        y_local = y[-n_local:]
        X = X[:-n_local]
        y = y[:-n_local]
        
        # Calculate split indices
        n_samples = len(X)
        n_train = int(n_samples * self.config.train_split)
        n_val = int(n_samples * self.config.val_split)
        
        # Split data (chronological order preserved)
        X_train = X[:n_train]
        y_train = y[:n_train]
        
        X_val = X[n_train:n_train + n_val]
        y_val = y[n_train:n_train + n_val]
        
        X_test = X[n_train + n_val:]
        y_test = y[n_train + n_val:]
        
        print(f"\nData splits:")
        print(f"  Train:      {X_train.shape[0]:5d} samples ({X_train.shape[0]/n_samples*100:.1f}%)")
        print(f"  Validation: {X_val.shape[0]:5d} samples ({X_val.shape[0]/n_samples*100:.1f}%)")
        print(f"  Test:       {X_test.shape[0]:5d} samples ({X_test.shape[0]/n_samples*100:.1f}%)")
        print(f"  Local test: {X_local.shape[0]:5d} samples (reserved)\n")
        
        # Save splits to disk
        np.save(self.data_dir / "train_X.npy", X_train)
        np.save(self.data_dir / "train_y.npy", y_train)
        np.save(self.data_dir / "val_X.npy", X_val)
        np.save(self.data_dir / "val_y.npy", y_val)
        np.save(self.data_dir / "test_X.npy", X_test)
        np.save(self.data_dir / "test_y.npy", y_test)
        np.save(self.data_dir / "local_test_X.npy", X_local)
        np.save(self.data_dir / "local_test_y.npy", y_local)
        
        print(f"Splits saved to {self.data_dir}/\n")
        
    def setup(self, stage: str | None = None):
        """Create datasets for each split."""
        if stage == "fit" or stage is None:
            X_train = np.load(self.data_dir / "train_X.npy")
            y_train = np.load(self.data_dir / "train_y.npy")
            X_val = np.load(self.data_dir / "val_X.npy")
            y_val = np.load(self.data_dir / "val_y.npy")
            
            self.train_dataset = StockDataset(X_train, y_train)
            self.val_dataset = StockDataset(X_val, y_val)
            
        if stage == "test" or stage is None:
            X_test = np.load(self.data_dir / "test_X.npy")
            y_test = np.load(self.data_dir / "test_y.npy")
            self.test_dataset = StockDataset(X_test, y_test)
            
        if stage == "predict" or stage is None:
            X_local = np.load(self.data_dir / "local_test_X.npy")
            y_local = np.load(self.data_dir / "local_test_y.npy")
            self.local_test_dataset = StockDataset(X_local, y_local)
    
    def train_dataloader(self) -> DataLoader:
        # Use 4 workers for CUDA, 0 for MPS/CPU
        num_workers = 4 if self.pin_memory else 0
        
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=num_workers > 0
        )
    
    def val_dataloader(self) -> DataLoader:
        num_workers = 4 if self.pin_memory else 0
        
        return DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=num_workers > 0
        )
    
    def test_dataloader(self) -> DataLoader:
        num_workers = 4 if self.pin_memory else 0
        
        return DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=num_workers > 0
        )
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.local_test_dataset,  # type: ignore
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
