"""PyTorch Dataset for stock prediction."""

import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """Dataset for time series stock prediction."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, noise_std: float = 0.0, is_training: bool = False):
        """
        Initialize dataset.
        
        Args:
            X: Input sequences (n_samples, seq_length, n_features)
            y: One-hot encoded labels (n_samples, n_classes)
            noise_std: Standard deviation of gaussian noise for augmentation
            is_training: Whether this is training set (applies augmentation)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.noise_std = noise_std
        self.is_training = is_training
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
