import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """Dataset for time series stock prediction."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Input sequences (n_samples, seq_length, n_features)
            y: One-hot encoded labels (n_samples, n_classes)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
