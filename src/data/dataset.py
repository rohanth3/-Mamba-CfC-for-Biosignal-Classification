import h5py
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np

class NinaproDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.X = f['X'][:]
            self.Y = f['Y'][:]
            
        self.X = torch.tensor(self.X, dtype=torch.float32)
        # Label is at index 3 of metadata
        self.labels = torch.tensor(self.Y[:, 3], dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]

def get_balanced_sampler(dataset):
    """
    Creates a WeightedRandomSampler that balances the classes (e.g. oversampling rare classes, undersampling majority class like rest).
    """
    labels = dataset.labels.numpy()
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-8)
    
    # Normalize weights so the max is not absurd, though WeightedRandomSampler handles it
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler
