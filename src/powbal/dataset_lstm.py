import torch
from torch.utils.data import Dataset
import numpy as np

class EventSequenceDataset(Dataset):
    """
    Dataset PyTorch per sequenze evento-centriche.

    Parametri
    ----------
    X : np.ndarray
        Array di input (n_seq, seq_len, n_features).
    y : np.ndarray
        Target associato (n_seq, ) oppure (n_seq, 1).

    Opzionale:
    - mask : np.ndarray (n_seq, seq_len) per Attention supervisionata o masking
    - user_ids : elenco utenti associati (debug o tracking)

    Basato su:
    - PyTorch Dataset best practices
    - Preparazione dati time-series (GitHub: pytorch-timeseries)
    """
    def __init__(self, X, y, mask=None, user_ids=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32) if mask is not None else None
        self.user_ids = user_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {
            'x': self.X[idx],
            'y': self.y[idx]
        }
        if self.mask is not None:
            sample['mask'] = self.mask[idx]
        if self.user_ids is not None:
            sample['user_id'] = self.user_ids[idx]
        return sample
