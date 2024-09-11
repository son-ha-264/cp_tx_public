import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class cp_tx_dataset(Dataset):
    def __init__(self, path_indices, path_cp_features, path_tx_features, norm=False):
        """Dataset class for contradtive learning between cp and tx
        NOTE xi: cp, xj: tx
        """
        super().__init__()
        self.df_indices = pd.read_csv(
            path_indices,
            usecols=["index_tx", "index_cp"],
            dtype={"index_tx": np.int64, "index_cp": np.int64},
        )
        self.cp_features = np.load(path_cp_features, mmap_mode="r")
        self.tx_features = np.load(path_tx_features, mmap_mode="r")
        if norm:
            scaler = StandardScaler()
            self.cp_features = scaler.fit_transform(self.cp_features)
            self.tx_features = scaler.fit_transform(self.tx_features)

    def __len__(self):
        return len(self.df_indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        cp_idx = self.df_indices.loc[idx, "index_cp"]
        tx_idx = self.df_indices.loc[idx, "index_tx"]
        xi = np.asarray(self.cp_features[cp_idx])
        xj = np.asarray(self.tx_features[tx_idx])
        xi = torch.tensor(xi, dtype=torch.float)
        xj = torch.tensor(xj, dtype=torch.float)
        return xi, xj
