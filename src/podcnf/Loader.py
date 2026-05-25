import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Union, Optional, Any

SEED = 42

def seed_worker(worker_id: int) -> int:
    """
    Initialization of the seed for each worker of the DataLoader to
    ensure the reproducibility of the experiments.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class PODCNFDataset(Dataset):
    """
    Combine conditioning parameter and data.
    """

    def __init__(self, mu: torch.Tensor, c: torch.Tensor) -> None:
        self.data = torch.cat((mu, c), axis=1)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

def make_loader(ds: Dataset, 
                batch_size: int, 
                shuffle: bool =True, drop_last: bool = True) -> DataLoader:
    """
    Optimized PyTorch DataLoader
    """

    # Determine the optimale number of worker process for data loading
    cpu_cores = os.cpu_count() or 2
    num_workers = max(2, min(4, cpu_cores))

    g = torch.Generator()
    g.manual_seed(SEED)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        # For optimization running
        pin_memory=True, # Faster GPU transfer
        pin_memory_device="cuda" if torch.cuda.is_available() else "",
        prefetch_factor=4, # Load 4 batches ahead
        # Initialization of the worker
        worker_init_fn=seed_worker,
        generator=g
    )


def LoadData(mu, c, 
            n_train, n_val, 
            BATCH_SIZE, 
            norm_scaler=None, 
            drop_last=False):
    """
    Normalization of the data, it returns directly the data normalized given the
    parameters mu and the solution c if norm_scaler!=None, otherwise it returns
    simply the data_loader.
    - norm_scaler==True: Standard Normalization
    - norm_scaler==False: MinMax Normalization
    """

    if isinstance(mu, np.ndarray):
        mu = torch.tensor(mu, dtype=torch.float32)
    if isinstance(c, np.ndarray):
        c = torch.tensor(c, dtype=torch.float32)

    # Divide the dataset in training, validation and test set - over c and not u
    mu_train, c_train = mu[:n_train, :], c[:n_train, :]
    mu_val, c_val = mu[n_train:n_val, :], c[n_train:n_val, :]
    mu_test, c_test = mu[n_val:, :], c[n_val:, :]

    if norm_scaler is None:
        data_train = torch.cat((mu_train, c_train), axis=1)
        data_val = torch.cat((mu_val, c_val), axis=1)
        data_test = torch.cat((mu_test, c_test), axis=1)

        # Building the dataset
        train_loader = make_loader(data_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = make_loader(data_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        test_loader = make_loader(data_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        return train_loader, val_loader, test_loader

    if norm_scaler is True:
        mu_scaler = StandardScaler()
        c_scaler = StandardScaler()
    else:
        mu_scaler = MinMaxScaler()
        c_scaler = MinMaxScaler()

    mu_scaler.fit(mu_train)
    c_scaler.fit(c_train)

    mu_train_scaled = torch.tensor(mu_scaler.transform(mu_train), dtype=torch.float32)
    c_train_scaled = torch.tensor(c_scaler.transform(c_train), dtype=torch.float32)

    mu_val_scaled = torch.tensor(mu_scaler.transform(mu_val), dtype=torch.float32)
    c_val_scaled = torch.tensor(c_scaler.transform(c_val), dtype=torch.float32)

    mu_test_scaled = torch.tensor(mu_scaler.transform(mu_test), dtype=torch.float32)
    c_test_scaled = torch.tensor(c_scaler.transform(c_test), dtype=torch.float32)

    # Combine parameters and solution c
    data_train_scaled = torch.cat((mu_train_scaled, c_train_scaled), axis=1)
    data_val_scaled = torch.cat((mu_val_scaled, c_val_scaled), axis=1)
    data_test_scaled = torch.cat((mu_test_scaled, c_test_scaled), axis=1)

    # Building the dataset
    train_loader = make_loader(data_train_scaled, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = make_loader(data_val_scaled, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = make_loader(data_test_scaled, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, mu_scaler, c_scaler