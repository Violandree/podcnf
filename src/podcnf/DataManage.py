import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from podcnf.roms import TorchScaler

SEED = 42

def seed_worker(worker_id):
    """
    Initialization of the seed for each worker of the DataLoader to
    ensure the reproducibility of the experiments.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
            drop_last=False):
    """
    Normalization of the data, it returns directly the data normalized given the
    parameters mu and the solution c.
    """

    # Divide the dataset in training, validation and test set - over c and not u
    mu_train, c_train = mu[:n_train, :], c[:n_train, :]
    mu_val, c_val = mu[n_train:n_val, :], c[n_train:n_val, :]
    mu_test, c_test = mu[n_val:, :], c[n_val:, :]

    mean_mu, scale_mu = mu_train.mean(dim = 0), torch.sqrt(mu_train.var(dim = 0))
    mean_c, scale_c = c_train.mean(dim = 0), torch.sqrt(c_train.var(dim = 0))

    mu_scaler = TorchScaler(mean_mu, scale_mu, mu.device)
    c_scaler = TorchScaler(mean_c, scale_c, c.device)

    mu_train_scaled = mu_scaler.transform(mu_train)
    c_train_scaled = c_scaler.transform(c_train)

    mu_val_scaled = mu_scaler.transform(mu_val)
    c_val_scaled = c_scaler.transform(c_val)

    mu_test_scaled = mu_scaler.transform(mu_test)
    c_test_scaled = c_scaler.transform(c_test)

    # Combine parameters and solution c
    data_train_scaled = torch.cat((mu_train_scaled, c_train_scaled), axis=1)
    data_val_scaled = torch.cat((mu_val_scaled, c_val_scaled), axis=1)
    data_test_scaled = torch.cat((mu_test_scaled, c_test_scaled), axis=1)

    # Building the dataset
    train_loader = make_loader(data_train_scaled, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = make_loader(data_val_scaled, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = make_loader(data_test_scaled, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, mu_scaler, c_scaler