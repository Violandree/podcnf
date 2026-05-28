import os
import torch
import pickle
import numpy as np
from scipy.linalg import svd

from dlroms.dnns import num2p

from podcnf.Loader import LoadData

def main():
    print("Start POD and Data Preprocessing:")

    input_file = "data/stokes_data_6400.pt"
    n_basis = 20 # to be choosen
    batch_size = 64

    dataset = torch.load(input_file, weights_only=True)
    u = dataset['u']
    mu = dataset['mu']

    n_samples = u.shape[0]
    n_train = int(n_samples * 0.75)
    n_val = int(n_train + n_samples * 0.20)

    print(f"SVD with {n_train} samples:")
    u_np = u.cpu().numpy()
    X, s, _ = svd(u_np[:n_train].T, full_matrices=False)

    V = torch.tensor(X[:, :n_basis], dtype=torch.float32)

    c = u @ V

    u_val_true = u[n_train:n_val]
    c_val = c[n_train:n_val]
    u_val_rec = c_val @ V.T
    
    residuals = u_val_true - u_val_rec
    errors = torch.linalg.norm(residuals, dim=1) / torch.linalg.norm(u_val_true, dim=1)
    print("Basis projection error on Validation: %s" % num2p(errors.mean().item()))
    
    os.makedirs('results/stokes', exist_ok=True)

    torch.save(V, 'results/stokes/V_POD_matrix.pt')

    torch.save({
        'c': c,
        'mu': mu,
        'eps': dataset['eps'],
        'theta': dataset['theta']
    }, f'data/stokes_data_reduced_{n_samples}.pt')

    print("Data Normalization:")

    train_loader, val_loader, test_loader, mu_scaler, c_scaler = LoadData(
        mu.numpy(), c.numpy(), 
        n_train=n_train, 
        n_val=n_val, 
        BATCH_SIZE=batch_size, 
        norm_scaler=True, 
        drop_last=False
    )

    with open('results/stokes/mu_scaler.pkl', 'wb') as f:
        pickle.dump(mu_scaler, f)
    with open('results/stokes/c_scaler.pkl', 'wb') as f:
        pickle.dump(c_scaler, f)

if __name__ == "__main__":
    main()