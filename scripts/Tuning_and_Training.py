import os
import pickle
import torch
from time import perf_counter

import numpy as np
from scipy.linalg import svd

from podcnf.Loader import LoadData
from podcnf.NFmodel import NormalizingFlow
from podcnf.Training import full_train, tuning_parameters

SEED = 42

def main(u, mu):
    print(">>> Starting Pipeline training NF:")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True
        scaler = torch.amp.GradScaler(enabled=True)
    else:
        device = torch.device("cpu")
        scaler = torch.amp.GradScaler(enabled=False)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")

    n_samples = u.shape[0]
    n_train = int(n_samples * 0.75)
    n_val = int(n_train + n_samples * 0.20)
    n_test = int(n_samples * 0.05)

    shuffle = True
    drop_last = False

    # norm_scaler = None # No normalization
    norm_scaler = True # StandardScaler()
    # norm_scaler = False # MinMaxScaler()

    print("SVD computation...")
    X, s, _ = svd(u[:n_train].T, full_matrices = False)
    n_basis = 20
    V = X[:, :n_basis]
    c = u @ V

    batch_size = 64
    
    dim_x = mu.shape[1]
    dim_y = c.shape[1]

    print(f"dim x:\t{dim_x}\ndim y:\t{dim_y}")

    if norm_scaler == None:
        train_loader, val_loader, test_loader = LoadData(
            mu, c, n_train=n_train, n_val=n_val,
            BATCH_SIZE=batch_size, norm_scaler=True, drop_last=False
        )
    else:
        train_loader, val_loader, test_loader, mu_scaler, c_scaler = LoadData(
            mu, c, n_train=n_train, n_val=n_val,
            BATCH_SIZE=batch_size, norm_scaler=True, drop_last=False
        )

    DO_TUNING = False

    if DO_TUNING:
        print("\nTuning Hyperparameters...")
        t0_tun = perf_counter()
        best_hyperparams = tuning_parameters(
            train_loader=train_loader, val_loader=val_loader,
            lr=[1e-3, 5e-4], num_flows=[12, 16, 24, 32],
            hidden_size=[64, 128, 256, 512], hidden_depth=[1, 2], weight_decay=[1e-5],
            epochs=50, dim_x=dim_x, dim_y=dim_y, device=device
        )
        print(f">>> Tuning\nExecution Time:\t{(perf_counter() - t0_tun)/60:.2f} min")
        
        lr = best_hyperparams['learning_rate']
        num_flows = best_hyperparams['num_flows']
        hidden_size = best_hyperparams['hidden_size']
        hidden_depth = best_hyperparams['hidden_depth']
        wd = best_hyperparams['weight_decay']
    else:
        lr = 0.001
        num_flows = 24
        hidden_size = 128
        hidden_depth = 2
        wd = 1e-05

    print(f"\nLearning Rate: {lr}\nnum_flows: {num_flows}\nhidden_size: {hidden_size}")
    print(f"Hidden Depth: {hidden_depth}\nweight_decay: {wd}")

    flow = NormalizingFlow(
        dim_x, dim_y, 
        num_flows=num_flows, 
        hidden_size=hidden_size, 
        hidden_depth=hidden_depth, 
        device=device
    ).to(device)

    print("\n>>>Start Training:")
    t0_tra = perf_counter()
    train_losses, val_losses = full_train(
        epochs=3,
        print_frequency=1,
        model=flow,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        weight_decay=wd,
        patience=20,
        device=device,
        model_save_path=model_save_path,
        show_plot=False
    )
    print(f">>> Training\nExecution Time:\t{(perf_counter() - t0_tra)/60:.2f} min")

    MODEL_NAME = 'NF_Stokes_trial.pth'
    destination_folder = "results/stokes"
    os.makedirs(destination_folder, exist_ok=True)
    save_path = os.path.join(destination_folder, MODEL_NAME)

    flow.load_state_dict(torch.load(model_save_path, weights_only=True))

    checkpoint = {
        'model_state_dict': flow.state_dict(),
        'hyperparameters': {
            'lr': lr,
            'num_flows': num_flows,
            'hidden_size': hidden_size,
            'hidden_depth': hidden_depth,
            'wd': wd
        }
    }

    torch.save(checkpoint, save_path)
    print(f"\nModel saved in: {save_path}")

    # Saving V_POD matrix and scalers
    path_V = '../results/stokes/V_POD_matrix.pt'
    torch.save(V, path_V)

    os.makedirs('../results/stokes', exist_ok=True)

    with open('../results/stokes/mu_scaler.pkl', 'wb') as f:
        pickle.dump(mu_scaler, f)
    with open('../results/stokes/c_scaler.pkl', 'wb') as f:
        pickle.dump(c_scaler, f)

if __name__ == "__main__":

    # Here one should upload the data and the conditioning parameters
    input_file = "../data/stokes_data_6400.pt"
    print(f"\nLoading data from {input_file}...")
    dataset = torch.load(input_file, weights_only=True)

    u = dataset['u']
    mu = dataset['mu']

    main(u, mu)