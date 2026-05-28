import os
import torch
from time import perf_counter

from podcnf.Loader import LoadData
from podcnf.NFmodel import NormalizingFlow
from podcnf.Training import full_train, tuning_parameters

SEED = 42

def main():
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

    input_file = "data/stokes_data_reduced_6400.pt" 
    print(f"\nLoading data from {input_file}...")

    dataset = torch.load(input_file, weights_only=True)
    mu = dataset['mu'].numpy()[:200]
    c = dataset['c'].numpy()[:200]

    n_samples = mu.shape[0]
    n_train = int(n_samples * 0.75)
    n_val = int(n_train + n_samples * 0.20)
    batch_size = 64
    
    dim_x = mu.shape[1]
    dim_y = c.shape[1]

    train_loader, val_loader, test_loader, mu_scaler, c_scaler = LoadData(
        mu, c, n_train=n_train, n_val=n_val,
        BATCH_SIZE=batch_size, norm_scaler=True, drop_last=False
    )

    DO_TUNING = False  # Imposta a True se vuoi ricalcolare i parametri

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

    save_dir_temp = "results/stokes"
    os.makedirs(save_dir_temp, exist_ok=True)
    model_save_path = os.path.join(save_dir_temp, "model_best_temp.pt")

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

    MODEL_NAME = 'NF_Stokes.pth'
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
    
    if os.path.exists(model_save_path):
        os.remove(model_save_path)
        print("File temporaneo eliminato.")

if __name__ == "__main__":
    main()