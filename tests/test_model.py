import os
import torch
import numpy as np
from podcnf.NFmodel import NormalizingFlow
from podcnf.Loader import LoadData
import tempfile

from podcnf.Training import full_train
from podcnf.NFmodel import NormalizingFlow
from podcnf.Loader import LoadData

import os
import gdown
import pickle
from time import perf_counter

import torch
import numpy as np
from scipy.stats import t

from dlroms.minns import Interpolate
from dlroms.dnns import num2p

from podcnf.NFmodel import NormalizingFlow

from podcnf.DataGenerationStokes import *

SEED = 42

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

# def test_flow_shapes():
#     """
#     Test to verify if the mdodel executes the forward step and the sampling
#     returning the tensors with the correct dimensions
#     """

#     batch_size = 10
#     dim_x = 4
#     dim_y = 6

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     model = NormalizingFlow(dim_x=dim_x, dim_y=dim_y, num_flows=4, device=device)
    
#     sample_x = torch.randn(batch_size, dim_x, device=device)
#     sample_y = torch.randn(batch_size, dim_y, device=device)

#     # Log-prob
#     log_p = model(sample_x, sample_y)
#     assert log_p.shape == (batch_size,), f"True shape {(batch_size,)}, obtained {log_px.shape}"
    
#     # sample
#     samples = model.sample(sample_x)
#     assert samples.shape == (batch_size, dim_y), f"True shape {(batch_size, dim_y)}, obtained {samples.shape}"

# def test_load_data_pipeline():
#     """
#     Test if the split and the standardization went well
#     """
#     total_samples = 60
#     dim_mu = 4
#     dim_c = 6
#     batch_size = 10
    
#     # random array
#     mock_mu = np.random.randn(total_samples, dim_mu)
#     mock_c = np.random.randn(total_samples, dim_c)
    
#     n_train = 30
#     n_val = 50
    
#     # Test with StandardScaler = True
#     train_l, val_l, test_l, mu_scale, c_scale = LoadData(
#         mock_mu, mock_c, n_train, n_val, BATCH_SIZE=batch_size, norm_scaler=True
#     )
    
#     first_batch = next(iter(train_l))
    
#     assert first_batch.shape == (batch_size, dim_mu + dim_c), f"Wrong shape: {first_batch.shape}"
#     assert mu_scale is not None
#     assert c_scale is not None

# def test_full_training_loop():
    
#     total_samples = 40
#     dim_x = 2
#     dim_y = 4
#     batch_size = 10
#     epochs = 2  
    
#     sample_mu = np.random.randn(total_samples, dim_x)
#     sample_c = np.random.randn(total_samples, dim_y)
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     train_l, val_l, _, _, _ = LoadData(
#         sample_mu, sample_c, n_train=20, n_val=35, BATCH_SIZE=batch_size, norm_scaler=True
#     )
    
#     model = NormalizingFlow(dim_x=dim_x, dim_y=dim_y, num_flows=2, device=device)
    
#     with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
#         temp_model_path = tmp_file.name
        
#     try:
#         train_losses, val_losses = full_train(
#             epochs=epochs,
#             print_frequency=1,
#             model=model,
#             train_loader=train_l,
#             val_loader=val_l,
#             lr=1e-3,
#             weight_decay=1e-4,
#             patience=5,
#             device=device,
#             model_save_path=temp_model_path
#         )
        
#         assert len(train_losses) == epochs, f"Attese {epochs} metriche registrate, ottenute {len(train_losses)}"
#         assert train_losses[0] != float('inf'), "La training loss è andata a infinito al primo step"
#         assert os.path.exists(temp_model_path), "Il modello non è stato salvato correttamente"

#     finally:
#         if os.path.exists(temp_model_path):
#             os.remove(temp_model_path)

from podcnf.roms import PODcnf

def main():
    # Trained model
    os.makedirs('../results/stokes', exist_ok=True)
    target_folder = os.path.join("..", "results/stokes")
    MODEL_NAME = os.path.join(target_folder, "NF_Stokes.pth")
    gdown.download(id = "1o3TPXo87eKOHRo82LjsOxeVTWZ6Od3mQ", quiet=True, output = MODEL_NAME)
    loaded_model = torch.load(MODEL_NAME, map_location = device)
    loaded_params = loaded_model['hyperparameters']

    # Data trasnformed through POD
    os.makedirs('../data', exist_ok=True)
    target_folder = os.path.join("..", "data") 
    reduced_input_file = os.path.join(target_folder, "stokes_data_reduced_6400.pt")
    gdown.download(id="19304ojlsmuL7hntN-m8KeR_CrAZD7wHb", quiet=True, output=reduced_input_file)
    reduced_dataset = torch.load(reduced_input_file, weights_only=True)
    mu = reduced_dataset['mu']
    c = reduced_dataset['c']

    # Raw data
    os.makedirs('../data', exist_ok=True)
    target_folder = os.path.join("..", "data")
    filename = os.path.join(target_folder, "stokes_data_6400.pt")
    gdown.download(id="1_E-gNMU9aMmWXHqm63JspI9i4kWy-wd1", quiet=True, output=filename)
    loaded_data = torch.load(filename, weights_only=True)

    u = loaded_data['u']
    eps = loaded_data['eps'].squeeze(1)
    theta = loaded_data['theta'].squeeze(1)
    mu = loaded_data['mu']

    dim_x = mu.shape[1]
    dim_y = c.shape[1]
    num_flows_loaded = loaded_params['num_flows']
    hidden_size_loaded = loaded_params['hidden_size']
    hidden_depth_loaded = loaded_params['hidden_depth']

    print(f"NF parameters flows: {num_flows_loaded}, size: {hidden_size_loaded}, depth: {hidden_depth_loaded}")

    NF_linear = NormalizingFlow(
        dim_x,
        dim_y,
        num_flows=num_flows_loaded,
        hidden_size=hidden_size_loaded,
        hidden_depth=hidden_depth_loaded,
        device=device
    ).to(device)

    NF_linear.load_state_dict(loaded_model['model_state_dict'])

    # c_scaler and mu_scaler
    c_scaler_path = os.path.join(target_folder, "c_scaler.pkl")
    if not os.path.exists(c_scaler_path):
        gdown.download(id="1m2RkT6bkPidlOOf9oB6EUgT4cYsCj9kd", quiet=True, output=c_scaler_path)
    with open(c_scaler_path, "rb") as file:
        c_scaler = pickle.load(file)

    mu_scaler_path = os.path.join(target_folder, "mu_scaler.pkl")
    if not os.path.exists(mu_scaler_path):
        gdown.download(id="1USRvaurzv0nSxT3vHQ98AaapTE498Uj6", quiet=True, output=mu_scaler_path)
    with open(mu_scaler_path, "rb") as file:
        mu_scaler = pickle.load(file)

    from podcnf.roms import TorchScaler
    mu_scaler = TorchScaler(mu_scaler.mean_, mu_scaler.scale_, device)
    c_scaler =  TorchScaler(c_scaler.mean_, c_scaler.scale_, device)

    # V_POD matrix
    V_file = os.path.join(target_folder, "V_POD_matrix.pt")
    if not os.path.exists(V_file):
        gdown.download(id="1dQ1QtqT8S96mJVZ-9axsPh5rW6tLO8RU", quiet=True, output=V_file)
    V = torch.load(V_file, map_location="cpu", weights_only=False)

    # Initialization of the object
    podcnf = PODcnf(V, NF_linear, mu_scaler, c_scaler)

    test_index = 6179 # np.random.randint(n_val, n_samples) # Userai questo per tutte le successive analisi
    # Corresponding value for mu and g
    mu_sele = mu[test_index - 1, :]
    u_true = u[test_index -1, :]
    print(f"mu_selected: {mu_sele}")
    print(test_index)

    u_sample = podcnf.sample(mu_sele, 2)
    c_sample = podcnf.sample_latent(mu_sele, 2)

    print(u_sample.shape)
    print(c_sample.shape)

if __name__ == "__main__":
    main()