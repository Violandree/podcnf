import os
import gdown
import numpy as np
from time import perf_counter
import random
import json

import torch

import logging
try:
    from dolfin import set_log_level, LogLevel
    set_log_level(LogLevel.WARNING)
    logging.getLogger('FFC').setLevel(logging.WARNING)
    logging.getLogger('UFL').setLevel(logging.WARNING)
except ImportError:
    pass

from podcnf.NFmodel import NormalizingFlow
from podcnf.AdaptiveMH import adaptive_metropolis_hastings
from podcnf.Utils import Wasser_dist
from podcnf.DataGenerationLinearElasticity import FOMsampler
from podcnf.roms import PODcnf

SEED = 42

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

def FOMgenerator(muj, nrep):
  if(isinstance(muj, torch.Tensor)):
    mu0 = muj.cpu().numpy()
  else:
    mu0 = muj
  return torch.tensor(np.stack([FOMsampler(j, *mu0, option = 1)[-2] for j in range(nrep)],
                               axis = 0), device = device)

def main():

    # Directories
    target_folder = os.path.join("..", "results", "elastic")
    os.makedirs(target_folder, exist_ok=True)

    target_folder_data = os.path.join("..", "data")
    os.makedirs(target_folder_data, exist_ok=True)

    results_dir = os.path.join("inverse_results")
    os.makedirs(results_dir, exist_ok=True)

    # Load reduced data
    test_data_file = 'test_data.pt'
    data_download_path = gdown.download(id="1eT8re3AeZaQdIen4R6iLkYcpk2Ynj2x9", quiet=True, output=test_data_file)
    test_data = torch.load(data_download_path, map_location=device, weights_only=True)

    mu = test_data['mu_test']
    u = test_data['u_test']

    n_samples = mu.shape[0]
    num_sensors = 31
    sur = [0,1] # top-left already added

    for j in range(num_sensors-1):
        xy = np.array([(j+2)**2 + j, (j+2)**2 + j + 1]) 
        sur.extend(xy)

    surface_idx = np.array(sur)
    print(f"Number of sensors on the surface: {len(surface_idx)}")
    u_surface_sensor = u[:, surface_idx] 
    print(f"Shape of u_surface_sensor: {u_surface_sensor.shape}")

    n_simulations = 2

    bounds = {
        'm_min': 1.0, 'm_max': 2.0,
        'd_min': 0.05, 'd_max': 0.25
    }

    verbose = True

    model_name = "elasticPODcnf.pt"
    downloaded_path = gdown.download(id="1M7Dx3tKViTRwUkbzoG1mMcMjUWnRmc8v", quiet=True, output=model_name)
    loaded_rom = PODcnf.load(downloaded_path, device)

    surface_idx_tensor = torch.tensor(sur, dtype=torch.long, device=device)

    V_sensors = loaded_rom.V[surface_idx_tensor, :]

    def podcnf_sensor_generator(mu, nrep):
        c_samples = loaded_rom.sample_latent(mu, nrep)
        return c_samples @ V_sensors.T

    def fom_sensor_generator(muj, nrep):
        mu0 = muj.cpu().numpy() if isinstance(muj, torch.Tensor) else muj
        u_sensors = []
        for j in range(nrep):
            full_u = FOMsampler(j, *mu0, option=1)[-2]
            u_sensors.append(full_u[surface_idx])
            
        return torch.tensor(np.stack(u_sensors, axis=0), dtype=torch.float32, device=device)

    Q = lambda u: u

    test_idx = np.random.randint(0, n_samples-1, n_simulations)

    Nexp = 10000
    Nref = 30000

    for i in range(n_simulations):
        print(f">>> Simulation {i+1}/{n_simulations} - Selected test index:\t{test_idx[i]}\n")

        mu_true_phys = mu[test_idx[i]].cpu().numpy()
        u_obs = u_surface_sensor[test_idx[i]].to(device)
        
        mu_0 = torch.tensor(
            [torch.rand(1).item() + 1.0, torch.rand(1).item() * 0.1 + 0.15],
            dtype=torch.float32,
            device=device
        )

        print(f"Index: {test_idx[i]} - True Mass={mu_true_phys[0]:.4f}, True Delta={mu_true_phys[1]:.4f}")
        print(f"Initial guess: {mu_0}")

        # PODCNF
        print(">>> Adaptive MH with PODCNF:")
        print("\n--- EXPLORATION ---")
        initial_cov_expl = torch.tensor([[0.001, 0.0], [0.0, 0.001]], dtype=torch.float32, device=device)

        t0 = perf_counter()
        chain_exploration, cov_learned = adaptive_metropolis_hastings(
            generator=loaded_rom.sample,
            Q=Q,
            mu_0=mu_0,
            obs=u_obs,
            N=Nexp,
            bounds=bounds,
            temperature=5.0,
            C_0=initial_cov_expl,
            n_0=500,
            s_d=2.4,
            nrep=100,
            h=0.1,
            verbose=verbose
        )
        t_exp = perf_counter() - t0
        best_guess = chain_exploration[-1].clone().detach()
        print(f"Best Guess after exploration: {best_guess}")

        print("\n--- REFINEMENT ---")
        cov_for_refinement = cov_learned + torch.eye(2, device=device) * 1e-6
        t1 = perf_counter()
        chain_refined, cov_refined = adaptive_metropolis_hastings(
            generator=loaded_rom.sample,
            Q=Q,
            mu_0=best_guess,
            obs=u_obs,
            N=Nref,
            bounds=bounds,
            temperature=1.0,
            C_0=cov_for_refinement,
            n_0=0,
            s_d=0.35,
            nrep=200,
            h=0.03,
            verbose=verbose
        )
        t_ref = perf_counter() - t1
        print(f">>> PODCNF\nExploration time:\t{t_exp}\nRefinement time:\t{t_ref}")
        
        full_chain_NF = torch.cat([chain_exploration, chain_refined])

        # FOM
        print("\n>>> Adaptive MH with FOM:")
        print("\n--- EXPLORATION ---")
        
        t0_FOM = perf_counter()
        chain_exploration_FOM, cov_learned_FOM = adaptive_metropolis_hastings(
            generator=FOMgenerator,
            Q=Q,
            mu_0=mu_0,
            obs=u_obs,
            N=Nexp,
            bounds=bounds,
            temperature=5.0,
            C_0=initial_cov_expl,
            n_0=200,
            s_d=2.4,
            nrep=10,
            h=0.1,
            verbose=verbose
        )
        t_exp_FOM = perf_counter() - t0_FOM
        best_guess_FOM = chain_exploration_FOM[-1].clone().detach()
        print(f"Exploration Time: {t_exp_FOM:.2f} s - Best Guess: {best_guess_FOM}")

        print("\n--- REFINEMENT ---")
        cov_for_refinement_FOM = cov_learned_FOM + torch.eye(2, device=device) * 1e-6
        t1_FOM = perf_counter()
        chain_refined_FOM, _ = adaptive_metropolis_hastings(
            generator=FOMgenerator,
            Q=Q,
            mu_0=best_guess_FOM,
            obs=u_obs,
            N=Nref,
            bounds=bounds,
            temperature=1.0,
            C_0=cov_for_refinement_FOM,
            n_0=0,
            s_d=0.35,
            nrep=25,
            h=0.03,
            verbose=verbose
        )
        t_ref_FOM = perf_counter() - t1_FOM
        print(f">>> FOM\nRefinement Time: {t_ref_FOM:.2f} s")
        
        full_chain_FOM = torch.cat([chain_exploration_FOM, chain_refined_FOM])

        np.save(
            os.path.join(results_dir, f'chain_NF_idx_{test_idx[i]}.npy'),
            full_chain_NF.detach().cpu().numpy()
        )
        
        np.save(
            os.path.join(results_dir, f'chain_FOM_idx_{test_idx[i]}.npy'),
            full_chain_FOM.detach().cpu().numpy()
        )

        results_dict = {
        'test_idx': int(test_idx[i]),
        'true_mass': float(mu_true_phys[0]),
        'true_delta': float(mu_true_phys[1]),
        'initial_guess': [float(ig) for ig in mu_0],
        'NF': {
            'time_exploration': float(t_exp),
            'time_refinement': float(t_ref),
        },
        'FOM': {
            'time_exploration': float(t_exp_FOM),
            'time_refinement': float(t_ref_FOM),
        }
    }
    
    with open(os.path.join(results_dir, f'results_idx_{test_idx[i]}.json'), 'w') as f:
        json.dump(results_dict, f, indent=4)
        
        print(f"Results for index {test_idx[i]} saved successfully!\n")

if __name__ == "__main__":
    main()