import os
import pickle
import json
import gdown
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter

import torch
from tqdm import tqdm

import logging
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
                               axis = 0), 
                device = device)

def plot_trace_and_posterior(chain_exp, chain_ref, mu_true, model_name, test_idx, results_dir):
    full_chain = np.vstack([chain_exp, chain_ref])
    split_point = len(chain_exp)

    # Burn-in e step
    if model_name == "NF":
        clean_samples = chain_ref[2000:]
        step = 200
    else:
        clean_samples = chain_ref[1000:]
        step = 20

    # TRACE PLOT
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(full_chain[:, 0], color='#1f77b4', linewidth=0.8, alpha=0.9, label='Chain')
    axes[0].axvline(x=split_point, color='black', linestyle='--', linewidth=1.5, label='Refinement')
    axes[0].axhline(y=mu_true[0], color='red', linestyle='--', linewidth=2, label='True target')
    axes[0].set_title(f"Trace Plot: Mass ({mu_true[0]:.4f})", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Mass")
    axes[0].set_xlabel("Iteration")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(full_chain[:, 1], color='#ff7f0e', linewidth=0.8, alpha=0.9, label='Chain')
    axes[1].axvline(x=split_point, color='black', linestyle='--', linewidth=1.5)
    axes[1].axhline(y=mu_true[1], color='red', linestyle='--', linewidth=2, label='True target')
    axes[1].set_title(f"Trace Plot: Delta ({mu_true[1]:.4f})", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Delta")
    axes[1].set_xlabel("Iteration")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'trace_{model_name}_idx_{test_idx}.png'))
    plt.close(fig1)

    # POSTERIOR DISTRIBUTION
    fig2, ax_post = plt.subplots(figsize=(7, 5))
    sns.kdeplot(
        x=clean_samples[:, 0],
        y=clean_samples[:, 1],
        cmap="Blues" if model_name == "NF" else "Oranges",
        fill=True,
        thresh=0.05,
        levels=10,
        ax=ax_post
    )
    ax_post.scatter(mu_true[0], mu_true[1], s=300, c='red', marker='X', label='True target', zorder=10)
    ax_post.scatter(clean_samples[::step, 0], clean_samples[::step, 1], s=5, c='black', alpha=0.1, label='Samples')
    ax_post.set_title(f"Posterior Distribution ({model_name})", fontsize=14, fontweight='bold')
    ax_post.set_xlabel("Mass")
    ax_post.set_ylabel("Delta")
    ax_post.legend()
    ax_post.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'posterior_{model_name}_idx_{test_idx}.png'))
    plt.close(fig2)

def main():

    # Directories
    target_folder = os.path.join("..", "results", "elastic")
    os.makedirs(target_folder, exist_ok=True)

    target_folder_data = os.path.join("..", "data")
    os.makedirs(target_folder_data, exist_ok=True)

    results_dir = os.path.join("results")
    os.makedirs(results_dir, exist_ok=True)

    # Load model
    MODEL_NAME = os.path.join(target_folder, 'MODEL_64_NEW.pth')
    if not os.path.exists(MODEL_NAME):
        gdown.download(id="1Mv9opjkEMDQaLBQx07Fqvh_ItyefLsCl", quiet=True, output=MODEL_NAME)
    loaded_model = torch.load(MODEL_NAME, map_location=device)

    # Load reduced data
    reduced_input_file = os.path.join(target_folder_data, "elastic_data_reduced_6400.pt")
    if not os.path.exists(reduced_input_file):
        gdown.download(id="1FX_9MQ1gAG4Xqyo3HGoGW0Tvokuouqdq", quiet=True, output=reduced_input_file)
    reduced_dataset = torch.load(reduced_input_file, map_location=device, weights_only=True)
    mu = reduced_dataset['mu']
    c = reduced_dataset['c']

    # Load raw data
    raw_data_file = os.path.join(target_folder_data, "data_linear_density_2pi.pt")
    if not os.path.exists(raw_data_file):
        gdown.download(id="1NYiXCqCNLhB87o3OlAav6YfiAO9qoBnM", quiet=True, output=raw_data_file)
    loaded_data = torch.load(raw_data_file, map_location=device)
    u = loaded_data['u_data']

    # c_scaler and mu_scaler
    c_scaler_path = os.path.join(target_folder, "c_scaler.pkl")
    if not os.path.exists(c_scaler_path):
        gdown.download(id="12XBewDNM8SSwPoJxmUuXjUXD3kC3lFBr", quiet=True, output=c_scaler_path)
    with open(c_scaler_path, "rb") as file:
        c_scal = pickle.load(file)

    mu_scaler = TorchScaler(mu_scal.mean_, mu_scal.scale_, device)

    mu_scaler_path = os.path.join(target_folder, "mu_scaler.pkl")
    if not os.path.exists(mu_scaler_path):
        gdown.download(id="1R2u27onqNzsbkvHPWFlrmRuzXyGod_NJ", quiet=True, output=mu_scaler_path)
    with open(mu_scaler_path, "rb") as file:
        mu_scal = pickle.load(file)

    c_scaler = TorchScaler(c_scal.mean_, c_scal.scale_, device)

    # V_POD
    V_file = os.path.join(target_folder, "V_POD_matrix.pt")
    if not os.path.exists(V_file):
        gdown.download(id="1k0WgaeP4hXHJEwOd8Ng9Tv7Pbi2iSa9g", quiet=True, output=V_file)
    V = torch.load(V_file, map_location=device, weights_only=False)

    dim_x = mu.shape[1]
    dim_y = c.shape[1]

    # Load Normalizing Flow
    num_flows = 16
    hidden_size = 256
    hidden_depth = 2

    NF_linear = NormalizingFlow(dim_x, dim_y, num_flows, hidden_size, hidden_depth, device).to(device)
    NF_linear.load_state_dict(loaded_model)
    flow = NF_linear

    num_sensors = 31
    sur = [0,1] # top-left already added

    for j in range(num_sensors-1):
        xy = np.array([(j+2)**2 + j, (j+2)**2 + j + 1]) 
        sur.extend(xy)

    surface_idx = np.array(sur)
    print(f"Number of sensors on the surface: {len(surface_idx)}")
    u_surface_sensor = u[:, surface_idx] 
    print(f"Shape of u_surface_sensor: {u_surface_sensor.shape}")   

    n_simulations = 10
    n_samples = mu.shape[0]
    n_val = 6081

    bounds = {
        'm_min': 1.0, 'm_max': 2.0,
        'd_min': 0.05, 'd_max': 0.25
    }

    V_sensors = V[surface_idx, :].to(device)

    Q1 = lambda c: c @ V_sensors.T # For generative model
    Q2 = lambda ufom: ufom[:, surface_idx] # For Full Order Modell

    test_idx = random.sample(range(n_val, n_samples + 1), n_simulations)

    verbose = True

    for i in range(n_simulations):
        print(f">>> Simulation {i+1}/{n_simulations} - Selected test index:\t{test_idx[i]}\n")

        mu_true_phys = mu[test_idx[i]].numpy()
        u_obs = u_surface_sensor[test_idx[i]].to(device)
        
        mu_0 = torch.tensor(
            [torch.rand(1).item() + 1.0, torch.rand(1).item() * 0.1 + 0.15],
            dtype=torch.float32,
            device=device
        )

        print(f"Index: {test_idx[i]} - True Mass={mu_true_phys[0]:.4f}, True Delta={mu_true_phys[1]:.4f}")
        print(f"Initial guess: {initial_guess}")

        # PODCNF
        print(">>> Adaptive MH with PODCNF:")
        print("\n--- EXPLORATION ---")
        initial_cov_expl = torch.tensor([[0.001, 0.0], [0.0, 0.001]], dtype=torch.float32, device=device)

        t0 = perf_counter()
        chain_exploration, cov_learned = adaptive_metropolis_hastings(
            generator=podcnf.sample_latent_same_mu,
            Q=Q,
            mu_0=mu_0,
            obs=u_obs,
            N=10000,
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
            generator=podcnf.sample_latent_same_mu,
            Q=Q,
            mu_0=best_guess,
            obs=u_obs,
            N=30000,
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
        
        plot_trace_and_posterior(chain_exploration, chain_refined, mu_true_phys, "NF", test_idx[i], results_dir)

        # FOM
        print("\n>>> Adaptive MH with FOM:")
        print("\n--- EXPLORATION ---")
        
        t0_FOM = perf_counter()
        chain_exploration_FOM, cov_learned_FOM = adaptive_metropolis_hastings(
            generator=FOMgenerator,
            Q=Q2,
            mu_0=mu_0,
            obs=u_obs,
            N=10000,
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
        cov_for_refinement_FOM = cov_learned + torch.eye(2, device=device) * 1e-6
        t1_FOM = perf_counter()
        chain_refined_FOM, _ = daptive_metropolis_hastings(
            generator=FOMgenerator,
            Q=lambda ufom: ufom[:, surface_idx],
            mu_0=mu_0,
            obs=u_obs,
            N=30000,
            bounds=bounds,
            temperature=1.0,
            C_0=initial_cov_expl,
            n_0=0,
            s_d=0.35,
            nrep=25,
            h=0.03,
            verbose=verbose
        )
        t_ref_FOM = perf_counter() - t1_FOM
        print(f">>> FOM\nRefinement Time: {t_ref_FOM:.2f} s")
        
        plot_trace_and_posterior(chain_exploration_FOM, chain_refined_FOM, mu_true_phys, "FOM", test_idx[i], results_dir)
        
        # Results
        clean_samples_NF = chain_refined[2000:]
        clean_samples_FOM = chain_refined_FOM[1000:]
        
        # Performance PODCNF
        mean_mass_NF = np.mean(clean_samples_NF[:, 0])
        std_mass_NF = np.std(clean_samples_NF[:, 0])
        err_mass_NF = abs(mean_mass_NF - mu_true_phys[0]) / mu_true_phys[0]

        mean_delta_NF = np.mean(clean_samples_NF[:, 1])
        std_delta_NF = np.std(clean_samples_NF[:, 1])
        err_delta_NF = abs(mean_delta_NF - mu_true_phys[1]) / mu_true_phys[1]

        # Performance FOM
        mean_mass_FOM = np.mean(clean_samples_FOM[:, 0])
        std_mass_FOM = np.std(clean_samples_FOM[:, 0])
        err_mass_FOM = abs(mean_mass_FOM - mu_true_phys[0]) / mu_true_phys[0]

        mean_delta_FOM = np.mean(clean_samples_FOM[:, 1])
        std_delta_FOM = np.std(clean_samples_FOM[:, 1])
        err_delta_FOM = abs(mean_delta_FOM - mu_true_phys[1]) / mu_true_phys[1]

        print("\n--- ERROR METRICS ---")
        print(f"PODCNF - Estimated Mass:  {mean_mass_NF:.4f} +/- {std_mass_NF:.4f} | Rel Error: {err_mass_NF:.4%}")
        print(f"PODCNF - Estimated Delta: {mean_delta_NF:.4f} +/- {std_delta_NF:.4f} | Rel Error: {err_delta_NF:.4%}")
        print(f"FOM    - Estimated Mass:  {mean_mass_FOM:.4f} +/- {std_mass_FOM:.4f} | Rel Error: {err_mass_FOM:.4%}")
        print(f"FOM    - Estimated Delta: {mean_delta_FOM:.4f} +/- {std_delta_FOM:.4f} | Rel Error: {err_delta_FOM:.4%}")

        print("\nCalculating Wasserstein distance:")
        w2_dist = Wasser_dist(
            torch.tensor(clean_samples_FOM, dtype=torch.float32), 
            torch.tensor(clean_samples_NF, dtype=torch.float32)
        )
        print(f"Wasserstein Distance (W2): {w2_dist:.4f}")
        
        fig3, ax_comp = plt.subplots(figsize=(7, 5))
        sns.kdeplot(x=clean_samples_NF[:, 0], y=clean_samples_NF[:, 1], cmap="Blues", fill=True, alpha=0.5)
        sns.kdeplot(x=clean_samples_FOM[:, 0], y=clean_samples_FOM[:, 1], cmap="Oranges", fill=True, alpha=0.5)
        
        ax_comp.plot([], [], color='blue', alpha=0.5, label='NF Posterior')
        ax_comp.plot([], [], color='orange', alpha=0.5, label='FOM Posterior')
        ax_comp.scatter(mu_true_phys[0], mu_true_phys[1], s=300, c='red', marker='X', label='True target', zorder=10)
        
        ax_comp.set_title(f"Posterior Comparison (W2 Dist: {w2_dist:.4f})", fontsize=14, fontweight='bold')
        ax_comp.set_xlabel("Mass")
        ax_comp.set_ylabel("Delta")
        ax_comp.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'posterior_comparison_idx_{test_idx[i]}.png'))
        plt.close(fig3)
        
        results_dict = {
            'test_idx': int(test_idx[i]),
            'true_mass': float(mu_true_phys[0]),
            'true_delta': float(mu_true_phys[1]),
            'initial_guess': [float(ig) for ig in initial_guess],
            'NF': {
                'time_exploration': float(t_exp),
                'time_refinement': float(t_ref),
                'mean_mass': float(mean_mass_NF),
                'std_mass': float(std_mass_NF),
                'rel_error_mass': float(err_mass_NF),
                'mean_delta': float(mean_delta_NF),
                'std_delta': float(std_delta_NF),
                'rel_error_delta': float(err_delta_NF)
            },
            'FOM': {
                'time_exploration': float(t_exp_FOM),
                'time_refinement': float(t_ref_FOM),
                'mean_mass': float(mean_mass_FOM),
                'std_mass': float(std_mass_FOM),
                'rel_error_mass': float(err_mass_FOM),
                'mean_delta': float(mean_delta_FOM),
                'std_delta': float(std_delta_FOM),
                'rel_error_delta': float(err_delta_FOM)
            },
            'Wasserstein_dist': float(w2_dist)
        }
        
        with open(os.path.join(results_dir, f'results_idx_{test_idx[i]}.json'), 'w') as f:
            json.dump(results_dict, f, indent=4)
            
        print(f"Results for index {test_idx[i]} saved successfully!\n")

if __name__ == "__main__":
    main()