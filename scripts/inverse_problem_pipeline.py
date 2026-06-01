import os
import pickle
import gdown
import numpy as np

import torch
from tqdm import tqdm

from podcnf.NFmodel import NormalizingFlow

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")


def main():

    # Load model: {'learning_rate': 0.001, 'num_flows': 16, 'hidden_size': 256, 'hidden_depth': 2, 'weight_decacy': 1e-05}
    target_folder = os.path.join("..", "results/elastic")
    MODEL_NAME = os.path.join(target_folder, 'MODEL_64_NEW.pth')
    gdown.download(id = "1Mv9opjkEMDQaLBQx07Fqvh_ItyefLsCl", quiet=True, output = MODEL_NAME)
    loaded_model = torch.load(MODEL_NAME, map_location=device)

    # Load data:
    reduced_input_file = "../data/elastic_data_reduced_6400.pt" 
    reduced_dataset = torch.load(reduced_input_file, weights_only=True)
    mu = reduced_dataset['mu']
    c = reduced_dataset['c']

    # Upload scaler for algorithm
    with open("../results/elastic/c_scaler.pkl", "rb") as file:
        c_scaler = pickle.load(file)

    with open("../results/elastic/mu_scaler.pkl", "rb") as file:
        mu_scaler = pickle.load(file)

    # V matrix from the POD decomposition
    V_file = "../results/elastic/V_POD_matrix.pt" 
    V = torch.load(V_file, weights_only=True)

    dim_x = mu.shape[1]
    dim_y = c.shape[1]  

    # Linear model
    num_flows = 16
    hidden_size = 256
    hidden_depth = 2

    NF_linear = NormalizingFlow(dim_x, dim_y, num_flows, hidden_size, hidden_depth, device).to(device)
    NF_linear.load_state_dict(loaded_model)
    flow = NF_linear

    num_sensors = 31

    sur = [0,1] # top-left already added
    for j in range(num_sensors-1):
        xy = np.array([np.pow(j+2,2)+j, np.pow(j+2,2)+j+1])
        sur.extend(xy)
    surface_idx = np.array(sur)
    print(f"Number of sensors on the surface: {len(surface_idx)}")
    u_surface_sensor = u[:, surface_idx] 
    print(u_surface_sensor.shape)   

    n_simulations = 10

    bounds = {
        'm_min': 1.0, 'm_max': 2.0,
        'd_min': 0.05, 'd_max': 0.25
    }

    for i in range(n_simulations):
        # Choose an index to test
        test_idx = np.random.randint(n_val, n_samples)
        print(f"Selected test index:\t{test_idx}")

        mu_true_phys = mu[test_idx].numpy()
        u_obs = u_surface_sensor[test_idx]

        initial_guess = [np.random.rand() + 1, np.random.rand()*0.1 + 0.15]

        print(f"Index: {test_idx} - True Mass={mu_true_phys[0]:.4f}, True Delta={mu_true_phys[1]:.4f}")
        print(f"Initial guess: {initial_guess}")

        print(">>> Adaptive MH with PODCNF:")

        print("\n--- EXPLORATION ---")
        initial_cov_expl = [[0.001, 0], [0, 0.001]]

        t0 = perf_counter()

        chain_exploration, cov_learned = adaptive_metropolis_hastings(
            flow_model=flow,
            u_obs=u_obs,
            theta_0=initial_guess,
            N=10000,
            bounds=bounds,
            mu_scaler=mu_scaler,
            c_scaler=c_scaler,
            device=device,
            V=V,
            surface_idx=surface_idx,
            temperature=5.0,
            C_0=initial_cov_expl,
            n_0=500,
            s_d=2.4,
            n_generations=100,
            h=0.1
        )

        t_exp = perf_counter() - t0

        best_guess = chain_exploration[-1]
        print(f"Best Guess after exploration: {best_guess}")

        print("\n--- REFINEMENT ---")
        cov_for_refinement = cov_learned + np.eye(2) * 1e-6

        t1 = perf_counter()

        chain_refined, _ = adaptive_metropolis_hastings(
            flow_model=flow,
            u_obs=u_obs,
            theta_0=best_guess,
            N=30000,
            bounds=bounds,
            mu_scaler=mu_scaler,
            c_scaler=c_scaler,
            device=device,
            V=V,
            surface_idx=surface_idx,
            temperature=1.0,
            C_0=cov_for_refinement,
            n_0=0,
            s_d=0.35,
            n_generations=200,
            h=0.03
        )

        t_ref = perf_counter() - t1

        print(f">>>PODCNF\nExploration time:\t{t_exp}\nRefinement time:\t{t_ref}")

        # QUI ANDRANNO SALVATI I PLOT I TEMPI DI ESECUZIONE DELL'ALGORITMO CON MH

        print(">>> Adaptive MH with PODCNF:")

        t0_FOM = perf_counter()

        chain_exploration_FOM, cov_learned_FOM = adaptive_metropolis_hastings_fom(
            u_obs=u_obs,
            theta_0=initial_guess,
            N=1000,
            bounds=bounds,
            surface_idx=surface_idx,
            temperature=5.0,
            C_0=initial_cov_expl,
            n_0=200,            # Adjusted n_0
            s_d=2.4,
            n_generations=10,
            h=0.1
        )

        t_exp_FOM = perf_counter() - t0_FOM
        print(f"Exploration Time: {t_exp_FOM:.2f} s")

        best_guess_FOM = chain_exploration_FOM[-1]
        print(f"Best Guess after exploration: {best_guess_FOM}")

        print("\n--- REFINEMENT ---")
        cov_for_refinement_FOM = cov_learned_FOM + np.eye(2) * 1e-6

        t1_FOM = perf_counter()

        chain_refined_FOM, _ = adaptive_metropolis_hastings_fom(
            u_obs=u_obs,
            theta_0=best_guess_FOM,
            N=3000,
            bounds=bounds,
            surface_idx=surface_idx,
            temperature=1.0,
            C_0=cov_for_refinement_FOM,
            n_0=0,
            s_d=0.35,
            n_generations=25,
            h=0.03
        )

        t_ref_FOM = perf_counter() - t1_FOM
        print(f"Refinement Time: {t_ref_FOM:.2f} s")

    return

if __name__ == "__main__":
    main()