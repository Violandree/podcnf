import os
import torch
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from podcnf.DataGenerationStokes import ADR, stokes, inflows, bi

def main():
    print("Starting generation Stokes Data:")

    n_samples = 2
    n_jobs = 2

    save_dir = 'data'
    os.makedirs(save_dir, exist_ok=True)

    eps_vals = np.random.uniform(0.01, 0.1, n_samples)
    theta_vals = np.random.uniform(0, 2*np.pi, n_samples)
    c1_vals = np.random.uniform(-1, 1, n_samples)
    c2_vals = np.random.uniform(-1, 1, n_samples)
    c3_vals = np.random.uniform(-1, 1, n_samples)

    def compute_sample(i):
        return ADR(eps_vals[i], theta_vals[i], c1_vals[i], c2_vals[i], c3_vals[i])

    u_results = Parallel(n_jobs=N_JOBS)(delayed(compute_sample)(i) for i in tqdm(range(n_samples)))
    u_array = np.array(u_results, dtype=np.float32)

    eps_t = torch.tensor(eps_vals, dtype=torch.float32).unsqueeze(1)
    theta_t = torch.tensor(theta_vals, dtype=torch.float32).unsqueeze(1)
    c1_t = torch.tensor(c1_vals, dtype=torch.float32).unsqueeze(1)
    c2_t = torch.tensor(c2_vals, dtype=torch.float32).unsqueeze(1)
    c3_t = torch.tensor(c3_vals, dtype=torch.float32).unsqueeze(1)

    mu_array = torch.cat((eps_t, theta_t, c1_t, c2_t, c3_t), dim=1).numpy()

    np.save(f'{save_dir}/u_data.npy', u_array)
    np.save(f'{save_dir}/mu_data.npy', mu_array)
    
    print(f"Generazione completata! Dati salvati in {save_dir}/")


if __name__ == "__main__":
        main()