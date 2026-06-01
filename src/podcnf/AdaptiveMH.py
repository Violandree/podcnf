import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.special import logsumexp

from podcnf.DataGenerationLinearElasticity import FOMsampler

def adaptive_metropolis_hastings(
    flow_model, u_obs, theta_0, N, bounds,
    mu_scaler, c_scaler, device,
    V, surface_idx,
    temperature=1.0, C_0=None, n_0=100, epsilon=1e-6, s_d=None,
    n_generations=100, h=0.1
):

    flow_model.eval()

    # Outside the loop I'll compute the "projection" matrix into the subspace (62x20)
    # In this way u_sensor is directly given by: u_sensor = c * V_sensor'
    V_sensors = V[surface_idx, :]
    V_sensors_torch = torch.tensor(V_sensors, dtype=torch.float32).to(device)

    u_obs_tensor = u_obs.to(device)
    mean_c = torch.tensor(c_scaler.mean_, dtype=torch.float32).to(device)
    scale_c = torch.tensor(c_scaler.scale_, dtype=torch.float32).to(device)

    # Initialization
    theta_n = np.array(theta_0, dtype=np.float32)
    d = len(theta_n)
    theta_n_scaled = mu_scaler.transform(theta_n.reshape(1, -1))
    theta_tensor = torch.tensor(theta_n_scaled, dtype=torch.float32).to(device)

    def compute_log_likelihood(theta_t):
        with torch.no_grad():
            theta_rep = theta_t.repeat(n_generations, 1)
            c_samples_norm = flow_model.sample(theta_rep) # sample ritorna (z, ldj), prendo z (c_samples)
            c_samples = c_samples_norm * scale_c + mean_c

            # Projection on the sensors
            # [N_gen, 20] @ [20, 62] -> [N_gen, 62]
            u_sensor_sample = c_samples @ V_sensors_torch.T

            # Da rivedere dal punto di vista teorico. In linea di prinicipio dal momento che il NF
            # non da un'unica soluzione per un dato valore di theta, considera n_generations (= 100)
            # sample, ognuno di questi fornisce una soluzione diversa. Per ogni soluzione fornita
            # dal NF nel calcolo l'errore in norma L2 con u_obs_tensor. Ottenute le distanze utilizza
            # un Kernel Gaussiano per vedere la bontà del theta selezionato
            # L2-distance
            diff = u_sensor_sample - u_obs_tensor.reshape(1, -1)
            dj2 = diff.pow(2).sum(dim=1) # [N_gen]

            # LogSumExp per stabilità (KDE Likelihood)
            # log( sum(exp(-d^2 / 2h^2)) ) - log(N)
            log_pi = torch.logsumexp(-dj2 / (2 * h**2), dim=0) - np.log(n_generations)

            return log_pi.item()

    # Initial value for the Log-likelihood
    log_pi_n = compute_log_likelihood(theta_tensor)

    chain = []
    accepted_count = 0

    # Covariance
    if C_0 is None:
        C_n = np.eye(d) * 1e-5
    else:
        C_n = np.array(C_0)

    theta_bar_n = theta_n.copy()

    # Scaling factor
    if s_d is None:
        scaling_val = (2.38**2) / d
    else:
        scaling_val = s_d

    # MCMC LOOP
    for n in tqdm(range(1, N + 1), desc="Adaptive MH"):

        # Proposal Covariance once n>n_0
        if n <= n_0:
            proposal_cov = C_n
        else:
            proposal_cov = scaling_val * C_n + epsilon * np.eye(d)

        perturbation = np.random.multivariate_normal(np.zeros(d), proposal_cov)
        Y = theta_n + perturbation

        # Check Prior (Bounds)
        if (Y[0] < bounds['m_min'] or Y[0] > bounds['m_max'] or
            Y[1] < bounds['d_min'] or Y[1] > bounds['d_max']):
            chain.append(theta_n)
            theta_next = theta_n
        else:
            # Compute the likelihood for the candidate
            Y_scaled = mu_scaler.transform(Y.reshape(1, -1))
            Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32).to(device)

            log_pi_Y = compute_log_likelihood(Y_tensor)

            # Acceptance ratio
            log_alpha = (log_pi_Y - log_pi_n) / temperature

            if np.log(np.random.rand()) < log_alpha:
                theta_n = Y
                log_pi_n = log_pi_Y
                accepted_count += 1

            chain.append(theta_n)
            theta_next = theta_n

        # Updating the covariance
        if n > n_0:
            theta_bar_prev = theta_bar_n.copy()
            theta_bar_n = (n * theta_bar_prev + theta_next) / (n + 1)

            dt = (theta_next - theta_bar_prev).reshape(-1, 1)
            term_update = np.dot(dt, dt.T)
            C_n = ((n - 1) / n) * C_n + (scaling_val / n) * (term_update * (n / (n + 1)) + epsilon * np.eye(d))

    chain = np.array(chain)
    acc_rate = accepted_count / N
    print(f"\nAcceptance Rate: {acc_rate:.2%}")
    return chain, C_n


def adaptive_metropolis_hastings_fom(
    u_obs, theta_0, N, bounds, surface_idx,
    temperature=1.0, C_0=None, n_0=100, epsilon=1e-6, s_d=None,
    n_generations=10, h=0.1
):

    u_obs_np = np.array(u_obs)

    # Initialization
    theta_n = np.array(theta_0, dtype=np.float64)
    d = len(theta_n)

    def compute_log_likelihood(theta_t):
        m_in, d_in = theta_t[0], theta_t[1]
        u_sensor_samples = np.zeros((n_generations, len(surface_idx)))

        # Marginalizing over the stochasticity of the PDE using the FOM
        for g_idx in range(n_generations):
            seed_in = np.random.randint(0, 2**32 - 1)

            # Generates data and select the sensors
            _, _, _, u_data_full, _ = FOMsampler(seed_in, m_in, d_in, option=1)
            u_sensor_samples[g_idx, :] = u_data_full[surface_idx]

        # L2 distance squared on sensors
        diff = u_sensor_samples - u_obs_np.reshape(1, -1)
        dj2 = np.sum(diff**2, axis=1) # [N_gen]

        # KDE Likelihood using LogSumExp for stability
        log_pi = logsumexp(-dj2 / (2 * h**2)) - np.log(n_generations)

        return log_pi

    # Initial value for the Log-likelihood
    print(f"Computing initial likelihood (requires {n_generations} FOM solves)...")
    log_pi_n = compute_log_likelihood(theta_n)
    print("Initial likelihood computed. Starting MCMC loop...")

    chain = []
    accepted_count = 0

    # Covariance initialization
    if C_0 is None:
        C_n = np.eye(d) * 1e-5
    else:
        C_n = np.array(C_0)

    theta_bar_n = theta_n.copy()

    # Scaling factor for the proposal distribution
    if s_d is None:
        scaling_val = (2.38**2) / d
    else:
        scaling_val = s_d

    # MCMC LOOP
    for n in tqdm(range(1, N + 1), desc="Adaptive MH (FOM)"):

        # Proposal Covariance once n > n_0
        if n <= n_0:
            proposal_cov = C_n
        else:
            proposal_cov = scaling_val * C_n + epsilon * np.eye(d)

        perturbation = np.random.multivariate_normal(np.zeros(d), proposal_cov)
        Y = theta_n + perturbation

        # Check Prior (Bounds)
        if (Y[0] < bounds['m_min'] or Y[0] > bounds['m_max'] or
            Y[1] < bounds['d_min'] or Y[1] > bounds['d_max']):
            chain.append(theta_n)
            theta_next = theta_n
        else:
            # Compute the likelihood for the candidate directly with the physical parameters
            log_pi_Y = compute_log_likelihood(Y)

            # Acceptance ratio
            log_alpha = (log_pi_Y - log_pi_n) / temperature

            if np.log(np.random.rand()) < log_alpha:
                theta_n = Y
                log_pi_n = log_pi_Y
                accepted_count += 1

            chain.append(theta_n)
            theta_next = theta_n

        # Updating the covariance
        if n > n_0:
            theta_bar_prev = theta_bar_n.copy()
            theta_bar_n = (n * theta_bar_prev + theta_next) / (n + 1)

            dt = (theta_next - theta_bar_prev).reshape(-1, 1)
            term_update = np.dot(dt, dt.T)

            # Update step
            C_n = ((n - 1) / n) * C_n + (1 / n) * (term_update * (n / (n + 1)) + epsilon * np.eye(d))

    chain = np.array(chain)
    acc_rate = accepted_count / N
    print(f"\nAcceptance Rate: {acc_rate:.2%}")
    return chain, C_n