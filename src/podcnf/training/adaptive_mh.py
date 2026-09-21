import torch
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np

from tqdm import tqdm

def adaptive_metropolis_hastings(
    generator, Q, mu_0, obs,
    N, bounds,
    temperature=1.0, C_0=None, n_0=100, epsilon=1e-6, s_d=None,
    nrep=100, h=0.1, verbose=True
):

    # Device
    device = mu_0.device

    # Initialization
    obs = obs.to(device).unsqueeze(0)
    mu_n = mu_0.clone().detach()
    d = mu_n.shape[0]

    # Pre-allocation
    chain = torch.empty((N, d), dtype=torch.float32, device=device)
    accepted_count = 0

    if C_0 is None:
      C_n = torch.eye(d, dtype=torch.float32, device = device)
    else:
      C_n = torch.as_tensor(C_0, dtype=torch.float32, device=device) * 1e-5

    mu_bar_n = mu_n.detach()
    zero_mean = torch.zeros(d, dtype=torch.float32, device=device)
    scaling_val = (2.38 ** 2) / d if s_d is None else s_d

    log_nrep = torch.log(torch.tensor(nrep, dtype=torch.float32, device=device))

    def compute_log_likelihood(mu_t):
        with torch.no_grad():

            u_samples = generator(mu_t, nrep)

            # Select the sensors
            sensor_sample = Q(u_samples)

            diff = sensor_sample - obs.reshape(1, -1)
            dj2 = diff.pow(2).sum(dim=1) # [N_gen]

            # LogSumExp per stabilità (KDE Likelihood)
            # log( sum(exp(-d^2 / 2h^2)) ) - log(N)
            log_pi = torch.logsumexp(-dj2 / (2 * h**2), dim=0) - np.log(nrep)

            return log_pi.item()

    # Initial value for the Log-likelihood
    log_pi_n = compute_log_likelihood(mu_n)

    # Iteretor configuration
    iterator = range(N)
    if verbose:
        iterator = tqdm(iterator, desc="Adaptive MH")

    # MCMC LOOP
    for i in iterator:
        n = i + 1

        # Proposal Covariance once n>n_0
        if n <= n_0:
            proposal_cov = C_n
        else:
            proposal_cov = scaling_val * C_n + epsilon * torch.eye(d, device=device)

        proposal_cov = (proposal_cov + proposal_cov.T) / 2.0

        proposal_dist = MultivariateNormal(zero_mean, proposal_cov)
        perturbation = proposal_dist.sample()

        Y = mu_n + perturbation

        # Check Prior (Bounds)
        if (Y[0] < bounds['m_min'] or Y[0] > bounds['m_max'] or
            Y[1] < bounds['d_min'] or Y[1] > bounds['d_max']):
            mu_next = mu_n
        else:
            log_pi_Y = compute_log_likelihood(Y)

            # Acceptance ratio
            log_alpha = (log_pi_Y - log_pi_n) / temperature

            if np.log(np.random.rand()) < log_alpha:
                mu_n = Y
                log_pi_n = log_pi_Y
                accepted_count += 1

            mu_next = mu_n

        # Save on the pre-allocated matrix
        chain[i] = mu_n

        # Updating the covariance
        if n > n_0:
            mu_bar_prev = mu_bar_n.clone()
            mu_bar_n = (n * mu_bar_prev + mu_next) / (n + 1)

            dt = (mu_next - mu_bar_prev).reshape(-1, 1)
            term_update = dt @ dt.T
            C_n = ((n - 1) / n) * C_n + (scaling_val / n) * (term_update * (n / (n + 1)) + epsilon * torch.eye(d, device=device))

    if verbose:
      acc_rate = accepted_count / N
      print(f"\nAcceptance Rate: {acc_rate:.2%}")
    return chain, C_n