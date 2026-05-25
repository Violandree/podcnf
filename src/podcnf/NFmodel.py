import torch
from torch import nn
from torch.distributions import Normal
import numpy as np
from typing import List, Tuple, Optional

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CouplingLayer(nn.Module):

    """
    Single level of Conditional Coupling Layer. It divides the latent space z in two halves.
    The first one will remain the same while the second one is transformed with a linear function that
    depends on the first half and on the input x, the conditioning parameter.

    Parameters
    ----------

    """

    def __init__(self, dim_x: int, dim_y: int, hidden_size: int = 256, hidden_depth: int = 1) -> None:
        super().__init__()

        input_dim = dim_y // 2 + dim_x
        output_dim = ((dim_y + 1) // 2) * 2

        layers = [
            nn.Linear(input_dim, hidden_size),
            nn.ReLU()
        ]

        for _ in range(hidden_depth):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ])

        layers.append(nn.Linear(hidden_size, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor, ldj: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward and Inverse with the coupling layer
        """
        id, z2 = z.chunk(2, dim=1)
        xid = torch.cat([x, id], dim=1)

        log_s_raw, b = self.network(xid).chunk(2, dim=1)
        log_scale_stable = torch.tanh(log_s_raw)
        scale = torch.exp(log_scale_stable)

        if not reverse:
            # Forward transformation (Z0 -> Zk)
            z2 = z2 * scale + b
            ldj += log_scale_stable.sum(dim=[1])
        else:
            # Inverse transformation (Zk -> Z0)
            z2 = (z2 - b) / scale
            ldj -= log_scale_stable.sum(dim=[1])

        z = torch.cat([id, z2], dim=1)
        return z, ldj

class NormalizingFlow(nn.Module):
    """
    Complete model for Conditional Normalizing Flow.
    It manage the seuquence of Coupling Layers with random permutations.

    Parameters
    ----------
    
    """
    def __init__(self, dim_x: int, dim_y: int, num_flows: int = 8, hidden_size: int = 256, hidden_depth: int = 1, device: torch.device = DEFAULT_DEVICE) -> None:
        super().__init__()

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.device = device

        self.flows = nn.ModuleList([
            CouplingLayer(dim_x, dim_y, hidden_size, hidden_depth) for _ in range(num_flows)
        ])

        self.base_dist = Normal(
            loc=torch.zeros(dim_y, device=device),
            scale=torch.ones(dim_y, device=device)
        )
        
        permutations = []
        for _ in range(num_flows):
            # Random permutation of dimension dim_y
            p = torch.randperm(self.dim_y, device=device)
            permutations.append(p)

        self.register_buffer('permutations', torch.stack(permutations))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.log_prob(x, y)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the conditional log-likelihood log p(y|x)
        """
        z, ldj = y, torch.zeros(x.shape[0], device=x.device)
        # Forward transformation -> reverse=True
        for i, flow in enumerate(self.flows):
            # Invert the order of the features to be able to transform all the dimensions
            z = z[:, self.permutations[i]]
            z, ldj = flow(x, z, ldj, reverse=True) # reverse=True per x -> z

        log_pz = self.base_dist.log_prob(z).sum(dim=1)
        log_px = log_pz + ldj
        return log_px

    # In the sample function we could pass also just one single value, the fact
    # is that each time we try to evaluate in x0 we obtain a different value due
    # to z that is randomly chosen each time
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        # Sample from the base distribution
        # 'x' acts as your \mu. Here, it is a vector with identical values so the model knows 
        # how many samples to produce. Subsequently, you generate a number of samples from  
        # the base distribution equal to the size of the vector 'x'.
        z = self.base_dist.sample((x.shape[0],)) # x.shape[0] is the number of points to be evaluated
        # Forward transformation -> reverse=False
        with torch.no_grad():
            ldj = torch.zeros(x.shape[0], device=z.device)
            for i, flow in reversed(list(enumerate(self.flows))):
                z, ldj = flow(x, z, ldj, reverse=False)
                # Inverse permutation
                inv_p = torch.argsort(self.permutations[i])
                z = z[:, inv_p]
        return z

    def sample_trajectory(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Sample from the base distribution
        z = self.base_dist.sample((x.shape[0],)) # x.shape[0] is the number of points to be evaluated
        # Initialization of the trajectory
        trajectory = [z.clone()]
        # Forward transformation -> reverse=False
        with torch.no_grad():
            ldj = torch.zeros(x.shape[0], device=z.device)
            for i, flow in reversed(list(enumerate(self.flows))):
                z, ldj = flow(x, z, ldj, reverse=False)
                # Inverse permutation
                inv_p = torch.argsort(self.permutations[i])
                z = z[:, inv_p]
                # Save the current state in the trajectory
                trajectory.append(z.clone())
                
        return trajectory

def sample_same_mu(model: NormalizingFlow,
                   mu, V,
                   device,
                   norm_scaler, mu_scaler, c_scaler,
                   n_generations):

    model.eval()

    mu_selected = torch.tensor(mu.reshape(1,-1), dtype=torch.float32).to(device)

    if norm_scaler != None:
        mu_selected = torch.tensor(mu_scaler.transform(mu_selected.cpu().reshape(1,-1)), dtype=torch.float32).to(device)

    # Generates samples starting give the value of mu_test
    with torch.no_grad():
        mu_repeated = mu_selected.repeat(n_generations, 1) # n_generations

        # Sample from the model
        c_samples = model.sample(mu_repeated)

        # Get back to the values non-normalized
        if norm_scaler != None:
            c_samples = c_scaler.inverse_transform(c_samples.cpu().numpy())

        # c = V.T * u  =>  u = c * V.T
        # Reconstruct the solution
        if norm_scaler != None:
            V_torch = torch.tensor(V, dtype=torch.float32)
        else:
            V_torch = torch.tensor(V, dtype=torch.float32).to(device) # without scaling put .to(device)

        u_reconstructed = torch.tensor(c_samples, dtype=torch.float32) @ V_torch.T

    return u_reconstructed