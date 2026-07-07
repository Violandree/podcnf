import itertools
import torch
from torch import nn
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

from IPython.display import clear_output
from podcnf.Training import train_one_epoch, validate_one_epoch

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
    Complete self for Conditional Normalizing Flow.
    It manage the seuquence of Coupling Layers with random permutations.

    Parameters
    ----------
    
    """
    def __init__(self, dim_x: int, dim_y: int, num_flows: int = 8, hidden_size: int = 256, hidden_depth: int = 1, device: torch.device = DEFAULT_DEVICE) -> None:
        super().__init__()

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.device = device
        self.hidden_size = hidden_size
        self.hidden_depth = hidden_depth
        self.num_flows = num_flows

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

    def log_likelihoods(self, x, y):
    
        self.eval()
        with torch.no_grad():
            loglikelihoods = [
                self.log_prob(x_i.unsqueeze(0), y_i.unsqueeze(0)) for x_i, y_i in zip(x,y)
            ]

        return torch.tensor(loglikelihoods)

    # In the sample function we could pass also just one single value, the fact
    # is that each time we try to evaluate in x0 we obtain a different value due
    # to z that is randomly chosen each time
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        # Sample from the base distribution
        # 'x' acts as your \mu. Here, it is a vector with identical values so the self knows 
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

    def sample_same_mu(self, muj, nrep):

        self.eval()
        mu_selected = muj.reshape(1,-1)

        # Generates samples starting give the value of mu_test
        with torch.no_grad():
            mu_repeated = mu_selected.repeat(nrep, 1) # n repetitions

            # Sample from the self
            c_samples = self.sample(mu_repeated)

        return c_samples

    def train_flow(self, epochs, print_frequency, train_data, val_data, lr, weight_decay, patience, model_save_path, show_plot=False, tuning = False):

        device = next(self.parameters()).device

        print(f"Training {epochs} epochs:")

        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=patience//2) # DA VEDERE

        # Initialization early stopping
        best_val_loss = float('inf')
        epochs_no_improve = 0 # to check the patience

        train_losses = []
        val_losses = []

        # Main training loop: iterate through epochs
        for epoch in range(1, epochs + 1):

            # Forward pass through training data, compute gradients, update weights
            train_loss = train_one_epoch(
                self, train_data, optimizer, device
                )

            if train_loss == float('inf'):
                print(f"Epoch {epoch}: Training diverged. Stopping this run.")
                break

            # Evaluate self on validation data without updating weights
            val_loss = validate_one_epoch(
                self, val_data, device
                )

            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Print progress every N epochs or on first epoch
            if epoch % print_frequency == 0 or epoch == 1:

                plt.figure(figsize=(10, 5))
                plt.plot(range(1, epoch + 1), train_losses, label='Train Loss', color='blue')
                plt.plot(range(1, epoch + 1), val_losses, label='Val Loss', color='orange')
                # plt.semilogy(range(1, epoch + 1), train_losses, label='Train Loss', color='blue')
                # plt.semilogy(range(1, epoch + 1), val_losses, label='Val Loss', color='orange')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss Progress')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)

                if show_plot:
                    clear_output(wait=True)
                    plt.show()
                else:
                    plt.close()

                print(f"Epoch {epoch:3d}/{epochs} | "
                    f"Train: Loss={train_loss:.4f} | "
                    f"Val: Loss={val_loss:.4f}")

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0

                if model_save_path is not None:
                    torch.save(self.state_dict(), model_save_path)

                if epoch % print_frequency == 0:
                    print(f"    -> New best self saved with Val Loss: {best_val_loss:.4f}")

            else: # If there is no improvment
                epochs_no_improve += 1

            # Se la pazienza è esaurita, fermati
            if epochs_no_improve >= patience:
                print(f"\n--- Early Stopping ---")
                print(f"Validation is not improving since {patience} epochs.")
                print(f"Interrupted training {epoch}.")
                break

        if model_save_path is not None:
            print(f"\nTraining completed. Best self saved to: '{model_save_path}'")

        print(f"Best Validation Loss achieved: {best_val_loss:.4f}")

        if tuning == True:
            return best_val_loss

        return train_losses, val_losses

    @staticmethod
    def tune_flow(train_data, val_data,
                  lr, num_flows, hidden_size, hidden_depth, weight_decay,
                  epochs, print_frequency, patience,
                  dim_x, dim_y):

        """
        Input:
            - lr, num_flows, hidden_size: lists of possible value for the hyperparameters
        """

        device = train_data.dataset.device

        # Test a sufficient number of parameters
        param_grid = {
            'learning_rate': lr,
            'num_flows': num_flows,
            'hidden_size': hidden_size,
            'hidden_depth': hidden_depth,
            'weight_decay': weight_decay
        }

        keys, values = zip(*param_grid.items())
        hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        print("\n Start Tuning:")

        best_val_loss = float('inf')
        best_hyperparams = None

        for i, params in enumerate(hyperparam_combinations):
            lr = params['learning_rate']
            num_flows = params['num_flows']
            hidden_size = params['hidden_size']
            hidden_depth = params['hidden_depth']
            w_d = params['weight_decay']

            run_name = f"run_{i+1}_flows_{num_flows}_hidden_{hidden_size}_depth_{hidden_depth}_lr_{lr}_wd_{w_d}"
            print(f"\n--- Execution {i+1}/{len(hyperparam_combinations)}: {run_name} ---")

            # Initialization of the model
            flow = NormalizingFlow(dim_x, dim_y, num_flows, hidden_size, hidden_depth, device).to(device)
        
            val_loss = flow.train_flow(epochs, print_frequency, train_data, val_data, lr, w_d, patience, None, show_plot=False, tuning=True)

            print(f"Best validation Loss for this run: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_hyperparams = params

        print(f"\n--- Tuning Finished ---")
        print(f"Best Global Validation Loss: {best_val_loss:.4f}")
        print("Best Hyperparameters:")
        print(best_hyperparams)

        return best_hyperparams

