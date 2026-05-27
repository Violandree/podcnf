import itertools
import os
import random
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from podcnf.NFmodel import NormalizingFlow

import matplotlib.pyplot as plt
from IPython.display import clear_output

def train_one_epoch(model, train_loader, optimizer, device):
    """
    Perform one complete training epoch through the entire training dataset.

    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): PyTorch DataLoader containing training data batches
        optimizer (torch.optim): Optimization algorithm (e.g., Adam, SGD)
        device (torch.device): Computing device ('cuda' for GPU, 'cpu' for CPU)

    Returns:
        tuple: (average_loss) - Training loss for this epoch
    """
    model.train()
    running_loss = 0.0

    # Iterate through training batches
    for xy_batch in train_loader:
        # Move data to device
        xy_batch = xy_batch.to(device) # Moved inside the loop

        # Forward step takes in input x and y separately
        x_batch, y_batch = xy_batch[:, :model.dim_x], xy_batch[:, model.dim_x:]

        # Clear gradient from previous step
        optimizer.zero_grad(set_to_none=True) # Instead of setting to zero, set the grads to None.
        # This will in general have lower memory footprint, and can modestly improve performance

        # Forward pass (in float32) - minimization of the negative Log-Likelihood
        loss = -model.log_prob(x_batch, y_batch).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected during training. Stopping run.")
            return float('inf')

        # Backward step
        loss.backward()
        # For gradient limit
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate metrics
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    return avg_train_loss

def validate_one_epoch(model, val_loader, device):
    """
    Perform one complete validation epoch through the entire validation dataset.

    Args:
        model (nn.Module): The neural network model to evaluate (must be in eval mode)
        val_loader (DataLoader): PyTorch DataLoader containing validation data batches
        device (torch.device): Computing device ('cuda' for GPU, 'cpu' for CPU)

    Returns:
        tuple: (average_loss) - Validation loss for this epoch

    Note:
        This function automatically sets the model to evaluation mode and disables
        gradient computation for efficiency during validation.
    """
    model.eval() # Set model to evaluate mode
    val_loss_sum = 0.0

    # Disable gradient computation for validation
    with torch.no_grad():
        for xy_batch in val_loader:
            # Move data to device
            xy_batch = xy_batch.to(device) # Moved inside the loop

            # Forward step takes in input x and y separately
            x_batch, y_batch = xy_batch[:, :model.dim_x], xy_batch[:, model.dim_x:]

            # Forward pass (in float32)
            val_loss = -model.log_prob(x_batch, y_batch).mean()

            # Check for Nan Values or INF
            if torch.isnan(val_loss) or torch.isinf(val_loss):
                print(f"Warning: NaN/Inf detected in validation loss. Model is unstable.")
                return float('inf')

            val_loss_sum += val_loss.item()

    avg_val_loss = val_loss_sum / len(val_loader)

    return avg_val_loss

def full_train(epochs, print_frequency,
               model, train_loader, val_loader,
               lr, weight_decay, patience,
               device, model_save_path):

    print(f"Training {epochs} epochs (with patience={patience}, lr={lr}, wd={weight_decay}):")

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay) # prva anche con 1e-4, 1e-5, 1e-6
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
            model, train_loader, optimizer, device
            )

        # Evaluate model on validation data without updating weights
        val_loss = validate_one_epoch(
            model, val_loader, device
            )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print progress every N epochs or on first epoch
        if epoch % print_frequency == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train: Loss={train_loss:.4f} | "
                  f"Val: Loss={val_loss:.4f}")

            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epoch + 1), train_losses, label='Train Loss', color='blue')
            plt.plot(range(1, epoch + 1), val_losses, label='Val Loss', color='orange')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Progress')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.show()

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the checkpoint of the optimal model
            torch.save(model.state_dict(), model_save_path)

            if epoch % print_frequency == 0:
                 print(f"    -> New best model saved with Val Loss: {best_val_loss:.4f}")

        else: # If there is no improvment
            epochs_no_improve += 1

        # Se la pazienza è esaurita, fermati
        if epochs_no_improve >= patience:
            print(f"\n--- Early Stopping ---")
            print(f"Validation is not improving since {patience} epochs.")
            print(f"Interrupted training {epoch}.")
            break

    # Save trained model state dict and close TensorBoard writer
    print(f"\nTraining completed. Best model saved to: '{model_save_path}'")
    print(f"Best Validation Loss achieved: {best_val_loss:.4f}")

    return train_losses, val_losses

def tuning_parameters(train_loader, val_loader, 
                      lr, num_flows, hidden_size, hidden_depth, weight_decay,
                      epochs,
                      dim_x, dim_y, device):
    """
    Input:
        - lr, num_flows, hidden_size: lists of possible value for the hyperparameters
    """
    # Test a sufficient number of parameters
    param_grid = {
        'learning_rate': lr,
        'num_flows': num_flows,
        'hidden_size': hidden_size,
        'hidden_depth': hidden_depth,
        'weight_decay': weight_decay
    }

    # All the possible combinations for the parameters
    keys, values = zip(*param_grid.items())
    hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print("\n Start Training:")

    best_val_loss = float('inf')
    best_hyperparams = None

    for i, params in enumerate(hyperparam_combinations):
        lr = params['learning_rate']
        num_flows = params['num_flows']
        hidden_size = params['hidden_size']
        hidden_depth = params['hidden_depth']
        w_d = params['weight_decay']

        # This name will appear in tensorboard to monitor it
        run_name = f"run_{i+1}_flows_{num_flows}_hidden_{hidden_size}_depth_{hidden_depth}_lr_{lr}_wd_{w_d}"
        print(f"\n--- Execution {i+1}/{len(hyperparam_combinations)}: {run_name} ---")

        # Initialization of the model
        flow = NormalizingFlow(dim_x, dim_y, num_flows, hidden_size, hidden_depth, device).to(device)
        optimizer = torch.optim.AdamW(flow.parameters(),
                                     lr=lr,
                                     weight_decay=w_d)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10) # DA VEDERE

        current_run_best_val_loss = float('inf')
        final_epoch_val_loss = float('inf') # Just for the final output

        for epoch in range(epochs):
            # Forward pass through training data, compute gradients, update weights
            train_loss = train_one_epoch(
                flow, train_loader, optimizer, device
                )

            # Interrupt the RUN if the training explode
            if train_loss == float('inf'):
                print(f"Epoch {epoch}: Training diverged. Stopping this run.")
                break

            # Evaluate model on validation data without updating weights
            val_loss = validate_one_epoch(
                flow, val_loader, device
                )

            # Interrupt the RUN if the validation explode
            if val_loss == float('inf'):
                print(f"Epoch {epoch}: Validation diverged. Stopping this run.")
                current_run_best_val_loss = float('inf') # Ensure that this Run will be never choosen
                break

            final_epoch_val_loss = val_loss
            scheduler.step(val_loss) # DA VEDERE

            # Check if this is the best val:loss
            if val_loss < current_run_best_val_loss:
                current_run_best_val_loss = val_loss

        print(f"Final epoch validation Loss: {final_epoch_val_loss:.4f}")
        print(f"Best validation Loss for this run: {current_run_best_val_loss:.4f}")

        # Confronta il MIGLIOR risultato di questo run con il MIGLIOR risultato globale
        if current_run_best_val_loss < best_val_loss:
            best_val_loss = current_run_best_val_loss
            best_hyperparams = params

    print(f"\n--- Tuning Finished ---")
    print(f"Best Global Validation Loss: {best_val_loss:.4f}")
    print("Best Hyperparameters:")
    print(best_hyperparams)

    return best_hyperparams