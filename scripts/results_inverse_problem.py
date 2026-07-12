import os
import pickle
import json
import gdown
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter
import random
import torch

from podcnf.Utils import Wasser_dist

def plot_trace(full_chain, mu_true, model_name, test_idx, results_dir):
    # full_chain = torch.cat([chain_exp, chain_ref])
    split_point = 10000

    # Burn-in e step
    if model_name == "NF":
        clean_samples = full_chain[12000:]
        step = 200
    else:
        clean_samples = full_chain[11000:]
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

def plot_posterior(clean_samples, mu_true, model_name, test_idx, results_dir, xlim, ylim):
    if isinstance(clean_samples, torch.Tensor):
        clean_samples = clean_samples.cpu().numpy()

    fig, ax_post = plt.subplots(figsize=(7, 5))
    
    step = 200 if model_name == "NF" else 20

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
    
    ax_post.set_xlim(xlim)
    ax_post.set_ylim(ylim)
    
    ax_post.legend()
    ax_post.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'posterior_{model_name}_idx_{test_idx}.png'))
    plt.close(fig)



links = {
    165: {
        'ROM': "",
        'FOM': ""
    },
    000: {
        'ROM': "",
        'FOM': ""
    }
}

def main():
    test_data_file = 'test_data.pt'
    data_download_path = gdown.download(id="1eT8re3AeZaQdIen4R6iLkYcpk2Ynj2x9", quiet=True, output=test_data_file)
    test_data = torch.load(data_download_path, weights_only=True)

    mu = test_data['mu_test']

    test_idx = 29
    mu_true_phys = mu[test_idx].cpu().numpy()

    results_dir = os.path.join("inverse_results")
    os.makedirs(results_dir, exist_ok=True)

    ROM_chain_name = "inverse_results/chain_NF_idx_%d.npy" % test_idx
    gdown.download(id=links[test_idx]['ROM'], quiet=True, output=ROM_chain_name)
    full_chain = np.load(ROM_chain_name)

    FOM_chain_name = "inverse_results/chain_FOM_idx_%d.npy" % test_idx
    gdown.download(id=links[test_idx]['FOM'], quiet=True, output=FOM_chain_name)
    full_chain_FOM = np.load(FOM_chain_name)

    # Results
    clean_samples_NF = full_chain[2000:]
    clean_samples_FOM = full_chain_FOM[1000:]

    nf_np = clean_samples_NF.cpu().numpy() if isinstance(clean_samples_NF, torch.Tensor) else clean_samples_NF
    fom_np = clean_samples_FOM.cpu().numpy() if isinstance(clean_samples_FOM, torch.Tensor) else clean_samples_FOM
    
    x_min = min(nf_np[:, 0].min(), fom_np[:, 0].min())
    x_max = max(nf_np[:, 0].max(), fom_np[:, 0].max())
    y_min = min(nf_np[:, 1].min(), fom_np[:, 1].min())
    y_max = max(nf_np[:, 1].max(), fom_np[:, 1].max())
    
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    xlim = (x_min - x_margin, x_max + x_margin)
    ylim = (y_min - y_margin, y_max + y_margin)

    # PLOT
    plot_trace(full_chain, mu_true_phys, "NF", test_idx, results_dir)
    plot_trace(full_chain_FOM, mu_true_phys, "FOM", test_idx, results_dir)

    plot_posterior(clean_samples_NF, mu_true_phys, "NF", test_idx, results_dir, xlim, ylim)
    plot_posterior(clean_samples_FOM, mu_true_phys, "FOM", test_idx, results_dir, xlim, ylim)
    
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
    
    results_dict = {
        'test_idx': int(test_idx),
        'true_mass': float(mu_true_phys[0]),
        'true_delta': float(mu_true_phys[1]),
        'NF': {
            'mean_mass': float(mean_mass_NF),
            'std_mass': float(std_mass_NF),
            'rel_error_mass': float(err_mass_NF),
            'mean_delta': float(mean_delta_NF),
            'std_delta': float(std_delta_NF),
            'rel_error_delta': float(err_delta_NF)
        },
        'FOM': {
            'mean_mass': float(mean_mass_FOM),
            'std_mass': float(std_mass_FOM),
            'rel_error_mass': float(err_mass_FOM),
            'mean_delta': float(mean_delta_FOM),
            'std_delta': float(std_delta_FOM),
            'rel_error_delta': float(err_delta_FOM)
        },
        'Wasserstein_dist': float(w2_dist)
    }
    
    with open(os.path.join(results_dir, f'results_idx_{test_idx}.json'), 'w') as f:
        json.dump(results_dict, f, indent=4)
        
    print(f"Results for index {test_idx} saved successfully!\n")

if __name__ == "__main__":
    main()