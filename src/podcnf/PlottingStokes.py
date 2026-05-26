import math

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import pearsonr
from scipy.linalg import svd
from tqdm import tqdm

import dolfin as fe
from podcnf.NFmodel import NormalizingFlow

def plot_stokes_solution(indices, data, Vh):

    n_plots = len(indices)
    cols = math.ceil(math.sqrt(n_plots))
    rows = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))

    if n_plots > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for i, index in enumerate(indices):
        ax = axes_flat[i]
        plt.sca(ax)

        u_tensor = data['u'][index].detach().cpu().numpy()
        mu_tensor = data['mu'][index].detach().cpu().numpy()
        eps_val = data['eps'][index].item()
        theta_val = data['theta'][index].item()

        c1, c2, c3 = mu_tensor[0], mu_tensor[1], mu_tensor[2]

        u_func = fe.Function(Vh)
        u_func.vector()[:] = u_tensor

        c = fe.plot(u_func, cmap='jet')
        plt.colorbar(c, ax=ax, shrink = 0.5, label='Concentration (u)')

        title_str = (
            f"$\\epsilon$ = {eps_val:.5f}, $\\theta$ = {theta_val:.3f} rad\n"
            f"$\\mu$ = [{c1:.2f}, {c2:.2f}, {c3:.2f}]"
        )
        plt.title(title_str, fontsize=14)
        plt.xlabel('x')
        plt.ylabel('y')

    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0.3)
    plt.show()

def analyze_stokes_residuals(u, V, mu):

    if isinstance(u, torch.Tensor): u = u.detach().cpu().numpy()
    if isinstance(V, torch.Tensor): V = V.detach().cpu().numpy()
    if isinstance(mu, torch.Tensor): mu = mu.detach().cpu().numpy()

    # u ~ c @ V.T
    c = np.dot(u, V)
    u_rec = np.dot(c, V.T)

    residuals = u - u_rec
    u_norms = np.linalg.norm(u, axis=1)
    error_norms = np.linalg.norm(residuals, axis=1)
    relative_errors = np.divide(error_norms, u_norms, out=np.zeros_like(error_norms), where=u_norms!=0)

    print(f"Mean Relative Error: {np.mean(relative_errors):.4%}")
    print(f"Max Relative Error:  {np.max(relative_errors):.4%}")

    params_dict = {
        'c1': mu[:, 0],
        'c2': mu[:, 1],
        'c3': mu[:, 2]
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    print("-" * 30)
    print("Correlation on known parameters:")

    for i, (name, values) in enumerate(params_dict.items()):
        corr, _ = pearsonr(values, relative_errors)
        print(f"{name}: {corr:.4f}")
        ax = axes[i]
        sc = ax.scatter(values, relative_errors, alpha=0.6, c=relative_errors, cmap='viridis')
        ax.set_title(f'Error vs {name}\nCorr: {corr:.2f}')
        ax.set_xlabel(name)
        if i == 0:
            ax.set_ylabel('L2 Relative Error')

        m, q = np.polyfit(values, relative_errors, 1)
        ax.plot(values, m*values + q, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    mean_abs_residual_field = np.mean(np.abs(residuals), axis=0)
    return mean_abs_residual_field

def analyze_bases_variation_stokes(u, mu, n_bases_list=range(1, 41)):

    if isinstance(u, torch.Tensor): u = u.detach().cpu().numpy()
    if isinstance(mu, torch.Tensor): mu = mu.detach().cpu().numpy()

    n_samples = u.shape[0]
    ntrain = int(n_samples * 0.75)
    nval = int(ntrain + n_samples * 0.2)
    nval = min(nval, n_samples)

    u_train = u[:ntrain]
    u_val = u[ntrain:nval]
    mu_val = mu[ntrain:nval]

    spatial_modes, s, _ = svd(u_train.T, full_matrices=False)

    s_energy = (s**2) / np.sum(s**2)
    cum_energy = np.cumsum(s_energy)

    mean_errors = []
    max_errors = []
    corrs = {'c1': [], 'c2': [], 'c3': []}

    for k in n_bases_list:
        V_k = spatial_modes[:, :k]
        coeffs = np.dot(u_val, V_k)
        u_rec = np.dot(coeffs, V_k.T)

        residuals = u_val - u_rec
        u_norms = np.linalg.norm(u_val, axis=1)
        err_norms = np.linalg.norm(residuals, axis=1)

        rel_errors = np.divide(err_norms, u_norms, out=np.zeros_like(err_norms), where=u_norms!=0)

        mean_errors.append(np.mean(rel_errors))
        max_errors.append(np.max(rel_errors))

        if mu_val.shape[1] >= 3:
            corrs['c1'].append(pearsonr(mu_val[:, 0], rel_errors)[0])
            corrs['c2'].append(pearsonr(mu_val[:, 1], rel_errors)[0])
            corrs['c3'].append(pearsonr(mu_val[:, 2], rel_errors)[0])

    fig = plt.figure(figsize=(18, 12))

    gs = gridspec.GridSpec(2, 4, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0:2])

    l1, = ax1.plot(n_bases_list, np.array(mean_errors)*100, 'b-o', label='Mean Relative Error (%)')
    ax1.set_ylabel('Mean Error (%)', color='b', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xlabel('Number of Bases (N)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_title('Reconstruction Error vs Number of Bases', fontsize=16)

    ax1_twin = ax1.twinx()
    energies_in_range = [cum_energy[k-1] for k in n_bases_list]
    l2, = ax1_twin.plot(n_bases_list, energies_in_range, 'g--', label='Cumulative Energy', linewidth=2)
    ax1_twin.set_ylabel('Cumulative Energy', color='g', fontsize=14)
    ax1_twin.tick_params(axis='y', labelcolor='g')

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    ax2 = fig.add_subplot(gs[0, 2:4])

    ax2.plot(n_bases_list, np.array(max_errors)*100, 'r-s', label='Max Relative Error (%)')
    ax2.set_ylabel('Max Error (%)', fontsize=14)
    ax2.set_xlabel('Number of Bases (N)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_title('Worst-Case Error vs Number of Bases', fontsize=16)
    ax2.legend(fontsize=12)

    ax3 = fig.add_subplot(gs[1, 1:3])

    styles = {
        'c1': {'color': 'purple', 'marker': 'o'},
        'c2': {'color': 'orange', 'marker': 'x'},
        'c3': {'color': 'teal', 'marker': 's'}
    }

    for param, values in corrs.items():
        if len(values) > 0:
            style = styles.get(param, {'color': 'black', 'marker': '.'})
            ax3.plot(n_bases_list, values, label=f'Corr(Err, {param})',
                     marker=style['marker'], color=style['color'], linewidth=2)

    ax3.set_ylabel('Pearson Correlation', fontsize=14)
    ax3.set_xlabel('Number of Bases (N)', fontsize=14)
    ax3.axhline(0, color='black', linewidth=1.5, alpha=0.7)
    ax3.legend(fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.set_title('Error Correlation vs Control Parameters (c1, c2, c3)', fontsize=16)

    plt.tight_layout()
    plt.show()

def plot_conditional_same_mu_stokes(n_generations,
                                    u_true, u_rec,
                                    Vh,
                                    mu_values=None):

    plt.figure(figsize=(9, 6))

    u_func_true = fe.Function(Vh)
    if isinstance(u_true, torch.Tensor):
        u_func_true.vector()[:] = u_true.detach().cpu().numpy()
    else:
        u_func_true.vector()[:] = u_true

    # True solution
    c = fe.plot(u_func_true, cmap='jet')
    plt.colorbar(c, shrink = 0.8, label='Concentration (u)')

    title_str = "True solution"
    if mu_values is not None:
        if isinstance(mu_values, torch.Tensor):
            mv = mu_values.detach().cpu().numpy().flatten()
        else:
            mv = np.array(mu_values).flatten()
        title_str += f"\n($c_1$={mv[0]:.2f}, $c_2$={mv[1]:.2f}, $c_3$={mv[2]:.2f})"

    plt.title(title_str)
    plt.show()

    print(f"{n_generations} generated samples:")

    cols = int(math.ceil(math.sqrt(n_generations)))
    rows = int(math.ceil(n_generations / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    if n_generations > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Generate sample
    for i in range(n_generations):
        ax = axes[i]
        plt.sca(ax)

        u_func_rec = fe.Function(Vh)

        sample_data = u_rec[i]
        if isinstance(sample_data, torch.Tensor):
            sample_data = sample_data.detach().cpu().numpy()

        u_func_rec.vector()[:] = sample_data

        c = fe.plot(u_func_rec, cmap='jet')
        plt.colorbar(c, shrink = 0.65, label='Concentration (u)')

        ax.set_title(f"Sample {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n_generations, len(axes)):
        axes[i].axis('off')

    mu_str = f"\n($c_1$={mv[0]:.2f}, $c_2$={mv[1]:.2f}, $c_3$={mv[2]:.2f})"
    fig.suptitle(f"Conditional Generated Samples (Same $\\mu$) \n{mu_str}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def Likelihood_Comparison_Stokes(flow, u, mu, V,
                                 mu_scaler, c_scaler, device,
                                 n_val, n_samples, ref_idx=None):

    if isinstance(mu, torch.Tensor):
        mu0_raw = mu[ref_idx].cpu().numpy().reshape(1, -1)
    else:
        mu0_raw = mu[ref_idx].reshape(1, -1)

    mu0_scaled = mu_scaler.transform(mu0_raw)
    mu0_tensor = torch.tensor(mu0_scaled, dtype=torch.float32).to(device)

    # Calcolo esplicito della Likelihood del Target (u0 | mu0)
    if isinstance(u, torch.Tensor):
        u0 = u[ref_idx].cpu().numpy()
    else:
        u0 = u[ref_idx]

    if isinstance(V, torch.Tensor):
        V_np = V.cpu().numpy()
    else:
        V_np = V

    # Proiezione PCA e Scaling per u0
    c0 = u0 @ V_np
    c0_scaled = c_scaler.transform(c0.reshape(1, -1))
    c0_tensor = torch.tensor(c0_scaled, dtype=torch.float32).to(device)

    flow.eval()
    with torch.no_grad():
        target_like = flow.log_prob(mu0_tensor, c0_tensor).item()

    test_indices = list(range(n_val, n_samples))
    all_likelihoods = []
    all_distances = []

    print(f"Comparing Log-Likelihoods against ref_idx {ref_idx}...")

    with torch.no_grad():
        for idx in tqdm(test_indices, desc="Execution"):
            if isinstance(u, torch.Tensor):
                u_k = u[idx].cpu().numpy()
                mu_k = mu[idx].cpu().numpy()
            else:
                u_k = u[idx]
                mu_k = mu[idx]

            dist_mu = np.linalg.norm(mu0_raw - mu_k.reshape(1, -1))

            # Proiezione u_k -> c_k
            c_k = u_k @ V_np
            c_k = c_k.reshape(1, -1)
            c_k_scaled = c_scaler.transform(c_k)
            c_k_tensor = torch.tensor(c_k_scaled, dtype=torch.float32).to(device)

            # Calcolo Log-Prob condizionata sempre su mu0_tensor!
            log_prob = flow.log_prob(mu0_tensor, c_k_tensor)

            all_likelihoods.append(log_prob.item())
            all_distances.append(dist_mu)

    arr_dist = np.array(all_distances)
    arr_like = np.array(all_likelihoods)
    min_like = np.min(arr_like)

    plt.figure(figsize=(15, 7))

    sc = plt.scatter(arr_dist, arr_like, c=arr_like, cmap='viridis', alpha=0.6, s=30)

    plt.scatter([0], [target_like], c='red', s=200, label='Target $(\mu_0, y_0)$', edgecolors='black', zorder=10)
    plt.axhline(y=target_like, color='red', linestyle='--', alpha=0.4)

    plt.yscale('symlog', linthresh=1000)

    top_lim = max(200, target_like + 50)
    bottom_lim = min_like * 1.1 if min_like < 0 else min_like * 0.9
    plt.ylim(bottom=bottom_lim, top=top_lim)

    ticks_candidates = [100, 0, -1000, -1e4, -1e6, -1e8, -1e10, -1e12, -1e13]
    ticks_visible = [t for t in ticks_candidates if t >= bottom_lim]
    if min_like < -1e13:
        ticks_visible.append(min_like)

    plt.yticks(ticks_visible)

    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)

    plt.xlabel(r'$||\mu_k - \mu_0||_2$', fontsize=15)
    plt.ylabel(r'Log-Likelihood $\log p(y_k | \mu_0)$', fontsize=15)
    plt.title(f'Global Robustness Analysis\nTarget Likelihood: {target_like:.2f}', fontsize=16)

    cbar = plt.colorbar(sc, label='Log-Likelihood', shrink=0.7)
    cbar.ax.tick_params(labelsize=12)

    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.legend(fontsize=14)

    plt.tight_layout()
    plt.show()

    return arr_dist, arr_like


def plot_marginal_conditional_density_stokes(u_replica_true, u_rec, indices):

    if isinstance(u_replica_true, torch.Tensor):
        u_replica_true = u_replica_true.detach().cpu().numpy()

    if isinstance(u_rec, torch.Tensor):
        u_rec = u_rec.detach().cpu().numpy()

    n_plots = len(indices)
    cols = min(n_plots, 3)
    rows = math.ceil(n_plots / cols) if n_plots > 0 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    if n_plots == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()

    for k, node_idx in enumerate(indices):
        ax = axes_flat[k]

        data_true = u_replica_true[:, node_idx]
        data_rec = u_rec[:, node_idx]

        sns.histplot(data_true, ax=ax, color='blue', stat='density', kde=True,
                     label='True Physical Dist.', element="step", alpha=0.3)

        sns.histplot(data_rec, ax=ax, color='red', stat='density', kde=True,
                     label='Generated (NF)', element="step", alpha=0.3)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

        if k == 0:
            ax.legend()

    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.show()