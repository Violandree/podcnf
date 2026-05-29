import ot
import torch
import numpy as np

def sliced_Wasser(u_replica_true, u_rec, n_projections):

    u_replica_true = u_replica_true.clone().detach().cpu().numpy().astype(np.float64)
    u_rec = u_rec.clone().detach().cpu().numpy().astype(np.float64)

    sw_dist = ot.sliced_wasserstein_distance(u_rec, u_replica_true, n_projections=n_projections)

    return sw_dist

def Wasser_dist(u_replica_true, u_rec):
    u_replica_true = u_replica_true.clone().detach().cpu().numpy().astype(np.float64)
    u_rec = u_rec.clone().detach().cpu().numpy().astype(np.float64)

    # weights for distribution
    n_samples_r = u_replica_true.shape[0]
    n_samples_gen = u_rec.shape[0]

    a = np.ones((n_samples_r,)) / n_samples_r
    b = np.ones((n_samples_gen,)) / n_samples_gen

    # Costs matrix
    # Euclidean distance
    M = ot.dist(u_replica_true, u_rec, metric='sqeuclidean')

    # EMD (W_2^2)
    w2_squared_dist = ot.emd2(a, b, M)
    w2_dist = np.sqrt(w2_squared_dist)

    return w2_dist