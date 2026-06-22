# Developing a class structure to generate samples
import torch
import numpy as np

from scipy.linalg import svd
from dlroms.dnns import num2p

from podcnf.Visualization import svdplot

from dlroms import euclidean

class TorchScaler(object):
    def __init__(self, vmean, scale):
        self.vmean = vmean.detach().clone()
        self.scale = scale.detach().clone()
    def inverse_transform(self, x):
        return x*self.scale + self.vmean
    def transform(self, xtilde):
        return (xtilde-self.vmean)/self.scale

class GenerativeROM(object):
    def sample_latent_same_mu(self, muj, nrep):
        raise RuntimeError("No latent sampler defined.")

    def sample_latent(self, mu, nrep = 100):
        mutilde = mu.reshape(-1, mu.shape[-1])
        result = torch.stack([self.sample_latent_same_mu(muj, nrep) for muj in mutilde])
        return result.squeeze(0) if(len(mu.shape)==1) else result

    def sample(self, mu, nrep = 100):
        return self.decode(self.sample_latent(mu, nrep))

    def decode(self, c):
        raise RuntimeError("No decoding method specified.")

class PODcnf(GenerativeROM):
    def __init__(self, pod_matrix, cnf_model, mu_scaler = None, c_scaler = None):
        self.__V = pod_matrix
        self.__cnf = cnf_model
        self.__mu_scaler = TorchScaler(0.0, 1.0, pod_matrix.device) if mu_scaler is None else mu_scaler
        self.__c_scaler = TorchScaler(0.0, 1.0, pod_matrix.device) if c_scaler is None else c_scaler

    @property
    def V(self):
        return self.__V

    @property
    def cnf(self):
        return self.__cnf

    @property
    def mu_scaler(self):
        return self.__mu_scaler

    @property
    def c_scaler(self):
        return self.__c_scaler

    def sample_latent_same_mu(self, muj, nrep = 100):
        return self.c_scaler.inverse_transform(self.cnf.sample_same_mu(self.mu_scaler.transform(muj), nrep))

    def decode(self, c):
        return c @ self.V.T    

    @staticmethod
    def svd(u):
        """
            svd computes the POD given in input the data matrix u

            input: u Torch.tensor
            output: projection matrix V and singular values s
        """
        V, s, _ = svd(u, full_matrices = False)
        return V, s

    @staticmethod
    def svdplot(svalues, nmax = 50, logscale = True):
        svdplot(svalues, nmax = 50, logscale = True)

    @staticmethod
    def nmin(svalues, threshold = 0.9):
        energy = (svalues ** 2) / np.sum(svalues ** 2)
        cum_energy = np.cumsum(energy)
        return np.where(cum_energy < threshold)[0][-1]

    @staticmethod
    def projection_errors(u, V, relative = True, norm = euclidean):

        if relative:
            return np.linalg.norm(u - u @ V @ V.T, axis = 1)/np.linalg.norm(u, axis = 1)
        
        return np.linalg.norm(u - u @ V @ V.T, axis = 1)

        