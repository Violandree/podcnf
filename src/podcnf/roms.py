# Developing a class structure to generate samples
import torch

class TorchScaler(object):
    def __init__(self, vmin, scale, device):
        self.vmin = torch.tensor(vmin, dtype=torch.float32).to(device)
        self.scale = torch.tensor(scale, dtype=torch.float32).to(device)
    def inverse_transform(self, x):
        return x*self.scale + self.vmin
    def transform(self, xtilde):
        return (xtilde-self.vmin)/self.scale

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