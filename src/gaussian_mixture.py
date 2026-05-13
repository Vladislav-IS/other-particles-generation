import torch
import torch.nn as nn


class GaussianMixtureLatent(nn.Module):
    """
    Base class for Gaussian Mixture
    """

    def __init__(self, n_modes, dim, max_len):
        super().__init__()
        self.d = dim
        self.n_modes = n_modes
        self.max_len = max_len

    def forward(self, batch_size):
        pass


class LearnableMixtureLatent(GaussianMixtureLatent):
    """
    Learnable Gaussian mixture
    """

    def __init__(self, n_modes, dim, max_len=None):
        """
        Parameters:
        - n_modes - number of Gaussian modes;
        - dim - latent dimensionality;
        - max_len - maximum number of particles in generating sequences
        """
        super().__init__(n_modes=n_modes, dim=dim, max_len=max_len)
        self.mu = nn.Parameter(torch.randn(n_modes, dim) * 0.02)
        self.log_sigma = nn.Parameter(torch.randn(n_modes, dim) * 0.02)
        self.register_buffer("weights", torch.ones(n_modes) / n_modes)

    def get_sigma(self):
        return torch.exp(torch.tanh(self.log_sigma))

    def forward(self, batch_size):
        """
        Parameters:
        - batch_size
        """
        sigma = self.get_sigma()
        comp_idx = torch.multinomial(self.weights, batch_size, replacement=True)
        if self.max_len is not None:
            z = torch.randn(batch_size, self.max_len, self.d, device=self.mu.device)
            mu_g = self.mu[comp_idx]
            sigma_g = sigma[comp_idx]
            return mu_g.unsqueeze(1) + sigma_g.unsqueeze(1) * z
        z = torch.randn(batch_size, self.d, device=self.mu.device)
        mu_g = self.mu[comp_idx]
        sigma_g = sigma[comp_idx]
        return mu_g + sigma_g * z


class StandardNormalLatent(GaussianMixtureLatent):
    """
    Standard normal distribution latent
    """

    def __init__(self, dim, max_len=None):
        """
        Parameters:
        - n_modes - number of Gaussian modes;
        - dim - latent dimensionality;
        - param_scale - for distribution parameters;
        - max_len - maximum number of particles in generating sequences
        """
        super().__init__(n_modes=1, dim=dim, max_len=max_len)
        self.register_buffer("weights", torch.ones(1))

    def forward(self, batch_size):
        """
        Parameters:
        - batch_size
        """
        if self.max_len is not None:
            z = torch.randn(
                batch_size, self.max_len, self.d, device=self.weights.device
            )
        else:
            z = torch.randn(batch_size, self.d, device=self.weights.device)
        return z
