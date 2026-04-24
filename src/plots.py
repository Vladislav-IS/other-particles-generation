import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance
from src.calculations import (
    calc_pseudorapidity,
    calc_energy,
    kl_divergence,
    w2_distance_1d,
    calc_vn_vs_pt,
    calc_pt_dist,
)
from src.config import *


def plot_losses(losses):
    """
    Function for losses plotting.
    Parameters:
    - losses - dictionary "losses type - epoch losses list"
    """
    plt.clf()
    len_ = len(losses.keys())
    fig, ax = plt.subplots(nrows=1, ncols=len_, figsize=(3 * len_, 3))
    for j, k in enumerate(losses.keys()):
        ax[j].plot([i + 1 for i in range(len(losses[k]))], losses[k])
        ax[j].set_title(k)
    plt.tight_layout()
    plt.show()


def plot_kde(real, fake, dim_names, part_type, num=None):
    """
    Fucntion for plotting and calculating distribution metric with respect of .
    Parameters:
    - real - real data;
    - fake - generated data;
    - dim_names - list of momenta components names (x, y, z);
    - part_type - particle type PDG;
    - num - number of selected particle in sequences sorted by momenta absolute value (if None, all particles in sequences are considered).
    Return:
    - kl - KL divergence calculated using KDE of real and fake data;
    - w1 - W1 distance calculated using KDE of real and fake data;
    - w2 - W2 distance calculated using KDE of real and fake data.
    """
    fig, axes = plt.subplots(2, 5, figsize=(25, 7))
    if num is not None:
        p_sq = real[..., 0] ** 2 + real[..., 1] ** 2 + real[..., 2] ** 2
        sorted_indices = torch.argsort(p_sq, dim=1, descending=True)
        kth_idx = sorted_indices[:, num]
        batch_idx = np.arange(real.shape[0])
        real = real[batch_idx, kth_idx]
        p_sq = fake[..., 0] ** 2 + fake[..., 1] ** 2 + fake[..., 2] ** 2
        sorted_indices = torch.argsort(p_sq, dim=1, descending=True)
        kth_idx = sorted_indices[:, num]
        batch_idx = np.arange(fake.shape[0])
        fake = fake[batch_idx, kth_idx]
    else:
        real = real.reshape(-1, real.shape[-1])
        fake = fake.reshape(-1, fake.shape[-1])
    real = real[~torch.all(real == 0.0, dim=1)]
    fake = fake[~torch.all(fake == 0.0, dim=1)]
    masses = torch.full_like(fake[..., -1], pion_masses[part_type])
    real_energy = real[..., -1]
    fake_energy = calc_energy(fake[..., 0], fake[..., 1], fake[..., 2], masses)
    real_eta = calc_pseudorapidity(real[..., 0], real[..., 1], real[..., 2])
    fake_eta = calc_pseudorapidity(fake[..., 0], fake[..., 1], fake[..., 2])
    for i in range(5):
        if i < 3:
            real_vals = real[:, i].numpy()
            fake_vals = fake[:, i].numpy()
            title_pre = f"{part_type} {dim_names[i]}"
        elif i == 3:
            real_vals = real_energy.numpy()
            fake_vals = fake_energy.numpy()
            title_pre = f"{part_type} energy"
        else:
            real_vals = real_eta.numpy()
            fake_vals = fake_eta.numpy()
            title_pre = f"{part_type} pseudorapidity"
        kde_real = stats.gaussian_kde(real_vals, bw_method="scott")
        kde_fake = stats.gaussian_kde(fake_vals, bw_method="scott")
        x_points = np.linspace(
            min(real_vals.min(), fake_vals.min()),
            max(real_vals.max(), fake_vals.max()),
            1000,
        )
        pdf_real = kde_real(x_points)
        pdf_fake = kde_fake(x_points)
        kl = kl_divergence(kde_real, kde_fake, x_points)
        w1 = wasserstein_distance(real_vals, fake_vals)
        w2 = w2_distance_1d(real_vals, fake_vals)
        axes[0, i].plot(x_points, pdf_real, label="Real")
        axes[0, i].plot(x_points, pdf_fake, label="Fake")
        axes[0, i].set_title(
            f'{title_pre} ({"all" if num is None else num}) KL={kl:.3f} W1={w1:.3f} W2={w2:.3f}'
        )
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        axes[1, i].hist(real_vals, bins=50, alpha=0.5, label="Real")
        axes[1, i].hist(fake_vals, bins=50, alpha=0.5, label="Fake")
        axes[1, i].set_title(f"{title_pre} histogram")
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return kl, w1, w2


def plot_scatter(real, fake):
    """
    Function for plotting particles in 3D momenta components space.
    Parameters:
    - real - real data;
    - fake - fake data.
    """
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    real_flat = real.reshape(-1, real.shape[-1])
    fake_flat = fake.reshape(-1, fake.shape[-1])
    real_flat = real_flat[~torch.all(real_flat == 0.0, dim=1)]
    fake_flat = fake_flat[~torch.all(fake_flat == 0.0, dim=1)]
    ax1.scatter(
        real_flat[:, 0], real_flat[:, 1], real_flat[:, 2], alpha=0.6, label="Real"
    )
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_zlim(-2, 2)
    ax1.set_title("Real")
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(
        fake_flat[:, 0],
        fake_flat[:, 1],
        fake_flat[:, 2],
        alpha=0.6,
        color="orange",
        label="Fake",
    )
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_zlim(-2, 2)
    ax2.set_title("Fake")
    plt.show()


def plot_v2(
    real_data,
    fake_data,
    n=2,
    pt_min=0.0,
    pt_max=2.0,
    eta_max=10.0,
    nbins=10,
    pt_bins=50,
):
    """
    Function for plotting v_n(p_T) dependency and p_T distribution.
    Parameters:
    - real_data - real momenta;
    - fake_data - fake momenta;
    - n - cosine coefficient (flow number);
    - pt_min - minimal value of transverse momentum;
    - pt_max - maximum value of transverse momentum;
    - eta_max - maximum value of pseudorapidity;
    - nbins - number of points for v_n(p_T) dependency;
    - pt_bins - number of bins for p_T distribution histogram
    """
    bin_centers, v2_real, err_real = calc_vn_vs_pt(
        real_data, pt_min, pt_max, eta_max, nbins, n
    )
    _, v2_fake, err_fake = calc_vn_vs_pt(fake_data, pt_min, pt_max, eta_max, nbins, n)
    pt_real_all = calc_pt_dist(real_data, pt_min=0, eta_max=eta_max)
    pt_fake_all = calc_pt_dist(fake_data, pt_min=0, eta_max=eta_max)
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(16, 5))
    ax[0].errorbar(
        bin_centers, v2_real, yerr=err_real, fmt="o-", label="Real", capsize=3
    )
    ax[0].errorbar(
        bin_centers, v2_fake, yerr=err_fake, fmt="s-", label="Fake", capsize=3
    )
    ax[0].set_xlabel("pT (GeV/c)")
    ax[0].set_ylabel(f"v{n}")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[0].set_title(f"v{n}(pT) with eta < {eta_max}")
    ax[1].hist(
        pt_real_all.numpy(),
        bins=pt_bins,
        density=True,
        alpha=0.5,
        label="Real",
        color="blue",
    )
    ax[1].hist(
        pt_fake_all.numpy(),
        bins=pt_bins,
        density=True,
        alpha=0.5,
        label="Fake",
        color="red",
    )
    ax[1].set_xlabel("pT, GeV/c")
    ax[1].set_ylabel("Normalized count")
    ax[1].set_title("pT distribution")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pt(real, fake, mode="pt"):
    """
    Function for plotting energy metrics (pT and energy fractions).
    Parameters:
    - real - real momenta;
    - fake - fake momenta;
    - mode - "pt" (transverse momenta) or "energy"
    """
    real_keys = [str(k) for k in real.keys()]
    real_values = [np.mean(v) for v in real.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(real_keys, real_values, label="Real", alpha=0.5)
    fake_keys = [str(k) for k in fake.keys()]
    fake_values = [np.mean(v) for v in fake.values()]
    bars = ax.bar(fake_keys, fake_values, label="Fake", alpha=0.5)
    real_values = np.array(real_values)
    fake_values = np.array(fake_values)
    ratios = [
        f"{round(val, 2)}%"
        for val in (np.abs(real_values - fake_values) / real_values * 100)
    ]
    ax.bar_label(bars, labels=ratios, padding=3)
    ax.set_xlabel("particle type")
    ax.set_ylabel("value")
    ax.set_title(
        "Transverse momentum modulus fraction" if mode == "pt" else "Energy fraction"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend()
    plt.show()
