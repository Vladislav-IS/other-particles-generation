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
    Plotting losses
    Parameters:
    - losses - dictionary {"loss type" : "value"}
    """
    len_ = len(losses.keys())
    fig, ax = plt.subplots(nrows=1, ncols=len_, figsize=(3 * len_, 3))
    for j, k in enumerate(losses.keys()):
        ax[j].plot([i + 1 for i in range(len(losses[k]))], losses[k])
        ax[j].set_title(k)
    plt.tight_layout()
    plt.show()


def plot_kde(real, fake, dim_names, part_type, num=None, show_plots=True):
    """
    Calculating and plotting distribution metrics (KL, W1, W2)
    Parameters:
    - real - real data;
    - fake - fake data;
    - dir_names - px, py, pz componenta names for plotting;
    - part_type - type of target particles;
    - num - number of selecting particles in sorted array (if None, metrics for all particles are calculated);
    - show_plots - flag if plots will be shown
    Return:
    - metrics - metrics dictionary
    """
    fig, axes = plt.subplots(2, 5, figsize=(25, 9))
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
    metrics = {}
    for i in range(5):
        if i < 3:
            real_vals = real[:, i].numpy()
            fake_vals = fake[:, i].numpy()
            title_pre = f"{part_names[part_type]} ({dim_names[i]},"
            axes[0, i].set_xlabel(f"$p_{dim_names[i]}$")
            axes[1, i].set_xlabel(f"$p_{dim_names[i]}$")
        elif i == 3:
            real_vals = real_energy.numpy()
            fake_vals = fake_energy.numpy()
            title_pre = f"{part_names[part_type]} (" + r"$E$" + ", "
            axes[0, i].set_xlabel(r"$E$")
            axes[1, i].set_xlabel(r"$E$")
        else:
            real_vals = real_eta.numpy()
            fake_vals = fake_eta.numpy()
            title_pre = f"{part_names[part_type]} (" + r"$\eta$" + ", "
            axes[0, i].set_xlabel(r"$\eta$")
            axes[1, i].set_xlabel(r"$\eta$")
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
        if i == 3:
            metrics["e_kl"] = kl
            metrics["e_w1"] = w1
        elif i > 3:
            metrics["eta_kl"] = kl
            metrics["eta_w1"] = w1
        axes[0, i].plot(x_points, pdf_real, label="UrQMD", color="tab:blue")
        axes[0, i].plot(x_points, pdf_fake, label="GAN", color="tab:red")
        axes[0, i].set_title(
            f'{title_pre} {"все частицы" if num is None else r"$p_{max}$" if num == 0 else num})\nKL={kl:.3f} W1={w1:.3f} W2={w2:.3f}'
        )
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        axes[1, i].hist(real_vals, bins=50, alpha=0.5, color="tab:blue", label="UrQMD")
        axes[1, i].hist(
            fake_vals, bins=50, histtype="step", color="tab:red", label="GAN"
        )
        axes[1, i].set_title(
            f'{title_pre} {"все частицы" if num is None else r"$p_{max}$" if num == 0 else num})'
        )
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    if show_plots:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()
    return metrics


def plot_scatter(real, fake):
    """
    Plotting particles points clouds 3D space of momenta components
    Parameters:
    - real - real momenta;
    - fake - fake momenta
    """
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    real_flat = real.reshape(-1, real.shape[-1])
    fake_flat = fake.reshape(-1, fake.shape[-1])
    real_flat = real_flat[~torch.all(real_flat == 0.0, dim=1)]
    fake_flat = fake_flat[~torch.all(fake_flat == 0.0, dim=1)]
    ax1.scatter(
        real_flat[:, 0],
        real_flat[:, 1],
        real_flat[:, 2],
        alpha=0.6,
    )
    ax1.set_xlim(-2, 2)
    ax1.set_xlabel(r"$p_x$", fontsize=14, labelpad=10)
    ax1.set_ylim(-2, 2)
    ax1.set_ylabel(r"$p_y$", fontsize=14, labelpad=10)
    ax1.set_zlim(-2, 2)
    ax1.set_zlabel(r"$p_z$", fontsize=14, labelpad=10)
    ax1.set_title("UrQMD")
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(
        fake_flat[:, 0],
        fake_flat[:, 1],
        fake_flat[:, 2],
        alpha=0.6,
        color="orange",
    )
    ax2.set_xlim(-2, 2)
    ax2.set_xlabel(r"$p_x$", fontsize=14, labelpad=10)
    ax2.set_ylim(-2, 2)
    ax2.set_ylabel(r"$p_y$", fontsize=14, labelpad=10)
    ax2.set_zlim(-2, 2)
    ax2.set_zlabel(r"$p_z$", fontsize=14, labelpad=10)
    ax2.set_title("GAN")
    plt.show()


def plot_vn(
    real_data,
    fake_data,
    pt_min=0.0,
    pt_max=1.2,
    eta_max=10.0,
    nbins=10,
    pt_bins=50,
    min_particles=15,
    n_interp=200,
):
    """
    Plotting vn(pt) and transverse momenta distibution
    Parameters:
    - real_data;
    - fake_data;
    - pt_min - minimal value of transverse momenta;
    - pt_max - maximum value of transverse momenta;
    - eta_max - maximum value of preusorapidity;
    - nbins - number of vn(pt) bins in actual data;
    - n - order of flow vn;
    - min_particles - minimal count of particles for calculating mean and std;
    - n_interp - number of interpolating points
    Return:
    - metrics - metrics dictionary
    """
    metrics = {}
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 10))
    harmonics = [1, 2, 3]
    for idx, n in enumerate(harmonics):
        row = int(idx == 2)
        col = idx % 2
        real_p, real_vn, real_err, real_is_interp = calc_vn_vs_pt(
            real_data,
            pt_min,
            pt_max,
            eta_max,
            nbins,
            n,
            min_particles=min_particles,
            n_interp=n_interp,
        )
        fake_p, fake_vn, fake_err, fake_is_interp = calc_vn_vs_pt(
            fake_data,
            pt_min,
            pt_max,
            eta_max,
            nbins,
            n,
            min_particles=min_particles,
            n_interp=n_interp,
        )
        metrics[f"v{n}"] = np.nanmean((real_vn - fake_vn) ** 2)
        mask_real_meas = ~real_is_interp
        ax[row][col].plot(
            real_p[mask_real_meas],
            real_vn[mask_real_meas],
            color="C0",
            linewidth=2,
            label="UrQMD (измер.)",
        )
        ax[row][col].fill_between(
            real_p, real_vn - real_err, real_vn + real_err, alpha=0.2, color="C0"
        )
        mask_real_interp = real_is_interp
        if np.any(mask_real_interp):
            ax[row][col].plot(
                real_p[mask_real_interp],
                real_vn[mask_real_interp],
                "--",
                color="C0",
                linewidth=1,
                alpha=0.5,
                label="UrQMD (интерп.)" if idx == 0 else "",
            )
        mask_fake_meas = ~fake_is_interp
        ax[row][col].plot(
            fake_p[mask_fake_meas],
            fake_vn[mask_fake_meas],
            color="C3",
            linewidth=2,
            label="GAN (измер.)",
        )
        ax[row][col].fill_between(
            fake_p, fake_vn - fake_err, fake_vn + fake_err, alpha=0.2, color="C3"
        )
        mask_fake_interp = fake_is_interp
        if np.any(mask_fake_interp):
            ax[row][col].plot(
                fake_p[mask_fake_interp],
                fake_vn[mask_fake_interp],
                "--",
                color="C3",
                linewidth=1,
                alpha=0.5,
                label="GAN (интерп.)" if idx == 0 else "",
            )
        ax[row][col].set_xlabel(r"$p_T$ (GeV/c)", fontsize=12)
        ax[row][col].set_ylabel(f"$v_{{{n}}}$", fontsize=12)
        ax[row][col].legend(fontsize=9)
        ax[row][col].grid(alpha=0.3)
        ax[row][col].set_title(f"$v_{{{n}}}(p_T)$", fontsize=13)
    pt_real_all = calc_pt_dist(real_data)
    pt_fake_all = calc_pt_dist(fake_data)
    ax[1][1].hist(
        pt_real_all.numpy(),
        bins=pt_bins,
        density=True,
        alpha=0.5,
        label="UrQMD",
        color="C0",
    )
    ax[1][1].hist(
        pt_fake_all.numpy(),
        bins=pt_bins,
        density=True,
        alpha=0.5,
        label="GAN",
        color="C3",
    )
    ax[1][1].set_xlabel(r"$p_T$ (GeV/c)", fontsize=12)
    ax[1][1].set_ylabel("Нормированное число частиц", fontsize=12)
    ax[1][1].set_title("Распределение по $p_T$", fontsize=13)
    ax[1][1].legend()
    ax[1][1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return metrics


def plot_pt(real, fake, mode="pt"):
    """
    Plotting total transverse momenta and energy fractions by particle type
    Parameters:
    - real - real data;
    - fake - fake data;
    - mode - "pt" or "energy"
    Return:
    - mean value of MAE of fraction errors
    """
    real_keys = [
        part_names[k] if k in part_names.keys() else str(k) for k in real.keys()
    ]
    real_values = [np.mean(v) for v in real.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(real_keys, real_values, label="UrQMD", alpha=0.5)
    fake_keys = [
        part_names[k] if k in part_names.keys() else str(k) for k in fake.keys()
    ]
    fake_values = [np.mean(v) for v in fake.values()]
    bars = ax.bar(fake_keys, fake_values, label="GAN", alpha=0.5)
    real_values = np.array(real_values)
    fake_values = np.array(fake_values)
    ratios = [
        round(val, 2) for val in (np.abs(real_values - fake_values) / real_values * 100)
    ]
    str_ratios = [f"{val}%" for val in ratios]
    ax.bar_label(bars, labels=str_ratios, padding=3)
    ax.set_xlabel("particle type")
    ax.set_ylabel("value")
    ax.set_title(
        "Доля поперечного импульса в событии"
        if mode == "pt"
        else "Доля полной энергии в событии"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend()
    plt.show()
    return np.mean(ratios)
