import matplotlib.pyplot as plt
from scipy.integrate import simpson
import torch
import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance
from src.preprocessing import calc_pseudorapidity, calc_energy
from src.settings import *


def plot_losses(losses):
    plt.clf()
    len_ = len(losses.keys())
    fig, ax = plt.subplots(nrows=1, ncols=len_, figsize=(3 * len_, 3))
    for j, k in enumerate(losses.keys()):
        ax[j].plot([i + 1 for i in range(len(losses[k]))], losses[k])
        ax[j].set_title(k)
    plt.tight_layout()
    plt.show()


def kl_divergence(real_kde, fake_kde, x, eps=1e-12):
    p = real_kde.evaluate(x)
    q = fake_kde.evaluate(x)
    return simpson(p * np.log((p + eps) / (q + eps)), x)


def w2_distance_1d(u_values, v_values):
    u_sorted = np.sort(u_values)
    v_sorted = np.sort(v_values)
    if len(u_sorted) != len(v_sorted):
        from scipy.interpolate import interp1d
        q = np.linspace(0, 1, min(len(u_sorted), len(v_sorted)))
        u_quant = np.quantile(u_sorted, q)
        v_quant = np.quantile(v_sorted, q)
        return np.sqrt(np.mean((u_quant - v_quant)**2))
    else:
        return np.sqrt(np.mean((u_sorted - v_sorted)**2))


def plot_kde(real, fake, dim_names, part_type, num=None):
    fig, axes = plt.subplots(2, 5, figsize=(25, 7))
    if num is not None:
        p_sq = real[..., 0]**2 + real[..., 1]**2 + real[..., 2]**2
        sorted_indices = torch.argsort(p_sq, dim=1, descending=True)
        kth_idx = sorted_indices[:, num]
        batch_idx = np.arange(real.shape[0])
        real = real[batch_idx, kth_idx]
        p_sq = fake[..., 0]**2 + fake[..., 1]**2 + fake[..., 2]**2
        sorted_indices = torch.argsort(p_sq, dim=1, descending=True)
        kth_idx = sorted_indices[:, num]
        batch_idx = np.arange(fake.shape[0])
        fake = fake[batch_idx, kth_idx]
    else:
        real = real.reshape(-1, real.shape[-1])
        fake = fake.reshape(-1, fake.shape[-1])
    real = real[~torch.all(real == 0., dim=1)]
    fake = fake[~torch.all(fake == 0., dim=1)]
    masses = torch.full_like(fake[..., -1], pion_masses[part_type])
    real_energy = real[..., -1]
    fake_energy = calc_energy(fake[..., 0], fake[..., 1], fake[..., 2], masses)
    real_eta = calc_pseudorapidity(real[..., 0], real[..., 1], real[..., 2])
    fake_eta = calc_pseudorapidity(fake[..., 0], fake[..., 1], fake[..., 2])
    for i in range(5):
        if i < 3:
            real_vals = real[:, i].numpy()
            fake_vals = fake[:, i].numpy()
            title_pre = f'{part_type} {dim_names[i]}'
        elif i == 3:
            real_vals = real_energy.numpy()
            fake_vals = fake_energy.numpy()
            title_pre = f'{part_type} energy'
        else:
            real_vals = real_eta.numpy()
            fake_vals = fake_eta.numpy()
            title_pre = f'{part_type} pseudorapidity'
        kde_real = stats.gaussian_kde(real_vals, bw_method='scott')
        kde_fake = stats.gaussian_kde(fake_vals, bw_method='scott')
        x_points = np.linspace(min(real_vals.min(), fake_vals.min()),
                               max(real_vals.max(), fake_vals.max()), 1000)
        pdf_real = kde_real(x_points)
        pdf_fake = kde_fake(x_points)
        kl = kl_divergence(kde_real, kde_fake, x_points)
        w1 = wasserstein_distance(real_vals, fake_vals)
        w2 = w2_distance_1d(real_vals, fake_vals)
        axes[0, i].plot(x_points, pdf_real, label='Real')
        axes[0, i].plot(x_points, pdf_fake, label='Fake')
        axes[0, i].set_title(f'{title_pre} ({"all" if num is None else num}) KL={kl:.3f} W1={w1:.3f} W2={w2:.3f}')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        axes[1, i].hist(real_vals, bins=50, alpha=0.5, label='Real')
        axes[1, i].hist(fake_vals, bins=50, alpha=0.5, label='Fake')
        axes[1, i].set_title(f'{title_pre} histogram')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return kl, w1, w2 


def plot_scatter(real, fake):
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    real_flat = real.reshape(-1, real.shape[-1])
    fake_flat = fake.reshape(-1, fake.shape[-1])
    real_flat = real_flat[~torch.all(real_flat == 0., dim=1)]
    fake_flat = fake_flat[~torch.all(fake_flat == 0., dim=1)]
    ax1.scatter(real_flat[:,0], real_flat[:,1], real_flat[:,2], alpha=0.6, label='Real')
    ax1.set_xlim(-2,2); ax1.set_ylim(-2,2); ax1.set_zlim(-2,2)
    ax1.set_title('Real')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(fake_flat[:,0], fake_flat[:,1], fake_flat[:,2], alpha=0.6, color='orange', label='Fake')
    ax2.set_xlim(-2,2); ax2.set_ylim(-2,2); ax2.set_zlim(-2,2)
    ax2.set_title('Fake')
    plt.show()


def get_pt(data):
    pt = torch.sqrt(data[...,0] ** 2 + data[...,1] ** 2)
    return pt[data[..., 0] !=0 ].numpy()


def compute_v2_vs_pt(data, pt_min, pt_max, eta_max, nbins, n):
    pt_all, phi_all = [], []
    for event in data:
        if isinstance(event, torch.Tensor):
            event = event.cpu().numpy()
        px = event[..., 0]
        py = event[..., 1]
        pz = event[..., 2]
        pt = np.sqrt(px ** 2 + py ** 2)
        p_total = np.sqrt(pt ** 2 + pz ** 2)
        eta = 0.5 * np.log((p_total + pz) / (p_total - pz + 1e-10))
        mask = (pt > pt_min) & (pt < pt_max) & (np.abs(eta) < eta_max)
        pt_sel = pt[mask]
        phi_sel = np.arctan2(py[mask], px[mask])
        pt_all.extend(pt_sel)
        phi_all.extend(phi_sel)
    pt_all = np.array(pt_all)
    phi_all = np.array(phi_all)
    bin_edges = np.linspace(pt_min, pt_max, nbins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    v2_vals, v2_errs = [], []
    for i in range(nbins):
        mask_bin = (pt_all >= bin_edges[i]) & (pt_all < bin_edges[i + 1])
        if np.sum(mask_bin) < 2:
            v2_vals.append(np.nan)
            v2_errs.append(np.nan)
            continue
        cos_nphi = np.cos(n * phi_all[mask_bin])
        v2 = np.mean(cos_nphi)
        err = np.std(cos_nphi) / np.sqrt(len(cos_nphi))
        v2_vals.append(v2)
        v2_errs.append(err)
    return bin_centers, np.array(v2_vals), np.array(v2_errs)


def get_masked_pt(p, pt_min=0.2, pt_max=None, eta_max=1.0):
    px, py, pz = p[..., 0], p[..., 1], p[..., 2]
    pt = torch.sqrt(px**2 + py**2)
    p_total = torch.sqrt(pt**2 + pz**2)
    eta = 0.5 * torch.log((p_total + pz) / (p_total - pz + 1e-10))
    mask = (pt > pt_min) & (torch.abs(eta) < eta_max)
    if pt_max is not None:
        mask = mask & (pt < pt_max)
    return mask, pt


def compute_pt_dist(data, pt_min=0.0, pt_max=2.0, eta_max=10.0):
    pt_list = []
    for event in data:
        mask, pt_event = get_masked_pt(event, pt_min, pt_max, eta_max)
        pt_sel = pt_event[mask]
        pt_list.append(pt_sel)
    return torch.cat(pt_list) if pt_list else torch.tensor([], device=data.device if isinstance(data, torch.Tensor) else data[0].device)


def plot_v2(real_data, fake_data, n=2, pt_min=0.0, pt_max=2.0, eta_max=10.0, nbins=10, pt_bins=50):
    bin_centers, v2_real, err_real = compute_v2_vs_pt(real_data, pt_min, pt_max, eta_max, nbins, n)
    _, v2_fake, err_fake = compute_v2_vs_pt(fake_data, pt_min, pt_max, eta_max, nbins, n)
    pt_real_all = compute_pt_dist(real_data, pt_min=0, eta_max=eta_max)
    pt_fake_all = compute_pt_dist(fake_data, pt_min=0, eta_max=eta_max)
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(16,5))
    ax[0].errorbar(bin_centers, v2_real, yerr=err_real, fmt='o-', label='Real', capsize=3)
    ax[0].errorbar(bin_centers, v2_fake, yerr=err_fake, fmt='s-', label='Fake', capsize=3)
    ax[0].set_xlabel('pT (GeV/c)')
    ax[0].set_ylabel(f'v{n}')
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[0].set_title(f'v{n}(pT) with eta < {eta_max}')
    ax[1].hist(pt_real_all.numpy(), bins=pt_bins, density=True, alpha=0.5, label='Real', color='blue')
    ax[1].hist(pt_fake_all.numpy(), bins=pt_bins, density=True, alpha=0.5, label='Fake', color='red')
    ax[1].set_xlabel('pT, GeV/c')
    ax[1].set_ylabel('Normalized count')
    ax[1].set_title('pT distribution')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
