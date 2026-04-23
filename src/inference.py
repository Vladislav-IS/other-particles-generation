import torch
import numpy as np
import matplotlib.pyplot as plt
from src.plots import plot_kde, plot_scatter, plot_v2
from src.preprocessing import calc_energy
from src.settings import *


def generate_my(G, batch_size, n_points, cond, labels, device, threshold=0.5):
    G.eval()
    cond = cond.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        impulses, mask_probs = G(labels, cond)
    mask = (mask_probs >= threshold).float()
    impulses = impulses * mask.unsqueeze(-1)
    return impulses.cpu()


def generate_epic(G, batch_size, n_points, cond, labels, device, zg_dim, zl_dim):
    G.eval()
    cond = cond.to(device)
    labels = labels.to(device)
    z_global = torch.randn(batch_size, zg_dim, device=device)
    z_local = torch.randn(batch_size, n_points, zl_dim, device=device)
    with torch.no_grad():
        impulses = G(z_global, z_local, cond, labels)
    return impulses.cpu()


def count_pt(particles):
    pt_parts = {}
    sum_pt = 0
    for k, v in particles.items():
        pt = torch.sum(torch.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2))
        sum_pt += pt
        pt_parts[k] = pt
    pt_parts = {k: (v / sum_pt).item() for k, v in pt_parts.items()}
    return pt_parts


def count_e(particles):
    e_parts = {}
    sum_e = 0
    for k, v in particles.items():
        e = torch.sum(v[:, -1])
        sum_e += e
        e_parts[k] = e
    e_parts = {k: (v / sum_e).item() for k, v in e_parts.items()}
    return e_parts


def count_e_with_mass(particles):
    e_parts = {}
    sum_e = 0
    for k, v in particles.items():
        if k == 'protons' or k == 'neutrons':
            e = torch.sum(v[:, -1])
            sum_e += e
            e_parts[k] = e
        else:
            e = torch.sum(calc_energy(v[:, 0], v[:, 1], v[:, 2], torch.full_like(v[:, 0], pion_masses[k])))
            sum_e += e
            e_parts[k] = e
    e_parts = {k: (v / sum_e).item() for k, v in e_parts.items()}
    return e_parts


def plot_pt(real, fake, mode='pt'):
    real_keys = [str(k) for k in real.keys()]
    real_values = [np.mean(v) for v in real.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(real_keys, real_values, label='Real', alpha=0.5)
    fake_keys = [str(k) for k in fake.keys()]
    fake_values = [np.mean(v) for v in fake.values()]
    bars = ax.bar(fake_keys, fake_values, label='Fake', alpha=0.5)
    real_values = np.array(real_values)
    fake_values = np.array(fake_values)
    ratios = [f'{round(val, 2)}%' for val in (np.abs(real_values - fake_values) / real_values * 100)]
    ax.bar_label(bars, labels=ratios, padding=3)
    ax.set_xlabel('particle type')
    ax.set_ylabel('value')
    ax.set_title('Transverse momentum modulus fraction' if mode == 'pt' else 'Energy fraction')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    plt.show()


def get_dist_metrics(G, generate_func, n_points, test_mom, test_nucl, test_nucl_mask, test_event, test_cond, test_label, test_mask, **kwargs):
    metrics = {}
    for i in range(len(num_to_label)):
        metrics[num_to_label[i]] = {}
        mask = torch.tensor(test_label) == i
        if mask.sum() == 0:
            continue
        other_test = torch.tensor(test_mom, dtype=torch.float32)[mask]
        label_test = torch.tensor(test_label)[mask]
        mask_test = torch.tensor(test_mask, dtype=torch.float32)[mask]
        cond_test = torch.tensor(test_cond, dtype=torch.float32)[mask]
        generated_other = generate_func(G, mask.sum(), n_points, cond_test, label_test, device, **kwargs)
        kl, w1, w2 = plot_kde(other_test, generated_other, ['Px', 'Py', 'Pz'], num_to_label[i])
        metrics[num_to_label[i]]['kl'] = kl
        metrics[num_to_label[i]]['w1'] = w1
        metrics[num_to_label[i]]['w2'] = w2
        kl_0, w1_0, w2_0 = plot_kde(other_test, generated_other, ['Px', 'Py', 'Pz'], num_to_label[i], num=0)
        metrics[num_to_label[i]]['kl_0'] = kl_0
        metrics[num_to_label[i]]['w1_0'] = w1_0
        metrics[num_to_label[i]]['w2_0'] = w2_0
        plot_kde(other_test, generated_other, ['Px', 'Py', 'Pz'], num_to_label[i], num=1)
        plot_kde(other_test, generated_other, ['Px', 'Py', 'Pz'], num_to_label[i], num=2)
        plot_scatter(other_test, generated_other)
        if num_to_label[i] != 111:
            plot_v2(other_test, generated_other)
    return metrics  


def get_energy_metrics(G, generate_func, n_points, test_mom, test_nucl, test_nucl_mask, test_event, test_cond, test_label, test_mask, **kwargs):
    all_real_pt = {}
    all_fake_pt = {}
    all_real_e = {}
    all_fake_e = {}
    for i in np.unique(test_event):
        mask = torch.tensor(test_event) == i
        if mask.sum() == 0:
            continue
        other_test = torch.tensor(test_mom, dtype=torch.float32)[mask]
        real = {}
        fake = {}
        nucl_1_test = torch.tensor([nucl[0] for nucl in test_nucl], dtype=torch.float32)[mask]
        nucl_2_test = torch.tensor([nucl[1] for nucl in test_nucl], dtype=torch.float32)[mask]
        nucl_1_test_mask = torch.tensor([nucl[0] for nucl in test_nucl_mask], dtype=torch.float32)[mask]
        nucl_2_test_mask = torch.tensor([nucl[1] for nucl in test_nucl_mask], dtype=torch.float32)[mask]
        real['protons'] = (nucl_1_test * nucl_1_test_mask.unsqueeze(-1))[0]
        real['neutrons'] = (nucl_2_test * nucl_2_test_mask.unsqueeze(-1))[0]
        fake['protons'] = (nucl_1_test * nucl_1_test_mask.unsqueeze(-1))[0]
        fake['neutrons'] = (nucl_2_test * nucl_2_test_mask.unsqueeze(-1))[0]
        label_test = torch.tensor(test_label)[mask]
        mask_test = torch.tensor(test_mask, dtype=torch.float32)[mask]
        cond_test = torch.tensor(test_cond, dtype=torch.float32)[mask]
        generated_other = generate_func(G, mask.sum(), n_points, cond_test, label_test, device, **kwargs)
        for i in range(len(label_test)):
            real[num_to_label[label_test[i].item()]] = other_test[i]
            fake[num_to_label[label_test[i].item()]] = generated_other[i]
        real_pt = count_pt(real)
        fake_pt = count_pt(fake)
        real_e = count_e(real)
        fake_e = count_e_with_mass(fake)
        for k in real_pt.keys():
            if k in all_real_pt:
                all_real_pt[k].append(real_pt[k])
                all_fake_pt[k].append(fake_pt[k])
                all_real_e[k].append(real_e[k])
                all_fake_e[k].append(fake_e[k])
            else:
                all_real_pt[k] = [real_pt[k]]
                all_fake_pt[k] = [fake_pt[k]]
                all_real_e[k] = [real_e[k]]
                all_fake_e[k] = [fake_e[k]]
    plot_pt(all_real_pt, all_fake_pt)
    plot_pt(all_real_e, all_fake_e, mode='energy')
