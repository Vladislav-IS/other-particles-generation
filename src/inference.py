import torch
import numpy as np
from src.plots import plot_kde, plot_scatter, plot_v2, plot_pt
from src.calculations import calc_energy
from src.config import *


def generate_transformer(G, batch_size, n_points, cond, labels, device, threshold=0.5):
    """
    Function for particles sets generation via EPiC-styled transformer.
    Parameters:
    - G - tramsformer generator;
    - batch_size, n_points - unused parameters;
    - cond - external condition;
    - labels - particles types;
    - device - "cuda" or "cpu";
    - threshold - threshold for generated mask.
    Return:
    - momenta tensor
    """
    G.eval()
    cond = cond.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        impulses, mask_probs = G(labels, cond)
    mask = (mask_probs >= threshold).float()
    impulses = impulses * mask.unsqueeze(-1)
    return impulses.cpu()


def generate_epic(G, batch_size, n_points, cond, labels, device, zg_dim, zl_dim):
    """
    Function for particles sets generation via vanilla EPiC-GAN.
    Parameters:
    - G - tramsformer generator;
    - batch_size,
    - n_points - length of padded particles sequences;
    - cond - external condition;
    - labels - particles types;
    - device - "cuda" or "cpu";
    - zg_dim - global noise dimensionality;
    - zl_dim - local noise dimensionality.
    Return:
    - momenta tensor
    """
    G.eval()
    cond = cond.to(device)
    labels = labels.to(device)
    z_global = torch.randn(batch_size, zg_dim, device=device)
    z_local = torch.randn(batch_size, n_points, zl_dim, device=device)
    with torch.no_grad():
        impulses = G(z_global, z_local, cond, labels)
    return impulses.cpu()


def count_pt(particles):
    """
    Function for calculating transverse momentum modulus fraction by particle type.
    Paeameters:
    - particles - dictionary containing raw tensors of particles momenta.
    Return:
    - pt_parts - dictionary containing transverse momentum modulus fraction
    """
    pt_parts = {}
    sum_pt = 0
    for k, v in particles.items():
        pt = torch.sum(torch.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2))
        sum_pt += pt
        pt_parts[k] = pt
    pt_parts = {k: (v / sum_pt).item() for k, v in pt_parts.items()}
    return pt_parts


def count_e(particles):
    """
    Function for calculating energy fraction by particle type (using real tabular data).
    Paeameters:
    - particles - dictionary containing raw tensors of particles energies.
    Return:
    - e_parts - dictionary containing energy fraction
    """
    e_parts = {}
    sum_e = 0
    for k, v in particles.items():
        e = torch.sum(v[:, -1])
        sum_e += e
        e_parts[k] = e
    e_parts = {k: (v / sum_e).item() for k, v in e_parts.items()}
    return e_parts


def count_e_with_mass(particles):
    """
    Function for calculating energy fraction by particle type (using momenta and particles masses).
    Paeameters:
    - particles - dictionary containing raw tensors of particles momenta.
    Return:
    - e_parts - dictionary containing energy fraction
    """
    e_parts = {}
    sum_e = 0
    for k, v in particles.items():
        if k == "protons" or k == "neutrons":
            e = torch.sum(v[:, -1])
            sum_e += e
            e_parts[k] = e
        else:
            e = torch.sum(
                calc_energy(
                    v[:, 0], v[:, 1], v[:, 2], torch.full_like(v[:, 0], pion_masses[k])
                )
            )
            sum_e += e
            e_parts[k] = e
    e_parts = {k: (v / sum_e).item() for k, v in e_parts.items()}
    return e_parts


def get_dist_metrics(
    G,
    generate_func,
    n_points,
    test_mom,
    test_cond,
    test_label,
    test_mask,
    device,
    **kwargs
):
    """
    Function for plotting and calculating distribution metrics (KL divergence, Wasserstein distance).
    Paeameters:
    - G - generator model;
    - generate_func - model-specific generation function;
    - n_points - length of padded particles sequences;
    - test_mom - tensors of thue "other" particles momenta (test subset);
    - test_cond - external condition (test subset);
    - test_label - particles types (test subset);
    - test_mask - padding mask (test subset);
    - device - "cuda" or "cpu";
    - kwargs - for model-specific generation function.
    Return:
    - metrics - dictionary containing metrics for all particles and the fastest partlice
    """
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
        generated_other = generate_func(
            G, mask.sum(), n_points, cond_test, label_test, device, **kwargs
        )
        kl, w1, w2 = plot_kde(
            other_test, generated_other, ["Px", "Py", "Pz"], num_to_label[i]
        )
        metrics[num_to_label[i]]["kl"] = kl
        metrics[num_to_label[i]]["w1"] = w1
        metrics[num_to_label[i]]["w2"] = w2
        kl_0, w1_0, w2_0 = plot_kde(
            other_test, generated_other, ["Px", "Py", "Pz"], num_to_label[i], num=0
        )
        metrics[num_to_label[i]]["kl_0"] = kl_0
        metrics[num_to_label[i]]["w1_0"] = w1_0
        metrics[num_to_label[i]]["w2_0"] = w2_0
        plot_kde(
            other_test, generated_other, ["Px", "Py", "Pz"], num_to_label[i], num=1
        )
        plot_kde(
            other_test, generated_other, ["Px", "Py", "Pz"], num_to_label[i], num=2
        )
        plot_scatter(other_test, generated_other)
        if num_to_label[i] != 111:
            plot_v2(other_test, generated_other)
    return metrics


def get_energy_metrics(
    G,
    generate_func,
    n_points,
    test_mom,
    test_nucl,
    test_nucl_mask,
    test_event,
    test_cond,
    test_label,
    test_mask,
    device,
    **kwargs
):
    """
    Function for plotting energy metrics (transverse momentum modulus and energy fraction).
    Paeameters:
    - G - generator model;
    - generate_func - model-specific generation function;
    - n_points - length of padded particles sequences;
    - test_mom - tensors of thue "other" particles momenta (test subset);
    - test_nucl - nucleons momenta from test subset;
    - test_nucl_mask - nucleons padding mask from test subset;
    - test_event - tensor of event numbers from test subset;
    - test_cond - external condition (test subset);
    - test_label - particles types (test subset);
    - test_mask - padding mask (test subset);
    - device - "cuda" or "cpu";
    - kwargs - for model-specific generation function.
    """
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
        nucl_1_test = torch.tensor(
            [nucl[0] for nucl in test_nucl], dtype=torch.float32
        )[mask]
        nucl_2_test = torch.tensor(
            [nucl[1] for nucl in test_nucl], dtype=torch.float32
        )[mask]
        nucl_1_test_mask = torch.tensor(
            [nucl[0] for nucl in test_nucl_mask], dtype=torch.float32
        )[mask]
        nucl_2_test_mask = torch.tensor(
            [nucl[1] for nucl in test_nucl_mask], dtype=torch.float32
        )[mask]
        real["protons"] = (nucl_1_test * nucl_1_test_mask.unsqueeze(-1))[0]
        real["neutrons"] = (nucl_2_test * nucl_2_test_mask.unsqueeze(-1))[0]
        fake["protons"] = (nucl_1_test * nucl_1_test_mask.unsqueeze(-1))[0]
        fake["neutrons"] = (nucl_2_test * nucl_2_test_mask.unsqueeze(-1))[0]
        label_test = torch.tensor(test_label)[mask]
        mask_test = torch.tensor(test_mask, dtype=torch.float32)[mask]
        cond_test = torch.tensor(test_cond, dtype=torch.float32)[mask]
        generated_other = generate_func(
            G, mask.sum(), n_points, cond_test, label_test, device, **kwargs
        )
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
    plot_pt(all_real_e, all_fake_e, mode="energy")
