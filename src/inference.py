import torch
import numpy as np
from src.plots import plot_kde, plot_scatter, plot_vn, plot_pt
from src.calculations import calc_energy
from src.config import *


def generate_transformer(G, batch_size, n_points, cond, labels, device, threshold=0.5):
    """
    Generating particles momenta for model inference
    Parameters:
    - G - generator model;
    - batch_size;
    - n_points - fixed size of momenta tensor;
    - cond - input external condition tensor;
    - labels - input labels tensor;
    - device - "cuda" or "cpu";
    - threshold - in order to get binary mask
    Return:
    - impulses - momenta tensor
    """
    G.eval()
    cond = cond.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        impulses, mask_probs = G(labels, cond)
    mask = (mask_probs >= threshold).float()
    impulses = impulses * mask.unsqueeze(-1)
    return impulses.cpu()


def count_pt(particles):
    """
    Getting fraction of total transverse momenta in events by particles type
    Parameters:
    - particles - dictionary where keys are particles types and values are momenergy tensors
    Return:
    - pt_parts - dictionary of fractions of total transverse momenta by particles type
    """
    pt_parts = {}
    sum_pt = 0
    for k, v in particles.items():
        pt = torch.sum(torch.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2))
        sum_pt += pt
        pt_parts[k] = pt
    pt_parts = {k: (v / sum_pt).item() for k, v in pt_parts.items()}
    return pt_parts


def count_e(particles):
    """
    Getting fraction of total energy in events by particles type
    Parameters:
    - particles - dictionary where keys are particles types and values are momenergy tensors
    Return:
    - e_parts - dictionary of fractions of total energy by particles type
    """
    e_parts = {}
    sum_e = 0
    for k, v in particles.items():
        e = torch.sum(v[..., -1])
        sum_e += e
        e_parts[k] = e
    e_parts = {k: (v / sum_e).item() for k, v in e_parts.items()}
    return e_parts


def count_e_with_mass(particles):
    """
    Getting fraction of total energy in events by particles type using particles masses
    Parameters:
    - particles - dictionary where keys are particles types and values are momenergy tensors
    Return:
    - e_parts - dictionary of fractions of total energy by particles type
    """
    e_parts = {}
    sum_e = 0
    for k, v in particles.items():
        if k == "нуклоны":
            e = torch.sum(v[..., -1])
            sum_e += e
            e_parts[k] = e
        else:
            e = torch.sum(
                calc_energy(
                    v[..., 0],
                    v[..., 1],
                    v[..., 2],
                    torch.full_like(v[:, 0], pion_masses[k]),
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
    Calculating and plotting distibution metrics (KL, W1, W2)
    Parameters:
    - G - generator model;
    - generate_func - function for generating momenta;
    - n_points - fixed size of generated tensors;
    - test_mom - target momenta array;
    - test_cond - input external conditon array;
    - test_label - input label array;
    - test_mask - true padding mask array;
    - device - "cuda" or "cpu";
    - **kwargs - aditional parameters for genetaring
    Return:
    - metrics - metrics dictionary {"particle type": {"metric type": "value"}}
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
        m = plot_kde(other_test, generated_other, ["Px", "Py", "Pz"], num_to_label[i])
        metrics[num_to_label[i]]["e_kl"] = m["e_kl"]
        metrics[num_to_label[i]]["e_w1"] = m["e_w1"]
        metrics[num_to_label[i]]["eta_kl"] = m["eta_kl"]
        metrics[num_to_label[i]]["eta_w1"] = m["eta_w1"]
        m_0 = plot_kde(
            other_test,
            generated_other,
            ["Px", "Py", "Pz"],
            num_to_label[i],
            num=0,
            show_plots=False,
        )
        metrics[num_to_label[i]]["e_kl_0"] = m_0["e_kl"]
        metrics[num_to_label[i]]["e_w1_0"] = m_0["e_w1"]
        metrics[num_to_label[i]]["eta_kl_0"] = m_0["eta_kl"]
        metrics[num_to_label[i]]["eta_w1_0"] = m_0["eta_w1"]
        plot_kde(
            other_test,
            generated_other,
            ["Px", "Py", "Pz"],
            num_to_label[i],
            num=1,
            show_plots=False,
        )
        plot_kde(
            other_test,
            generated_other,
            ["Px", "Py", "Pz"],
            num_to_label[i],
            num=2,
            show_plots=False,
        )
        plot_scatter(other_test, generated_other)
        vn_metrics = plot_vn(other_test, generated_other)
        for k, v in vn_metrics.items():
            metrics[num_to_label[i]][k] = v
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
    Calculating and plotting aggregated energy metrics (total transverse momenta and energy fractions)
    Parameters:
    - G - generator model;
    - generate_func - function for generating momenta;
    - n_points - fixed size of generated tensors;
    - test_mom - target momenta array;
    - test_nucl - nucleons momenta array;
    - test_nucl_mask - nucleons mask array;
    - test_event - events numbers array;
    - test_cond - input external conditon array;
    - test_label - input label array;
    - test_mask - true padding mask array;
    - device - "cuda" or "cpu";
    - **kwargs - aditional parameters for genetaring
    Return:
    - metrics - metrics dictionary {"particle type": {"metric type": "value"}}
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
        real["нуклоны"] = torch.tensor(test_nucl, dtype=torch.float32)[mask][0]
        fake["нуклоны"] = torch.tensor(test_nucl, dtype=torch.float32)[mask][0]
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
    metrics = {}
    pt_ = plot_pt(all_real_pt, all_fake_pt)
    e_ = plot_pt(all_real_e, all_fake_e, mode="energy")
    for p in pions:
        metrics[p] = {}
        metrics[p]["pt"] = pt_
        metrics[p]["e"] = e_
    return metrics
