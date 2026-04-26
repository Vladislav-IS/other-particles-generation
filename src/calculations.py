import numpy as np
import torch
import pandas as pd
from scipy import stats
import random
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from src.config import *


def set_seed(seed):
    """
    Function forrandom seed fixation.
    Paramerers:
    - seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calc_vn_vs_pt(data, pt_min, pt_max, eta_max, nbins, n, n_interp=500):
    """
    Function for calculating v_n(p_T) dependency with error bars.

    Parameters:
    - data - particles momenta;
    - pt_min - minimal value of transverse momentum;
    - pt_max - maximum value of transverse momentum;
    - eta_max - maximum value of pseudorapidity;
    - nbins - number of bins for v_n(p_T) dependency;
    - n - cosine coefficient (flow number);
    - n_interp - number of points for interpolation.

    Return:
    - p_interp - interpolated p_T values;
    - vn_interp - interpolated v_n values;
    - err_interp - interpolated errors (standard error of the mean).
    """
    pt_all, phi_all = [], []
    for event in data:
        if isinstance(event, torch.Tensor):
            event = event.cpu().numpy()
        px = event[..., 0]
        py = event[..., 1]
        pz = event[..., 2]
        pt = np.sqrt(px**2 + py**2)
        p_total = np.sqrt(pt**2 + pz**2)
        eta = 0.5 * np.log((p_total + pz) / (p_total - pz + 1e-10))
        mask = (pt > pt_min) & (pt < pt_max) & (np.abs(eta) < eta_max)
        pt_all.extend(pt[mask])
        phi_all.extend(np.arctan2(py[mask], px[mask]))
    pt_all = np.array(pt_all)
    phi_all = np.array(phi_all)
    bin_edges = np.linspace(pt_min, pt_max, nbins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    vn_vals = []
    err_vals = []
    for i in range(nbins):
        mask_bin = (pt_all >= bin_edges[i]) & (pt_all < bin_edges[i + 1])
        if np.sum(mask_bin) == 0:
            vn_vals.append(np.nan)
            err_vals.append(np.nan)
        else:
            cos_nphi = np.cos(n * phi_all[mask_bin])
            vn_vals.append(np.mean(cos_nphi))
            err_vals.append(np.std(cos_nphi) / np.sqrt(len(cos_nphi)))
    vn_vals = np.array(vn_vals)
    err_vals = np.array(err_vals)
    valid = ~np.isnan(vn_vals)
    if np.sum(valid) < 2:
        raise ValueError("Недостаточно бинов с частицами для интерполяции")
    bin_centers_valid = bin_centers[valid]
    vn_valid = vn_vals[valid]
    err_valid = err_vals[valid]
    f_vn = interp1d(
        bin_centers_valid,
        vn_valid,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    f_err = interp1d(
        bin_centers_valid,
        err_valid,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    p_interp = np.linspace(pt_min, pt_max, n_interp)
    vn_interp = f_vn(p_interp)
    err_interp = f_err(p_interp)
    return p_interp, vn_interp, err_interp


def get_masked_pt(p, pt_min=0.2, pt_max=None, eta_max=1.0):
    """
    Function for calculating transverse momenta with mask.
    Parameters:
    - p - particles momenta;
    - pt_min - minimal value of transverse momentum;
    - pt_max - maximum value of transverse momentum;
    - eta_max - maximum value of pseudorapidity.
    Return:
    - mask - mask depending on pt_min, pt_max amd eta_max;
    - pt - transverse momenta tensor.
    """
    px, py, pz = p[..., 0], p[..., 1], p[..., 2]
    pt = torch.sqrt(px**2 + py**2)
    p_total = torch.sqrt(pt**2 + pz**2)
    eta = 0.5 * torch.log((p_total + pz) / (p_total - pz + 1e-10))
    mask = (pt > pt_min) & (torch.abs(eta) < eta_max)
    if pt_max is not None:
        mask = mask & (pt < pt_max)
    return mask, pt


def calc_pt_dist(data, pt_min=0.0, pt_max=2.0, eta_max=10.0):
    """
    Function for calculating transverse momenta distribution.
    Parameters:
    - data - particles momenta;
    - pt_min - minimal value of transverse momentum;
    - pt_max - maximum value of transverse momentum;
    - eta_max - maximum value of pseudorapidity.
    Return:
    - tensor of selected transverse momenta
    """
    pt_list = []
    for event in data:
        mask, pt_event = get_masked_pt(event, pt_min, pt_max, eta_max)
        pt_sel = pt_event[mask]
        pt_list.append(pt_sel)
    return (
        torch.cat(pt_list)
        if pt_list
        else torch.tensor(
            [], device=data.device if isinstance(data, torch.Tensor) else data[0].device
        )
    )


def kl_divergence(real_kde, fake_kde, x, eps=1e-12):
    """
    Function for calculating KL divergence.
    Parameters:
    - real_kde - KDE of real data;
    - fake_kde - KDE of generated data;
    - x - points;
    - eps - for numeric stability.
    Return:
    - KL divergence value
    """
    p = real_kde.evaluate(x)
    q = fake_kde.evaluate(x)
    return simpson(p * np.log((p + eps) / (q + eps)), x)


def w2_distance_1d(real, fake):
    """
    Function for calculating W2 distance.
    Parameters:
    - real_values - real data;
    - fake_values - generated data.
    Return:
    - W2 distance depending on data lengths
    """
    real_sorted = np.sort(real)
    fake_sorted = np.sort(fake)
    if len(real_sorted) != len(fake_sorted):
        q = np.linspace(0, 1, min(len(real_sorted), len(fake_sorted)))
        real_quant = np.quantile(real_sorted, q)
        fake_quant = np.quantile(fake_sorted, q)
        return np.sqrt(np.mean((real_quant - fake_quant) ** 2))
    else:
        return np.sqrt(np.mean((real_sorted - fake_sorted) ** 2))


def calc_energy(px, py, pz, masses):
    """
    Fucntion for calculating energies of particles (using momenta and masses).
    Parameters:
    - px - x momenta component;
    - py - y momenta component;
    - pz - z moment component;
    - masses - tensor of particles masses.
    Return:
    - tensor of particles energies
    """
    return torch.sqrt(px**2 + py**2 + pz**2 + masses**2)


def calc_m_inv(row, mass_number=131):
    """
    Function for calculating invariant mass per nucleon.
    Parameters:
    - row - row of ROOT data DataFrame;
    - mass_number - mass number of colliding nuclei.
    Return:
    - invariant mass per nucleon
    """
    energy = np.sum(row["fParticles.fE"])
    px = np.sum(row["fParticles.fPx"])
    py = np.sum(row["fParticles.fPy"])
    pz = np.sum(row["fParticles.fPz"])
    M_inv = np.sqrt(np.maximum(0, energy**2 - (px**2 + py**2 + pz**2)))
    return M_inv / mass_number


def select_particles(row, target_col, mask_cols):
    """
    Function for selecting particles using mask.
    Parameters:
    - row - row of ROOT data DataFrame;
    - target_col - column of selecting data;
    - mask_cols - columns for mask.
    Return:
    - selected values
    """
    feature_arr = np.array(row[target_col])
    mask_arr = np.zeros_like(feature_arr)
    for col in mask_cols:
        mask_arr += row[col]
    return feature_arr[mask_arr.astype(bool)]


def calc_pseudorapidity(px=None, py=None, pz=None, eps=1e-5):
    """
    Fucntion for calculating pseudorapidity.
    Parameters:
    - px - momenta x component;
    - py - momenta y component;
    - pz - momenta z component;
    - eps - for numeric stability.
    Return:
    - pseudorapidity array
    """
    norm = np.sqrt(px**2 + py**2 + pz**2)
    return 0.5 * np.log(eps + (norm + pz) / (eps + norm - pz))


def calc_pT(px, py):
    """
    Fucntion for calculating transverse momenta.
    Parameters:
    - px - momenta x component;
    - py - momenta y component.
    Return:
    - transverse momenta array
    """
    return np.sqrt(px**2 + py**2)


def calc_flows(px, py, weights=None):
    """
    Function for calculating v1, v2, v3 flows.
    Parameters:
    - px - momenta x component;
    - py - momenta y component;
    - weights - for vn correction.
    Return:
    - v1 flow;
    - v2 flow;
    -v3 flow
    """
    phi = np.arctan2(py, px)
    n = len(phi)
    w = np.ones(n) if weights is None else weights
    sum_w = np.sum(w)
    v1 = np.sum(w * np.cos(phi)) / sum_w
    v2 = np.sum(w * np.cos(2 * phi)) / sum_w
    v3 = np.sum(w * np.cos(3 * phi)) / sum_w
    return v1, v2, v3


def get_nucleons_by_type(px, py, pz, part_type="spectators", threshold=2.0):
    """
    Function for creating spectators or participants mask (using pseudorapidity).
    Parameters:
    - px - momenta x component;
    - py - momenta y component;
    - pz - momenta z component;
    - part_type - spectators or participants;
    - threshold - for separating spectators and participants.
    Return:
    - nucleons mask
    """
    eta = calc_pseudorapidity(px, py, pz)
    if part_type == "spectators":
        return eta > threshold
    else:
        return eta <= threshold


def get_part_spec(row, target_col, part_type="spectators"):
    """
    Function for selecting nucleons features depending on type.
    Parameters:
    - row - row of ROOT data DataFrame;
    - target_col - target particles features;
    - part_type - spectators or participants.
    Return:
    - selected features
    """
    nucleon_mask = np.array(row[N]) | np.array(row[P])
    nucleon_indices = np.where(nucleon_mask)[0]
    px = np.array(row["fParticles.fPx"])[nucleon_indices]
    py = np.array(row["fParticles.fPy"])[nucleon_indices]
    pz = np.array(row["fParticles.fPz"])[nucleon_indices]
    type_mask = get_nucleons_by_type(px, py, pz, part_type)
    selected_indices = nucleon_indices[type_mask]
    return np.array(row[target_col])[selected_indices]


def get_final_moms(mom_x, mom_y, mom_z, energy, max_size):
    """
    Function for preprocessing particles momenta.
    Parameters:
    - mom_x - momenta x component;
    - mom_y - momenta y component;
    - mom_z - momenta z component;
    - energy - partclies energies;
    - max_size - maximum length of particles sequences.
    Return:
    - mom_merged - padded momenergy;
    - mask - padding mask
    """
    if mom_x.shape[0] >= max_size:
        mask = np.ones(max_size, dtype=float)
        mom_x = mom_x[:max_size]
        mom_y = mom_y[:max_size]
        mom_z = mom_z[:max_size]
        energy = energy[:max_size]
        mom_merged = np.vstack((mom_x, mom_y, mom_z, energy)).T
    else:
        mask = np.zeros(max_size)
        mask[: len(mom_x)] = 1.0
        zeros_x = np.zeros(max_size)
        zeros_x[: len(mom_x)] = mom_x
        zeros_y = np.zeros(max_size)
        zeros_y[: len(mom_y)] = mom_y
        zeros_z = np.zeros(max_size)
        zeros_z[: len(mom_z)] = mom_z
        zeros_e = np.zeros(max_size)
        zeros_e[: len(energy)] = energy
        mom_merged = np.vstack((zeros_x, zeros_y, zeros_z, zeros_e)).T
    return mom_merged, mask


def preprocess_urqmd(urqmd, max_size_t=80, max_size_c=250, return_cols=False):
    """
    Function for preprocessing ROOT dataset.
    Parameters:
    - urqmd - ROOT dataset;
    - max_size_t - maximum length of target "other" particles sequences;
    - max_size_c - maximum length of nucleons sequences;
    - return cols - flag if condition columns names is needed.
    Return:
    - df - preprocessed DataFrame;
    - cond_cols - external colition columns names;
    - nucleons_cols - nucleons padded features columns names;
    - nucleons_mask_cols - nucleons padding mask columns names
    """
    urqmd_new = pd.DataFrame()
    if "fB" in urqmd.columns:
        urqmd_new["B"] = urqmd["fB"]
    urqmd_new["m_inv"] = urqmd.apply(calc_m_inv, axis=1)
    for p in particles:
        urqmd[p] = urqmd["fParticles.fPdg"].apply(lambda x: np.array(x) == p)
    for d in directions:
        urqmd_new[f"mom_{d}"] = urqmd.apply(
            lambda x: select_particles(x, f"fParticles.fP{d}", nucleons), axis=1
        )
        urqmd_new[f"coord_{d}"] = urqmd.apply(
            lambda x: select_particles(x, f"fParticles.f{d.upper()}", nucleons), axis=1
        )
        urqmd_new[f"mom_part_{d}"] = urqmd.apply(
            lambda x: get_part_spec(x, f"fParticles.fP{d}", "participants"), axis=1
        )
        urqmd_new[f"coord_part_{d}"] = urqmd.apply(
            lambda x: get_part_spec(x, f"fParticles.f{d.upper()}", "participants"),
            axis=1,
        )
    urqmd_new["energy_part"] = urqmd.apply(
        lambda x: get_part_spec(x, "fParticles.fE", "participants"), axis=1
    )
    urqmd_new["energy"] = urqmd.apply(
        lambda x: select_particles(x, "fParticles.fE", nucleons), axis=1
    )
    urqmd_new["mom_norm"] = urqmd_new.apply(
        lambda x: np.sqrt(x["mom_x"] ** 2 + x["mom_y"] ** 2 + x["mom_z"] ** 2), axis=1
    )
    urqmd_new["pseudorapidity"] = urqmd_new.apply(
        lambda x: calc_pseudorapidity(px=x["mom_x"], py=x["mom_y"], pz=x["mom_z"]),
        axis=1,
    )
    urqmd_new["pseudorapidity_part"] = urqmd_new.apply(
        lambda x: calc_pseudorapidity(
            px=x["mom_part_x"], py=x["mom_part_y"], pz=x["mom_part_z"]
        ),
        axis=1,
    )
    urqmd_new["mom_T_part"] = urqmd_new.apply(
        lambda x: calc_pT(x["mom_part_x"], x["mom_part_y"]), axis=1
    )
    flows = urqmd_new.apply(
        lambda x: calc_flows(x["mom_part_x"], x["mom_part_y"]),
        axis=1,
        result_type="expand",
    )
    urqmd_new["v1"] = flows[0]
    urqmd_new["v2"] = flows[1]
    urqmd_new["v3"] = flows[2]
    urqmd_new["sum_mom"] = np.sqrt(
        urqmd_new["mom_x"].apply(np.sum) ** 2 + urqmd_new["mom_y"].apply(np.sum) ** 2
    )
    urqmd_new["rms_mom_T"] = np.sqrt(
        urqmd_new["mom_x"].apply(lambda arr: np.mean(arr**2))
        + urqmd_new["mom_y"].apply(lambda arr: np.mean(arr**2))
    )
    urqmd_new["n_p_ratio"] = urqmd.apply(lambda x: np.sum(x[N]) / np.sum(x[P]), axis=1)
    list_cols = [
        "mom_x",
        "mom_y",
        "mom_z",
        "mom_part_x",
        "mom_part_y",
        "mom_part_z",
        "mom_T_part",
        "coord_x",
        "coord_y",
        "coord_z",
        "coord_part_x",
        "coord_part_y",
        "coord_part_z",
        "energy_part",
        "mom_norm",
        "pseudorapidity_part",
        "pseudorapidity",
    ]
    percentiles = [0.5]
    for col in list_cols:
        urqmd_new[f"{col}_mean"] = urqmd_new[col].apply(np.mean)
        urqmd_new[f"{col}_var"] = urqmd_new[col].apply(np.var)
        urqmd_new[f"{col}_kurtosis"] = urqmd_new[col].apply(stats.kurtosis)
        urqmd_new[f"{col}_skew"] = urqmd_new[col].apply(stats.skew)
        for p in percentiles:
            urqmd_new[f"{col}_{p}_perc"] = urqmd_new[col].apply(
                lambda x: np.percentile(x, p)
            )
    for col in ["psi1", "psi2", "mom_T", "phi"]:
        if col in urqmd_new.columns:
            urqmd_new.drop(columns=col, inplace=True)
    cond_cols = ["B", "v1", "v2", "v3"]
    urqmd[cond_cols] = urqmd_new[cond_cols]
    for p in particles:
        for d in directions:
            urqmd[f"{p}_mom_{d}"] = urqmd.apply(
                lambda x: select_particles(x, f"fParticles.fP{d}", [p]), axis=1
            )
        urqmd[f"{p}_energy"] = urqmd.apply(
            lambda x: select_particles(x, "fParticles.fE", [p]), axis=1
        )
    nucleons_cols = [f"{c}_moms" for c in nucleons if c != "nuclei"]
    nucleons_mask_cols = [f"{c}_mask" for c in nucleons if c != "nuclei"]
    df_rows = []
    for i, row in urqmd.iterrows():
        base_cond = {k: row[k] for k in cond_cols}
        for pion in pions:
            cond = base_cond.copy()
            cond["label"] = label_to_num[pion]
            cond["event"] = i
            mom_x = row[f"{pion}_mom_x"]
            mom_y = row[f"{pion}_mom_y"]
            mom_z = row[f"{pion}_mom_z"]
            energy = row[f"{pion}_energy"]
            for bc in nucleons:
                if bc == "nuclei":
                    continue
                baryon_mom_merged, baryon_mask = get_final_moms(
                    mom_x, mom_y, mom_z, energy, max_size_c
                )
                cond[f"{bc}_moms"] = baryon_mom_merged
                cond[f"{bc}_mask"] = baryon_mask
            target_mom_merged, target_mask = get_final_moms(
                mom_x, mom_y, mom_z, energy, max_size_t
            )
            if target_mask.sum() == 0:
                continue
            cond["target_moms"] = target_mom_merged
            cond["target_mask"] = target_mask
            df_rows.append(cond)
    df = pd.DataFrame(df_rows)
    if return_cols:
        return df, cond_cols, nucleons_cols, nucleons_mask_cols
    return df
