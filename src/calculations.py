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
    Set the random seed.
    Parameters:
    - seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_masked_pt(p, pt_min=0.2, pt_max=None, eta_max=1.0):
    """
    Calculating mask for transverse momenta based on pt and pseudorapidity extreme values
    Parameters:
    - p - tensor of 3-D momenta;
    - pt_min - minimal value of transverse momenta;
    - pt_max - maximum value of transverse momenta;
    - eta_max - maximum value of preusorapidity
    Return:
    - mask;
    - pt - transverse momenta tensor
    """
    px, py, pz = p[..., 0], p[..., 1], p[..., 2]
    pt = torch.sqrt(px**2 + py**2)
    p_total = torch.sqrt(pt**2 + pz**2)
    eta = 0.5 * torch.log((p_total + pz) / (p_total - pz + 1e-10))
    mask = (pt > pt_min) & (torch.abs(eta) < eta_max)
    if pt_max is not None:
        mask = mask & (pt < pt_max)
    return mask, pt


def calc_pt_dist(data, pt_min=0.0, pt_max=1.2, eta_max=10.0):
    """
    Getting transverse momenta tensor for all input events
    Parameters:
    - data - list of momenta tensors (each momenta tensor corresponds to single event);
    - pt_min - minimal value of transverse momenta;
    - pt_max - maximum value of transverse momenta;
    - eta_max - maximum value of preusorapidity
    Return:
    - transverse momenta tansor for all input events
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
    Calculating KL divergence for KDE
    Parameters:
    - real_kde - KDE for real data;
    - fake_kde - KDE for fake data
    - x - values for which KDE is calculated
    - eps - for numerical stability
    Return:
    - KL divergence value
    """
    p = real_kde.evaluate(x)
    q = fake_kde.evaluate(x)
    return simpson(p * np.log((p + eps) / (q + eps)), x)


def w2_distance_1d(real, fake):
    """
    Calculating W2 distance for real and fake data
    Parameters:
    - real - real data np.array;
    - fake - fake data np.array
    Return:
    - W2 value
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
    Calculating particles energy based on momenta amd masses
    Parameters:
    - px - tensor of px momenta componenta;
    - py - tensor of py momenta componenta;
    - pz - tensor of pz momenta componenta;
    - masses - tensor of particles masses
    Return:
    - energies tensor
    """
    return torch.sqrt(px**2 + py**2 + pz**2 + masses**2)


def calc_m_inv(row, mass_number=131):
    """
    Calculating invariant mass for event (as a row of pd.DataFrame)
    Parameters:
    - row - row of pd.DataFrame of events;
    - mass_number - mass number of colliding heavy ions
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
    Selecting particles features from target_col based on mask_cols
    Parameters:
    - row - row of pd.DataFrame of events;
    - target_col - column of pd.Dataframe from which features are selected;
    - mask_cols - columns for joined mask
    Return:
    - array of selected features
    """
    feature_arr = np.array(row[target_col])
    mask_arr = np.zeros_like(feature_arr)
    for col in mask_cols:
        mask_arr += row[col]
    return feature_arr[mask_arr.astype(bool)]


def calc_pseudorapidity(px=None, py=None, pz=None, eps=1e-5):
    """
    Calculating pseudorapidity
    Parameters:
    - px - tensor of px momenta componenta;
    - py - tensor of py momenta componenta;
    - pz - tensor of pz momenta componenta;
    - eps - for numerical stability
    Return:
    - pseudorapidity np.array or torch.Tensor (it depends on input data type)
    """
    if isinstance(px, torch.Tensor):
        norm = torch.sqrt(px**2 + py**2 + pz**2)
        return 0.5 * torch.log(eps + (norm + pz) / (eps + norm - pz))
    else:
        norm = np.sqrt(px**2 + py**2 + pz**2)
        return 0.5 * np.log(eps + (norm + pz) / (eps + norm - pz))


def calc_pT(px, py):
    """
    Calculating transverse momenta
    Parameters:
    - px - np.array of px momenta componenta;
    - py - np.array of py momenta componenta
    Return:
    - transverse momenta np.array
    """
    return np.sqrt(px**2 + py**2)


def calc_flows(px, py, weights=None):
    """
    Calculating v1, v2, v3 flows
    Parameters:
    - px - np.array of px momenta componenta;
    - py - np.array of py momenta componenta;
    - weight - for weighted flows
    Return:
    - v1 - directed flow;
    - v2 - elliptic flow;
    - v3 - triangular flow
    """
    phi = np.arctan2(py, px)
    n = len(phi)
    w = np.ones(n) if weights is None else weights
    sum_w = np.sum(w)
    v1 = np.sum(w * np.cos(phi)) / sum_w
    v2 = np.sum(w * np.cos(2 * phi)) / sum_w
    v3 = np.sum(w * np.cos(3 * phi)) / sum_w
    return v1, v2, v3


def get_nucleons_by_type(status, part_type="spectators"):
    """
    Getting nucleons mask based on status
    Parameters:
    - status - array of particles statuses;
    - part_type - "spectators" of "participants"
    Return:
    - nucleons statuses mask
    """
    if part_type == "spectators":
        return status == 0
    else:
        return status != 0


def get_part_spec(row, target_col, part_type="spectators"):
    """
    Selecting nucleons features from target_col based on their type
    Parameters:
    - row - row of pd.DataFrame of events;
    - target_col - column of pd.Dataframe from which features are selected;
    - part_type - "spectators" of "participants"
    Return:
    - array of selected features
    """
    nucleon_mask = np.array(row[N]) | np.array(row[P])
    nucleon_indices = np.where(nucleon_mask)[0]
    status = np.array(row["fParticles.fStatus"])[nucleon_indices]
    type_mask = get_nucleons_by_type(status, part_type)
    selected_indices = nucleon_indices[type_mask]
    return np.array(row[target_col])[selected_indices]


def get_final_moms(mom_x, mom_y, mom_z, energy, max_size):
    """
    Reshaping momenergy arrays
    Parameters:
    - mom_x - np.array of px momenta componenta;
    - mom_y - np.array of py momenta componenta;
    - mom_z - np.array of pz momenta componenta;
    - enegy - np.array of energies;
    - max_size - fixed size of final array
    Return:
    - mom_merged - reshaped momenergy array;
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
    Preprocessing UrQMD dataset
    Parameters:
    - urqmd - input pd.DataFrame;
    - max_size_t - fixed size of target particles momenergy;
    - max_size_c - fixed size of nucleons particles momenergy;
    - return_cols - flag if condition columns names are returned
    Return:
    - df - final pd.DataFrame;
    - cond_cols - external conidion columns;
    - nucleons momenta columns;
    - nucleons mask columns
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
    cond_cols = ["B"]
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
                    row[f"{bc}_mom_x"],
                    row[f"{bc}_mom_y"],
                    row[f"{bc}_mom_z"],
                    row[f"{bc}_energy"],
                    max_size_c,
                )
                cond[f"{bc}_moms"] = baryon_mom_merged
                cond[f"{bc}_mask"] = baryon_mask
            cond["nucleons_moms"] = np.concatenate(
                [cond["2112_moms"], cond["2212_moms"]], axis=0
            )
            cond["nucleons_mask"] = np.concatenate(
                [cond["2112_mask"], cond["2212_mask"]], axis=0
            )
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
        return df, cond_cols, ["nucleons_moms"], ["nucleons_mask"]
    return df


def calc_vn_vs_pt(
    data, pt_min, pt_max, eta_max, nbins, n, min_particles=15, n_interp=200
):
    """
    Calculating vn(pt) values for plotting
    Parameters:
    - data - list of momenta tensors (each momenta tensor corresponds to single event);
    - pt_min - minimal value of transverse momenta;
    - pt_max - maximum value of transverse momenta;
    - eta_max - maximum value of preusorapidity;
    - nbins - number of vn(pt) bins in actual data;
    - n - order of flow vn;
    - min_particles - minimal count of particles for calculating mean and std;
    - n_interp - number of interpolating points
    Return:
    - p_interp - transverse momenta values;
    - vn_interp - flow values;
    - err_interp - error values;
    - is_interp - flag if value is interpolated
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
    vn_vals = np.full(nbins, np.nan)
    err_vals = np.full(nbins, np.nan)
    for i in range(nbins):
        mask_bin = (pt_all >= bin_edges[i]) & (pt_all < bin_edges[i + 1])
        count = np.sum(mask_bin)
        if count >= min_particles:
            cos_nphi = np.cos(n * phi_all[mask_bin])
            vn_vals[i] = np.mean(cos_nphi)
            err_vals[i] = np.std(cos_nphi) / np.sqrt(count)
    p_interp = np.linspace(pt_min, pt_max, n_interp)
    valid = ~np.isnan(vn_vals)
    if np.sum(valid) < 2:
        vn_interp = np.zeros_like(p_interp)
        err_interp = np.full_like(p_interp, 1.0)
        is_interp = np.ones_like(p_interp, dtype=bool)
        return p_interp, vn_interp, err_interp, is_interp
    f_vn = interp1d(
        bin_centers[valid],
        vn_vals[valid],
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    f_err = interp1d(
        bin_centers[valid],
        err_vals[valid],
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    vn_interp = f_vn(p_interp)
    err_interp = f_err(p_interp)
    bin_width = bin_edges[1] - bin_edges[0]
    is_interp = np.ones_like(p_interp, dtype=bool)
    for j, p_j in enumerate(p_interp):
        dist = np.min(np.abs(bin_centers[valid] - p_j))
        if dist <= bin_width * 0.6:
            is_interp[j] = False
    mean_err = np.nanmean(err_vals[valid])
    err_interp[is_interp] = np.maximum(err_interp[is_interp], 2.0 * mean_err)
    return p_interp, vn_interp, err_interp, is_interp
