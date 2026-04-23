import numpy as np
import torch
import pandas as pd
from scipy import stats
from src.settings import *


def calc_energy(px, py, pz, masses):
    return torch.sqrt(px**2 + py**2 + pz**2 + masses**2)


def calc_s_nn(row, mass_number = 131):
    energy = np.sum(row['fParticles.fE'])
    px = np.sum(row['fParticles.fPx'])
    py = np.sum(row['fParticles.fPy'])
    pz = np.sum(row['fParticles.fPz'])
    M_inv = np.sqrt(np.maximum(0, energy ** 2 - (px ** 2 + py ** 2 + pz ** 2)))
    return M_inv / mass_number


def select_particles(row, target_col, mask_cols):
    feature_arr = np.array(row[target_col])
    mask_arr = np.zeros_like(feature_arr)
    for col in mask_cols:
        mask_arr += row[col]
    return feature_arr[mask_arr.astype(bool)]


def calc_pseudorapidity(px=None, py=None, pz=None, eps=1e-5):
    norm = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    return 0.5 * np.log(eps + (norm + pz) / (eps + norm - pz))


def calc_pT(px, py):
    return np.sqrt(px ** 2 + py ** 2)


def calc_flows(px, py, weights=None):
    phi = np.arctan2(py, px)
    n = len(phi)
    w = np.ones(n) if weights is None else weights
    sum_w = np.sum(w)
    v1 = np.sum(w * np.cos(phi)) / sum_w
    v2 = np.sum(w * np.cos(2 * phi)) / sum_w
    v3 = np.sum(w * np.cos(3 * phi)) / sum_w
    return v1, v2, v3


def get_nucleons_by_type(px, py, pz, part_type='spectators', threshold=2.):
    eta = calc_pseudorapidity(px, py, pz)
    if part_type == 'spectators':
        return eta > threshold
    else:
        return eta <= threshold


def get_part_spec(row, target_col, part_type='spectators'):
    nucleon_mask = np.array(row[N]) | np.array(row[P])
    nucleon_indices = np.where(nucleon_mask)[0]
    px = np.array(row['fParticles.fPx'])[nucleon_indices]
    py = np.array(row['fParticles.fPy'])[nucleon_indices]
    pz = np.array(row['fParticles.fPz'])[nucleon_indices]
    type_mask = get_nucleons_by_type(px, py, pz, part_type)
    selected_indices = nucleon_indices[type_mask]
    return np.array(row[target_col])[selected_indices]


def get_final_moms(mom_x, mom_y, mom_z, energy, max_size):
    if mom_x.shape[0] >= max_size:
        mask = np.ones(max_size, dtype=float)
        mom_x = mom_x[:max_size]
        mom_y = mom_y[:max_size]
        mom_z = mom_z[:max_size]
        energy = energy[:max_size]
        mom_merged = np.vstack((mom_x, mom_y, mom_z, energy)).T
    else:
        mask = np.zeros(max_size)
        mask[:len(mom_x)] = 1.0
        zeros_x = np.zeros(max_size)
        zeros_x[:len(mom_x)] = mom_x
        zeros_y = np.zeros(max_size)
        zeros_y[:len(mom_y)] = mom_y
        zeros_z = np.zeros(max_size)
        zeros_z[:len(mom_z)] = mom_z
        zeros_e = np.zeros(max_size)
        zeros_e[:len(energy)] = energy
        mom_merged = np.vstack((zeros_x, zeros_y, zeros_z, zeros_e)).T
    return mom_merged, mask


def preprocess_urqmd(urqmd, max_size_t=80, max_size_c=250, return_cols=False):
    urqmd_new = pd.DataFrame()
    if 'fB' in urqmd.columns:
        urqmd_new['B'] = urqmd['fB']
    urqmd_new['s_nn'] = urqmd.apply(calc_s_nn, axis=1)
    for p in particles:
        urqmd[p] = urqmd['fParticles.fPdg'].apply(lambda x: np.array(x) == p)
    for d in directions:
        urqmd_new[f'mom_{d}'] = urqmd.apply(lambda x: select_particles(x, f'fParticles.fP{d}', nucleons), axis=1)
        urqmd_new[f'coord_{d}'] = urqmd.apply(lambda x: select_particles(x, f'fParticles.f{d.upper()}', nucleons), axis=1)
        urqmd_new[f'mom_part_{d}'] = urqmd.apply(lambda x: get_part_spec(x, f'fParticles.fP{d}', 'participants'), axis=1)
        urqmd_new[f'coord_part_{d}'] = urqmd.apply(lambda x: get_part_spec(x, f'fParticles.f{d.upper()}', 'participants'), axis=1)
    urqmd_new['energy_part'] = urqmd.apply(lambda x: get_part_spec(x, 'fParticles.fE', 'participants'), axis=1)
    urqmd_new['energy'] = urqmd.apply(lambda x: select_particles(x, 'fParticles.fE', nucleons), axis=1)
    urqmd_new['mom_norm'] = urqmd_new.apply(lambda x: np.sqrt(x['mom_x'] ** 2 + x['mom_y'] ** 2 + x['mom_z'] ** 2), axis=1)
    urqmd_new['pseudorapidity'] = urqmd_new.apply(lambda x: calc_pseudorapidity(px=x["mom_x"], py=x["mom_y"], pz=x["mom_z"]), axis=1)
    urqmd_new['pseudorapidity_part'] = urqmd_new.apply(lambda x: calc_pseudorapidity(px=x["mom_part_x"], py=x["mom_part_y"], pz=x["mom_part_z"]), axis=1)
    urqmd_new['mom_T_part'] = urqmd_new.apply(lambda x: calc_pT(x["mom_part_x"], x["mom_part_y"]), axis=1)
    flows = urqmd_new.apply(lambda x: calc_flows(x['mom_part_x'], x['mom_part_y']), axis=1, result_type='expand')
    urqmd_new['v1'] = flows[0]
    urqmd_new['v2'] = flows[1]
    urqmd_new['v3'] = flows[2]
    urqmd_new['sum_mom'] = np.sqrt(urqmd_new['mom_x'].apply(np.sum) ** 2 + urqmd_new['mom_y'].apply(np.sum) ** 2)
    urqmd_new['rms_mom_T'] = np.sqrt(urqmd_new['mom_x'].apply(lambda arr: np.mean(arr ** 2)) + urqmd_new['mom_y'].apply(lambda arr: np.mean(arr ** 2)))
    urqmd_new['n_p_ratio'] = urqmd.apply(lambda x: np.sum(x[N]) / np.sum(x[P]), axis=1)
    list_cols = [
        'mom_x',
        'mom_y',
        'mom_z',
        'mom_part_x',
        'mom_part_y',
        'mom_part_z',
        'mom_T_part',
        'coord_x',
        'coord_y',
        'coord_z',
        'coord_part_x',
        'coord_part_y',
        'coord_part_z',
        'energy_part',
        'mom_norm',
        'pseudorapidity_part',
        'pseudorapidity'
    ]
    percentiles = [0.5]
    for col in list_cols:
        urqmd_new[f'{col}_mean'] = urqmd_new[col].apply(np.mean)
        urqmd_new[f'{col}_var'] = urqmd_new[col].apply(np.var)
        urqmd_new[f'{col}_kurtosis'] = urqmd_new[col].apply(stats.kurtosis)
        urqmd_new[f'{col}_skew'] = urqmd_new[col].apply(stats.skew)
        for p in percentiles:
            urqmd_new[f'{col}_{p}_perc'] = urqmd_new[col].apply(lambda x: np.percentile(x, p))
    for col in ['psi1', 'psi2', 'mom_T', 'phi']:
        if col in urqmd_new.columns:
            urqmd_new.drop(columns=col, inplace=True)
    cond_cols = [col for col in urqmd_new.columns if col not in list_cols]
    urqmd[cond_cols] = urqmd_new[cond_cols]
    for p in particles:
        for d in directions:
            urqmd[f'{p}_mom_{d}'] = urqmd.apply(
                lambda x: select_particles(x, f'fParticles.fP{d}', [p]), axis=1)
        urqmd[f'{p}_energy'] = urqmd.apply(
            lambda x: select_particles(x, 'fParticles.fE', [p]), axis=1)
    nucleons_cols = [f'{c}_moms' for c in nucleons if c != 'nuclei']
    nucleons_mask_cols = [f'{c}_mask' for c in nucleons if c != 'nuclei']
    df_rows = []
    for i, row in urqmd.iterrows():
        base_cond = {k: row[k] for k in cond_cols}
        for pion in pions:
            cond = base_cond.copy()
            cond['label'] = label_to_num[pion]
            cond['event'] = i
            mom_x = row[f'{pion}_mom_x']
            mom_y = row[f'{pion}_mom_y']
            mom_z = row[f'{pion}_mom_z']
            energy = row[f'{pion}_energy']
            for bc in nucleons:
                if bc == 'nuclei':
                    continue
                baryon_mom_merged, baryon_mask = get_final_moms(
                    mom_x, mom_y, mom_z, energy, max_size_c)
                cond[f'{bc}_moms'] = baryon_mom_merged
                cond[f'{bc}_mask'] = baryon_mask
            target_mom_merged, target_mask = get_final_moms(
                mom_x, mom_y, mom_z, energy, max_size_t)
            if target_mask.sum() == 0:
                continue
            cond['target_moms'] = target_mom_merged
            cond['target_mask'] = target_mask
            df_rows.append(cond)
    df = pd.DataFrame(df_rows)
    if return_cols:
        return df, cond_cols, nucleons_cols, nucleons_mask_cols
    return df