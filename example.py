import gdown
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser
import os
import uproot
import numpy as np
import pandas as pd
from src.preprocessing import preprocess_urqmd
from src.config import *
from src.training import train_epoch_transformer, train_epoch_epic
from src.inference import generate_transformer, get_dist_metrics, get_energy_metrics
from src.transformer import EPiC_Transformer_Discriminator, EPiC_Transformer_Generator


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = ArgumentParser()
    parser.add_argument("latent_dim", type=int, default=32)
    parser.add_argument("latent_local_dim", type=int, default=32)
    parser.add_argument("d_ff", type=int, default=64)
    parser.add_argument("d_model", type=int, default=128)
    parser.add_argument("n_heads", type=int, default=8)
    parser.add_argument("num_layers_gen", type=int, default=5)
    parser.add_argument("num_layers_disc", type=int, default=5)
    parser.add_argument("lr", type=float, default=1e-4)
    parser.add_argument("beta_1", type=float, default=0.5)
    parser.add_argument("beta_2", type=float, default=0.999)
    parser.add_argument("batch_size", type=int, default=128)
    parser.add_argument("n_modes", type=int, default=2)
    parser.add_argument("d_iters", type=int, default=1)
    parser.add_argument("epochs", type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists('xexe_eos_1_bmn.root'):
        gdown.download('https://drive.google.com/uc?id=1EhDJ0DSe1AuHNRxUIbQV5r8AXo4TP8Kp')
    if not os.path.exists('xexe_urqmd_5fm.root'):
        gdown.download('https://drive.google.com/uc?id=1mBNT9X2qJRgRFHAhRIfv6kFSHQRnbZ4T')
    urqmd_1_7 = uproot.open("xexe_eos_1_bmn.root:events").arrays(library="pd")
    urqmd_5 = uproot.open("xexe_urqmd_5fm.root:events").arrays(library="pd")
    n_events_by_b = int(len(urqmd_1_7) * 0.8)
    ids_1_7 = list(range(n_events_by_b))
    ids_5 = list(range(n_events_by_b))
    np.random.shuffle(ids_1_7)
    np.random.shuffle(ids_5)
    urqmd = pd.concat([urqmd_1_7.iloc[ids_1_7], urqmd_5.iloc[ids_5]], ignore_index=True)
    urqmd_1_7_test = urqmd_1_7.iloc[~np.isin(np.arange(len(urqmd_1_7)), ids_1_7)]
    urqmd_5_test = urqmd_5.iloc[~np.isin(np.arange(len(urqmd_5)), ids_5)].head(len(urqmd_1_7) - n_events_by_b)
    COUNT = {}
    for code in particles:
        selected_count = urqmd.apply(lambda x: (np.array(x['fParticles.fPdg']) == code).sum(), axis=1).values
        final_count = int(np.percentile(selected_count, 0.95))
        if final_count == 0:
            final_count = int(selected_count.max())
        COUNT[code] = final_count
    meson_count_max = max({k: v for k, v in COUNT.items() if k not in nucleons}.values())
    nucleon_count_max = max({k: v for k, v in COUNT.items() if k in nucleons}.values())
    train, cond_cols, nucl_cols, nucl_mask_cols = preprocess_urqmd(urqmd, max_size_t=meson_count_max, max_size_c=nucleon_count_max, return_cols=True)
    test_1_7 = preprocess_urqmd(urqmd_1_7_test, max_size_t=meson_count_max, max_size_c=nucleon_count_max)
    test_5 = preprocess_urqmd(urqmd_5_test, max_size_t=meson_count_max, max_size_c=nucleon_count_max)
    means = {}
    stds = {}
    for col in cond_cols:
        means[col] = train[col].mean()
        stds[col] = train[col].std()
        train[col] = (train[col] - means[col]) / stds[col]
        test_1_7[col] = (test_1_7[col] - means[col]) / stds[col]
        test_5[col] = (test_5[col] - means[col]) / stds[col]
    train_cond = train[cond_cols].values.astype(float)
    train_nucl = list(train[nucl_cols].values)
    train_nucl_mask = list(train[nucl_mask_cols].values)
    train_label = train['label'].values 
    train_event = train['event'].values
    train_mom = list(train['target_moms'].values)
    train_mask = list(train['target_mask'].values)
    test_1_7_cond = test_1_7[cond_cols].values.astype(float)
    test_1_7_nucl = list(test_1_7[nucl_cols].values)
    test_1_7_nucl_mask = list(test_1_7[nucl_mask_cols].values)
    test_1_7_label = test_1_7['label'].values
    test_1_7_event = test_1_7['event'].values
    test_1_7_mom = list(test_1_7['target_moms'].values)
    test_1_7_mask = list(test_1_7['target_mask'].values)
    test_5_cond = test_5[cond_cols].values.astype(float)
    test_5_nucl = list(test_5[nucl_cols].values)
    test_5_nucl_mask = list(test_5[nucl_mask_cols].values)
    test_5_label = test_5['label'].values
    test_5_event = test_5['event'].values
    test_5_mom = list(test_5['target_moms'].values)
    test_5_mask = list(test_5['target_mask'].values)
    urqmd_train_dataset = TensorDataset(torch.tensor(train_mom, dtype=torch.float32),
                                        torch.tensor(train_cond, dtype=torch.float32),
                                        torch.tensor(train_label),
                                        torch.tensor(train_mask, dtype=torch.float32))
    urqmd_train_dataloader = DataLoader(urqmd_train_dataset, batch_size=args.batch_size, shuffle=True)
    
    G = EPiC_Transformer_Generator(
        latent_global=args.latent_dim,
        latent_local=args.latent_local_dim,
        num_labels=num_classes,
        feats=feat_dim,
        max_len=meson_count_max,
        d_model=args.d_model,
        n_heads=args.n_heads,
        extern_cond_d=extern_cond_d,
        d_ff=args.d_ff,
        n_modes=args.n_modes,
        n_layers=args.num_layers_gen
    ).to(device)

    D = EPiC_Transformer_Discriminator(
        num_labels=num_classes,
        feats=feat_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        extern_cond_d=extern_cond_d,
        n_layers=args.num_layers_disc
    ).to(device)

    opt_g = optim.Adam(G.parameters(), lr=lr, betas=(args.beta_1, args.beta_2))
    opt_d = optim.Adam(D.parameters(), lr=lr, betas=(args.beta_1, args.beta_2))

    losses_my = train_model(
        G=G, D=D,
        dataloader=urqmd_train_dataloader,
        optim_G=opt_g, optim_D=opt_d,
        device=device,
        mode='lsgan',
        d_iters=args.d_iters,
        epochs=args.epochs,
        train_epoch_func=train_epoch_transformer,
        generate_func=generate_transformer,
        test_mom=test_1_7_mom,
        test_cond=test_1_7_cond,
        test_label=test_1_7_label,
        test_mask=test_1_7_mask,
        num_classes=num_classes,
        meson_count_max=meson_count_max
    )

    get_dist_metrics(G, generate_my, test_1_7_mom,
                     test_1_7_nucl, test_1_7_nucl_mask, test_1_7_event,
                     test_1_7_cond, test_1_7_label, test_1_7_mask)
    get_energy_metrics(G, generate_my, test_1_7_mom,
                       test_1_7_nucl, test_1_7_nucl_mask, test_1_7_event,
                       test_1_7_cond, test_1_7_label, test_1_7_mask)
