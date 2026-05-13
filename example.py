import gdown
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import os
import warnings
import uproot
import numpy as np
import pandas as pd
from src.calculations import preprocess_urqmd, set_seed
from src.config import *
from src.training import train_epoch_transformer, train_and_test
from src.inference import generate_transformer
from src.pion_gan import Pion_Gan_Discriminator, Pion_Gan_Generator


def main():
    warnings.filterwarnings("ignore")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    latent_dim = 32
    latent_local_dim = 32
    hid_dim = 64
    num_classes = len(pions)
    feat_dim = 3
    num_layers_gen = 5
    num_layers_disc = 5
    lr = 1e-4
    betas = (0.5, 0.999)
    n_modes = 3
    n_heads = 8
    d_iters = 1
    epochs = 70
    METRICS = {}

    if not os.path.exists("xexe_eos_1_bmn.root"):
        gdown.download(
            "https://drive.google.com/uc?id=1EhDJ0DSe1AuHNRxUIbQV5r8AXo4TP8Kp"
        )
    if not os.path.exists("xexe_urqmd_5fm.root"):
        gdown.download(
            "https://drive.google.com/uc?id=1mBNT9X2qJRgRFHAhRIfv6kFSHQRnbZ4T"
        )
    urqmd_1_7 = uproot.open("xexe_eos_1_bmn.root:events").arrays(library="pd")
    urqmd_1_7_inv = urqmd_1_7.copy()
    urqmd_1_7_inv["fParticles.fPx"] = urqmd_1_7_inv["fParticles.fPx"].apply(
        lambda x: (-np.array(x)).tolist()
    )
    urqmd_1_7_inv["fParticles.fPy"] = urqmd_1_7_inv["fParticles.fPy"].apply(
        lambda x: (-np.array(x)).tolist()
    )
    urqmd_1_7 = pd.concat([urqmd_1_7, urqmd_1_7_inv], ignore_index=True)
    urqmd_5 = uproot.open("xexe_urqmd_5fm.root:events").arrays(library="pd")
    seed = seeds[0]
    set_seed(seed)
    ratio = 1
    train_n_1_7 = int(0.8 * len(urqmd_1_7))
    val_n_1_7 = (len(urqmd_1_7) - train_n_1_7) // 2
    train_n_5 = ratio * train_n_1_7
    val_n_5 = ratio * val_n_1_7
    ids_1_7 = np.random.permutation(len(urqmd_1_7))
    ids_5 = np.random.permutation(len(urqmd_5))
    urqmd_train_1_7 = urqmd_1_7.iloc[ids_1_7[:train_n_1_7]]
    urqmd_val_1_7 = urqmd_1_7.iloc[ids_1_7[train_n_1_7 : train_n_1_7 + val_n_1_7]]
    urqmd_test_1_7 = urqmd_1_7.iloc[
        ids_1_7[train_n_1_7 + val_n_1_7 : train_n_1_7 + val_n_1_7 * 2]
    ]
    urqmd_train_5 = urqmd_5.iloc[ids_5[:train_n_5]]
    urqmd_val_5 = urqmd_5.iloc[ids_5[train_n_5 : train_n_5 + val_n_5]]
    urqmd_test_5 = urqmd_5.iloc[ids_5[train_n_5 + val_n_5 : train_n_5 + val_n_5 * 2]]
    urqmd_train = pd.concat([urqmd_train_1_7, urqmd_train_5], ignore_index=True)
    urqmd_val = pd.concat([urqmd_val_1_7, urqmd_val_5], ignore_index=True)
    COUNT = {}
    COUNT_2 = {}
    for code in particles:
        selected_count = urqmd_train.apply(
            lambda x: (np.array(x["fParticles.fPdg"]) == code).sum(), axis=1
        ).values
        final_count = (
            int(np.percentile(selected_count, 0.95))
            if code in pions
            else selected_count.max()
        )
        mean_ = np.mean(selected_count)
        std_ = np.std(selected_count)
        if final_count == 0:
            final_count = int(selected_count.max())
        COUNT[code] = final_count
        COUNT_2[code] = {"mean": mean_, "std": std_}
    meson_count_max = max(
        {k: v for k, v in COUNT.items() if k not in nucleons}.values()
    )
    nucleon_count_max = max({k: v for k, v in COUNT.items() if k in nucleons}.values())
    train, cond_cols, nucl_cols, nucl_mask_cols = preprocess_urqmd(
        urqmd_train,
        max_size_t=meson_count_max,
        max_size_c=nucleon_count_max,
        return_cols=True,
    )
    val = preprocess_urqmd(
        urqmd_val, max_size_t=meson_count_max, max_size_c=nucleon_count_max
    )
    test_1_7 = preprocess_urqmd(
        urqmd_test_1_7, max_size_t=meson_count_max, max_size_c=nucleon_count_max
    )
    test_5 = preprocess_urqmd(
        urqmd_test_5, max_size_t=meson_count_max, max_size_c=nucleon_count_max
    )
    means = {}
    stds = {}
    for col in cond_cols:
        means[col] = train[col].mean()
        stds[col] = train[col].std()
        train[col] = (train[col] - means[col]) / stds[col]
        val[col] = (val[col] - means[col]) / stds[col]
        test_1_7[col] = (test_1_7[col] - means[col]) / stds[col]
        test_5[col] = (test_5[col] - means[col]) / stds[col]
    train_cond = train[cond_cols].values.astype(float)
    train_nucl = list(train[nucl_cols[0]].values)
    train_nucl_mask = list(train[nucl_mask_cols[0]].values)
    train_label = train["label"].values
    train_event = train["event"].values
    train_mom = list(train["target_moms"].values)
    train_mask = list(train["target_mask"].values)
    val_cond = val[cond_cols].values.astype(float)
    val_nucl = list(val[nucl_cols[0]].values)
    val_nucl_mask = list(val[nucl_mask_cols[0]].values)
    val_label = val["label"].values
    val_event = val["event"].values
    val_mom = list(val["target_moms"].values)
    val_mask = list(val["target_mask"].values)
    test_1_7_cond = test_1_7[cond_cols].values.astype(float)
    test_1_7_nucl = list(test_1_7[nucl_cols[0]].values)
    test_1_7_nucl_mask = list(test_1_7[nucl_mask_cols[0]].values)
    test_1_7_label = test_1_7["label"].values
    test_1_7_event = test_1_7["event"].values
    test_1_7_mom = list(test_1_7["target_moms"].values)
    test_1_7_mask = list(test_1_7["target_mask"].values)
    test_5_cond = test_5[cond_cols].values.astype(float)
    test_5_nucl = list(test_5[nucl_cols[0]].values)
    test_5_nucl_mask = list(test_5[nucl_mask_cols[0]].values)
    test_5_label = test_5["label"].values
    test_5_event = test_5["event"].values
    test_5_mom = list(test_5["target_moms"].values)
    test_5_mask = list(test_5["target_mask"].values)
    urqmd_train_dataset = TensorDataset(
        torch.tensor(train_mom, dtype=torch.float32),
        torch.tensor(train_cond, dtype=torch.float32),
        torch.tensor(train_label),
        torch.tensor(train_mask, dtype=torch.float32),
    )
    urqmd_train_dataloader = DataLoader(
        urqmd_train_dataset, batch_size=batch_size, shuffle=True
    )
    num_classes = len(pions)
    extern_cond_d = len(cond_cols)
    G = Pion_Gan_Generator(
        latent_global=latent_dim,
        latent_local=latent_local_dim,
        num_labels=num_classes,
        feats=feat_dim,
        max_len=meson_count_max,
        d_model=hid_dim,
        n_heads=n_heads,
        extern_cond_d=extern_cond_d,
        d_ff=hid_dim * 2,
        n_modes=n_modes,
        n_layers=num_layers_gen,
        use_learnable_noise=False,
        #  use_weight_norm=True
    ).to(device)
    D = Pion_Gan_Discriminator(
        num_labels=num_classes,
        feats=feat_dim,
        d_model=hid_dim,
        n_heads=n_heads,
        d_ff=hid_dim * 2,
        extern_cond_d=extern_cond_d,
        n_layers=num_layers_disc,
        #  use_weight_norm=True
    ).to(device)

    opt_g = optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_d = optim.Adam(D.parameters(), lr=lr, betas=betas)

    G, G_shadow, D = train_and_test(
        G=G,
        D=D,
        train_loader=urqmd_train_dataloader,
        optim_G=opt_g,
        optim_D=opt_d,
        device=device,
        mode="lsgan",
        d_iters=d_iters,
        epochs=epochs,
        train_epoch_func=train_epoch_transformer,
        generate_func=generate_transformer,
        val_mom_obj={0: val_mom},
        val_cond_obj={0: val_cond},
        val_label_obj={0: val_label},
        val_mask_obj={0: val_mask},
        val_nucl_mask_obj={0: val_nucl_mask},
        val_nucl_obj={0: val_nucl},
        val_event_obj={0: val_event},
        test_mom_obj={1.7: test_1_7_mom, 5: test_5_mom},
        test_cond_obj={1.7: test_1_7_cond, 5: test_5_cond},
        test_label_obj={1.7: test_1_7_label, 5: test_5_label},
        test_mask_obj={1.7: test_1_7_mask, 5: test_5_mask},
        test_nucl_mask_obj={1.7: test_1_7_nucl_mask, 5: test_5_nucl_mask},
        test_nucl_obj={1.7: test_1_7_nucl, 5: test_5_nucl},
        test_event_obj={1.7: test_1_7_event, 5: test_5_event},
        metrics_mode_name=G.__class__.__name__,
        final_metrics=METRICS,
        seeds=[seed],
        num_classes=num_classes,
        meson_count_max=meson_count_max,
        use_shadow=True,
        ema_beta=0.999,
        model_name="pion_gan",
    )


if __name__ == "__main__":
    main()
