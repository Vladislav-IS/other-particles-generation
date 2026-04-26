import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
from src.calculations import set_seed
from src.inference import get_dist_metrics, get_energy_metrics
from src.plots import plot_kde, plot_scatter, plot_vn, plot_losses
from src.config import *


def compute_gradient_penalty(
    discriminator, real_data, fake_data, cond, labels, mask_real, mask_fake
):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=real_data.device)
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated.requires_grad_(True)
    combined_mask = (mask_real.bool() | mask_fake.bool()).float()
    d_interpolated = discriminator(interpolated, labels, cond, combined_mask)
    grad = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad * combined_mask.unsqueeze(-1)
    grad = grad.view(batch_size, -1)
    grad_norm = grad.norm(2, dim=1)
    gp = ((grad_norm - 1) ** 2).mean()
    return gp


def wgan_gp_loss(
    d_real,
    d_fake,
    real_data,
    fake_data,
    discriminator,
    cond,
    labels,
    mask_real,
    mask_fake,
    lambda_gp=10,
):
    gp = compute_gradient_penalty(
        discriminator, real_data, fake_data, cond, labels, mask_real, mask_fake
    )
    d_loss = d_fake.mean() - d_real.mean() + lambda_gp * gp
    g_loss = -d_fake.mean()
    return d_loss, g_loss


def update_average(model_tgt, model_src, beta):
    with torch.no_grad():
        params_src = dict(model_src.named_parameters())
        params_tgt = dict(model_tgt.named_parameters())
        for k in params_src.keys():
            params_tgt[k].data.copy_(
                torch.lerp(params_tgt[k].data, params_src[k].data, 1.0 - beta)
            )


def train_epoch_transformer(
    G,
    D,
    dataloader,
    optim_G,
    optim_D,
    device,
    mode="lsgan",
    d_iters=1,
    lambda_mask=1.0,
    lambda_len=1.0,
    clip_val=1.0,
    use_shadow=False,
    G_shadow=None,
    ema_beta=0.5,
):
    """
    Function for EPiC transformer training (1 epoch).
    Parameters:
    - G - generator;
    - D - discriminator;
    - dataloader - train dataloader;
    - optim_G - generator optimizer;
    - optim_D - discriminator optimizer;
    - device - "cuda" or "cpu";
    - mode - "lsgan", other modes are unused;
    - d_iters - number of iteration steps for discriminator;
    - lambda_mask - weight of mask_loss;
    - lambda_len - weight of length loss;
    - clip_val - clipping value.
    Return:
    - losses dictionary
    """
    G.train()
    D.train()
    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    epoch_mask_loss = 0.0
    epoch_len_loss = 0.0
    num_batches = 0
    for real_impulses, cond, labels, mask in dataloader:
        real_impulses = real_impulses[..., :-1].to(device)
        cond = cond.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        batch_size, max_len, feats = real_impulses.shape
        for _ in range(d_iters):
            optim_D.zero_grad()
            d_real = D(real_impulses, labels, cond, mask)
            with torch.no_grad():
                fake_impulses, fake_mask = G(labels, cond)
            d_fake = D(fake_impulses * fake_mask.unsqueeze(-1), labels, cond, fake_mask)
            if mode == "lsgan":
                d_real_mse = F.mse_loss(d_real, torch.ones(batch_size, device=device))
                d_fake_mse = F.mse_loss(d_fake, torch.zeros(batch_size, device=device))
                d_loss = (d_real_mse + d_fake_mse) / 2
            else:
                d_loss, _ = wgan_gp_loss(
                    d_real,
                    d_fake,
                    real_impulses,
                    fake_impulses,
                    D,
                    cond,
                    labels,
                    mask,
                    fake_mask,
                )
            d_loss.backward()
            nn.utils.clip_grad_norm_(D.parameters(), clip_val)
            optim_D.step()
        optim_G.zero_grad()
        fake_impulses, fake_mask = G(labels, cond)
        d_fake = D(fake_impulses * fake_mask.unsqueeze(-1), labels, cond, fake_mask)
        if mode == "lsgan":
            g_loss = F.mse_loss(d_fake, torch.ones(batch_size, device=device))
        else:
            g_loss = -d_fake.mean()
        mask_loss = F.binary_cross_entropy(fake_mask, mask)
        real_len = mask.sum(dim=1)
        fake_len = fake_mask.sum(dim=1)
        len_loss = F.mse_loss(fake_len, real_len)
        total_g_loss = g_loss + lambda_mask * mask_loss + lambda_len * len_loss
        total_g_loss.backward()
        nn.utils.clip_grad_norm_(G.parameters(), clip_val)
        optim_G.step()
        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()
        epoch_mask_loss += mask_loss.item()
        epoch_len_loss += len_loss.item()
        num_batches += 1
        if use_shadow and G_shadow is not None:
            update_average(G_shadow, G, ema_beta)
    return {
        "d_loss": epoch_d_loss / num_batches,
        "g_loss": epoch_g_loss / num_batches,
        "mask_loss": epoch_mask_loss / num_batches,
        "len_loss": epoch_len_loss / num_batches,
    }


def train_epoch_epic(
    G,
    D,
    dataloader,
    optim_G,
    optim_D,
    device,
    mode="lsgan",
    d_iters=1,
    zg_dim=128,
    zl_dim=16,
    clip_val=1.0,
    use_shadow=False,
    G_shadow=None,
    ema_beta=0.5
):
    """
    Function for EPiC transformer training (1 epoch).
    Parameters:
    - G - generator;
    - D - discriminator;
    - dataloader - train dataloader;
    - optim_G - generator optimizer;
    - optim_D - discriminator optimizer;
    - device - "cuda" or "cpu";
    - mode - "lsgan", other modes are unused;
    - d_iters - number of iteration steps for discriminator;
    - zg_dim - global noise dimensionalty;
    - zl_dim - local noise dimensionalty;
    - clip_val - clipping value.
    Return:
    - losses dictionary
    """
    G.train()
    D.train()
    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    num_batches = 0
    for real_impulses, cond, labels, mask in dataloader:
        real_impulses = real_impulses[..., :3].to(device)
        cond = cond.to(device)
        labels = labels.to(device)
        batch_size, n_points, _ = real_impulses.shape
        for _ in range(d_iters):
            optim_D.zero_grad()
            d_real = D(real_impulses, cond, labels)
            z_global = torch.randn(batch_size, zg_dim, device=device)
            z_local = torch.randn(batch_size, n_points, zl_dim, device=device)
            with torch.no_grad():
                fake_impulses = G(z_global, z_local, cond, labels)
            d_fake = D(fake_impulses.detach(), cond, labels)
            if mode == "lsgan":
                d_loss = 0.5 * (
                    F.mse_loss(d_real, torch.ones_like(d_real))
                    + F.mse_loss(d_fake, torch.zeros_like(d_fake))
                )
            else:
                d_label_real = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
                d_label_fake = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)
                d_label_cat = torch.cat((d_label_real, d_label_fake), dim=0)
                out_cat = torch.cat((real_impulses, fake_impulses), dim=0)
                discr_out_cat = D(out_cat)
                d_loss = F.binary_cross_entropy_with_logits(discr_out_cat, d_label_cat)
            d_loss.backward()
            clip_grad_norm_(D.parameters(), clip_val)
            optim_D.step()
        optim_G.zero_grad()
        z_global = torch.randn(batch_size, zg_dim, device=device)
        z_local = torch.randn(batch_size, n_points, zl_dim, device=device)
        fake_impulses = G(z_global, z_local, cond, labels)
        d_fake = D(fake_impulses, cond, labels)
        if mode == "lsgan":
            g_loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
        else:
            d_label = torch.full((batch_size,1), 1, dtype=torch.float, device=device)
            discr_out = D(fake_impulses)
            g_loss = F.binary_cross_entropy_with_logits(discr_out, d_label)
        g_loss.backward()
        clip_grad_norm_(G.parameters(), clip_val)
        optim_G.step()
        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()
        num_batches += 1
        if use_shadow and G_shadow is not None:
            update_average(G_shadow, G, ema_beta)
    return {"d_loss": epoch_d_loss / num_batches, "g_loss": epoch_g_loss / num_batches}


def train_model(
    G,
    D,
    dataloader,
    optim_G,
    optim_D,
    device,
    mode,
    d_iters,
    epochs,
    train_epoch_func,
    generate_func,
    test_mom,
    test_cond,
    test_label,
    test_mask,
    num_classes,
    meson_count_max,
    scheduler_g=None,
    scheduler_d=None,
    use_shadow=False,
    ema_beta=0.5,
    **kwargs
):
    """
    Unified function for training.
    Parameters:
    - G - generator;
    - D - discriminator;
    - dataloader - train dataloader;
    - optim_G - generator optimizer;
    - optim_D - discriminator optimizer;
    - device - "cuda" or "cpu";
    - d_iters - number of iteration steps for discriminator;
    - epochs - number of training epochs;
    - train_epoch_func - model-specific training function;
    - generate_fucn - model-specific generation function;
    - test_mom - real momenta from test subset;
    - test_cond - external condition from test subset;
    - test_label - particle types from test subset;
    - test_mask - padding masks from test subset;
    - num_classes - number of particles types;
    - meson_count_max - maximum length of padded particles sequences;
    - scheduler_g - generator scheduler;
    - scheduler_d - discriminator scheduler;
    - kwargs - for model-specific training and generation function.
    Return:
    - losses dictionary
    """
    if use_shadow:
        G_shadow = copy.deepcopy(G).eval()
        update_average(beta=0.0)
    else:
        G_shadow = None
    losses = {}
    for epoch in range(epochs):
        epoch_losses = train_epoch_func(
            G=G,
            D=D, 
            dataloader=dataloader, 
            optim_G=optim_G, 
            optim_D=optim_D, 
            device=device, 
            mode=mode, 
            d_iters=d_iters, 
            ema_beta=ema_beta,
            G_shadow=G_shadow,
            use_shadow=use_shadow,
            **kwargs
        )
        best_w1 = float('inf')
        for k, v in epoch_losses.items():
            if k in losses.keys():
                losses[k].append(v)
            else:
                losses[k] = [v]
        if (epoch + 1) % 10 == 0:
            plot_losses(losses)
            G.eval()
            if use_shadow:
                G_shadow.eval()
            gen = G_shadow if use_shadow else G
            metrics = {}
            for b in test_mom.keys():
                metrics[b] = get_dist_metrics(
                    gen,
                    generate_func,
                    test_mom[b],
                    test_cond[b],
                    test_label[b],
                    test_mask[b],
                    **kwargs
                )
                w1 = [metrics[b][part_type]['w1_0'] for b in metrics for part_type in metrics[b]]
                average_w1 = sum(w1) / len(w1) if w1 else float('inf')
                if average_w1 < best_w1:
                    torch.save(D.state_dict(), 'best_d.pt')
                    torch.save(G.state_dict(), 'best_g.pt')
                    if use_shadow:
                        torch.save(G_shadow.state_dict(), 'best_g_shadow.pt')
            G.train()
            if use_shadow:
                G_shadow.train()
        if scheduler_g is not None:
            scheduler_g.step()
        if scheduler_d is not None:
            scheduler_d.step()
    return losses, G, G_shadow, D


def train_and_test(
    G,
    D,
    train_loader,
    optim_G,
    optim_D,
    device,
    mode,
    d_iters,
    epochs,
    train_epoch_func,
    generate_func,
    test_mom_obj,
    test_cond_obj,
    test_label_obj,
    test_mask_obj,
    test_nucl_obj,
    test_nucl_mask_obj,
    test_event_obj,
    num_classes,
    meson_count_max,
    metrics_mode_name,
    final_metrics,
    scheduler_g=None,
    scheduler_d=None,
    use_shadow=False,
    ema_beta=0.5,
    **kwargs
):
    final_metrics[metrics_mode_name] = {}
    for seed in seeds:
        final_metrics[metrics_mode_name][seed] = {}
        set_seed(seed)
        _, G, G_shadow, D = train_model(
            G=G,
            D=D,
            dataloader=train_loader,
            optim_G=optim_G,
            optim_D=optim_D,
            device=device,
            mode=mode,
            d_iters=d_iters,
            epochs=epochs,
            train_epoch_func=train_epoch_func,
            generate_func=generate_func,
            test_mom=test_mom_obj,
            test_cond=test_cond_obj,
            test_label=test_label_obj,
            test_mask=test_mask_obj,
            num_classes=num_classes,
            meson_count_max=meson_count_max,
            scheduler_d=scheduler_d,
            scheduler_g=scheduler_g,
            use_shadow=use_shadow,
            ema_beta=ema_beta,
            **kwargs
        )
        G.load_state_dict(torch.load('best_g.pt'))
        D.load_state_dict(torch.load('best_d.pt'))
        if use_shadow:
            G_shadow.load_state_dict(torch.load('best_g_shadow.pt'))
        gen = G_shadow if use_shadow else G
        plt.clf()
        for b in test_mom_obj.keys():
            final_metrics[metrics_mode_name][seed][b] = get_dist_metrics(
                G,
                generate_func,
                test_mom_obj[b],
                test_cond_obj[b],
                test_label_obj[b],
                test_mask_obj[b],
                **kwargs
            )
            get_energy_metrics(G, 
                               generate_func, 
                               test_mom_obj[b],
                               test_nucl_obj[b], 
                               test_nucl_mask_obj[b], 
                               test_event_obj[b],
                               test_cond_obj[b], 
                               test_label_obj[b], 
                               test_mask_obj[b])
