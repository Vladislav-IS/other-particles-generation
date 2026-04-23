import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from src.plots import plot_kde, plot_scatter, plot_v2, plot_losses
from src.settings import *


def train_epoch_transformer(G,
                            D,
                            dataloader,
                            optim_G,
                            optim_D,
                            device,
                            mode='lsgan',
                            d_iters=1,
                            lambda_mask=1.0,
                            lambda_len=1.0,
                            clip_val=1.0):
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
            if mode == 'lsgan':
                d_real_mse = F.mse_loss(d_real, torch.ones(batch_size, device=device))
                d_fake_mse = F.mse_loss(d_fake, torch.zeros(batch_size, device=device))
                d_loss = (d_real_mse + d_fake_mse) / 2
            else:
                raise NotImplementedError
            d_loss.backward()
            nn.utils.clip_grad_norm_(D.parameters(), clip_val)
            optim_D.step()
        optim_G.zero_grad()
        fake_impulses, fake_mask = G(labels, cond)
        d_fake = D(fake_impulses * fake_mask.unsqueeze(-1), labels, cond, fake_mask)
        if mode == 'lsgan':
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
    return {'d_loss': epoch_d_loss / num_batches,
            'g_loss': epoch_g_loss / num_batches,
            'mask_loss': epoch_mask_loss / num_batches, 
            'len_loss': epoch_len_loss / num_batches}


def train_epoch_epic(G, D, dataloader, optim_G, optim_D, device,
                     mode='lsgan', d_iters=1, zg_dim=128, zl_dim=16, clip_val=1.0):
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
            if mode == 'lsgan':
                d_loss = 0.5 * (F.mse_loss(d_real, torch.ones_like(d_real)) +
                                F.mse_loss(d_fake, torch.zeros_like(d_fake)))
            else:
                raise NotImplementedError("Only LSGAN implemented for EPiC")
            d_loss.backward()
            clip_grad_norm_(D.parameters(), clip_val)
            optim_D.step()
        optim_G.zero_grad()
        z_global = torch.randn(batch_size, zg_dim, device=device)
        z_local = torch.randn(batch_size, n_points, zl_dim, device=device)
        fake_impulses = G(z_global, z_local, cond, labels)
        d_fake = D(fake_impulses, cond, labels)
        if mode == 'lsgan':
            g_loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
        else:
            g_loss = -d_fake.mean()
        g_loss.backward()
        clip_grad_norm_(G.parameters(), clip_val)
        optim_G.step()
        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()
        num_batches += 1
    return {'d_loss': epoch_d_loss / num_batches,
            'g_loss': epoch_g_loss / num_batches}


def train_model(G, D, dataloader, optim_G, optim_D, device, mode, d_iters,
                epochs, train_epoch_func, generate_func, test_mom, test_cond,
                test_label, test_mask, num_classes, meson_count_max,
                scheduler_g=None, scheduler_d=None, **kwargs):
    losses = {}
    for epoch in range(epochs):
        epoch_losses = train_epoch_func(G, D, dataloader, optim_G, optim_D, device, mode, d_iters, **kwargs)
        for k, v in epoch_losses.items():
            if k in losses.keys():
                losses[k].append(v)
            else:
                losses[k] = [v]
        if (epoch + 1) % 10 == 0:
            plot_losses(losses)
            G.eval()
            for i in range(num_classes):
                mask_cls = test_label == i
                if mask_cls.sum() == 0:
                    continue
                real = torch.tensor(test_mom, dtype=torch.float32)[mask_cls]
                cond = torch.tensor(test_cond, dtype=torch.float32)[mask_cls]
                labels = torch.tensor(test_label)[mask_cls]
                mask_data = torch.tensor(test_mask, dtype=torch.bool)[mask_cls]
                fake_masked = generate_func(G, mask_cls.sum(), meson_count_max, cond, labels, device, **kwargs)
                real_masked = real * mask_data.unsqueeze(-1)
                plot_kde(real_masked, fake_masked, ['Px', 'Py', 'Pz'], num_to_label[i])
                plot_kde(real_masked, fake_masked, ['Px', 'Py', 'Pz'], num_to_label[i], num=0)
                plot_kde(real_masked, fake_masked, ['Px', 'Py', 'Pz'], num_to_label[i], num=1)
                plot_kde(real_masked, fake_masked, ['Px', 'Py', 'Pz'], num_to_label[i], num=2)
                plot_scatter(real_masked, fake_masked)
                if num_to_label[i] != 111:
                    plot_v2(real_masked, fake_masked)
            G.train()
        if scheduler_g is not None:
            scheduler_g.step()
        if scheduler_d is not None:
            scheduler_d.step()
    return losses
