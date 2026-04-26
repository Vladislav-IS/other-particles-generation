import torch
from torch import nn
from torch.nn.utils import weight_norm
from src.gaussian_mixture import GaussianMixtureLatent


class TransformerBlock(nn.Module):
    """
    EPiC transformer layer
    """

    def __init__(self, d_model, n_heads, d_ff, neg_slope=0.2, use_weight_norm=False):
        """
        Parameters:
        - d_model - hidden dimensionality in multi-head attention;
        - n_heads - number of heads in multi-head attention;
        - d_ff - hidden dimensionality in feed-forward network;
        - neg_slope - negative slope for LeakyReLU
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            (
                weight_norm(nn.Linear(d_model, d_ff))
                if use_weight_norm
                else nn.Linear(d_model, d_ff)
            ),
            nn.LeakyReLU(neg_slope),
            (
                weight_norm(nn.Linear(d_ff, d_model))
                if use_weight_norm
                else nn.Linear(d_ff, d_model)
            ),
        )

    def forward(self, x_local, x_global, mask=None):
        """
        Parameters:
        - x_local - local features;
        - x_global - global features;
        - mask - padding mask
        """
        residual = x_local
        key_padding_mask = None if mask is None else (mask == 0)
        attn_out, _ = self.self_attn(
            x_local, x_local, x_local, key_padding_mask=key_padding_mask
        )
        x_local = residual + attn_out
        residual = x_local
        if x_global.dim() == 2:
            x_global_expanded = x_global.unsqueeze(1)
        else:
            x_global_expanded = x_global
        attn_out, _ = self.cross_attn(x_local, x_global_expanded, x_global_expanded)
        x_local = residual + attn_out
        residual = x_local
        x_local = residual + self.ff(x_local)
        return x_local


class EPiC_Transformer_Generator(nn.Module):
    """
    EPiC transformer generator
    """

    def __init__(
        self,
        latent_global=128,
        latent_local=16,
        num_labels=5,
        extern_cond_d=6,
        feats=3,
        d_model=256,
        n_heads=8,
        d_ff=512,
        max_len=80,
        n_layers=6,
        n_modes=3,
        neg_slope=0.2,
        use_weight_norm=False,
    ):
        """
        Parameters:
        - latent_global - global noise dimensionality;
        - latent_local - local_noise dimensionality;
        - num_labels - number of particles types;
        - extern_cond_d - external condition dimensionality;
        - feats - output data dimensionality;
        - d_model - multi-head attention dimensionalty;
        - n_heads - number of heads in multi-head attention;
        - d_ff - feed-forward dimensionality;
        - max_len - maximum length of padded particles sequences;
        - n_layers - number of generator layers;
        - n_modes - number of modes in Gaussian mixture;
        - neg_slope - negative slope for LeakyReLU
        """
        super().__init__()
        self.latent_local = latent_local
        self.feats = feats
        self.extern_cond_d = extern_cond_d
        self.emb = nn.Embedding(num_labels, d_model)
        if extern_cond_d is not None:
            self.cond_proj = (
                weight_norm(nn.Linear(extern_cond_d, d_model))
                if use_weight_norm
                else nn.Linear(extern_cond_d, d_model)
            )
        self.global_proj = (
            weight_norm(nn.Linear(latent_global, d_model))
            if use_weight_norm
            else nn.Linear(latent_global, d_model)
        )
        self.local_proj = (
            weight_norm(nn.Linear(latent_local, d_model))
            if use_weight_norm
            else nn.Linear(latent_local, d_model)
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, neg_slope)
                for _ in range(n_layers)
            ]
        )
        self.out_impulse = (
            weight_norm(nn.Linear(d_model, feats))
            if use_weight_norm
            else nn.Linear(d_model, feats)
        )
        self.out_mask = (
            weight_norm(nn.Linear(d_model, 1))
            if use_weight_norm
            else nn.Linear(d_model, 1)
        )
        if not use_weight_norm:
            self.apply(self._init_weights)
        self.local_sampler = GaussianMixtureLatent(n_modes, latent_local, max_len)
        self.global_sampler = GaussianMixtureLatent(n_modes, latent_global)

    def _init_weights(self, module):
        """
        Unused.
        Parameters:
        - module - network module
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, labels, cond=None):
        """
        Parameters:
        - labels - tensor of particles types;
        - cond - tensor of external condition
        """
        batch_size = labels.size(0)
        z_local = self.local_sampler(batch_size)
        z_global = self.global_sampler(batch_size)
        labels_emb = self.emb(labels).mean(dim=1).unsqueeze(1)
        x_global = self.global_proj(z_global).unsqueeze(1)
        if self.extern_cond_d is not None:
            x_cond = self.cond_proj(cond).unsqueeze(1)
            context = torch.cat([x_global, labels_emb, x_cond], dim=1)
        else:
            context = torch.cat([x_global, labels_emb], dim=1)
        slots = self.local_proj(z_local)
        for block in self.blocks:
            slots = block(slots, context, mask=None)
        impulses = self.out_impulse(slots)
        mask_logits = self.out_mask(slots).squeeze(-1)
        mask = torch.sigmoid(mask_logits)
        return impulses, mask


class EPiC_Transformer_Discriminator(nn.Module):
    """
    EPiC transformer discriminator
    """

    def __init__(
        self,
        feats=3,
        num_labels=5,
        d_model=256,
        extern_cond_d=6,
        n_heads=8,
        d_ff=512,
        n_layers=6,
        neg_slope=0.2,
        use_weight_norm=False,
    ):
        """
        Parameters:
        - feats - input data dimensionality;
        - num_labels - number of particles types;
        - d_model - multi-head attention dimensionalty;
        - n_heads - number of heads in multi-head attention;
        - d_ff - feed-forward dimensionality;
        - n_layers - number of generator layers;
        - neg_slope - negative slope for LeakyReLU
        """
        super().__init__()
        self.d_model = d_model
        self.extern_cond_d = extern_cond_d
        self.input_proj = (
            weight_norm(nn.Linear(feats, d_model))
            if use_weight_norm
            else nn.Linear(feats, d_model)
        )
        self.emb = nn.Embedding(num_labels, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, neg_slope)
                for _ in range(n_layers)
            ]
        )
        if extern_cond_d is not None:
            self.cond_proj = (
                weight_norm(nn.Linear(extern_cond_d, d_model))
                if use_weight_norm
                else nn.Linear(extern_cond_d, d_model)
            )
        self.in_global_proj = (
                weight_norm(nn.Linear(2 * d_model, d_model))
                if use_weight_norm
                else nn.Linear(2 * d_model, d_model)
            )
        self.out_global_proj = nn.Sequential(
            (
                weight_norm(nn.Linear(3 * d_model, d_model))
                if use_weight_norm
                else nn.Linear(3 * d_model, d_model)
            ),
            nn.LeakyReLU(neg_slope),
            (
                weight_norm(nn.Linear(d_model, 1))
                if use_weight_norm
                else nn.Linear(d_model, 1)
            ),
        )
        if not use_weight_norm:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Unused.
        Parameters:
        - module - network module
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x, labels, cond=None, mask=None):
        """
        Parameters:
        - x - particles momenta;
        - labels - tensor of particles types;
        - cond - tensor of condition;
        - mask - tensor of passing mask
        """
        B, N, _ = x.shape
        x_local = self.input_proj(x)
        label_emb = self.emb(labels).mean(dim=1).unsqueeze(1)
        x_sum = x_local.sum(dim=1)
        if mask is not None:
            x_mean = x_sum / mask.sum(dim=1, keepdim=True)
        else:
            x_mean = x_local.mean(dim=1)
        x_global = self.in_global_proj(torch.cat([x_mean, x_sum], dim=-1)).unsqueeze(1)
        if self.extern_cond_d is not None:
            x_cond = self.cond_proj(cond).unsqueeze(1)
            x_global = torch.cat([x_global, label_emb, x_cond], dim=1)
        else:
            x_global = torch.cat([x_global, label_emb], dim=1)
        for block in self.blocks:
            x_local = block(x_local, x_global, mask)
        if mask is not None:
            masked = x_local * mask.unsqueeze(-1)
            final_mean = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            final_sum = masked.sum(dim=1)
        else:
            final_mean = x_local.mean(dim=1)
            final_sum = x_local.sum(dim=1)
        final_feat = torch.cat([final_mean, final_sum, x_global], dim=-1)
        out = self.out_global_proj(final_feat).squeeze(-1)
        return out
