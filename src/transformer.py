import torch
from torch import nn
from src.gaussian_mixture import GaussianMixtureLatent


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, neg_slope=0.2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.LeakyReLU(neg_slope),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x_local, x_global, mask=None):
        residual = x_local
        key_padding_mask = None if mask is None else (mask == 0)
        attn_out, _ = self.self_attn(x_local, x_local, x_local, key_padding_mask=key_padding_mask)
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
    def __init__(self,
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
                 dropout=0.1):
        super().__init__()
        self.latent_local = latent_local
        self.feats = feats
        self.extern_cond_d = extern_cond_d
        self.emb = nn.Embedding(num_labels, d_model)
        if extern_cond_d is None:
            self.cond_proj = nn.Linear(2 * d_model, d_model)
        else:
            self.cond_proj = nn.Linear(extern_cond_d + 2 * d_model, d_model)
        self.global_proj = nn.Linear(latent_global, d_model)
        self.local_proj = nn.Linear(latent_local, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.out_impulse = nn.Linear(d_model, feats)
        self.out_mask = nn.Linear(d_model, 1)
        self.apply(self._init_weights)
        self.local_sampler = GaussianMixtureLatent(n_modes, latent_local, max_len)
        self.global_sampler = GaussianMixtureLatent(n_modes, latent_global)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, labels, cond=None):
        batch_size = labels.size(0)
        z_local = self.local_sampler(batch_size)
        z_global = self.global_sampler(batch_size)
        labels_emb = self.emb(labels)
        x_global = self.global_proj(z_global)
        if self.extern_cond_d is not None:
            context = torch.cat([x_global, labels_emb, cond], dim=-1)
        else:
            context = torch.cat([x_global, labels_emb], dim=-1)
        context = self.cond_proj(context.unsqueeze(1))
        slots = self.local_proj(z_local)
        for block in self.blocks:
            slots = block(slots, context, mask=None)
        impulses = self.out_impulse(slots)
        mask_logits = self.out_mask(slots).squeeze(-1)
        mask = torch.sigmoid(mask_logits)
        return impulses, mask


class EPiC_Transformer_Discriminator(nn.Module):
    def __init__(self,
                 feats=3,
                 num_labels=5,
                 d_model=256,
                 extern_cond_d=6,
                 n_heads=8,
                 d_ff=512,
                 n_layers=6,
                 dropout=0.1,
                 neg_slope=0.2):
        super().__init__()
        self.d_model = d_model
        self.extern_cond_d = extern_cond_d
        self.input_proj = nn.Linear(feats, d_model)
        self.emb = nn.Embedding(num_labels, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        if extern_cond_d is not None:
            self.cond_proj = nn.Linear(extern_cond_d + 2 * d_model, d_model)
        else:
            self.cond_proj = nn.Linear(2 * d_model, d_model)
        self.global_proj = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.LeakyReLU(neg_slope),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x, labels, cond=None, mask=None):
        B, N, _ = x.shape
        x_local = self.input_proj(x)
        label_emb = self.emb(labels).unsqueeze(1)
        if mask is not None:
        #    masked = x_local * mask.unsqueeze(-1)
            x_global = x_local.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            x_global = x_local.mean(dim=1)
        if self.extern_cond_d is not None:
            x_global = torch.cat([x_global, label_emb.squeeze(1), cond], dim=-1)
        else:
            x_global = torch.cat([x_global, label_emb.squeeze(1)], dim=-1)
        x_global = self.cond_proj(x_global)
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
        out = self.global_proj(final_feat).squeeze(-1)
        return out
    