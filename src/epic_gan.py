import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm


class EPiC_layer(nn.Module):
    def __init__(self, local_in_dim, hid_dim, latent_dim):
        super(EPiC_layer, self).__init__()
        self.fc_global1 = weight_norm(nn.Linear(int(2*hid_dim)+latent_dim, hid_dim))
        self.fc_global2 = weight_norm(nn.Linear(hid_dim, latent_dim))
        self.fc_local1 = weight_norm(nn.Linear(local_in_dim+latent_dim, hid_dim))
        self.fc_local2 = weight_norm(nn.Linear(hid_dim, hid_dim))

    def forward(self, x_global, x_local):
        batch_size, n_points, latent_local = x_local.size()
        latent_global = x_global.size(1)
        x_pooled_mean = x_local.mean(1, keepdim=False)
        x_pooled_sum = x_local.sum(1, keepdim=False)
        x_pooledCATglobal = torch.cat([x_pooled_mean, x_pooled_sum, x_global], 1)
        x_global1 = F.leaky_relu(self.fc_global1(x_pooledCATglobal))
        x_global = F.leaky_relu(self.fc_global2(x_global1) + x_global)
        x_global2local = x_global.view(-1,1,latent_global).repeat(1,n_points,1)
        x_localCATglobal = torch.cat([x_local, x_global2local], 2)
        x_local1 = F.leaky_relu(self.fc_local1(x_localCATglobal))
        x_local = F.leaky_relu(self.fc_local2(x_local1) + x_local)
        return x_global, x_local


class EPiC_generator(nn.Module):
    def __init__(self, latent, latent_local, hid_d, feats, equiv_layers_generator, extern_cond_d, num_labels):
        super(EPiC_generator, self).__init__()
        self.equiv_layers = equiv_layers_generator
        self.local_0 = weight_norm(nn.Linear(latent_local, hid_d))
        self.global_0 = weight_norm(nn.Linear(latent + extern_cond_d + num_labels, hid_d))
        self.global_1 = weight_norm(nn.Linear(hid_d, latent))
        self.nn_list = nn.ModuleList()
        for _ in range(self.equiv_layers):
            self.nn_list.append(EPiC_layer(hid_d, hid_d, latent))
        self.local_1 = weight_norm(nn.Linear(hid_d, feats))
        self.emb = nn.Embedding(num_labels, num_labels)

    def forward(self, z_global, z_local, cond, label):
        batch_size, _, _= z_local.size()
        latent_tensor = z_global.clone().reshape(batch_size, 1, -1)
        z_local = F.leaky_relu(self.local_0(z_local))
        label_emb = self.emb(label)
        z_global = F.leaky_relu(self.global_0(torch.cat([z_global, cond, label_emb], dim=-1)))
        z_global = F.leaky_relu(self.global_1(z_global))
        latent_tensor = torch.cat([latent_tensor, z_global.clone().reshape(batch_size, 1, -1)], 1)
        z_global_in, z_local_in = z_global.clone(), z_local.clone()
        for i in range(self.equiv_layers):
            z_global, z_local = self.nn_list[i](z_global, z_local)
            z_global, z_local = z_global + z_global_in, z_local + z_local_in
            latent_tensor = torch.cat([latent_tensor, z_global.clone().reshape(batch_size, 1, -1)], 1)
        out = self.local_1(z_local)
        return out


class EPiC_discriminator(nn.Module):
    def __init__(self, latent, hid_d, feats, equiv_layers_discriminator, extern_cond_d, num_labels):
        super(EPiC_discriminator, self).__init__()
        self.equiv_layers = equiv_layers_discriminator
        self.fc_l1 = weight_norm(nn.Linear(feats, hid_d))
        self.fc_l2 = weight_norm(nn.Linear(hid_d, hid_d))
        self.fc_g1 = weight_norm(nn.Linear(int(2 * hid_d + extern_cond_d + num_labels), hid_d))
        self.fc_g2 = weight_norm(nn.Linear(hid_d, latent))
        self.nn_list = nn.ModuleList()
        for _ in range(self.equiv_layers):
            self.nn_list.append(EPiC_layer(hid_d, hid_d, latent))
        self.fc_g3 = weight_norm(nn.Linear(int(2 * hid_d + latent), hid_d))
        self.fc_g4 = weight_norm(nn.Linear(hid_d, hid_d))
        self.fc_g5 = weight_norm(nn.Linear(hid_d, 1))
        self.emb = nn.Embedding(num_labels, num_labels)

    def forward(self, x, cond, label):
        label_emb = self.emb(label)
        x_local = F.leaky_relu(self.fc_l1(x))
        x_local = F.leaky_relu(self.fc_l2(x_local) + x_local)
        x_mean = x_local.mean(1, keepdim=False)
        x_sum = x_local.sum(1, keepdim=False)
        x_global = torch.cat([x_mean, x_sum, cond, label_emb], 1)
        x_global = F.leaky_relu(self.fc_g1(x_global))
        x_global = F.leaky_relu(self.fc_g2(x_global))
        for i in range(self.equiv_layers):
            x_global, x_local = self.nn_list[i](x_global, x_local)
        x_mean = x_local.mean(1, keepdim=False)
        x_sum = x_local.sum(1, keepdim=False)
        x = torch.cat([x_mean, x_sum, x_global], 1)
        x = F.leaky_relu(self.fc_g3(x))
        x = F.leaky_relu(self.fc_g4(x) + x)
        x = self.fc_g5(x)
        return x
    