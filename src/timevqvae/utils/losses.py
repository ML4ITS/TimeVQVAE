import numpy as np
import torch
import torch.nn.functional as F


def compute_var_loss(z):
    return torch.relu(1.0 - torch.sqrt(z.var(dim=0) + 1e-4)).mean()


def compute_cov_loss(z):
    norm_z = z - z.mean(dim=0)
    norm_z = F.normalize(norm_z, p=2, dim=0)
    fxf_cov_z = torch.mm(norm_z.T, norm_z)
    ind = np.diag_indices(fxf_cov_z.shape[0])
    fxf_cov_z[ind[0], ind[1]] = torch.zeros(fxf_cov_z.shape[0]).to(norm_z.device)
    return (fxf_cov_z**2).mean()


def compute_emb_loss(codebook, x, use_cosine_sim, esm_max_codes):
    embed = codebook.embed
    flatten = x.reshape(-1, x.shape[-1])

    if use_cosine_sim:
        flatten = F.normalize(flatten, p=2, dim=-1)
        embed = F.normalize(embed, p=2, dim=-1)

    ind = torch.randint(0, embed.shape[0], size=(min(esm_max_codes, embed.shape[0]),))
    embed = embed[ind]

    cov_embed = torch.cov(embed.t())
    cov_x = torch.cov(flatten.t())

    mean_embed = torch.mean(embed, dim=0)
    mean_x = torch.mean(flatten, dim=0)

    return F.mse_loss(cov_x.detach(), cov_embed) + F.mse_loss(mean_x.detach(), mean_embed)
