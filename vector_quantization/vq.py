"""
The code is taken from https://github.com/lucidrains/vector-quantize-pytorch
"""
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as distributed
from torch.cuda.amp import autocast
from torch.distributions.categorical import Categorical

from einops import rearrange, repeat
from contextlib import contextmanager

from utils import *


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def noop(*args, **kwargs):
    pass


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    if temperature == 0:
        return t.argmax(dim=dim)

    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def softmax_sample(t, temperature, dim=-1):
    if isinstance(temperature, type(None)):
        return t.argmax(dim=dim)

    m = Categorical(logits=t / temperature)
    return m.sample()


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


# regularization losses

def orthgonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    n = t.shape[0]
    normed_codes = l2norm(t)
    identity = torch.eye(n, device=t.device)
    cosine_sim = einsum('i d, j d -> i j', normed_codes, normed_codes)
    return ((cosine_sim - identity) ** 2).sum() / (n ** 2)


# distance types

class EuclideanCodebook(nn.Module):
    """
    source: https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py
    """
    def __init__(
            self,
            dim,
            codebook_size,
            kmeans_init=False,
            kmeans_iters=10,
            decay=0.8,
            eps=1e-5,
            threshold_ema_dead_code=2,
            use_ddp=False,
            learnable_codebook=False,
            sample_codebook_temp=0,
            emb_dropout=0.,
    ):
        super().__init__()
        self.decay = decay
        init_fn = torch.randn if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp
        self.emb_dropout = emb_dropout

        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

        self.embed_onehot = None
        self.perplexity = None

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None],
            sample_vectors(samples, self.codebook_size),
            self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace(batch_samples, mask=expired_codes)

    @autocast(enabled=False)
    def forward(self, x, svq_temp:Union[float,None]=None):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')

        self.init_embed_(flatten)

        embed = self.embed if not self.learnable_codebook else self.embed.detach()
        embed = embed.t()

        if self.emb_dropout and self.training:
            embed = F.dropout(embed, self.emb_dropout)

        dist = -(
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ embed
                + embed.pow(2).sum(0, keepdim=True)
        )
        # embed_ind = gumbel_sample(dist, dim=-1, temperature=self.sample_codebook_temp)
        temp = svq_temp
        if self.training:
            embed_ind = softmax_sample(dist, dim=-1, temperature=temp)
        else:
            embed_ind = softmax_sample(dist, dim=-1, temperature=temp)  # no stochasticity
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed)

        if self.training:
            cluster_size = embed_onehot.sum(0)
            self.all_reduce_fn(cluster_size)

            ema_inplace(self.cluster_size, cluster_size, self.decay)

            embed_sum = flatten.t() @ embed_onehot
            self.all_reduce_fn(embed_sum)

            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        # perplexity
        avg_probs = torch.mean(embed_onehot, dim=0)  # (K,)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        self.embed_onehot = embed_onehot.detach()  # .cpu()
        self.perplexity = perplexity.detach()  # .cpu()

        return quantize, embed_ind


# main class
class VectorQuantize(nn.Module):
    def __init__(
            self,
            dim,
            codebook_size,
            codebook_dim=None,
            heads=1,
            decay=0.8,
            eps=1e-5,
            kmeans_init=False,
            kmeans_iters=10,
            use_cosine_sim=False,
            threshold_ema_dead_code=0,
            channel_last=True,
            accept_image_fmap=False,
            commitment_weight=1.,
            orthogonal_reg_weight=0.,
            orthogonal_reg_active_codes_only=False,
            orthogonal_reg_max_codes=None,
            sample_codebook_temp=0.,
            sync_codebook=False,
            emb_dropout=0.,
            **kwargs
    ):
        super().__init__()
        self.heads = heads
        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.eps = eps
        self.commitment_weight = commitment_weight

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        codebook_class = EuclideanCodebook

        self._codebook = codebook_class(
            dim=codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_ddp=sync_codebook,
            learnable_codebook=has_codebook_orthogonal_loss,
            sample_codebook_temp=sample_codebook_temp,
            emb_dropout=emb_dropout,
        )

        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

    @property
    def codebook(self):
        return self._codebook.embed

    def forward(self, x, svq_temp:Union[float,None]=None):
        """
        x: (B, N, D)
        """
        shape, device, heads, is_multiheaded, codebook_size = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size
        need_transpose = not self.channel_last and not self.accept_image_fmap
        vq_loss = {'loss': torch.tensor([0.], device=device, requires_grad=self.training),
                   'commit_loss': 0.,
                   'orthogonal_reg_loss': 0.,
                   }

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        x = self.project_in(x)

        if is_multiheaded:
            x = rearrange(x, 'b n (h d) -> (b h) n d', h=heads)

        quantize, embed_ind = self._codebook(x, svq_temp)

        if self.training:
            quantize = x + (quantize - x).detach()  # allows `z`-part to be trainable while `z_q`-part is un-trainable. `z_q` is updated by the EMA.

        if self.training:
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                vq_loss['commit_loss'] = commit_loss
                vq_loss['loss'] = vq_loss['loss'] + commit_loss * self.commitment_weight

            if self.orthogonal_reg_weight > 0:
                codebook = self.codebook

                if self.orthogonal_reg_active_codes_only:
                    # only calculate orthogonal loss for the activated codes for this batch
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[unique_code_ids]

                num_codes = codebook.shape[0]
                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device=device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[rand_ids]

                orthogonal_reg_loss = orthgonal_loss_fn(codebook)
                vq_loss['orthogonal_reg_loss'] = orthogonal_reg_loss
                vq_loss['loss'] = vq_loss['loss'] + orthogonal_reg_loss * self.orthogonal_reg_weight

        if is_multiheaded:
            quantize = rearrange(quantize, '(b h) n d -> b n (h d)', h=heads)
            embed_ind = rearrange(embed_ind, '(b h) n -> b n h', h=heads)

        quantize = self.project_out(quantize)

        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=height, w=width)
            embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h=height, w=width)

        return quantize, embed_ind, vq_loss, self._codebook.perplexity


if __name__ == '__main__':
    torch.manual_seed(0)

    B, N, D = 1024, 32, 128
    x = torch.rand((B, N, D))

    vq = VectorQuantize(dim=D, codebook_size=512)

    quantize, vq_ind, vq_loss, perplexity = vq(x)
    print(vq_ind[0])  # `vq_ind` is a set of codebook indices; e.g., 87 denotes the 88-th code in the codebook which can be accessed by `vq.codebook[87]`.

    # you can fetch the codebook weight by `vq.codebook`
    print('vq.codebook.shape:', vq.codebook.shape)  # (codebook_size, dim)

