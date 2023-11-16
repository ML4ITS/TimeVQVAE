"""
reference for the U-Net implementation:
    - https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch_1d.py
"""
import math
from functools import partial
from collections import namedtuple

import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from vector_quantization import VectorQuantize
from utils import quantize, time_to_timefreq, timefreq_to_time
from encoder_decoders.vq_vae_encdec1d import VQVAEEncoder, VQVAEDecoder

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1, padding_mode='zeros')
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1, padding_mode='zeros')

class WeightStandardizedConv2d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

# model

class Unet1D(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 1,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        attention_ups_downs=False,
        **kwargs,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention_ups_downs else nn.Identity(),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1, padding_mode='zeros')
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention_ups_downs else nn.Identity(),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1, padding_mode='zeros')
            ]))
        self.last_up = Upsample(dim_in, dim_in)

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

        # self.vq_model_mid = VectorQuantize(dim * dim_mults[-1], codebook_sizes['mid'])
        # self.vq_model_end = VectorQuantize(dim, codebook_sizes['end'], learnable_codebook=True)

    def forward(self, x):
        # if self.self_condition:
        #     x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
        #     x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        # t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            h.append(x)

            x = block2(x)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        # x, indices, vq_loss_mid, perplexity = quantize(x, self.vq_model_mid, transpose_channel_length_axes=True)
        # vq_loss_mid = vq_loss_mid['loss']

        for block1, block2, attn, upsample in self.ups:
            h_ = F.interpolate(h.pop(), size=x.shape[-1], mode='linear', align_corners=False)
            x = torch.cat((x, h_), dim = 1)
            x = block1(x)

            h_ = F.interpolate(h.pop(), size=x.shape[-1], mode='linear', align_corners=False)
            x = torch.cat((x, h_), dim = 1)
            x = block2(x)
            x = attn(x)

            x = upsample(x)
        x = self.last_up(x)
        x = F.interpolate(x, size=r.shape[-1], mode='linear', align_corners=False)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x)
        # x, indices, vq_loss_end, perplexity = quantize(x, self.vq_model_end, transpose_channel_length_axes=True)
        # vq_loss_end = vq_loss_end['loss']
        x = self.final_conv(x)

        # if return_vq_output:
        #     vq_loss = vq_loss_mid + vq_loss_end
            # vq_loss = vq_loss_end
            # return x, (vq_loss, indices, perplexity)
        # else:
        return x


# class DomainShifter(nn.Module):
#     def __init__(self, input_length, in_channels, n_fft, config):
#         super(DomainShifter, self).__init__()
#         self.n_fft = n_fft
#         self.input_length = input_length
#         self.domain_shifter = Unet1D(channels=in_channels, **config['domain_shifter'])
#
#     def forward(self, xhat):
#         """
#         :param xhat: (b c n)
#         :return:
#         """
#         xhat = F.upsample(xhat, size=self.input_length, mode='linear', align_corners=False)
#         xfhat = torch.stft(xhat[:, 0, :], n_fft=self.n_fft, return_complex=False, normalized=True)  # (b h w 2)
#         xfhat = rearrange(xfhat, 'b h w c -> b (h c) w')  # (b h2 w)
#
#         xfhat_c = self.domain_shifter(xfhat)
#         xfhat_c = rearrange(xfhat_c, 'b (h c) w -> b h w c', c=2)  # (b h w 2)
#
#         xhat_c = torch.istft(xfhat_c, n_fft=self.n_fft, normalized=True)  # (b l)
#         xhat_c = xhat_c[:, None, :]  # (b 1 l); adding a channel dim
#         xhat_c = F.upsample(xhat_c, size=self.input_length, mode='linear', align_corners=False)
#         return xhat_c


class Discretizer(nn.Module):
    def __init__(self, codebook_size, dim, input_length, **kwargs):
        super(Discretizer, self).__init__()
        """
        - dim: it should be kept large enough so that codes are easily distinguishable. 
        """
        self.input_length = input_length

        self.in_conv = nn.Sequential(nn.Conv1d(1, dim, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
                                     nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
                                     nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
                                     nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
                                     )
        # self.in_conv = nn.Conv1d(1, dim, kernel_size=1, stride=1)
        self.vq = VectorQuantize(dim, codebook_size)
        self.out_conv = nn.Sequential(nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
                                      nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
                                      nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
                                      nn.Conv1d(dim, 1, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
                                      )
        # self.out_conv = nn.Conv1d(dim, 1, kernel_size=1, stride=1)

    def forward(self, xhat_c, xhat, xhat_l, xhat_h):
        """
        :param xhat_c: (b 1 l)
        :return:
        """
        # print('xhat_c.shape:', xhat_c.shape)
        # print('xhat.shape:', xhat.shape)
        # print('xhat_l.shape:', xhat_l.shape)
        # print('xhat_h.shape:', xhat_h.shape)
        # input = torch.cat((xhat_c, xhat, xhat_l, xhat_h), dim=1)
        input = xhat_c

        out = F.upsample(input, size=self.input_length, mode='linear', align_corners=False)
        out = self.in_conv(out)
        out, indices, vq_loss, perplexity = quantize(out, self.vq, transpose_channel_length_axes=True)
        vq_loss = vq_loss['loss']
        out = self.out_conv(out)
        out = F.upsample(out, size=self.input_length, mode='linear', align_corners=False)
        return out, vq_loss

    # def forward(self, xhat_c):
    #     """
    #     :param xhat_c: (b 1 l)
    #     :return:
    #     """
    #     out = time_to_timefreq(xhat_c, self.n_fft, C=1)  # (b c h w)
    #     width, height = out.shape[2], out.shape[3]
    #     out = self.in_conv(out)
    #     out, indices, vq_loss, perplexity = quantize(out, self.vq)
    #     vq_loss = vq_loss['loss']
    #     out = self.out_conv(out)  # (b c h w)
    #     out = F.upsample(out, size=(width, height), mode='bilinear', align_corners=False)  # (b c h w)
    #     out = timefreq_to_time(out, self.n_fft, C=1)  # (b 1 l)
    #     out = F.upsample(out, size=self.input_length, mode='linear', align_corners=False)  # (b 1 l)
    #     return out, vq_loss


class Sum(nn.Module):
    def forward(self, x):
        return torch.sum(x, dim=1, keepdim=True)


class Refiner(nn.Module):
    def __init__(self, n_layers, dim, input_length):
        super(Refiner, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(nn.Linear(input_length, input_length),
                                             nn.Linear(input_length, input_length),
                                             nn.Linear(input_length, input_length),
                                             ))

    def forward(self, input):
        out = input.detach()
        outs = []
        for layer in self.layers:
            # residual = layer(out)  # (b 1 l)
            # residual = layer(out.detach())  # (b 1 l)
            # out = out + residual
            out = layer(out)  # (b 1 l)
            outs.append(out)
        return outs


class InfoRemover(nn.Module):
    def __init__(self, dim, removal_rate=0.99):
        super().__init__()
        self.removal_rate = removal_rate

        self.conv = nn.Conv1d(in_channels=1, out_channels=dim, kernel_size=1, stride=1)
        self.mask_token = nn.Parameter(torch.randn(dim,))

    def forward(self, x):
        out = self.conv(x)  # (b d l)
        # mask_token = repeat(self.mask_token, 'd -> b d l', b=out.shape[0], l=out.shape[2])  # (b d l)

        if self.training:
            mask = torch.rand((out.shape[0], out.shape[2],)).to(x.device)  # (b l)
            removal_rate = np.random.uniform(0.5, 0.99)
            mask = mask > removal_rate  # 0 (False) for removing
            mask = repeat(mask, 'b l -> b d l', d=out.shape[1])  # (b d l)
            mask_token = repeat(self.mask_token, 'd -> b d l', b=out.shape[0], l=out.shape[2])  # (b d l)
            out = mask * out + (~mask) * mask_token
        return out


class DomainShifter(nn.Module):
    def __init__(self, input_length, in_channels, n_fft, config):
        super(DomainShifter, self).__init__()
        self.n_fft = n_fft
        self.input_length = input_length

        # self.info_remover = InfoRemover(dim=4)
        self.projector = Unet1D(channels=1, **config['domain_shifter'], out_dim=1)
        # self.projector2 = Unet1D(channels=2*in_channels, **config['domain_shifter'], out_dim=1)
        # self.discretizer = Discretizer(**config['domain_shifter']['discretizer'], input_length=input_length)
        # self.refiner = Refiner(n_layers=1, dim=16, input_length=self.input_length)

        # self.encoder_real = VQVAEEncoder(d=32, num_channels=1, downsample_rate=1, n_resnet_blocks=4)
        # self.encoder_fake = VQVAEEncoder(d=32, num_channels=1, downsample_rate=1, n_resnet_blocks=4)
        # self.decoder = VQVAEDecoder(d=32, num_channels=1, downsample_rate=4, n_resnet_blocks=4)
        #self.encoder_real = Unet1D(channels=1, **config['domain_shifter'], out_dim=32)
        #self.decoder = Unet1D(channels=32, **config['domain_shifter'], out_dim=1)
        #self.vq = VectorQuantize(dim=32, codebook_size=128)

    def forward_projector(self, x_a, x_l_a, x_h_a):
        #xhat_comb = torch.cat((xhat, xhat_l, xhat_h), dim=1)  # (b 3c l)
        # xhat_comb = torch.cat((xhat_l, xhat_h), dim=1)  # (b 3c l)
        # input = F.upsample(x_h_a, size=self.input_length, mode='linear', align_corners=True)
        # xhat = self.projector(input)
        # xhat = F.upsample(xhat, size=self.input_length, mode='linear', align_corners=True)
        #
        # xhat2 = self.projector2(torch.cat((xhat, x_l_a), dim=1))
        # xhat2 = F.upsample(xhat2, size=self.input_length, mode='linear', align_corners=True)

        # return xhat, xhat2
        # xf_a = time_to_timefreq(x_a, self.n_fft, C=1)  # (b c h w)
        # xf_a = rearrange(xf_a, 'b c h w -> b (c h) w')
        # x_a = self.info_remover(x_a)
        xhat = self.projector(x_a)
        xhat = F.upsample(xhat, size=self.input_length, mode='linear')
        return xhat

    #def forward_encoder(self, x, kind):
    #    if kind == 'real':
    #        z = self.encoder_real(x)
    #    elif kind == 'fake':
    #        z = self.encoder_fake(x)
    #    else:
    #        raise ValueError

        #z = F.normalize(z, p=2)

        # register `upsample_size` in the decoders
        # if not self.decoder.is_upsample_size_updated:
        #     self.decoder.register_upsample_size(torch.tensor(x.shape[2]))

    #    return z

    #def forward_decoder(self, z):
    #    # xhat_comb = torch.cat((xhat, xhat_l, xhat_h), dim=1)  # (b 3c l)
    #    # xhat = F.upsample(xhat, size=self.input_length, mode='linear', align_corners=False)
    #    x = self.decoder(z)
    #    x = F.upsample(x, size=self.input_length, mode='linear', align_corners=False)
    #    return x

    # def forward_refiner(self, xhat_p):
    #     outs = self.refiner(xhat_p)
    #     return outs

    def forward(self, xhat):
        """
        :param xhat: (b 1 l)
        :return:
        """
        zhat = self.forward_encoder(xhat, 'real')
        self.vq.eval()
        zqhat, indices, vq_loss, perplexity = quantize(zhat, self.vq, transpose_channel_length_axes=True)
        x_recons = self.forward_decoder(zqhat)
        return x_recons


if __name__ == '__main__':
    # unet
    x = torch.rand((1, 1, 301))
    model = Unet1D(dim = 8)
    xhat = model(x)
    print(xhat.shape)
