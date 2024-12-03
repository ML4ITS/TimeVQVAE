import math
from typing import Union

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from einops import repeat, rearrange
# from x_transformers import ContinuousTransformerWrapper, Encoder as TFEncoder


def calculate_padding(kernel_size, stride, dilation):
    """
    Calculate the padding size for a convolutional layer to achieve 'same' padding.

    Args:
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        dilation (int, optional): Dilation rate. Defaults to 1.

    Returns:
        int: Calculated padding size.
    """
    # Effective kernel size considering dilation
    effective_kernel_size = dilation * (kernel_size - 1) + 1

    # Calculate padding
    padding = math.floor((effective_kernel_size - stride) / 2)

    return padding

class Upscale(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, h_dim:int) -> None:
        super().__init__()
        self.conv = nn.Sequential(weight_norm(nn.Conv1d(in_channels, h_dim, kernel_size=7, stride=1, dilation=1, padding=calculate_padding(7,1,1))),
                                  nn.GELU(),
                                  nn.BatchNorm1d(h_dim),
                                  weight_norm(nn.Conv1d(h_dim, out_channels, kernel_size=7, stride=1, dilation=2, padding=calculate_padding(7,1,2))),
                                  )

    def forward(self, x, upscale_size:int):
        """
        x: (b n d)
        """
        x = rearrange(x, 'b n d -> b d n')  # (b d n)
        x = F.interpolate(x, size=(upscale_size,), mode='nearest')  # (b d m)
        x = self.conv(x)  # (b d m)
        x = rearrange(x, 'b d m -> b m d')
        return x


# class BidirectionalTransformer(nn.Module):
#     def __init__(self,
#                  kind: str,
#                  num_tokens: int,
#                  codebook_sizes: dict,
#                  embed_dim: int,
#                  hidden_dim: int,
#                  n_layers: int,
#                  heads: int,
#                  ff_mult: int,
#                  use_rmsnorm: bool,
#                  p_unconditional: float,
#                  n_classes: int,
#                  model_dropout:float=0.3,
#                  emb_dropout:float=0.3,
#                  **kwargs):
#         """
#         :param kind:
#         :param num_tokens:
#         :param codebook_sizes:
#         :param embed_dim:
#         :param hidden_dim:
#         :param n_layers:
#         :param heads:
#         :param ff_mult:
#         :param use_rmsnorm:
#         :param p_unconditional:
#         :param n_classes:
#         :param num_tokens_l:
#         :param kwargs:
#         """
#         super().__init__()
#         kind = kind.lower()
#         assert kind in ['lf', 'hf'], 'invalid `kind`.'
#         self.kind = kind
#         self.num_tokens = num_tokens
#         self.n_classes = n_classes
#         self.p_unconditional = p_unconditional
#         in_dim = embed_dim if kind == 'lf' else 2 * embed_dim
#         out_dim = embed_dim
#         self.emb_dropout = emb_dropout
#         self.mask_token_ind = {'lf':codebook_sizes['lf'], 'hf':codebook_sizes['hf']}

#         # token embeddings
#         self.tok_emb_l = nn.Embedding(codebook_sizes['lf'] + 1, embed_dim)  # `+1` is for mask-token
#         if kind == 'hf':
#             self.tok_emb_h = nn.Embedding(codebook_sizes['hf'] + 1, embed_dim)  # `+1` is for mask-token

#         # transformer
#         self.pos_emb = nn.Embedding(self.num_tokens + 1, in_dim)
#         self.class_condition_emb = nn.Embedding(n_classes + 1, in_dim)  # `+1` is for no-condition
#         self.blocks = ContinuousTransformerWrapper(dim_in=in_dim,
#                                                    dim_out=in_dim,
#                                                    max_seq_len=self.num_tokens + 1,
#                                                    use_abs_pos_emb=False,
#                                                    post_emb_norm=True,
#                                                    attn_layers=TFEncoder(
#                                                        pre_norm=True,
#                                                        dim=hidden_dim,
#                                                        depth=n_layers,
#                                                        heads=heads,
#                                                        attn_dim_head=64,
#                                                        use_rmsnorm=use_rmsnorm,
#                                                        ff_mult=ff_mult,
#                                                        layer_dropout=model_dropout,
#                                                        attn_dropout=model_dropout, 
#                                                        ff_dropout=model_dropout,
#                                                    ))
#         self.pred_head = nn.Sequential(*[
#             nn.Linear(in_features=in_dim, out_features=out_dim),
#             nn.GELU(),
#             nn.LayerNorm(out_dim, eps=1e-12)
#         ])
#         codebook_size = codebook_sizes['lf'] if kind == 'lf' else codebook_sizes['hf']
#         self.bias = nn.Parameter(torch.zeros(self.num_tokens, codebook_size + 1))

#         if kind == 'hf':
#             self.projector = Upscale(embed_dim, embed_dim, 2*embed_dim)

#     def class_embedding(self, class_condition: Union[None, torch.Tensor], batch_size: int, device):
#         cond_type = 'uncond' if isinstance(class_condition, type(None)) else 'class-cond'

#         if cond_type == 'uncond':
#             class_uncondition = repeat(torch.Tensor([self.n_classes]).long().to(device), 'i -> b i', b=batch_size)  # (b 1)
#             cls_emb = self.class_condition_emb(class_uncondition)  # (b 1 dim)
#             return cls_emb
#         elif cond_type == 'class-cond':
#             if self.training:
#                 ind = torch.rand(class_condition.shape).to(device) > self.p_unconditional  # to enable classifier-free guidance
#             else:
#                 ind = torch.ones_like(class_condition, dtype=torch.bool).to(device)
#             class_condition = torch.where(ind, class_condition.long(), self.n_classes)  # (b 1)
#             cls_emb = self.class_condition_emb(class_condition)  # (b 1 dim)
#             return cls_emb

#     def _token_emb_dropout(self, s:torch.LongTensor, token_emb:torch.FloatTensor, freq_type:str, p:float):
#         mask_ind = (s == self.mask_token_ind[freq_type])[:,:,None]  # (b n 1)
#         token_emb_dropout = F.dropout(token_emb, p=p)  # (b n d); to make the prediction process more robust during sampling
#         token_emb = torch.where(mask_ind, token_emb, token_emb_dropout)  # (b n d)
#         return token_emb

#     def forward_lf(self, s_M_l, class_condition: Union[None, torch.Tensor] = None):
#         device = s_M_l.device

#         token_embeddings = self.tok_emb_l(s_M_l)  # (b n dim)
#         if self.training:
#             token_embeddings = self._token_emb_dropout(s_M_l, token_embeddings, 'lf', p=self.emb_dropout)  # (b n d)

#         cls_emb = self.class_embedding(class_condition, s_M_l.shape[0], device)  # (b 1 dim)

#         n = token_embeddings.shape[1]
#         position_embeddings = self.pos_emb.weight[:n, :]
#         embed = token_embeddings + position_embeddings  # (b, n, dim)
#         embed = torch.cat((cls_emb, embed), dim=1)  # (b, 1+n, dim)
#         embed = self.blocks(embed)  # (b, 1+n, dim)
#         embed = self.pred_head(embed)[:, 1:, :]  # (b, n, dim)

#         logits = torch.matmul(embed, self.tok_emb_l.weight.T) + self.bias  # (b, n, codebook_size+1)
#         logits = logits[:, :, :-1]  # remove the logit for the mask token.  # (b, n, codebook_size)
#         return logits

#     def forward_hf(self, s_l, s_M_h, class_condition=None):
#         """
#         s_M_l (b n)
#         s_M_h (b m); m > n
#         """
#         device = s_l.device

#         token_embeddings_l = self.tok_emb_l(s_l)  # (b n dim)
#         token_embeddings_h = self.tok_emb_h(s_M_h)  # (b m dim)

#         if self.training:
#             token_embeddings_l = self._token_emb_dropout(s_l, token_embeddings_l, 'lf', p=self.emb_dropout)
#             token_embeddings_h = self._token_emb_dropout(s_M_h, token_embeddings_h, 'hf', p=self.emb_dropout)

#         token_embeddings_l = self.projector(token_embeddings_l, upscale_size=token_embeddings_h.shape[1])  # (b m dim)
#         token_embeddings = torch.cat((token_embeddings_l, token_embeddings_h), dim=-1)  # (b m 2*dim)

#         cls_emb = self.class_embedding(class_condition, s_l.shape[0], device)  # (b 1 2*dim)

#         n = token_embeddings.shape[1]
#         position_embeddings = self.pos_emb.weight[:n, :]
#         embed = token_embeddings + position_embeddings
#         embed = torch.cat((cls_emb, embed), dim=1)  # (b, 1+m, 2*dim)
#         embed = self.blocks(embed)  # (b, 1+m, 2*dim)
#         embed = self.pred_head(embed)[:, 1:, :]  # (b, m, dim)

#         logits = torch.matmul(embed, self.tok_emb_h.weight.T) + self.bias  # (b, m, codebook_size+1)
#         logits = logits[:, :, :-1]  # remove the logit for the mask token.  # (b, m, codebook_size)
#         return logits

#     def forward(self, s_M_l, s_M_h=None, class_condition: Union[None, torch.Tensor] = None):
#         """
#         embed_ind: indices for embedding; (b n)
#         class_condition: (b 1); if None, unconditional sampling is operated.
#         """
#         if self.kind == 'lf':
#             logits = self.forward_lf(s_M_l, class_condition)
#         elif self.kind == 'hf':
#             logits = self.forward_hf(s_M_l, s_M_h, class_condition)
#         else:
#             raise ValueError
#         return logits


def FeedForward(dim, mult = 4):
    """ https://arxiv.org/abs/2110.09456 """

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.LayerNorm(dim),
        weight_norm(nn.Linear(dim, inner_dim, bias = False)),
        nn.GELU(),
        weight_norm(nn.LayerNorm(inner_dim)),
        nn.Linear(inner_dim, dim, bias = False)
    )

def exists(val):
    return val is not None

def l2norm(t):
    return F.normalize(t, dim = -1)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        cross_attend = False,
        # scale = 8,
    ):
        super().__init__()
        # self.scale = scale
        self.heads =  heads
        inner_dim = dim_head * heads

        self.cross_attend = cross_attend
        self.norm = nn.LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        self.to_q = weight_norm(nn.Linear(dim, inner_dim, bias = False))
        self.to_kv = weight_norm(nn.Linear(dim, inner_dim * 2, bias = False))

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = weight_norm(nn.Linear(inner_dim, dim, bias = False))
        self.norm_out = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        context=None,
        mask=None
        ):

        if (exists(context) and not self.cross_attend) or (not exists(context) and self.cross_attend):
            raise AssertionError("Context and cross_attend must either both be present or both be absent.")

        n = x.shape[-2]
        h = self.heads

        x = self.norm(x)

        kv_input = context if self.cross_attend else x

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, 'h 1 d -> b h 1 d', b = x.shape[0]), (nk, nv))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        out = F.scaled_dot_product_attention(q, k, v)  # scale by 1/√d is a default setting.

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.norm_out(out)
        return out

class TransformerBlocks(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult),
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = x + attn(x, mask=mask)
            x = x + ff(x)
        return x

class BidirectionalTransformer(nn.Module):
    def __init__(self,
                 kind: str,
                 num_tokens: int,
                 codebook_sizes: dict,
                 embed_dim: int,
                 hidden_dim: int,
                 n_layers: int,
                 heads: int,
                 ff_mult: int,
                 use_rmsnorm: bool,
                 p_unconditional: float,
                 n_classes: int,
                 model_dropout:float=0.3,
                 emb_dropout:float=0.3,
                 **kwargs):
        """
        :param kind:
        :param num_tokens:
        :param codebook_sizes:
        :param embed_dim:
        :param hidden_dim:
        :param n_layers:
        :param heads:
        :param ff_mult:
        :param use_rmsnorm:
        :param p_unconditional:
        :param n_classes:
        :param num_tokens_l:
        :param kwargs:
        """
        super().__init__()
        kind = kind.lower()
        assert kind in ['lf', 'hf'], 'invalid `kind`.'
        self.kind = kind
        self.num_tokens = num_tokens
        self.n_classes = n_classes
        self.p_unconditional = p_unconditional
        in_dim = embed_dim if kind == 'lf' else 2 * embed_dim
        out_dim = embed_dim
        self.emb_dropout = emb_dropout
        self.mask_token_ind = {'lf':codebook_sizes['lf'], 'hf':codebook_sizes['hf']}

        # token embeddings
        self.tok_emb_l = nn.Embedding(codebook_sizes['lf'] + 1, embed_dim)  # `+1` is for mask-token
        if kind == 'hf':
            self.tok_emb_h = nn.Embedding(codebook_sizes['hf'] + 1, embed_dim)  # `+1` is for mask-token

        # transformer
        self.pos_emb = nn.Embedding(self.num_tokens + 1, in_dim)
        self.class_condition_emb = nn.Embedding(n_classes + 1, in_dim)  # `+1` is for no-condition
        # self.blocks = ContinuousTransformerWrapper(dim_in=in_dim,
        #                                            dim_out=in_dim,
        #                                            max_seq_len=self.num_tokens + 1,
        #                                            use_abs_pos_emb=False,
        #                                            post_emb_norm=True,
        #                                            attn_layers=TFEncoder(
        #                                                pre_norm=True,
        #                                                dim=hidden_dim,
        #                                                depth=n_layers,
        #                                                heads=heads,
        #                                                attn_dim_head=64,
        #                                                use_rmsnorm=use_rmsnorm,
        #                                                ff_mult=ff_mult,
        #                                                layer_dropout=model_dropout,
        #                                                attn_dropout=model_dropout, 
        #                                                ff_dropout=model_dropout,
        #                                            ))
        self.blocks = TransformerBlocks(dim=in_dim, depth=n_layers, dim_head=64, heads=heads, ff_mult=ff_mult)
        self.pred_head = nn.Sequential(*[
            weight_norm(nn.Linear(in_features=in_dim, out_features=out_dim)),
            nn.GELU(),
            nn.LayerNorm(out_dim, eps=1e-12)
        ])
        codebook_size = codebook_sizes['lf'] if kind == 'lf' else codebook_sizes['hf']
        self.bias = nn.Parameter(torch.zeros(self.num_tokens, codebook_size + 1))

        if kind == 'hf':
            self.projector = Upscale(embed_dim, embed_dim, 2*embed_dim)

    def class_embedding(self, class_condition: Union[None, torch.Tensor], batch_size: int, device):
        cond_type = 'uncond' if isinstance(class_condition, type(None)) else 'class-cond'

        if cond_type == 'uncond':
            class_uncondition = repeat(torch.Tensor([self.n_classes]).long().to(device), 'i -> b i', b=batch_size)  # (b 1)
            cls_emb = self.class_condition_emb(class_uncondition)  # (b 1 dim)
            return cls_emb
        elif cond_type == 'class-cond':
            if self.training:
                ind = torch.rand(class_condition.shape).to(device) > self.p_unconditional  # to enable classifier-free guidance
            else:
                ind = torch.ones_like(class_condition, dtype=torch.bool).to(device)
            class_condition = torch.where(ind, class_condition.long(), self.n_classes)  # (b 1)
            cls_emb = self.class_condition_emb(class_condition)  # (b 1 dim)
            return cls_emb

    def _token_emb_dropout(self, s:torch.LongTensor, token_emb:torch.FloatTensor, freq_type:str, p:float):
        mask_ind = (s == self.mask_token_ind[freq_type])[:,:,None]  # (b n 1)
        token_emb_dropout = F.dropout(token_emb, p=p)  # (b n d); to make the prediction process more robust during sampling
        token_emb = torch.where(mask_ind, token_emb, token_emb_dropout)  # (b n d)
        return token_emb

    def forward_lf(self, s_M_l, class_condition: Union[None, torch.Tensor] = None):
        device = s_M_l.device

        token_embeddings = self.tok_emb_l(s_M_l)  # (b n dim)
        if self.training:
            token_embeddings = self._token_emb_dropout(s_M_l, token_embeddings, 'lf', p=self.emb_dropout)  # (b n d)

        cls_emb = self.class_embedding(class_condition, s_M_l.shape[0], device)  # (b 1 dim)

        n = token_embeddings.shape[1]
        position_embeddings = self.pos_emb.weight[:n, :]
        embed = token_embeddings + position_embeddings  # (b, n, dim)
        embed = torch.cat((cls_emb, embed), dim=1)  # (b, 1+n, dim)
        embed = self.blocks(embed)  # (b, 1+n, dim)
        embed = self.pred_head(embed)[:, 1:, :]  # (b, n, dim)

        logits = torch.matmul(embed, self.tok_emb_l.weight.T) + self.bias  # (b, n, codebook_size+1)
        logits = logits[:, :, :-1]  # remove the logit for the mask token.  # (b, n, codebook_size)
        return logits

    def forward_hf(self, s_l, s_M_h, class_condition=None):
        """
        s_M_l (b n)
        s_M_h (b m); m > n
        """
        device = s_l.device

        token_embeddings_l = self.tok_emb_l(s_l)  # (b n dim)
        token_embeddings_h = self.tok_emb_h(s_M_h)  # (b m dim)

        if self.training:
            token_embeddings_l = self._token_emb_dropout(s_l, token_embeddings_l, 'lf', p=self.emb_dropout)
            token_embeddings_h = self._token_emb_dropout(s_M_h, token_embeddings_h, 'hf', p=self.emb_dropout)

        token_embeddings_l = self.projector(token_embeddings_l, upscale_size=token_embeddings_h.shape[1])  # (b m dim)
        token_embeddings = torch.cat((token_embeddings_l, token_embeddings_h), dim=-1)  # (b m 2*dim)

        cls_emb = self.class_embedding(class_condition, s_l.shape[0], device)  # (b 1 2*dim)

        n = token_embeddings.shape[1]
        position_embeddings = self.pos_emb.weight[:n, :]
        embed = token_embeddings + position_embeddings
        embed = torch.cat((cls_emb, embed), dim=1)  # (b, 1+m, 2*dim)
        embed = self.blocks(embed)  # (b, 1+m, 2*dim)
        embed = self.pred_head(embed)[:, 1:, :]  # (b, m, dim)

        logits = torch.matmul(embed, self.tok_emb_h.weight.T) + self.bias  # (b, m, codebook_size+1)
        logits = logits[:, :, :-1]  # remove the logit for the mask token.  # (b, m, codebook_size)
        return logits

    def forward(self, s_M_l, s_M_h=None, class_condition: Union[None, torch.Tensor] = None):
        """
        embed_ind: indices for embedding; (b n)
        class_condition: (b 1); if None, unconditional sampling is operated.
        """
        if self.kind == 'lf':
            logits = self.forward_lf(s_M_l, class_condition)
        elif self.kind == 'hf':
            logits = self.forward_hf(s_M_l, s_M_h, class_condition)
        else:
            raise ValueError
        return logits