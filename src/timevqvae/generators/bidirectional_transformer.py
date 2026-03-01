import math
from typing import Union

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch import Tensor

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


class DiffAttn(nn.Module):
    """
    Differential Attention module.
    
    This module computes attention weights based on the difference between two sets of queries and keys.
    
    Attributes:
    - d (int): The dimensionality of the attention weights.
    - embedding_dim (int): The dimensionality of the input embeddings.
    - W_q (nn.Linear): Linear layer for transforming queries.
    - W_k (nn.Linear): Linear layer for transforming keys.
    - W_v (nn.Linear): Linear layer for transforming values.
    """
    def __init__(self, d: int, embedding_dim: int):
        super(DiffAttn, self).__init__()
        self.d = d
        self.W_q = nn.Linear(embedding_dim, 2 * d)
        self.W_k = nn.Linear(embedding_dim, 2 * d)
        self.W_v = nn.Linear(embedding_dim, d)  # Changed to output d dimensions

    def forward(self, X: Tensor, λ: float) -> Tensor:
        """
        Forward pass of the Differential Attention module.
        
        Args:
        - X (Tensor): Input tensor.
        - λ (float): Scaling factor for the difference.
        
        Returns:
        - Tensor: Output tensor.
        """
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        Q1, Q2 = self.split(Q)
        K1, K2 = self.split(K)

        s = 1 / math.sqrt(self.d)
        
        A1 = (Q1 @ K1.transpose(-1, -2)) * s
        A2 = (Q2 @ K2.transpose(-1, -2)) * s
        
        A1_softmax = F.softmax(A1, dim=-1)
        A2_softmax = F.softmax(A2, dim=-1)
        
        result = (A1_softmax - λ * A2_softmax) @ V
        return result

    @staticmethod
    def split(X: Tensor) -> (Tensor, Tensor):
        """
        Splits the input tensor into two halves along the last dimension.
        
        Args:
        - X (Tensor): Input tensor.
        
        Returns:
        - Tuple[Tensor, Tensor]: Two tensors, each containing half of the input dimensions.
        """
        half_dim = X.shape[-1] // 2
        return X[..., :half_dim], X[..., half_dim:]


class MultiHeadDifferentialAttention(nn.Module):
    """
    Multi-Head Differential Attention module.
    
    This module applies the Differential Attention mechanism multiple times in parallel.
    
    Attributes:
    - h (int): The number of attention heads.
    - d (int): The dimensionality of the attention weights.
    - embedding_dim (int): The dimensionality of the input embeddings.
    - λinit (float): The initial scaling factor for the difference.
    - diff_attn_heads (nn.ModuleList): List of Differential Attention modules.
    - W_o (nn.Linear): Linear layer for output transformation.
    - norm (nn.LayerNorm): Layer normalization module.
    """
    def __init__(self, h: int, d: int, embedding_dim: int, λinit: float=0.05):
        super(MultiHeadDifferentialAttention, self).__init__()
        self.h = h
        self.d = d
        self.λinit = λinit
        self.embedding_dim = embedding_dim
        self.diff_attn_heads = nn.ModuleList([DiffAttn(d, embedding_dim) for _ in range(h)])
        self.W_o = nn.Linear(h * d, embedding_dim)  # Changed to h * d
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, X: Tensor, λ: float=0.1) -> Tensor:
        """
        Forward pass of the Multi-Head Differential Attention module.
        
        Args:
        - X (Tensor): Input tensor.
        - λ (float): Scaling factor for the difference.
        
        Returns:
        - Tensor: Output tensor.
        """
        O_list = [head(X, λ) for head in self.diff_attn_heads]
        
        O_concat = torch.cat(O_list, dim=-1)

        # Apply the output transformation
        result = self.W_o(O_concat)

        # Apply LayerNorm
        result = self.norm(result)

        # Scale by λinit
        result = result * (1 - self.λinit)

        return result
    
    
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

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
    
class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        # args,
        embed_dim:int,
        depth:int,
        num_heads:int,
        head_dim:int,
    ):
        super().__init__()
        # self.args = args
        self.embed_dim = embed_dim
        
        # arg num_heads set to half of Transformer's num_heads
        self.num_heads = num_heads
        
        # arg decoder_kv_attention_heads set to half of Transformer's num_kv_heads if use GQA
        # set to same as num_heads if use normal MHA
        self.num_kv_heads = num_heads  #args.decoder_kv_attention_heads if args.decoder_kv_attention_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = head_dim #embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = weight_norm(nn.Linear(embed_dim, self.head_dim*num_heads*2, bias=False))
        self.k_proj = weight_norm(nn.Linear(embed_dim, self.head_dim*num_heads*2 // self.n_rep, bias=False))
        self.v_proj = weight_norm(nn.Linear(embed_dim, self.head_dim*num_heads*2 // self.n_rep, bias=False))
        self.out_proj = weight_norm(nn.Linear(self.head_dim*num_heads*2, embed_dim, bias=False))

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    
    def forward(
        self,
        x,
        # rel_pos,
        attn_mask=None,
    ):
        """
        x (Tensor)
        """
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        # q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        # k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += attn_mask   
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        return attn
    
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

        for dep_i in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads),
                # MultiHeadDifferentialAttention(h=heads, d=dim_head, embedding_dim=dim),
                # MultiheadDiffAttn(dim, dep_i+1, heads, dim_head),
                FeedForward(dim = dim, mult = ff_mult),
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = x + attn(x)
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
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = TransformerBlocks(dim=hidden_dim, depth=n_layers, dim_head=64, heads=heads, ff_mult=ff_mult)
        codebook_size = codebook_sizes['lf'] if kind == 'lf' else codebook_sizes['hf']
        self.pred_head = nn.Sequential(*[
            weight_norm(nn.Linear(in_features=hidden_dim, out_features=hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            weight_norm(nn.Linear(in_features=hidden_dim, out_features=codebook_size)),
        ])
        # self.bias = nn.Parameter(torch.zeros(self.num_tokens, codebook_size + 1))

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
        embed = self.proj_in(embed)
        embed = self.blocks(embed)  # (b, 1+n, dim)
        logits = self.pred_head(embed)[:, 1:, :]  # (b, n, dim)

        # logits = torch.matmul(embed, self.tok_emb_l.weight.T) + self.bias  # (b, n, codebook_size+1)
        # logits = logits[:, :, :-1]  # remove the logit for the mask token.  # (b, n, codebook_size)
        # return logits
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
        embed = self.proj_in(embed)
        embed = self.blocks(embed)  # (b, 1+m, 2*dim)
        logits = self.pred_head(embed)[:, 1:, :]  # (b, m, dim)

        # logits = torch.matmul(embed, self.tok_emb_h.weight.T) + self.bias  # (b, m, codebook_size+1)
        # logits = logits[:, :, :-1]  # remove the logit for the mask token.  # (b, m, codebook_size)
        # return logits
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