import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from x_transformers import ContinuousTransformerWrapper, Encoder as TFEncoder


def load_pretrained_tok_emb(pretrained_tok_emb, tok_emb, freeze_pretrained_tokens: bool):
    """
    :param pretrained_tok_emb: pretrained token embedding from stage 1
    :param tok_emb: token embedding of the transformer
    :return:
    """
    with torch.no_grad():
        if pretrained_tok_emb != None:
            tok_emb.weight[:-1, :] = pretrained_tok_emb
            if freeze_pretrained_tokens:
                tok_emb.weight[:-1, :].requires_grad = False


class Upscale(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, h_dim:int) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=1, padding=1),
                                  nn.GELU(),
                                  nn.BatchNorm1d(h_dim),
                                  nn.Conv1d(h_dim, out_channels, kernel_size=3, stride=1, padding=1),
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
                 pretrained_tok_emb_l: nn.Parameter = None,
                 pretrained_tok_emb_h: nn.Parameter = None,
                 freeze_pretrained_tokens: bool = False,
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
        :param pretrained_tok_emb_l: if given, the embedding of the transformer is initialized with the pretrained embedding from stage 1; low-frequency
        :param pretrained_tok_emb_h: if given, the embedding of the transformer is initialized with the pretrained embedding from stage 1; high-frequency
        :param freeze_pretrained_tokens:
        :param num_tokens_l:
        :param kwargs:
        """
        super().__init__()
        assert kind in ['LF', 'HF']
        self.kind = kind
        self.num_tokens = num_tokens
        self.n_classes = n_classes
        self.p_unconditional = p_unconditional
        in_dim = embed_dim if kind == 'LF' else 2 * embed_dim
        out_dim = embed_dim
        self.emb_dropout = emb_dropout

        # token embeddings
        self.tok_emb_l = nn.Embedding(codebook_sizes['lf'] + 1, embed_dim)  # `+1` is for mask-token
        load_pretrained_tok_emb(pretrained_tok_emb_l, self.tok_emb_l, freeze_pretrained_tokens)
        if kind == 'HF':
            self.tok_emb_h = nn.Embedding(codebook_sizes['hf'] + 1, embed_dim)  # `+1` is for mask-token
            load_pretrained_tok_emb(pretrained_tok_emb_h, self.tok_emb_h, freeze_pretrained_tokens)

        # transformer
        self.pos_emb = nn.Embedding(self.num_tokens + 1, in_dim)
        self.class_condition_emb = nn.Embedding(n_classes + 1, in_dim)  # `+1` is for no-condition
        self.blocks = ContinuousTransformerWrapper(dim_in=in_dim,
                                                   dim_out=in_dim,
                                                   max_seq_len=self.num_tokens + 1,
                                                   use_abs_pos_emb=False,
                                                   post_emb_norm=True,
                                                   attn_layers=TFEncoder(
                                                       pre_norm=True,
                                                       dim=hidden_dim,
                                                       depth=n_layers,
                                                       heads=heads,
                                                       attn_dim_head=64,
                                                       use_rmsnorm=use_rmsnorm,
                                                       ff_mult=ff_mult,
                                                       layer_dropout=model_dropout,
                                                       attn_dropout=model_dropout, 
                                                       ff_dropout=model_dropout,
                                                   ))
        self.Token_Prediction = nn.Sequential(*[
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim, eps=1e-12)
        ])
        codebook_size = codebook_sizes['lf'] if kind == 'LF' else codebook_sizes['hf']
        self.bias = nn.Parameter(torch.zeros(self.num_tokens, codebook_size + 1))

        if kind == 'HF':
            # self.projector = nn.Conv1d(num_tokens_l, self.num_tokens, kernel_size=1)
            self.projector = Upscale(embed_dim, embed_dim, 2*embed_dim)

    def class_embedding(self, class_condition: Union[None, torch.Tensor], batch_size: int, device):
        if isinstance(class_condition, torch.Tensor):
            # if condition is given (conditional sampling)
            conditional_ind = torch.rand(class_condition.shape).to(device) > self.p_unconditional
            class_uncondition = repeat(torch.Tensor([self.n_classes]).long().to(device), 'i -> b i', b=batch_size)  # (b 1)
            class_condition = torch.where(conditional_ind, class_condition.long(), class_uncondition)  # (b 1)
        else:
            # if condition is not given (unconditional sampling)
            class_uncondition = repeat(torch.Tensor([self.n_classes]).long().to(device), 'i -> b i', b=batch_size)  # (b 1)
            class_condition = class_uncondition
        cls_emb = self.class_condition_emb(class_condition)  # (b 1 dim)
        return cls_emb

    def forward_lf(self, embed_ind_l, class_condition: Union[None, torch.Tensor] = None):
        device = embed_ind_l.device

        token_embeddings = self.tok_emb_l(embed_ind_l)  # (b n dim)
        if self.training:
            token_embeddings = F.dropout(token_embeddings, p=self.emb_dropout)  # to make the LF prediction process more robust during sampling
        cls_emb = self.class_embedding(class_condition, embed_ind_l.shape[0], device)  # (b 1 dim)

        n = token_embeddings.shape[1]
        position_embeddings = self.pos_emb.weight[:n, :]
        embed = token_embeddings + position_embeddings  # (b, n, dim)
        embed = torch.cat((cls_emb, embed), dim=1)  # (b, 1+n, dim)
        embed = self.blocks(embed)  # (b, 1+n, dim)
        embed = self.Token_Prediction(embed)[:, 1:, :]  # (b, n, dim)

        logits = torch.matmul(embed, self.tok_emb_l.weight.T) + self.bias  # (b, n, codebook_size+1)
        logits = logits[:, :, :-1]  # remove the logit for the mask token.  # (b, n, codebook_size)
        return logits

    def forward_hf(self, embed_ind_l, embed_ind_h, class_condition=None):
        """
        embed_ind_l (b n)
        embed_ind_h (b m); m > n
        """
        device = embed_ind_l.device

        token_embeddings_l = self.tok_emb_l(embed_ind_l)  # (b n dim)
        token_embeddings_h = self.tok_emb_h(embed_ind_h)  # (b m dim)

        if self.training:
            token_embeddings_l = F.dropout(token_embeddings_l, p=self.emb_dropout)  # to make the HF prediction process more robust during sampling
            token_embeddings_h = F.dropout(token_embeddings_h, p=self.emb_dropout)  # to make the HF prediction process more robust during sampling

        token_embeddings_l = self.projector(token_embeddings_l, upscale_size=token_embeddings_h.shape[1])  # (b m dim)
        token_embeddings = torch.cat((token_embeddings_l, token_embeddings_h), dim=-1)  # (b m 2*dim)

        cls_emb = self.class_embedding(class_condition, embed_ind_l.shape[0], device)  # (b 1 2*dim)

        n = token_embeddings.shape[1]
        position_embeddings = self.pos_emb.weight[:n, :]
        embed = token_embeddings + position_embeddings
        embed = torch.cat((cls_emb, embed), dim=1)  # (b, 1+m, 2*dim)
        embed = self.blocks(embed)  # (b, 1+m, 2*dim)
        embed = self.Token_Prediction(embed)[:, 1:, :]  # (b, m, dim)

        logits = torch.matmul(embed, self.tok_emb_h.weight.T) + self.bias  # (b, m, codebook_size+1)
        logits = logits[:, :, :-1]  # remove the logit for the mask token.  # (b, m, codebook_size)
        return logits

    def forward(self, embed_ind_l, embed_ind_h=None, class_condition: Union[None, torch.Tensor] = None):
        """
        embed_ind: indices for embedding; (b n)
        class_condition: (b 1); if None, unconditional sampling is operated.
        """
        if self.kind == 'LF':
            logits = self.forward_lf(embed_ind_l, class_condition)
        elif self.kind == 'HF':
            logits = self.forward_hf(embed_ind_l, embed_ind_h, class_condition)
        else:
            raise ValueError
        return logits
