import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from x_transformers import ContinuousTransformerWrapper, Encoder as TFEncoder


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
                 n_enc_layers: int,
                 n_dec_layers: int,
                 heads: int,
                 ff_mult: int,
                 use_rmsnorm: bool,
                 p_unconditional: float,
                 n_classes: int,
                 model_dropout:float=0.3,
                 emb_dropout:float=0.3,
                 n_register_tokens:int=4,
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
        self.mask_token_l = self.tok_emb_l.weight[self.mask_token_ind['lf'],:][None,None,:]  # (1 1 d)
        if kind == 'hf':
            self.tok_emb_h = nn.Embedding(codebook_sizes['hf'] + 1, embed_dim)  # `+1` is for mask-token
            self.mask_token_h = self.tok_emb_h.weight[self.mask_token_ind['hf'],:][None,None,:]  # (1 1 d)

        # transformer
        self.n_aux_tokens = 1 + n_register_tokens
        self.class_condition_emb = nn.Embedding(n_classes + 1, in_dim)  # `+1` is for no-condition
        self.register_emb = nn.Parameter(torch.randn(n_register_tokens, in_dim))
        
        self.pos_emb = nn.Embedding(self.num_tokens, in_dim)
        self.pos_emb_proj = nn.Linear(in_dim, out_dim) if kind == 'hf' else nn.Identity()
        
        self.enc_blocks = ContinuousTransformerWrapper(dim_in=in_dim,
                                                   dim_out=out_dim,
                                                   max_seq_len=self.num_tokens + self.n_aux_tokens,
                                                   use_abs_pos_emb=False,
                                                   post_emb_norm=True,
                                                   attn_layers=TFEncoder(
                                                       pre_norm=True,
                                                       dim=hidden_dim,
                                                       depth=n_enc_layers,
                                                       heads=heads,
                                                       attn_dim_head=64,
                                                       use_rmsnorm=use_rmsnorm,
                                                       ff_mult=ff_mult,
                                                       layer_dropout=model_dropout,
                                                       attn_dropout=model_dropout, 
                                                       ff_dropout=model_dropout,
                                                   ))
        self.dec_blocks = ContinuousTransformerWrapper(dim_in=out_dim,
                                                   dim_out=out_dim,
                                                   max_seq_len=self.num_tokens + self.n_aux_tokens,
                                                   use_abs_pos_emb=False,
                                                   post_emb_norm=True,
                                                   attn_layers=TFEncoder(
                                                       pre_norm=True,
                                                       dim=hidden_dim,
                                                       depth=n_dec_layers,
                                                       heads=heads,
                                                       attn_dim_head=64,
                                                       use_rmsnorm=use_rmsnorm,
                                                       ff_mult=ff_mult,
                                                       layer_dropout=model_dropout,
                                                       attn_dropout=model_dropout, 
                                                       ff_dropout=model_dropout,
                                                   ))
        self.pred_head = nn.Sequential(*[
            nn.Linear(in_features=out_dim, out_features=embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim, eps=1e-12)
        ])
        # codebook_size = codebook_sizes['lf'] if kind == 'lf' else codebook_sizes['hf']
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

    def forward_decoder(self, x, ids_restore, mask_token, skip_restore_seq_dim:bool=False):
        """
        x: (b, n`, d) where n` < n
        ids_restore: (b n)
        """
        if not skip_restore_seq_dim:
            # Append mask tokens to sequence
            mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] + self.n_aux_tokens - x.shape[1], 1)  # Create mask tokens (b, num_patches + 1 - n, decoder_embed_dim)
            x_ = torch.cat([x[:, self.n_aux_tokens:, :], mask_tokens], dim=1)  # Concatenate mask tokens (without class token) (b, num_patches, decoder_embed_dim)
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # Unshuffle (b, num_patches, decoder_embed_dim)
            x = torch.cat([x[:, :self.n_aux_tokens, :], x_], dim=1)  # Append class token back (b, num_patches + 1, decoder_embed_dim)

        # Add positional embedding
        # n = x.shape[1]
        # x = x + self.decoder_pos_embed[:,:n,:]
        # n = x.shape[1] - self.n_aux_tokens
        # pos_emb = self.pos_emb.weight[:n, :]   # (n d)
        pos_emb = self.pos_emb.weight[None,:,:]   # (1 n d)
        pos_emb = self.pos_emb_proj(pos_emb)  # (1 n d)
        x[:,self.n_aux_tokens:,:] = x[:,self.n_aux_tokens:,:] + pos_emb

        # Apply Transformer blocks
        x = self.dec_blocks(x)
        return x
    
    def forward_lf(self, s_M_l, class_condition:Union[None,torch.Tensor]=None, mask=None, ids_restore=None):
        """
        mask: (b n`); 0 is keep, 1 is remove
        """
        b = s_M_l.shape[0]
        device = s_M_l.device

        token_embeddings = self.tok_emb_l(s_M_l)  # (b n dim)
        if self.training:
            token_embeddings = self._token_emb_dropout(s_M_l, token_embeddings, 'lf', p=self.emb_dropout)  # (b n d)

        cls_emb = self.class_embedding(class_condition, b, device)  # (b 1 dim)
        registers = repeat(self.register_emb, 'k d -> b k d', b=b)  # (b, n_regs, d)

        # add pos emb
        position_embeddings = repeat(self.pos_emb.weight, 'n d -> b n d', b=b)  # (b n d)
        d = position_embeddings.shape[-1]
        position_embeddings = position_embeddings[repeat(~mask, 'b n -> b n d', d=d)]
        position_embeddings = rearrange(position_embeddings, '(b n d) -> b n d', b=b, d=d)
        embed = token_embeddings + position_embeddings  # (b, n, dim)
        
        # concat aux tokens
        embed = torch.cat((registers, cls_emb, embed), dim=1)  # (b, n_regs+1+n, dim)
        
        embed = self.enc_blocks(embed)
        embed = self.forward_decoder(embed, ids_restore, self.mask_token_l)

        embed = self.pred_head(embed)[:, self.n_aux_tokens:, :]  # (b, n, dim)
        logits = torch.matmul(embed, self.tok_emb_l.weight.T)  # (b, n, codebook_size+1)
        logits = logits[:, :, :-1]  # remove the logit for the mask token.  # (b, n, codebook_size)
        return logits

    def forward_hf(self, s_l, s_M_h, class_condition=None, mask=None, ids_restore=None):
        """
        s_l (b n)
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
        registers = repeat(self.register_emb, 'k d -> b k d', b=s_l.shape[0])  # (b, n_regs, 2*d)

        # add pos emb
        position_embeddings = self.pos_emb.weight
        embed = token_embeddings + position_embeddings

        # concat aux tokens
        embed = torch.cat((registers, cls_emb, embed), dim=1)  # (b, n_regs+1+m, 2*dim)
        
        embed = self.enc_blocks(embed)
        embed = self.forward_decoder(embed, ids_restore, self.mask_token_h, skip_restore_seq_dim=True)

        embed = self.pred_head(embed)[:, self.n_aux_tokens:, :]  # (b, m, dim)
        logits = torch.matmul(embed, self.tok_emb_h.weight.T)  # (b, m, codebook_size+1)
        logits = logits[:, :, :-1]  # remove the logit for the mask token.  # (b, m, codebook_size)
        return logits

    def forward(self, s_M_l, s_M_h=None, class_condition: Union[None, torch.Tensor] = None, mask=None, ids_restore=None):
        """
        embed_ind: indices for embedding; (b n)
        class_condition: (b 1); if None, unconditional sampling is operated.
        """
        if isinstance(ids_restore, type(None)):
            assert False, 'ids_restore must be given.'
        if isinstance(mask, type(None)):
            assert False, 'mask must be given.'

        if self.kind == 'lf':
            logits = self.forward_lf(s_M_l, class_condition, mask, ids_restore)
        elif self.kind == 'hf':
            logits = self.forward_hf(s_M_l, s_M_h, class_condition, mask, ids_restore)
        else:
            raise ValueError
        return logits
