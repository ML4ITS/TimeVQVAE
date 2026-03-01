import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import tempfile
from typing import Union
from collections import deque

from einops import repeat, rearrange
from typing import Callable
from timevqvae.generators.bidirectional_transformer import BidirectionalTransformer

from timevqvae.encoder_decoders.vq_vae_encdec import VQVAEEncoder
from timevqvae.vector_quantization.vq import VectorQuantize

from timevqvae.experiments.exp_stage1 import ExpStage1
from timevqvae.utils.nn import freeze, quantize
from timevqvae.utils.signal import (
    time_to_timefreq,
    timefreq_to_time,
    zero_pad_high_freq,
    zero_pad_low_freq,
)


class MaskGIT(nn.Module):
    """
    ref: https://github.com/dome272/MaskGIT-pytorch/blob/cff485ad3a14b6ed5f3aa966e045ea2bc8c68ad8/transformer.py#L11
    """

    def __init__(self,
                 dataset_name: str,
                 in_channels:int,
                 input_length: int,
                 choice_temperatures: dict,
                 T: dict,
                 config: dict,
                 n_classes: int,
                 **kwargs):
        super().__init__()
        self.choice_temperature_l = choice_temperatures['lf']
        self.choice_temperature_h = choice_temperatures['hf']
        self.T = T
        self.config = config
        self.n_classes = n_classes
        self.n_fft = config['VQ-VAE']['n_fft']
        self.cfg_scale = config['MaskGIT']['cfg_scale']

        self.mask_token_ids = {'lf': config['VQ-VAE']['codebook_sizes']['lf'], 'hf': config['VQ-VAE']['codebook_sizes']['hf']}
        self.gamma = self.gamma_func("cosine")

        # load the staeg1 model
        self.stage1 = ExpStage1.load_from_checkpoint(os.path.join('saved_models', f'stage1-{dataset_name}.ckpt'), 
                                                     in_channels=in_channels,
                                                     input_length=input_length, 
                                                     config=config,
                                                     map_location='cpu')
        freeze(self.stage1)
        self.stage1.eval()

        self.encoder_l = self.stage1.encoder_l
        self.decoder_l = self.stage1.decoder_l
        self.vq_model_l = self.stage1.vq_model_l
        self.encoder_h = self.stage1.encoder_h
        self.decoder_h = self.stage1.decoder_h
        self.vq_model_h = self.stage1.vq_model_h

        # token lengths
        self.num_tokens_l = self.encoder_l.num_tokens.item()
        self.num_tokens_h = self.encoder_h.num_tokens.item()

        # latent space dim
        self.H_prime_l, self.H_prime_h = self.encoder_l.H_prime.item(), self.encoder_h.H_prime.item()
        self.W_prime_l, self.W_prime_h = self.encoder_l.W_prime.item(), self.encoder_h.W_prime.item()

        # transformers / prior models
        emb_dim = self.config['encoder']['hid_dim']
        self.transformer_l = BidirectionalTransformer('lf',
                                                      self.num_tokens_l,
                                                      config['VQ-VAE']['codebook_sizes'],
                                                      emb_dim,
                                                      **config['MaskGIT']['prior_model_l'],
                                                      n_classes=n_classes,
                                                      )

        self.transformer_h = BidirectionalTransformer('hf',
                                                      self.num_tokens_h,
                                                      config['VQ-VAE']['codebook_sizes'],
                                                      emb_dim,
                                                      **config['MaskGIT']['prior_model_h'],
                                                      n_classes=n_classes,
                                                      num_tokens_l=self.num_tokens_l,
                                                      )

    def load(self, model, dirname, fname):
        """
        model: instance
        path_to_saved_model_fname: path to the ckpt file (i.e., trained model)
        """
        try:
            model.load_state_dict(torch.load(dirname.joinpath(fname)))
        except FileNotFoundError:
            dirname = Path(tempfile.gettempdir())
            model.load_state_dict(torch.load(dirname.joinpath(fname)))

    @torch.no_grad()
    def encode_to_z_q(self, x, encoder: VQVAEEncoder, vq_model: VectorQuantize, svq_temp:Union[float,None]=None):
        """
        encode x to zq

        x: (b c l)
        """
        z = encoder(x)
        zq, s, _, _ = quantize(z, vq_model, svq_temp=svq_temp)  # (b c h w), (b (h w) h), ...
        return zq, s
    
    def masked_prediction(self, transformer, class_condition, *s_in):
        """
        masked prediction with classifier-free guidance
        """
        if isinstance(class_condition, type(None)):
            # unconditional 
            logits_null = transformer(*s_in, class_condition=None)  # (b n k)
            return logits_null
        else:
            # class-conditional
            if self.cfg_scale == 1.0:
                logits = transformer(*s_in, class_condition=class_condition)  # (b n k)
            else:
                # with CFG
                logits_null = transformer(*s_in, class_condition=None)
                logits = transformer(*s_in, class_condition=class_condition)  # (b n k)
                logits = logits_null + self.cfg_scale * (logits - logits_null)
            return logits

    def forward(self, x, y):
        """
        x: (B, C, L)
        y: (B, 1)
        straight from [https://github.com/dome272/MaskGIT-pytorch/blob/main/transformer.py]
        """
        self.encoder_l.eval()
        self.vq_model_l.eval()
        self.encoder_h.eval()
        self.vq_model_h.eval()
        
        device = x.device
        _, s_l = self.encode_to_z_q(x, self.encoder_l, self.vq_model_l)  # (b n)
        _, s_h = self.encode_to_z_q(x, self.encoder_h, self.vq_model_h)  # (b m)

        # mask tokens
        s_l_M, mask_l = self._randomly_mask_tokens(s_l, self.mask_token_ids['lf'], device)  # (b n), (b n) where 0 for masking and 1 for un-masking
        s_h_M, mask_h = self._randomly_mask_tokens(s_h, self.mask_token_ids['hf'], device)  # (b n), (b n) where 0 for masking and 1 for un-masking

        # prediction
        logits_l = self.masked_prediction(self.transformer_l, y, s_l_M)  # (b n k)
        logits_h = self.masked_prediction(self.transformer_h, y, s_l, s_h_M)
        
        # maksed prediction loss
        logits_l_on_mask = logits_l[~mask_l]  # (bm k) where m < n
        s_l_on_mask = s_l[~mask_l]  # (bm) where m < n
        mask_pred_loss_l = F.cross_entropy(logits_l_on_mask.float(), s_l_on_mask.long())
        
        logits_h_on_mask = logits_h[~mask_h]  # (bm k) where m < n
        s_h_on_mask = s_h[~mask_h]  # (bm) where m < n
        mask_pred_loss_h = F.cross_entropy(logits_h_on_mask.float(), s_h_on_mask.long())

        mask_pred_loss = mask_pred_loss_l + mask_pred_loss_h
        return mask_pred_loss, (mask_pred_loss_l, mask_pred_loss_h)

    def _randomly_mask_tokens(self, s, mask_token_id, device):
        """
        s: token set
        """
        b, n = s.shape
        
        # sample masking indices
        ratio = np.random.uniform(0, 1, (b,))  # (b,)
        n_unmasks = np.floor(self.gamma(ratio) * n)  # (b,)
        n_unmasks = np.clip(n_unmasks, a_min=0, a_max=n-1).astype(int)  # ensures that there's at least one masked token
        rand = torch.rand((b, n), device=device)  # (b n)
        mask = torch.zeros((b, n), dtype=torch.bool, device=device)  # (b n)

        for i in range(b):
            ind = rand[i].topk(n_unmasks[i], dim=-1).indices
            mask[i].scatter_(dim=-1, index=ind, value=True)

        # mask the token set
        masked_indices = mask_token_id * torch.ones((b, n), device=device)  # (b n)
        s_M = mask * s + (~mask) * masked_indices  # (b n); `~` reverses bool-typed data
        return s_M.long(), mask
    
    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError

    def create_input_tokens_normal(self, num, num_tokens, mask_token_ids, device):
        """
        returns masked tokens
        """
        blank_tokens = torch.ones((num, num_tokens), device=device)
        masked_tokens = mask_token_ids * blank_tokens
        return masked_tokens.to(torch.int64)

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0, device='cpu'):
        """
        mask_len: (b 1)
        probs: (b n); also for the confidence scores

        This version keeps `mask_len` exactly.
        """
        def log(t, eps=1e-20):
            return torch.log(t.clamp(min=eps))

        def gumbel_noise(t):
            """
            Gumbel max trick: https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch
            """
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -log(-log(noise))

        confidence = torch.log(probs + 1e-5) + temperature * gumbel_noise(probs).to(device)  # Gumbel max trick; 1e-5 for numerical stability; (b n)
        mask_len_unique = int(mask_len.unique().item())
        masking_ind = torch.topk(confidence, k=mask_len_unique, dim=-1, largest=False).indices  # (b k)
        masking = torch.zeros_like(confidence).to(device)  # (b n)
        for i in range(masking_ind.shape[0]):
            masking[i, masking_ind[i].long()] = 1.
        masking = masking.bool()
        return masking

    def first_pass(self,
                   s_l: torch.Tensor,
                   unknown_number_in_the_beginning_l,
                   class_condition: Union[torch.Tensor, None],
                   gamma: Callable,
                   device):
        for t in range(self.T['lf']):
            logits_l = self.masked_prediction(self.transformer_l, class_condition, s_l)  # (b n k)

            sampled_ids = torch.distributions.categorical.Categorical(logits=logits_l).sample()  # (b n)
            unknown_map = (s_l == self.mask_token_ids['lf'])  # which tokens need to be sampled; (b n)
            sampled_ids = torch.where(unknown_map, sampled_ids, s_l)  # keep the previously-sampled tokens; (b n)

            # create masking according to `t`
            ratio = 1. * (t + 1) / self.T['lf']  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)

            probs = F.softmax(logits_l, dim=-1)  # convert logits into probs; (b n K)
            selected_probs = torch.gather(probs, dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze()  # get probability for the selected tokens; p(\hat{s}(t) | \hat{s}_M(t)); (b n)
            _CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to(device)
            selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS)  # assign inf probability to the previously-selected tokens; (b n)

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning_l * mask_ratio), 1)  # number of tokens that are to be masked;  (b,)
            mask_len = torch.clip(mask_len, min=0.)  # `mask_len` should be equal or larger than zero.

            # Adds noise for randomness
            masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=self.choice_temperature_l * (1. - ratio), device=device)

            # Masks tokens with lower confidence.
            s_l = torch.where(masking, self.mask_token_ids['lf'], sampled_ids)  # (b n)

        return s_l

    def second_pass(self,
                    s_l: torch.Tensor,
                    s_h: torch.Tensor,
                    unknown_number_in_the_beginning_h,
                    class_condition: Union[torch.Tensor, None],
                    gamma: Callable,
                    device):
        for t in range(self.T['hf']):
            logits_h = self.masked_prediction(self.transformer_h, class_condition, s_l, s_h)  # (b m k)

            sampled_ids = torch.distributions.categorical.Categorical(logits=logits_h).sample()  # (b m)
            unknown_map = (s_h == self.mask_token_ids['hf'])  # which tokens need to be sampled; (b m)
            sampled_ids = torch.where(unknown_map, sampled_ids, s_h)  # keep the previously-sampled tokens; (b m)

            # create masking according to `t`
            ratio = 1. * (t + 1) / self.T['hf']  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)

            probs = F.softmax(logits_h, dim=-1)  # convert logits into probs; (b m K)
            selected_probs = torch.gather(probs, dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze()  # get probability for the selected tokens; p(\hat{s}(t) | \hat{s}_M(t)); (b m)
            _CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to(device)
            selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS)  # assign inf probability to the previously-selected tokens; (b m)

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning_h * mask_ratio), 1)  # number of tokens that are to be masked;  (b,)
            mask_len = torch.clip(mask_len, min=0.)  # `mask_len` should be equal or larger than zero.

            # Adds noise for randomness
            masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=self.choice_temperature_h * (1. - ratio), device=device)

            # Masks tokens with lower confidence.
            s_h = torch.where(masking, self.mask_token_ids['hf'], sampled_ids)  # (b n)
        
        return s_h

    @torch.no_grad()
    def iterative_decoding(self, num=1, mode="cosine", class_index=None, device='cpu'):
        """
        It performs the iterative decoding and samples token indices for LF and HF.
        :param num: number of samples
        :return: sampled token indices for LF and HF
        """
        s_l = self.create_input_tokens_normal(num, self.num_tokens_l, self.mask_token_ids['lf'], device)  # (b n)
        s_h = self.create_input_tokens_normal(num, self.num_tokens_h, self.mask_token_ids['hf'], device)  # (b n)

        unknown_number_in_the_beginning_l = torch.sum(s_l == self.mask_token_ids['lf'], dim=-1)  # (b,)
        unknown_number_in_the_beginning_h = torch.sum(s_h == self.mask_token_ids['hf'], dim=-1)  # (b,)
        gamma = self.gamma_func(mode)
        class_condition = repeat(torch.Tensor([class_index]).int().to(device), 'i -> b i', b=num) if class_index != None else None  # (b 1)

        s_l = self.first_pass(s_l, unknown_number_in_the_beginning_l, class_condition, gamma, device)
        s_h = self.second_pass(s_l, s_h, unknown_number_in_the_beginning_h, class_condition, gamma, device)
        return s_l, s_h

    def decode_token_ind_to_timeseries(self, s: torch.Tensor, frequency: str, return_representations: bool = False):
        """
        It takes token embedding indices and decodes them to time series.
        :param s: token embedding index
        :param frequency:
        :param return_representations:
        :return:
        """
        self.eval()
        frequency = frequency.lower()
        assert frequency in ['lf', 'hf']

        vq_model = self.vq_model_l if frequency == 'lf' else self.vq_model_h
        decoder = self.decoder_l if frequency == 'lf' else self.decoder_h
        zero_pad = zero_pad_high_freq if frequency == 'lf' else zero_pad_low_freq

        zq = F.embedding(s, vq_model._codebook.embed)  # (b n d)
        zq = vq_model.project_out(zq)  # (b n c)
        zq = rearrange(zq, 'b n c -> b c n')  # (b c n) == (b c (h w))
        H_prime = self.H_prime_l if frequency == 'lf' else self.H_prime_h
        W_prime = self.W_prime_l if frequency == 'lf' else self.W_prime_h
        zq = rearrange(zq, 'b c (h w) -> b c h w', h=H_prime, w=W_prime)

        xhat = decoder(zq)

        if return_representations:
            return xhat, zq
        else:
            return xhat

