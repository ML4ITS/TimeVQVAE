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
from generators.bidirectional_transformer import BidirectionalTransformer

from encoder_decoders.vq_vae_encdec import VQVAEEncoder
from vector_quantization.vq import VectorQuantize

from experiments.exp_stage1 import ExpStage1
from utils import freeze, timefreq_to_time, time_to_timefreq, quantize, zero_pad_low_freq, zero_pad_high_freq


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

    # def mask_by_random_topk(self, mask_len, probs, temperature=1.0, device='cpu'):
    #     """
    #     mask_len: (b 1)
    #     probs: (b n); also for the confidence scores
    #     """
    #     def log(t, eps=1e-20):
    #         return torch.log(t.clamp(min=eps))
    #
    #     def gumbel_noise(t):
    #         """
    #         Gumbel max trick: https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch
    #         """
    #         noise = torch.zeros_like(t).uniform_(0, 1)
    #         return -log(-log(noise))
    #
    #     confidence = torch.log(probs) + temperature * gumbel_noise(probs).to(device)  # Gumbel max trick
    #     sorted_confidence, _ = torch.sort(confidence, dim=-1)
    #     # Obtains cut off threshold given the mask lengths.
    #     cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
    #     # Masks tokens with lower confidence.
    #     masking = (confidence < cut_off)
    #     # NB! it can mask more than mask_len when there are several confidence scores identical to cut_off.
    #     # the advantage is that we can sample all the lowest scores at once.
    #     return masking

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

        # use ESS (Enhanced Sampling Scheme)
        if self.config['MaskGIT']['ESS']['use']:
            print(' ===== ESS: LF =====')
            t_star, s_star = self.critical_reverse_sampling(s_l, unknown_number_in_the_beginning_l, class_condition, 'lf')
            s_l = self.iterative_decoding_with_self_token_critic(t_star, s_star, 'lf', unknown_number_in_the_beginning_l, class_condition, device)

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
        
        # use ESS (Enhanced Sampling Scheme)
        if self.config['MaskGIT']['ESS']['use']:
            print(' ===== ESS: HF =====')
            t_star, s_star = self.critical_reverse_sampling(s_l, unknown_number_in_the_beginning_h, class_condition, 'hf', s_h=s_h)
            s_h = self.iterative_decoding_with_self_token_critic(t_star, s_star, 'hf', unknown_number_in_the_beginning_h, class_condition, device, s_l=s_l)

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

    # def critical_reverse_sampling(self,
    #                               s_l: torch.Tensor,
    #                               unknown_number_in_the_beginning,
    #                               class_condition: Union[torch.Tensor, None],
    #                               kind: str,
    #                               s_h: torch.Tensor=None,
    #                               ):
    #     """
    #     s: sampled token sequence from the naive iterative decoding.
    #     """
    #     if kind == 'lf':
    #         mask_token_ids = self.mask_token_ids['lf']
    #         transformer = self.transformer_l
    #         vq_model = self.vq_model_l
    #         s = s_l
    #     elif kind == 'hf':
    #         mask_token_ids = self.mask_token_ids['hf']
    #         transformer = self.transformer_h
    #         vq_model = self.vq_model_h
    #         s = s_h
    #     else:
    #         raise ValueError

    #     # compute the confidence scores for s_T
    #     # the scores are used for the step retraction by iteratively removing unrealistic tokens.
    #     confidence_scores = self.compute_confidence_score(kind, s_l, mask_token_ids, vq_model, transformer, class_condition, s_h=s_h)  # (b n)

    #     # find s_{t*}
    #     # t* denotes the step where unrealistic tokens have been removed.
    #     t_star = 1
    #     s_star = None
    #     prev_error = None
    #     error_ratio_hist = deque(maxlen=round(self.T[kind] * self.config['MaskGIT']['ESS']['error_ratio_ma_rate']))
    #     for t in range(1, self.T[kind])[::-1]:
    #         # masking ratio according to the masking scheduler
    #         ratio_t = 1. * (t + 1) / self.T[kind]  # just a percentage e.g. 1 / 12
    #         ratio_tm1 = 1. * t / self.T[kind]  # tm1: t - 1
    #         mask_ratio_t = self.gamma(ratio_t)
    #         mask_ratio_tm1 = self.gamma(ratio_tm1)  # tm1: t - 1

    #         # mask length
    #         mask_len_t = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio_t), 1)
    #         mask_len_tm1 = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio_tm1), 1)

    #         # masking matrices: {True: masking, False: not-masking}
    #         masking_t = self.mask_by_random_topk(mask_len_t, confidence_scores, temperature=0., device=s.device)  # (b n)
    #         masking_tm1 = self.mask_by_random_topk(mask_len_tm1, confidence_scores, temperature=0., device=s.device)  # (b n)
    #         masking = ~((masking_tm1.float() - masking_t.float()).bool())  # (b n); True for everything except the area of interest with False.

    #         # if there's no difference between t-1 and t, ends the retraction.
    #         if masking_t.float().sum() == masking_tm1.float().sum():
    #             t_star = t
    #             s_star = torch.where(masking_t, mask_token_ids, s)  # (b n)
    #             print('no difference between t-1 and t.')
    #             break

    #         # predict s_t given s_{t-1}
    #         s_tm1 = torch.where(masking_tm1, mask_token_ids, s)  # (b n)
    #         # logits = self.masked_prediction(transformer, class_condition, s_tm1)  # (b m k)
    #         if kind == 'lf':
    #             logits = self.masked_prediction(transformer, class_condition, s_tm1)  # (b m k)
    #         elif kind == 'hf':
    #             logits = self.masked_prediction(transformer, class_condition, s_l, s_tm1)  # (b m k)
            
    #         s_t_hat = logits.argmax(dim=-1)  # (b n)

    #         # leave the tokens of interest -- i.e., ds/dt -- only at t
    #         s_t = torch.where(masking, mask_token_ids, s)  # (b n)
    #         s_t_hat = torch.where(masking, mask_token_ids, s_t_hat)  # (b n)

    #         # measure error: distance between z_q_t and z_q_t_hat
    #         z_q_t = F.embedding(s_t[~masking], vq_model._codebook.embed)  # (b n d)
    #         z_q_t_hat = F.embedding(s_t_hat[~masking], vq_model._codebook.embed)  # (b n d)
    #         # error = ((z_q_t - z_q_t_hat) ** 2).mean().cpu().detach().item()
    #         error = (-1*F.cosine_similarity(z_q_t, z_q_t_hat, dim=-1)+1).mean().cpu().detach().item()
    #         # print(f't:{t}, error:{error}')

    #         # error ratio
    #         if t + 1 == self.T[kind]:
    #             error_ratio_ma = 0.
    #             prev_error = error
    #         else:
    #             error_ratio = error / (prev_error + 1e-5)
    #             error_ratio_hist.append(error_ratio)
    #             error_ratio_ma = np.mean(error_ratio_hist)
    #             print(f't:{t} | error:{round(error, 6)} | error_ratio_ma:{round(error_ratio_ma, 6)}')
    #             prev_error = error

    #         # stopping criteria
    #         stopping_threshold = 1.0
    #         if error_ratio_ma > stopping_threshold and (t + 1 != self.T['lf']):
    #             t_star = t
    #             s_star = torch.where(masking_t, mask_token_ids, s)  # (b n)
    #             print('stopped by `error_ratio_ma > threshold`.')
    #             break
    #         if t == 1:
    #             t_star = t
    #             s_star = torch.where(masking_t, mask_token_ids, s)  # (b n)
    #             print('t_star has reached t=1.')
    #             break
    #     print('t_star:', t_star)
    #     return t_star, s_star

    def critical_reverse_sampling(self,
                                  s_l: torch.Tensor,
                                  unknown_number_in_the_beginning,
                                  class_condition: Union[torch.Tensor, None],
                                  kind: str,
                                  s_h: torch.Tensor=None,
                                  ):
        """
        s: sampled token sequence from the naive iterative decoding.
        """
        if kind == 'lf':
            mask_token_ids = self.mask_token_ids['lf']
            transformer = self.transformer_l
            vq_model = self.vq_model_l
            s = s_l
            s_star = s.clone()
            s_star_prev = s.clone()
            temperature = self.choice_temperature_l
        elif kind == 'hf':
            mask_token_ids = self.mask_token_ids['hf']
            transformer = self.transformer_h
            vq_model = self.vq_model_h
            s = s_h
            s_star = s.clone()
            s_star_prev = s.clone()
            temperature = 0.
        else:
            raise ValueError

        # compute the confidence scores for s_T
        # the scores are used for the step retraction by iteratively removing unrealistic tokens.
        confidence_scores = self.compute_confidence_score(kind, s_l, mask_token_ids, vq_model, transformer, class_condition, s_h=s_h)  # (b n)

        # find s_{t*}
        # t* denotes the step where unrealistic tokens have been removed.
        t_star = self.T[kind]
        logprob_prev = -torch.inf
        for t in range(1, self.T[kind])[::-1]:
            # masking ratio according to the masking scheduler
            # ratio_t = 1. * (t + 1) / self.T[kind]  # just a percentage e.g. 1 / 12
            ratio_tm1 = 1. * t / self.T[kind]  # tm1: t - 1
            # mask_ratio_t = self.gamma(ratio_t)
            mask_ratio_tm1 = self.gamma(ratio_tm1)  # tm1: t - 1

            # mask length
            # mask_len_t = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio_t), 1)
            mask_len_tm1 = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio_tm1), 1)

            # masking matrices: {True: masking, False: not-masking}
            # masking_t = self.mask_by_random_topk(mask_len_t, confidence_scores, temperature=0., device=s.device)  # (b n)
            masking_tm1 = self.mask_by_random_topk(mask_len_tm1, confidence_scores, temperature=temperature, device=s.device)  # (b n)
            # masking = ~((masking_tm1.float() - masking_t.float()).bool())  # (b n); True for everything except the area of interest with False.

            s_star = torch.where(masking_tm1, mask_token_ids, s)

            # predict s_t given s_{t-1}
            s_tm1 = torch.where(masking_tm1, mask_token_ids, s)  # (b n)
            if kind == 'lf':
                logits = self.masked_prediction(transformer, class_condition, s_tm1)  # (b n k)
            elif kind == 'hf':
                logits = self.masked_prediction(transformer, class_condition, s_l, s_tm1)  # (b n k)
            prob = torch.nn.functional.softmax(logits, dim=-1)  # (b n K)
            logprob = prob.clamp_min(1.e-5).log10()  # (b n k)
            logprob = torch.gather(logprob, dim=-1, index=s.unsqueeze(-1)).squeeze(-1)  # (b n)
            print('t:', t)
            # print('masking_tm1:', masking_tm1)
            # print('masking_tm1.int().mean():', masking_tm1.float().mean().item())
            logprob = logprob[masking_tm1]
            logprob = logprob.mean().cpu().detach().item()

            # stopping criteria
            if (t == self.T[kind]-1) or (logprob > logprob_prev - 0.05*logprob_prev):
            # if (t != 1):
                logprob_prev = logprob
                t_star = t
                s_star_prev = s_star.clone()
                pass
            else:
                break

            # # measure error: distance between z_q_t and z_q_t_hat
            # z_q_t = F.embedding(s_t[~masking], vq_model._codebook.embed)  # (b n d)
            # z_q_t_hat = F.embedding(s_t_hat[~masking], vq_model._codebook.embed)  # (b n d)
            # error = (-1*F.cosine_similarity(z_q_t, z_q_t_hat, dim=-1)+1).mean().cpu().detach().item()

            # # error ratio
            # if t + 1 == self.T[kind]:
            #     error_ratio_ma = 0.
            #     prev_error = error
            # else:
            #     error_ratio = error / (prev_error + 1e-5)
            #     error_ratio_hist.append(error_ratio)
            #     error_ratio_ma = np.mean(error_ratio_hist)
            #     print(f't:{t} | error:{round(error, 6)} | error_ratio_ma:{round(error_ratio_ma, 6)}')
            #     prev_error = error

            # # stopping criteria
            # stopping_threshold = 1.0
            # if error_ratio_ma > stopping_threshold and (t + 1 != self.T['lf']):
            #     t_star = t
            #     s_star = torch.where(masking_t, mask_token_ids, s)  # (b n)
            #     print('stopped by `error_ratio_ma > threshold`.')
            #     break
            # if t == 1:
            #     t_star = t
            #     s_star = torch.where(masking_t, mask_token_ids, s)  # (b n)
            #     print('t_star has reached t=1.')
            #     break
        print('t_star:', t_star)
        return t_star, s_star_prev
    
    # def iterative_decoding_with_self_token_critic(self,
    #                                               t_star,
    #                                               s_star,
    #                                               kind: str,
    #                                               unknown_number_in_the_beginning,
    #                                               class_condition: Union[torch.Tensor, None],
    #                                               device,
    #                                               s_l=None
    #                                               ):
    #     if kind == 'lf':
    #         mask_token_ids = self.mask_token_ids['lf']
    #         transformer = self.transformer_l
    #         vq_model = self.vq_model_l
    #         choice_temperature = self.choice_temperature_l
    #     elif kind == 'hf':
    #         mask_token_ids = self.mask_token_ids['hf']
    #         transformer = self.transformer_h
    #         vq_model = self.vq_model_h
    #         choice_temperature = self.choice_temperature_h
    #     else:
    #         raise ValueError

    #     s = s_star
    #     for t in range(t_star, self.T[kind]):
    #         if kind == 'lf':
    #             logits = self.masked_prediction(transformer, class_condition, s)  # (b n k)
    #         elif kind == 'hf':
    #             logits = self.masked_prediction(transformer, class_condition, s_l, s)  # (b n k)

    #         sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()  # (b n)

    #         # create masking according to `t`
    #         ratio = 1. * (t + 1) / self.T[kind]  # just a percentage e.g. 1 / 12
    #         mask_ratio = self.gamma(ratio)

    #         # compute the confidence scores for s_t
    #         if kind == 'lf':
    #             confidence_scores = self.compute_confidence_score(kind, sampled_ids, mask_token_ids, vq_model, transformer, class_condition)  # (b n)
    #         elif kind == 'hf':
    #             confidence_scores = self.compute_confidence_score(kind, s_l, mask_token_ids, vq_model, transformer, class_condition, s_h=sampled_ids)  # (b n)

    #         mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # number of tokens that are to be masked;  (b,)
    #         mask_len = torch.clip(mask_len, min=0.)  # `mask_len` should be equal or larger than zero.

    #         # Adds noise for randomness
    #         masking = self.mask_by_random_topk(mask_len, confidence_scores, temperature=choice_temperature * (1. - ratio), device=device)

    #         # Masks tokens with lower confidence.
    #         s = torch.where(masking, mask_token_ids, sampled_ids)  # (b n)
    #     return s

    def iterative_decoding_with_self_token_critic(self,
                                                  t_star,
                                                  s_star,
                                                  kind: str,
                                                  unknown_number_in_the_beginning,
                                                  class_condition: Union[torch.Tensor, None],
                                                  device,
                                                  s_l=None
                                                  ):
        if kind == 'lf':
            mask_token_ids = self.mask_token_ids['lf']
            transformer = self.transformer_l
            vq_model = self.vq_model_l
            choice_temperature = self.choice_temperature_l
        elif kind == 'hf':
            mask_token_ids = self.mask_token_ids['hf']
            transformer = self.transformer_h
            vq_model = self.vq_model_h
            choice_temperature = self.choice_temperature_h
        else:
            raise ValueError

        s = s_star
        for t in range(t_star, self.T[kind]):
            if kind == 'lf':
                logits = self.masked_prediction(transformer, class_condition, s)  # (b n k)
            elif kind == 'hf':
                logits = self.masked_prediction(transformer, class_condition, s_l, s)  # (b n k)
            
            sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()  # (b n)
            unknown_map = (s == mask_token_ids)  # which tokens need to be sampled; (b n)
            sampled_ids = torch.where(unknown_map, sampled_ids, s)  # keep the previously-sampled tokens; (b n)

            # create masking according to `t`
            ratio = 1. * (t + 1) / self.T[kind]  # just a percentage e.g. 1 / 12
            mask_ratio = self.gamma(ratio)

            if kind == 'lf':
                selected_probs = self.compute_confidence_score(kind, sampled_ids, mask_token_ids, vq_model, transformer, class_condition)  # (b n)
            elif kind == 'hf':
                selected_probs = self.compute_confidence_score(kind, s_l, mask_token_ids, vq_model, transformer, class_condition, s_h=sampled_ids)  # (b n)
            _CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to(device)
            selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS)  # assign inf probability to the previously-selected tokens; (b n)

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # number of tokens that are to be masked;  (b,)
            mask_len = torch.clip(mask_len, min=0.)  # `mask_len` should be equal or larger than zero.

            # Adds noise for randomness
            masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=choice_temperature * (1. - ratio), device=device)
            # masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=0., device=device)

            # Masks tokens with lower confidence.
            s = torch.where(masking, mask_token_ids, sampled_ids)  # (b n)

        return s
    
    # def compute_confidence_score(self, s, mask_token_ids, vq_model, transformer, class_condition):
    #     confidence_scores = torch.zeros_like(s).float()  # (b n)
    #     for n in range(confidence_scores.shape[-1]):
    #         s_m = copy.deepcopy(s)  # (b n)
    #         s_m[:, n] = mask_token_ids  # (b n); masking the n-th token to measure the confidence score for that token.
    #         logits = self.masked_prediction(transformer, class_condition, s_m)  # (b n k)
    #         logits = torch.nn.functional.softmax(logits, dim=-1)  # (b n K)

    #         true_tokens = s[:, n]  # (b,)
    #         logits = logits[:, n]  # (b, K)
    #         pred_tokens = logits.argmax(dim=-1)  # (b,)

    #         z_q_true = vq_model._codebook.embed[true_tokens]  # (b, dim)
    #         z_q_pred = vq_model._codebook.embed[pred_tokens]  # (b, dim)
    #         dist = torch.sum((z_q_true - z_q_pred) ** 2, dim=-1)  # (b,)
    #         confidence_scores[:, n] = -1 * dist  # confidence score for the n-th token
    #     confidence_scores = torch.nn.functional.softmax(confidence_scores, dim=-1)  # (b n)
    #     return confidence_scores

    def compute_confidence_score(self, kind, s_l, mask_token_ids, vq_model, transformer, class_condition, s_h=None):
        
        if kind == 'lf':
            s = s_l
        elif kind == 'hf':
            s = s_h

        confidence_scores = torch.zeros_like(s).float()  # (b n)
        for n in range(confidence_scores.shape[-1]):
            s_m = copy.deepcopy(s)  # (b n)
            s_m[:, n] = mask_token_ids  # (b n); masking the n-th token to measure the confidence score for that token.
            if kind == 'lf':
                logits = self.masked_prediction(transformer, class_condition, s_l)  # (b n k)
            elif kind == 'hf':
                logits = self.masked_prediction(transformer, class_condition, s_l, s_h)  # (b n k)
            prob = torch.nn.functional.softmax(logits, dim=-1)  # (b n K)
            # logprob = prob.clamp_min(1e-5).log10()  # (b n k)

            selected_prob = torch.gather(prob, dim=2, index=s.unsqueeze(-1)).squeeze(-1)  # (b n)
            selected_prob = selected_prob[:, n]  # (b,)

            confidence_scores[:, n] = selected_prob
        return confidence_scores





        #     true_tokens = s[:, n]  # (b,)
        #     logits = logits[:, n]  # (b, K)
        #     pred_tokens = logits.argmax(dim=-1)  # (b,)

        #     z_q_true = vq_model._codebook.embed[true_tokens]  # (b, dim)
        #     z_q_pred = vq_model._codebook.embed[pred_tokens]  # (b, dim)
        #     dist = torch.sum((z_q_true - z_q_pred) ** 2, dim=-1)  # (b,)
        #     confidence_scores[:, n] = -1 * dist  # confidence score for the n-th token
        # confidence_scores = torch.nn.functional.softmax(confidence_scores, dim=-1)  # (b n)
        # return confidence_scores
