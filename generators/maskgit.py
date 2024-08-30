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
    
    def masked_prediction(self, transformer, mask, ids_restore, class_condition, *s_in):
        """
        masked prediction with classifier-free guidance
        """
        if isinstance(class_condition, type(None)):
            # unconditional 
            logits_null = transformer(*s_in, class_condition=None, mask=mask, ids_restore=ids_restore)  # (b n k)
            return logits_null
        else:
            # class-conditional
            if self.cfg_scale == 1.0:
                logits = transformer(*s_in, class_condition=class_condition, mask=mask, ids_restore=ids_restore)  # (b n k)
            else:
                # with CFG
                logits_null = transformer(*s_in, class_condition=None, mask=mask, ids_restore=ids_restore)
                logits = transformer(*s_in, class_condition=class_condition, mask=mask, ids_restore=ids_restore)  # (b n k)
                logits = logits_null + self.cfg_scale * (logits - logits_null)
            return logits

    def random_masking(self, x, mask_ratio, freq:str):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L], sequence = (b n)
        """
        N, L = x.shape  # batch, length (N, L)
        # len_keep = round(L * (1 - mask_ratio))  # Number of patches to keep
        # len_keep = np.clip(round(L * (1 - mask_ratio)), 1, L-1)  # Number of patches to keep; keep at least 1 token and 1 masked token.
        len_keep = np.clip(round(L * (1 - mask_ratio)), 0, L-1)  # Number of patches to keep; keep at least 1 token.


        noise = torch.rand(N, L, device=x.device)  # Generate random noise for masking (N, L)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # Ascend: small is keep, large is remove (N, L)
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # Restore indices after shuffling (N, L)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # Indices to keep (N, len_keep)
        x_masked = torch.gather(x, dim=1, index=ids_keep)  # Gather kept patches (N, len_keep)

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)  # Initialize mask with ones (N, L)
        mask[:, :len_keep] = 0  # Set first len_keep entries to 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)  # Unshuffle mask to match original input order (N, L)
        mask = mask.bool()

        x_masked_dim_kept = x.clone()  # (N, L)
        x_masked_dim_kept[mask] = self.mask_token_ids[freq]  # while keeping the original dim, the maksed tokens are simply indexed by mask_token_id

        return (x_masked, x_masked_dim_kept), mask, ids_restore.long()

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

        b = x.shape[0]
        device = x.device
        _, s_l = self.encode_to_z_q(x, self.encoder_l, self.vq_model_l)  # (b n)
        _, s_h = self.encode_to_z_q(x, self.encoder_h, self.vq_model_h)  # (b m)

        # mask tokens
        r = np.random.uniform(0., 1.)
        # r = np.random.uniform(1., 1.)  # no masking test; computational time should be longer with this
        gamma = self.gamma_func()
        masking_ratio = gamma(r)  # {1: no masking}
        (s_l_M, _), mask_l, ids_restore_l = self.random_masking(s_l, masking_ratio, 'lf')  # (b, len_keep), (b n), (b n)
        (_, s_h_M_dim_kept), mask_h, ids_restore_h = self.random_masking(s_h, masking_ratio, 'hf')  # (b, n), (b n), (b n)

        # prediction
        logits_l = self.masked_prediction(self.transformer_l, mask_l, ids_restore_l, y, s_l_M)  # (b n k)
        logits_h = self.masked_prediction(self.transformer_h, mask_h, ids_restore_h, y, s_l, s_h_M_dim_kept)

        # maksed prediction loss
        logits_l_on_mask = logits_l[mask_l]  # (bm k) where m < n
        s_l_on_mask = s_l[mask_l]  # (bm) where m < n
        mask_pred_loss_l = F.cross_entropy(logits_l_on_mask.float(), s_l_on_mask.long())
        
        logits_h_on_mask = logits_h[mask_h]  # (bm k) where m < n
        s_h_on_mask = s_h[mask_h]  # (bm) where m < n
        mask_pred_loss_h = F.cross_entropy(logits_h_on_mask.float(), s_h_on_mask.long())

        mask_pred_loss = mask_pred_loss_l + mask_pred_loss_h
        return mask_pred_loss, (mask_pred_loss_l, mask_pred_loss_h)        
    
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

        # # use ESS (Enhanced Sampling Scheme)
        # if self.config['MaskGIT']['ESS']['use']:
        #     t_star, s_star = self.critical_reverse_sampling(s_l, unknown_number_in_the_beginning_l, class_condition, 'lf')
        #     s_l = self.iterative_decoding_with_self_token_critic(t_star, s_star, 'lf',
        #                                                          unknown_number_in_the_beginning_l, class_condition, 
        #                                                          device)

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

    def critical_reverse_sampling(self,
                                  s: torch.Tensor,
                                  unknown_number_in_the_beginning_l,
                                  class_condition: Union[torch.Tensor, None],
                                  kind: str
                                  ):
        """
        s: sampled token sequence from the naive iterative decoding.
        """
        if kind == 'lf':
            mask_token_ids = self.mask_token_ids['lf']
            transformer = self.transformer_l
            vq_model = self.vq_model_l
        elif kind == 'hf':
            mask_token_ids = self.mask_token_ids['hf']
            transformer = self.transformer_h
            vq_model = self.vq_model_h
        else:
            raise ValueError

        # compute the confidence scores for s_T
        # the scores are used for the step retraction by iteratively removing unrealistic tokens.
        confidence_scores = self.compute_confidence_score(s, mask_token_ids, vq_model, transformer, class_condition)  # (b n)

        # find s_{t*}
        # t* denotes the step where unrealistic tokens have been removed.
        t_star = 1
        s_star = None
        prev_error = None
        error_ratio_hist = deque(maxlen=round(self.T[kind] * self.config['MaskGIT']['ESS']['error_ratio_ma_rate']))
        for t in range(1, self.T[kind])[::-1]:
            # masking ratio according to the masking scheduler
            ratio_t = 1. * (t + 1) / self.T[kind]  # just a percentage e.g. 1 / 12
            ratio_tm1 = 1. * t / self.T[kind]  # tm1: t - 1
            mask_ratio_t = self.gamma(ratio_t)
            mask_ratio_tm1 = self.gamma(ratio_tm1)  # tm1: t - 1

            # mask length
            mask_len_t = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning_l * mask_ratio_t), 1)
            mask_len_tm1 = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning_l * mask_ratio_tm1), 1)

            # masking matrices: {True: masking, False: not-masking}
            masking_t = self.mask_by_random_topk(mask_len_t, confidence_scores, temperature=0., device=s.device)  # (b n)
            masking_tm1 = self.mask_by_random_topk(mask_len_tm1, confidence_scores, temperature=0., device=s.device)  # (b n)
            masking = ~((masking_tm1.float() - masking_t.float()).bool())  # (b n); True for everything except the area of interest with False.

            # if there's no difference between t-1 and t, ends the retraction.
            if masking_t.float().sum() == masking_tm1.float().sum():
                t_star = t
                s_star = torch.where(masking_t, mask_token_ids, s)  # (b n)
                print('no difference between t-1 and t.')
                break

            # predict s_t given s_{t-1}
            s_tm1 = torch.where(masking_tm1, mask_token_ids, s)  # (b n)
            logits = self.masked_prediction(transformer, class_condition, s_tm1)  # (b m k)
            
            s_t_hat = logits.argmax(dim=-1)  # (b n)

            # leave the tokens of interest -- i.e., ds/dt -- only at t
            s_t = torch.where(masking, mask_token_ids, s)  # (b n)
            s_t_hat = torch.where(masking, mask_token_ids, s_t_hat)  # (b n)

            # measure error: distance between z_q_t and z_q_t_hat
            z_q_t = F.embedding(s_t[~masking], vq_model._codebook.embed)  # (b n d)
            z_q_t_hat = F.embedding(s_t_hat[~masking], vq_model._codebook.embed)  # (b n d)
            error = ((z_q_t - z_q_t_hat) ** 2).mean().cpu().detach().item()

            # error ratio
            if t + 1 == self.T[kind]:
                error_ratio_ma = 0.
                prev_error = error
            else:
                error_ratio = error / (prev_error + 1e-5)
                error_ratio_hist.append(error_ratio)
                error_ratio_ma = np.mean(error_ratio_hist)
                print(f't:{t} | error:{round(error, 6)} | error_ratio_ma:{round(error_ratio_ma, 6)}')
                prev_error = error

            # stopping criteria
            stopping_threshold = 1.0
            if error_ratio_ma > stopping_threshold and (t + 1 != self.T['lf']):
                t_star = t
                s_star = torch.where(masking_t, mask_token_ids, s)  # (b n)
                print('stopped by `error_ratio_ma > threshold`.')
                break
            if t == 1:
                t_star = t
                s_star = torch.where(masking_t, mask_token_ids, s)  # (b n)
                print('t_star has reached t=1.')
                break
        print('t_star:', t_star)
        return t_star, s_star

    def iterative_decoding_with_self_token_critic(self,
                                                  t_star,
                                                  s_star,
                                                  kind: str,
                                                  unknown_number_in_the_beginning,
                                                  class_condition: Union[torch.Tensor, None],
                                                  device,
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
            logits = self.masked_prediction(transformer, class_condition, s)

            sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()  # (b n)

            # create masking according to `t`
            ratio = 1. * (t + 1) / self.T[kind]  # just a percentage e.g. 1 / 12
            mask_ratio = self.gamma(ratio)

            # compute the confidence scores for s_t
            confidence_scores = self.compute_confidence_score(sampled_ids, mask_token_ids, vq_model, transformer, class_condition)  # (b n)

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # number of tokens that are to be masked;  (b,)
            mask_len = torch.clip(mask_len, min=0.)  # `mask_len` should be equal or larger than zero.

            # Adds noise for randomness
            masking = self.mask_by_random_topk(mask_len, confidence_scores, temperature=choice_temperature * (1. - ratio), device=device)

            # Masks tokens with lower confidence.
            s = torch.where(masking, mask_token_ids, sampled_ids)  # (b n)
        return s

    def compute_confidence_score(self, s, mask_token_ids, vq_model, transformer, class_condition):
        confidence_scores = torch.zeros_like(s).float()  # (b n)
        for n in range(confidence_scores.shape[-1]):
            s_m = copy.deepcopy(s)  # (b n)
            s_m[:, n] = mask_token_ids  # (b n); masking the n-th token to measure the confidence score for that token.
            logits = self.masked_prediction(transformer, class_condition, s_m)  # (b n k)
            logits = torch.nn.functional.softmax(logits, dim=-1)  # (b n K)

            true_tokens = s[:, n]  # (b,)
            logits = logits[:, n]  # (b, K)
            pred_tokens = logits.argmax(dim=-1)  # (b,)

            z_q_true = vq_model._codebook.embed[true_tokens]  # (b, dim)
            z_q_pred = vq_model._codebook.embed[pred_tokens]  # (b, dim)
            dist = torch.sum((z_q_true - z_q_pred) ** 2, dim=-1)  # (b,)
            confidence_scores[:, n] = -1 * dist  # confidence score for the n-th token
        confidence_scores = torch.nn.functional.softmax(confidence_scores, dim=-1)  # (b n)
        return confidence_scores
