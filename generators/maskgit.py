import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
import tempfile
from typing import Union

from einops import repeat, rearrange
from typing import Callable
from generators.bidirectional_transformer import BidirectionalTransformer

from encoder_decoders.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from vector_quantization.vq import VectorQuantize

from utils import compute_downsample_rate, get_root_dir, freeze, timefreq_to_time, time_to_timefreq, quantize, zero_pad_low_freq, zero_pad_high_freq


class MaskGIT(nn.Module):
    """
    ref: https://github.com/dome272/MaskGIT-pytorch/blob/cff485ad3a14b6ed5f3aa966e045ea2bc8c68ad8/transformer.py#L11
    """

    def __init__(self,
                 input_length: int,
                 choice_temperatures: dict,
                 stochastic_sampling: int,
                 T: int,
                 config: dict,
                 n_classes: int,
                 **kwargs):
        super().__init__()
        self.choice_temperature_l = choice_temperatures['lf']
        self.choice_temperature_h = choice_temperatures['hf']
        self.T = T
        self.config = config
        self.n_classes = n_classes

        self.mask_token_ids = {'LF': config['VQ-VAE']['codebook_sizes']['lf'], 'HF': config['VQ-VAE']['codebook_sizes']['hf']}
        self.gamma = self.gamma_func("cosine")

        # define encoder, decoder, vq_models
        dim = config['encoder']['dim']
        in_channels = config['dataset']['in_channels']
        downsampled_width_l = config['encoder']['downsampled_width']['lf']
        downsampled_width_h = config['encoder']['downsampled_width']['hf']
        self.n_fft = config['VQ-VAE']['n_fft']
        downsample_rate_l = compute_downsample_rate(input_length, self.n_fft, downsampled_width_l)
        downsample_rate_h = compute_downsample_rate(input_length, self.n_fft, downsampled_width_h)
        self.encoder_l = VQVAEEncoder(dim, 2 * in_channels, downsample_rate_l, config['encoder']['n_resnet_blocks'])
        self.decoder_l = VQVAEDecoder(dim, 2 * in_channels, downsample_rate_l, config['decoder']['n_resnet_blocks'])
        self.vq_model_l = VectorQuantize(dim, config['VQ-VAE']['codebook_sizes']['lf'], **config['VQ-VAE'])
        self.encoder_h = VQVAEEncoder(dim, 2 * in_channels, downsample_rate_h, config['encoder']['n_resnet_blocks'])
        self.decoder_h = VQVAEDecoder(dim, 2 * in_channels, downsample_rate_h, config['decoder']['n_resnet_blocks'])
        self.vq_model_h = VectorQuantize(dim, config['VQ-VAE']['codebook_sizes']['hf'], **config['VQ-VAE'])

        # load trained models for encoder, decoder, and vq_models
        dataset_name = self.config['dataset']['dataset_name']
        self.load(self.encoder_l, get_root_dir().joinpath('saved_models'), f'encoder_l-{dataset_name}.ckpt')
        self.load(self.decoder_l, get_root_dir().joinpath('saved_models'), f'decoder_l-{dataset_name}.ckpt')
        self.load(self.vq_model_l, get_root_dir().joinpath('saved_models'), f'vq_model_l-{dataset_name}.ckpt')
        self.load(self.encoder_h, get_root_dir().joinpath('saved_models'), f'encoder_h-{dataset_name}.ckpt')
        self.load(self.decoder_h, get_root_dir().joinpath('saved_models'), f'decoder_h-{dataset_name}.ckpt')
        self.load(self.vq_model_h, get_root_dir().joinpath('saved_models'), f'vq_model_h-{dataset_name}.ckpt')

        # freeze the models for encoder, decoder, and vq_models
        freeze(self.encoder_l)
        freeze(self.decoder_l)
        freeze(self.vq_model_l)
        freeze(self.encoder_h)
        freeze(self.decoder_h)
        freeze(self.vq_model_h)

        # evaluation model for encoder, decoder, and vq_models
        self.encoder_l.eval()
        self.decoder_l.eval()
        self.vq_model_l.eval()
        self.encoder_h.eval()
        self.decoder_h.eval()
        self.vq_model_h.eval()

        # token lengths
        self.num_tokens_l = self.encoder_l.num_tokens.item()
        self.num_tokens_h = self.encoder_h.num_tokens.item()

        # latent space dim
        self.H_prime_l, self.H_prime_h = self.encoder_l.H_prime, self.encoder_h.H_prime
        self.W_prime_l, self.W_prime_h = self.encoder_l.W_prime, self.encoder_h.W_prime

        # pretrained discrete tokens
        embed_l = nn.Parameter(copy.deepcopy(self.vq_model_l._codebook.embed))  # pretrained discrete tokens (LF)
        embed_h = nn.Parameter(copy.deepcopy(self.vq_model_h._codebook.embed))  # pretrained discrete tokens (HF)

        self.transformer_l = BidirectionalTransformer('LF',
                                                      self.num_tokens_l,
                                                      config['VQ-VAE']['codebook_sizes'],
                                                      config['VQ-VAE']['codebook_dim'],
                                                      **config['MaskGIT']['prior_model'],
                                                      n_classes=n_classes,
                                                      pretrained_tok_emb_l=embed_l,
                                                      )

        self.transformer_h = BidirectionalTransformer('HF',
                                                      self.num_tokens_h,
                                                      config['VQ-VAE']['codebook_sizes'],
                                                      config['VQ-VAE']['codebook_dim'],
                                                      **config['MaskGIT']['prior_model'],
                                                      n_classes=n_classes,
                                                      pretrained_tok_emb_l=embed_l,
                                                      pretrained_tok_emb_h=embed_h,
                                                      num_tokens_l=self.num_tokens_l,
                                                      )

        # stochastic codebook sampling
        self.vq_model_l._codebook.sample_codebook_temp = stochastic_sampling
        self.vq_model_h._codebook.sample_codebook_temp = stochastic_sampling

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
    def encode_to_z_q(self, x, encoder: VQVAEEncoder, vq_model: VectorQuantize, spectrogram_padding: Callable = None):
        """
        x: (B, C, L)
        """
        C = x.shape[1]
        xf = time_to_timefreq(x, self.n_fft, C)  # (B, C, H, W)
        if spectrogram_padding is not None:
            xf = spectrogram_padding(xf)
        z = encoder(xf)  # (b c h w)
        z_q, indices, vq_loss, perplexity = quantize(z, vq_model)  # (b c h w), (b (h w) h), ...
        return z_q, indices

    def forward(self, x, y):
        """
        x: (B, C, L)
        y: (B, 1)
        straight from [https://github.com/dome272/MaskGIT-pytorch/blob/main/transformer.py]
        """
        device = x.device
        _, s_l = self.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq)  # (b n)
        _, s_h = self.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq)  # (b m)

        # randomly sample `t`
        t = np.random.uniform(0, 1)

        # create masks
        n_masks_l = math.floor(self.gamma(t) * s_l.shape[1])
        rand = torch.rand(s_l.shape, device=device)  # (b n)
        mask_l = torch.zeros(s_l.shape, dtype=torch.bool, device=device)
        mask_l.scatter_(dim=1, index=rand.topk(n_masks_l, dim=1).indices, value=True)

        n_masks_h = math.floor(self.gamma(t) * s_h.shape[1])
        rand = torch.rand(s_h.shape, device=device)  # (b m)
        mask_h = torch.zeros(s_h.shape, dtype=torch.bool, device=device)
        mask_h.scatter_(dim=1, index=rand.topk(n_masks_h, dim=1).indices, value=True)

        # masked tokens
        masked_indices_l = self.mask_token_ids['LF'] * torch.ones_like(s_l, device=device)  # (b n)
        s_l_M = mask_l * s_l + (~mask_l) * masked_indices_l  # (b n); `~` reverses bool-typed data
        masked_indices_h = self.mask_token_ids['HF'] * torch.ones_like(s_h, device=device)
        s_h_M = mask_h * s_h + (~mask_h) * masked_indices_h  # (b m)

        # prediction
        logits_l = self.transformer_l(s_l_M.detach(), class_condition=y)  # (b n codebook_size)
        target_l = s_l  # (b n)

        logits_h = self.transformer_h(s_l.detach(), s_h_M.detach(), class_condition=y)  # (b m codebook_size)
        target_h = s_h  # (b m)

        return [logits_l, logits_h], [target_l, target_h]

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
        def log(t, eps=1e-20):
            return torch.log(t.clamp(min=eps))

        def gumbel_noise(t):
            """
            Gumbel max trick: https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch
            """
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -log(-log(noise))

        confidence = torch.log(probs) + temperature * gumbel_noise(probs).to(device)  # Gumbel max trick
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking

    def first_pass(self,
                   s_l: torch.Tensor,
                   unknown_number_in_the_beginning_l,
                   class_condition: Union[torch.Tensor, None],
                   guidance_scale: float,
                   gamma: Callable,
                   device):
        for t in range(self.T):
            logits_l = self.transformer_l(s_l, class_condition=class_condition)  # (b n codebook_size) == (b n K)
            if isinstance(class_condition, torch.Tensor):
                logits_l_null = self.transformer_l(s_l, class_condition=None)
                logits_l = logits_l_null + guidance_scale * (logits_l - logits_l_null)
            sampled_ids = torch.distributions.categorical.Categorical(logits=logits_l).sample()  # (b n)
            unknown_map = (s_l == self.mask_token_ids['LF'])  # which tokens need to be sampled; (b n)
            sampled_ids = torch.where(unknown_map, sampled_ids, s_l)  # keep the previously-sampled tokens; (b n)

            # create masking according to `t`
            ratio = 1. * (t + 1) / self.T  # just a percentage e.g. 1 / 12
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
            s_l = torch.where(masking, self.mask_token_ids['LF'], sampled_ids)  # (b n)
        return s_l

    def second_pass(self,
                    s_l: torch.Tensor,
                    s_h: torch.Tensor,
                    unknown_number_in_the_beginning_h,
                    class_condition: Union[torch.Tensor, None],
                    guidance_scale: float,
                    gamma: Callable,
                    device):
        for t in range(self.T):
            logits_h = self.transformer_h(s_l, s_h, class_condition=class_condition)  # (b m codebook_size) == (b m K)
            if isinstance(class_condition, torch.Tensor) and (guidance_scale > 1):
                logits_h_null = self.transformer_h(s_l, s_h, class_condition=None)
                logits_h = logits_h_null + guidance_scale * (logits_h - logits_h_null)
            sampled_ids = torch.distributions.categorical.Categorical(logits=logits_h).sample()  # (b m)
            unknown_map = (s_h == self.mask_token_ids['HF'])  # which tokens need to be sampled; (b m)
            sampled_ids = torch.where(unknown_map, sampled_ids, s_h)  # keep the previously-sampled tokens; (b m)

            # create masking according to `t`
            ratio = 1. * (t + 1) / self.T  # just a percentage e.g. 1 / 12
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
            s_h = torch.where(masking, self.mask_token_ids['HF'], sampled_ids)  # (b n)
        return s_h

    @torch.no_grad()
    def iterative_decoding(self, num=1, mode="cosine", class_index=None, device='cpu', guidance_scale: float = 1.):
        """
        It performs the iterative decoding and samples token indices for LF and HF.
        :param num: number of samples
        :return: sampled token indices for LF and HF
        """
        s_l = self.create_input_tokens_normal(num, self.num_tokens_l, self.mask_token_ids['LF'], device)  # (b n)
        s_h = self.create_input_tokens_normal(num, self.num_tokens_h, self.mask_token_ids['HF'], device)  # (b n)

        unknown_number_in_the_beginning_l = torch.sum(s_l == self.mask_token_ids['LF'], dim=-1)  # (b,)
        unknown_number_in_the_beginning_h = torch.sum(s_h == self.mask_token_ids['HF'], dim=-1)  # (b,)
        gamma = self.gamma_func(mode)
        class_condition = repeat(torch.Tensor([class_index]).int().to(device), 'i -> b i', b=num) if class_index != None else None  # (b 1)

        s_l = self.first_pass(s_l, unknown_number_in_the_beginning_l, class_condition, guidance_scale, gamma, device)
        s_h = self.second_pass(s_l, s_h, unknown_number_in_the_beginning_h, class_condition, guidance_scale, gamma, device)
        return s_l, s_h

    def decode_token_ind_to_timeseries(self, s: torch.Tensor, frequency: str, return_representations: bool = False):
        """
        It takes token embedding indices and decodes them to time series.
        :param s: token embedding index
        :param frequency:
        :param return_representations:
        :return:
        """
        assert frequency in ['LF', 'HF']

        vq_model = self.vq_model_l if frequency == 'LF' else self.vq_model_h
        decoder = self.decoder_l if frequency == 'LF' else self.decoder_h
        zero_pad = zero_pad_high_freq if frequency == 'LF' else zero_pad_low_freq

        quantize = F.embedding(s, vq_model._codebook.embed)  # (b n d)
        quantize = vq_model.project_out(quantize)  # (b n c)
        quantize = rearrange(quantize, 'b n c -> b c n')  # (b c n) == (b c (h w))
        H_prime = self.H_prime_l if frequency == 'LF' else self.H_prime_h
        W_prime = self.W_prime_l if frequency == 'LF' else self.W_prime_h
        quantize = rearrange(quantize, 'b c (h w) -> b c h w', h=H_prime, w=W_prime)

        xfhat = decoder(quantize)

        uhat = zero_pad(xfhat)
        xhat = timefreq_to_time(uhat, self.n_fft, self.config['dataset']['in_channels'])  # (B, C, L)

        if return_representations:
            return xhat, quantize
        else:
            return xhat
