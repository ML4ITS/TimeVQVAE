import os
from typing import Callable
from pathlib import Path
import tempfile
import math
from copy import deepcopy
from collections import Counter

import matplotlib.pyplot as plt
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn

from generators.maskgit import MaskGIT
from experiments.exp_base import ExpBase, detach_the_unnecessary
from generators.domain_shifter import DomainShifter
from encoder_decoders.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from vector_quantization.vq import VectorQuantize
from utils import get_root_dir, unfreeze, freeze, compute_downsample_rate, timefreq_to_time, time_to_timefreq, quantize, zero_pad_low_freq, zero_pad_high_freq


class ExpDomainShifter(ExpBase):
    def __init__(self,
                 dataset_name: str,
                 input_length: int,
                 config: dict,
                 n_train_samples: int,
                 n_classes: int,
                 ):
        super().__init__()
        self.config = config
        self.T_max = config['trainer_params']['max_epochs']['stage2'] * (np.ceil(n_train_samples / config['dataset']['batch_sizes']['stage2']) + 1)
        self.n_fft = config['VQ-VAE']['n_fft']

        # domain shifter
        self.domain_shifter = DomainShifter(input_length, 1, self.n_fft, config)
        # self.real_domain_keeper = deepcopy(self.domain_shifter)

        # load maskgit
        self.maskgit = MaskGIT(dataset_name, input_length, **self.config['MaskGIT'], config=self.config, n_classes=n_classes).to(self.device)
        fname = f"maskgit-{dataset_name}.ckpt"
        self.maskgit.load_state_dict(torch.load(os.path.join('saved_models', fname)), strict=False)
        self.maskgit.eval()

        self.encoder_l = self.maskgit.encoder_l
        self.decoder_l = self.maskgit.decoder_l
        self.vq_model_l = self.maskgit.vq_model_l
        self.encoder_h = self.maskgit.encoder_h
        self.decoder_h = self.maskgit.decoder_h
        self.vq_model_h = self.maskgit.vq_model_h

        # stochastic codebook sampling
        self.vq_model_l._codebook.sample_codebook_temp = config['MaskGIT']['stochastic_sampling']
        self.vq_model_h._codebook.sample_codebook_temp = config['MaskGIT']['stochastic_sampling']

        self.s_count = None

    def forward(self, x):
        """
        :param x: (B, C, L)
        """
        pass

    def domain_shifter_loss_fn(self, x, s_l_stoch, s_h_stoch):
        # s -> z -> x
        xhat_l_stoch = self.maskgit.decode_token_ind_to_timeseries(s_l_stoch, 'LF')  # (b 1 l)
        xhat_h_stoch = self.maskgit.decode_token_ind_to_timeseries(s_h_stoch, 'HF')  # (b 1 l)
        xhat = xhat_l_stoch + xhat_h_stoch  # (b c l)
        xhat = xhat.detach()

        xhat_c = self.domain_shifter.forward_projector(xhat)
        domain_shifter_loss = F.l1_loss(xhat_c, x)

        xhat_cd, vq_loss = self.domain_shifter.forward_vq(xhat_c.detach())
        domain_shifter_loss += F.l1_loss(xhat_cd, x) + vq_loss.mean()

        return domain_shifter_loss

    def create_masked_token_set(self, s_l, s_h, mask_ratio: float, device):
        n_masks_l = math.floor(mask_ratio * s_l.shape[1])
        rand = torch.rand(s_l.shape, device=device)  # (b n)
        mask_l = torch.zeros(s_l.shape, dtype=torch.bool, device=device)
        mask_l.scatter_(dim=1, index=rand.topk(n_masks_l, dim=1).indices, value=True)

        n_masks_h = math.floor(mask_ratio * s_h.shape[1])
        rand = torch.rand(s_h.shape, device=device)  # (b m)
        mask_h = torch.zeros(s_h.shape, dtype=torch.bool, device=device)
        mask_h.scatter_(dim=1, index=rand.topk(n_masks_h, dim=1).indices, value=True)

        masked_indices_l = self.maskgit.mask_token_ids['LF'] * torch.ones_like(s_l, device=device)  # (b n)
        s_l_M = mask_l * s_l + (~mask_l) * masked_indices_l  # (b n); `~` reverses bool-typed data
        masked_indices_h = self.maskgit.mask_token_ids['HF'] * torch.ones_like(s_h, device=device)
        s_h_M = mask_h * s_h + (~mask_h) * masked_indices_h  # (b m)
        return s_l_M, s_h_M

    def training_step(self, batch, batch_idx):
        x, y = batch
        device = x.device

        with torch.no_grad():
            # get `s`
            _, s_l = self.maskgit.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq)  # (b n)
            _, s_h = self.maskgit.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq)  # (b m)

            # `s_M` <- randomly-mask `s`
            mask_ratio = np.random.uniform(0., self.config['domain_shifter']['max_mask_ratio'])
            s_l_M, s_h_M = self.create_masked_token_set(s_l, s_h, mask_ratio, device)

            # p_theta(s_a | s_M)
            logits_l = self.maskgit.transformer_l(s_l_M)  # (b n codebook_size)
            logits_h = self.maskgit.transformer_h(s_l, s_h_M)  # (b m codebook_size)
            s_a_l = torch.distributions.categorical.Categorical(logits=logits_l).sample()  # (b n)
            s_a_h = torch.distributions.categorical.Categorical(logits=logits_h).sample()  # (b m)

        # domain shift loss
        domain_shifter_loss = self.domain_shifter_loss_fn(x, s_a_l, s_a_h)

        # clustering loss
        # codebook_embed = self.domain_shifter.domain_shifter.vq_model_end._codebook.embed[None, :, :]  # (1 K d)
        # dist_btn_codes = torch.cdist(codebook_embed, codebook_embed, p=2)  # (1 K K)
        # dist_btn_codes = dist_btn_codes.mean()
        # codebook_embed = self.domain_shifter.domain_shifter.vq_model_end._codebook.embed  # (K d)
        # corr_btn_codes = torch.corrcoef(codebook_embed)
        # corr_btn_codes_loss = (corr_btn_codes ** 2).mean()
        # codebook_embed_val = self.domain_shifter.domain_shifter.vq_model_end._codebook.embed.abs().mean()

        # # count use of tokens
        # s_ind = s_ind.cpu().numpy().flatten()
        # if batch_idx == 0:
        #     self.s_count = Counter(s_ind)
        # else:
        #     self.s_count += Counter(s_ind)
        # n_used_tokens = len(self.s_count)
        # ratio_used_tokens_to_total_tokens = n_used_tokens / self.config['domain_shifter']['codebook_sizes']['end']

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss = domain_shifter_loss #+ codebook_embed_val
        loss_hist = {'loss': loss,
                     'domain_shifter_loss': domain_shifter_loss,
                     # 'codebook_embed_val': codebook_embed_val,
                     # 'ratio_used_tokens_to_total_tokens:': ratio_used_tokens_to_total_tokens,
                     }

        detach_the_unnecessary(loss_hist)
        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        device = x.device

        with torch.no_grad():
            # get `s`
            _, s_l = self.maskgit.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq)  # (b n)
            _, s_h = self.maskgit.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq)  # (b m)

            # `s_M` <- randomly-mask `s`
            mask_ratio = np.random.uniform(0., self.config['domain_shifter']['max_mask_ratio'])
            s_l_M, s_h_M = self.create_masked_token_set(s_l, s_h, mask_ratio, device)

            # p_theta(s_a | s_M)
            logits_l = self.maskgit.transformer_l(s_l_M)  # (b n codebook_size)
            logits_h = self.maskgit.transformer_h(s_l, s_h_M)  # (b m codebook_size)
            s_a_l = torch.distributions.categorical.Categorical(logits=logits_l).sample()  # (b n)
            s_a_h = torch.distributions.categorical.Categorical(logits=logits_h).sample()  # (b m)

        # domain shift loss
        domain_shifter_loss = self.domain_shifter_loss_fn(x, s_a_l, s_a_h)

        # log
        loss = domain_shifter_loss
        loss_hist = {'loss': loss,
                     'domain_shifter_loss': domain_shifter_loss,
                     }
        detach_the_unnecessary(loss_hist)

        # maskgit sampling
        if batch_idx == 0:
            class_index = np.random.choice(np.concatenate(([None], np.unique(y.cpu()))))

            # unconditional sampling
            s_l, s_h = self.maskgit.iterative_decoding(device=x.device, class_index=class_index)
            x_new_l = self.maskgit.decode_token_ind_to_timeseries(s_l, 'LF').cpu()
            x_new_h = self.maskgit.decode_token_ind_to_timeseries(s_h, 'HF').cpu()
            x_new = x_new_l + x_new_h
            x_new_c, x_new_cd = self.domain_shifter(x_new.to(x.device), return_all=True)
            x_new_c = x_new_c.detach().cpu().numpy()
            x_new_cd = x_new_cd.detach().cpu().numpy()

            b = 0
            fig, axes = plt.subplots(5, 1, figsize=(4, 2 * 5))
            axes[0].plot(x_new_l[b, 0, :], label='x_l')
            axes[0].legend()
            axes[1].plot(x_new_h[b, 0, :], label='x_h')
            axes[1].legend()
            axes[2].plot(x_new[b, 0, :], label='x')
            axes[2].legend()
            axes[3].plot(x_new_c[b, 0, :], label=r'$D_s$(x)')
            axes[3].legend()
            axes[4].plot(x_new_cd[b, 0, :], label=r'VQ($D_s$(x))')
            axes[4].legend()
            for ax in axes:
                ax.set_ylim(-4, 4)
            plt.title(f'ep_{self.current_epoch}; class-{class_index}')
            plt.tight_layout()
            wandb.log({f"maskgit sample": wandb.Image(plt)})
            plt.close()

        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params': self.domain_shifter.parameters(), 'lr': self.config['exp_params']['LR']},
                                 # {'params': self.real_domain_keeper.parameters(), 'lr': self.config['exp_params']['LR']},
                                 ],
                                weight_decay=self.config['exp_params']['weight_decay'])
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max)}
