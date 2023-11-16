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
        self.vq_model_l._codebook.sample_codebook_temp = config['domain_shifter']['stochastic_sampling']
        self.vq_model_h._codebook.sample_codebook_temp = config['domain_shifter']['stochastic_sampling']

        self.s_count = None

    def forward(self, x):
        """
        :param x: (B, C, L)
        """
        pass

    def domain_shifter_loss_fn(self, x, s_a_l, s_a_h):
        # s -> z -> x
        x_l_a = self.maskgit.decode_token_ind_to_timeseries(s_a_l, 'LF')  # (b 1 l)
        x_h_a = self.maskgit.decode_token_ind_to_timeseries(s_a_h, 'HF')  # (b 1 l)
        x_a = x_l_a + x_h_a  # (b c l)

        x_l_a = x_l_a.detach()
        x_h_a = x_h_a.detach()
        x_a = x_a.detach()

        xhat = self.domain_shifter.forward_projector(x_a, x_l_a, x_h_a)
        # domain_shifter_proj_loss = F.l1_loss(xhat_p, x)
        #
        # xhat_prs = self.domain_shifter.forward_refiner(xhat_p)
        # refine_loss = 0
        # for i in range(len(xhat_prs)):
        #     refine_loss += F.l1_loss(xhat_prs[i], x)
        # refine_loss /= len(xhat_prs)
        # domain_shifter_refiner_loss = refine_loss

        # similarity loss
        #z = self.domain_shifter.forward_encoder(x, 'real')
        # zhat = self.domain_shifter.forward_encoder(xhat, 'fake')
        #zhat = self.domain_shifter.forward_encoder(xhat, 'real')
        #sim_loss = torch.tensor(0., requires_grad=True)  #F.l1_loss(zhat, z)
        # sim_loss = torch.tensor(0., requires_grad=True)

        # reconstruction loss
        #zq, indices, vq_loss, perplexity = quantize(z, self.domain_shifter.vq, transpose_channel_length_axes=True)
        #vq_loss = vq_loss['loss']
        #x_recons = self.domain_shifter.forward_decoder(zq)

        #self.domain_shifter.vq.eval()
        #zhatq, _, _, _ = quantize(zhat, self.domain_shifter.vq, transpose_channel_length_axes=True)
        #xhat_recons = self.domain_shifter.forward_decoder(zhatq)
        #self.domain_shifter.vq.train()

        #recons_loss = F.l1_loss(x_recons, x) + F.l1_loss(xhat_recons, x)
        # recons_loss = F.l1_loss(xhat_recons, x)
        recons_loss = F.l1_loss(xhat, x)

        return recons_loss, (x_a, xhat)

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
        x = x.float()
        device = x.device

        with torch.no_grad():
            # get `s`
            _, s_a_l = self.maskgit.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq)  # (b n)
            _, s_a_h = self.maskgit.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq)  # (b m)

            # # `s_M` <- randomly-mask `s`
            # # mask_ratio = np.random.uniform(self.config['domain_shifter']['min_mask_ratio'], self.config['domain_shifter']['max_mask_ratio'])
            # mask_ratio = self.config['domain_shifter']['mask_ratio']  # constant masking_ratio leads to better performance in terms of FID and IS. varying masking_ratio makes the model training harder.
            # s_l_M, s_h_M = self.create_masked_token_set(s_l, s_h, mask_ratio, device)
            #
            # # p_theta(s_a | s_M)
            # logits_l = self.maskgit.transformer_l(s_l_M)  # (b n codebook_size)
            # logits_h = self.maskgit.transformer_h(s_l, s_h_M)  # (b m codebook_size)
            # s_a_l = torch.distributions.categorical.Categorical(logits=logits_l).sample()  # (b n)
            # s_a_h = torch.distributions.categorical.Categorical(logits=logits_h).sample()  # (b m)

        # domain shift loss
        recons_loss, (x_a, xhat) = self.domain_shifter_loss_fn(x, s_a_l, s_a_h)
        domain_shifter_loss = recons_loss

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss = domain_shifter_loss
        loss_hist = {'loss': loss,
                     'domain_shifter_loss': domain_shifter_loss,
                     #'sim_loss': sim_loss,
                     'recons_loss': recons_loss,
                     #'vq_loss': vq_loss
                     }
        detach_the_unnecessary(loss_hist)
        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        device = x.device

        with torch.no_grad():
            # get `s`
            _, s_a_l = self.maskgit.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq)  # (b n)
            _, s_a_h = self.maskgit.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq)  # (b m)

            # # `s_M` <- randomly-mask `s`
            # # mask_ratio = np.random.uniform(0., self.config['domain_shifter']['max_mask_ratio'])
            # mask_ratio = self.config['domain_shifter']['mask_ratio']
            # s_l_M, s_h_M = self.create_masked_token_set(s_l, s_h, mask_ratio, device)
            #
            # # p_theta(s_a | s_M)
            # logits_l = self.maskgit.transformer_l(s_l_M)  # (b n codebook_size)
            # logits_h = self.maskgit.transformer_h(s_l, s_h_M)  # (b m codebook_size)
            # s_a_l = torch.distributions.categorical.Categorical(logits=logits_l).sample()  # (b n)
            # s_a_h = torch.distributions.categorical.Categorical(logits=logits_h).sample()  # (b m)

        # domain shift loss
        recons_loss, (x_a, xhat) = self.domain_shifter_loss_fn(x, s_a_l, s_a_h)
        domain_shifter_loss = recons_loss

        # log
        loss = domain_shifter_loss
        loss_hist = {'loss': loss,
                     'domain_shifter_loss': domain_shifter_loss,
                     #'sim_loss': sim_loss,
                     'recons_loss': recons_loss,
                     #'vq_loss': vq_loss
                     }
        detach_the_unnecessary(loss_hist)

        # maskgit sampling
        if batch_idx == 0:
            class_index = np.random.choice(np.concatenate(([None], np.unique(y.cpu()))))

            # unconditional sampling
            s_l, s_h = self.maskgit.iterative_decoding(device=x.device, class_index=class_index, num=x.shape[0])
            x_new_l = self.maskgit.decode_token_ind_to_timeseries(s_l, 'LF').cpu()
            x_new_h = self.maskgit.decode_token_ind_to_timeseries(s_h, 'HF').cpu()
            x_new = x_new_l + x_new_h
            # x_new_c, x_new_cd = self.domain_shifter(x_new.to(x.device), return_all=True)
            # x_new_p = self.domain_shifter(x_new.to(x.device))
            # x_new_p = self.domain_shifter.forward_projector(x_new.to(x.device), x_new_l.to(x.device), x_new_h.to(x.device))
            x_new_p = self.domain_shifter.forward_projector(x_new.to(x.device), x_new_l.to(x.device),x_new_h.to(x.device))
            #z_gen = self.domain_shifter.forward_encoder(x_new_p, 'real')
            x_new_p = x_new_p.detach().cpu().numpy()
            # x_new_pr = x_new_pr.detach().cpu().numpy()

            b = np.random.randint(0, x.shape[0])
            n_subfigs = 8
            fig, axes = plt.subplots(n_subfigs, 1, figsize=(4, 2 * n_subfigs))
            axes[0].plot(x_new_l[b, 0, :], label='x_l')
            axes[0].legend()
            axes[1].plot(x_new_h[b, 0, :], label='x_h')
            axes[1].legend()
            axes[2].plot(x_new[b, 0, :], label='x')
            axes[2].legend()
            axes[3].plot(x_new_p[b, 0, :], label=r'$P$(x)')
            axes[3].legend()

            #axes[4].plot(x[b,0].cpu(), label=r'x')
            #axes[4].plot(x_recons[b, 0].cpu(), label=r'x_recons')
            #axes[4].legend()

            axes[5].plot(x[b,0].cpu(), label=r'x', alpha=0.5)
            axes[5].plot(x_a[b, 0].cpu(), label=r'x_a', alpha=0.5)
            axes[5].legend()

            # from sklearn.decomposition import PCA
            # pca = PCA(n_components=2)
            #z = rearrange(z, 'b c l -> b (c l)').cpu().numpy()
            #zhat = rearrange(zhat, 'b c l -> b (c l)').cpu().numpy()
            #z = pca.fit_transform(z)
            #zhat = pca.transform(zhat)

            #z_gen = rearrange(z_gen, 'b c l -> b (c l)').cpu().numpy()
            #z_gen = pca.transform(z_gen)

            #axes[6].scatter(z[:, 0], z[:, 1], label=r'z', alpha=0.5)
            #axes[6].scatter(zhat[:,0], zhat[:,1], label=r'zhat', alpha=0.5)
            #axes[6].scatter(z_gen[:,0], z_gen[:,1], label=r'z_gen', alpha=0.5)
            #axes[6].legend()

            axes[6].plot(x_a[b, 0].cpu(), label=r'x_a', alpha=0.7)
            axes[6].plot(xhat[b, 0].cpu(), label=r'xhat', alpha=0.7)
            axes[6].legend()

            axes[7].plot(x[b, 0].cpu(), label=r'x', alpha=0.7)
            axes[7].plot(xhat[b, 0].cpu(), label=r'xhat', alpha=0.7)
            axes[7].legend()

            # for i, ax in enumerate(axes):
            #     if i != 6:
            #         ax.set_ylim(-4, 4)

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
