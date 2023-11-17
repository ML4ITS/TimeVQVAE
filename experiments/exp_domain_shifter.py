import os
from typing import Callable
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F
from einops import rearrange

from generators.maskgit import MaskGIT
from experiments.exp_base import ExpBase, detach_the_unnecessary
from generators.domain_shifter import DomainShifter, Discriminator
from encoder_decoders.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from vector_quantization.vq import VectorQuantize
from utils import get_root_dir, freeze, compute_downsample_rate, timefreq_to_time, time_to_timefreq, quantize, zero_pad_low_freq, zero_pad_high_freq


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

        # discriminator
        self.discriminator = Discriminator(self.n_fft)

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def forward(self, x):
        """
        :param x: (B, C, L)
        """
        pass

    def domain_shifter_loss_fn(self, x, s_a_l, s_a_h):
        # s -> z -> x
        x_a_l = self.maskgit.decode_token_ind_to_timeseries(s_a_l, 'LF')  # (b 1 l)
        x_a_h = self.maskgit.decode_token_ind_to_timeseries(s_a_h, 'HF')  # (b 1 l)
        x_a = x_a_l + x_a_h  # (b c l)
        x_a = x_a.detach()

        xhat = self.domain_shifter(x_a)
        recons_loss = F.l1_loss(xhat, x)

        domain_shifter_loss = recons_loss
        return domain_shifter_loss, (x_a, xhat)

    def training_step(self, batch, batch_idx):
        opt_domain, opt_disc = self.optimizers()
        x, y = batch
        x = x.float()

        # train domain shifter
        _, s_a_l = self.maskgit.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq)  # (b n)
        _, s_a_h = self.maskgit.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq)  # (b m)

        domain_shifter_loss, (x_a, xhat) = self.domain_shifter_loss_fn(x, s_a_l, s_a_h)

        out = self.discriminator(xhat).flatten()  # (bl)
        labels = torch.ones(out.shape[0]).to(x.device)  # (bl)
        disc_loss_G = F.binary_cross_entropy_with_logits(out, labels)

        opt_domain.zero_grad()
        self.manual_backward(domain_shifter_loss + disc_loss_G)
        opt_domain.step()

        # train discriminator
        xhat = xhat.detach()  # (b 1 l)
        out_x = self.discriminator(x).flatten()  # (bl)
        out_xhat = self.discriminator(xhat).flatten()  # (bl)
        out = torch.cat((out_x, out_xhat), dim=0)
        labels = torch.cat((torch.ones(out_x.shape[0]), torch.zeros(out_xhat.shape[0])), dim=0).to(x.device)  # (2b)
        disc_loss_D = F.binary_cross_entropy_with_logits(out, labels)

        opt_disc.zero_grad()
        self.manual_backward(disc_loss_D)
        opt_disc.step()

        acc = (torch.round(F.sigmoid(out)) == labels).float().mean()

        # log
        loss_hist = {'domain_shifter_loss': domain_shifter_loss,
                     'disc_loss_G': disc_loss_G,
                     'disc_loss_D': disc_loss_D,
                     'acc': acc,
                     }
        detach_the_unnecessary(loss_hist)
        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()

        # train domain shifter
        _, s_a_l = self.maskgit.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq)  # (b n)
        _, s_a_h = self.maskgit.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq)  # (b m)

        domain_shifter_loss, (x_a, xhat) = self.domain_shifter_loss_fn(x, s_a_l, s_a_h)

        out = self.discriminator(xhat).flatten()  # (bl)
        labels = torch.ones(out.shape[0]).to(x.device)  # (bl)
        disc_loss_G = F.binary_cross_entropy_with_logits(out, labels)

        # train discriminator
        xhat = xhat.detach()  # (b 1 l)
        out_x = self.discriminator(x).flatten()  # (bl)
        out_xhat = self.discriminator(xhat).flatten()  # (bl)
        out = torch.cat((out_x, out_xhat), dim=0)
        labels = torch.cat((torch.ones(out_x.shape[0]), torch.zeros(out_xhat.shape[0])), dim=0).to(x.device)  # (2b)
        disc_loss_D = F.binary_cross_entropy_with_logits(out, labels)

        acc = (torch.round(F.sigmoid(out)) == labels).float().mean()

        # log
        loss_hist = {'domain_shifter_loss': domain_shifter_loss,
                     'disc_loss_G': disc_loss_G,
                     'disc_loss_D': disc_loss_D,
                     'acc': acc,
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
            x_new_corrected = self.domain_shifter(x_new.to(x.device)).detach().cpu().numpy()

            b = 0
            n_figs = 6
            fig, axes = plt.subplots(n_figs, 1, figsize=(4, 2 * n_figs))
            axes[0].plot(x_new_l[b, 0, :], label='x_l')
            axes[0].legend()
            axes[1].plot(x_new_h[b, 0, :], label='x_h')
            axes[1].legend()
            axes[2].plot(x_new[b, 0, :], label='x')
            axes[2].legend()
            axes[3].plot(x_new_corrected[b, 0, :], label=r'$D_s$(x)')
            axes[3].legend()

            x = x.cpu().numpy()
            x_a = x_a.cpu().numpy()
            xhat = xhat.cpu().numpy()
            b_ = np.random.randint(0, x.shape[0])
            axes[4].plot(x[b_, 0, :], label='x')
            axes[4].plot(xhat[b_, 0, :], label='xhat')
            axes[4].legend()

            axes[5].plot(x_a[b_, 0, :], label='x_a')
            axes[5].plot(xhat[b_, 0, :], label='xhat')
            axes[5].legend()

            for ax in axes:
                ax.set_ylim(-4, 4)
            plt.title(f'ep_{self.current_epoch}; class-{class_index}')
            plt.tight_layout()
            wandb.log({f"maskgit sample": wandb.Image(plt)})
            plt.close()

        return loss_hist

    def configure_optimizers(self):
        # opt = torch.optim.AdamW([{'params': self.domain_shifter.parameters(), 'lr': self.config['exp_params']['LR']},
        #                          ],
        #                         weight_decay=self.config['exp_params']['weight_decay'])
        # return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max)}
        opt_domain = torch.optim.AdamW(self.domain_shifter.parameters(), lr=self.config['exp_params']['LR'])
        opt_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=self.config['exp_params']['LR'])
        return opt_domain, opt_disc

    def test_step(self, batch, batch_idx):
        x, y = batch

        _, s_l = self.maskgit.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq)  # (b n)
        _, s_h = self.maskgit.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq)  # (b m)

        domain_shifter_loss = self.domain_shifter_loss_fn(x, s_l, s_h)

        # log
        loss_hist = {'loss': domain_shifter_loss,
                     }

        detach_the_unnecessary(loss_hist)
        return loss_hist
