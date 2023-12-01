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
from generators.domain_shifter import DomainShifter
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

    def forward(self, x):
        """
        :param x: (B, C, L)
        """
        pass

    def domain_shifter_loss_fn(self, x, s_a_l, s_a_h):
        # s -> z -> x
        self.vq_model_l.train()
        self.vq_model_h.train()
        x_a_l = self.maskgit.decode_token_ind_to_timeseries(s_a_l, 'LF')  # (b 1 l)
        x_a_h = self.maskgit.decode_token_ind_to_timeseries(s_a_h, 'HF')  # (b 1 l)
        self.vq_model_l.eval()
        self.vq_model_h.eval()
        x_a = x_a_l + x_a_h  # (b c l)
        x_a = x_a.detach()

        xhat = self.domain_shifter(x_a)
        recons_loss = F.l1_loss(xhat, x)

        domain_shifter_loss = recons_loss
        return domain_shifter_loss, (x_a, xhat)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()

        _, s_a_l = self.maskgit.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq)  # (b n)
        _, s_a_h = self.maskgit.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq)  # (b m)

        domain_shifter_loss, (x_a, xhat) = self.domain_shifter_loss_fn(x, s_a_l, s_a_h)

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {'loss': domain_shifter_loss,
                     }

        detach_the_unnecessary(loss_hist)
        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()

        _, s_a_l = self.maskgit.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq)  # (b n)
        _, s_a_h = self.maskgit.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq)  # (b m)

        domain_shifter_loss, (x_a, xhat) = self.domain_shifter_loss_fn(x, s_a_l, s_a_h)

        # log
        loss_hist = {'loss': domain_shifter_loss,
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
        opt = torch.optim.AdamW([{'params': self.domain_shifter.parameters(), 'lr': self.config['exp_params']['LR']},
                                 ],
                                weight_decay=self.config['exp_params']['weight_decay'])
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max)}
