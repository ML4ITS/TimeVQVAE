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
import pytorch_lightning as pl

from generators.fidelity_enhancer import FidelityEnhancer
from experiments.exp_maskgit import ExpMaskGIT
from utils import get_root_dir, freeze, compute_downsample_rate, timefreq_to_time, time_to_timefreq, quantize, zero_pad_low_freq, zero_pad_high_freq


class ExpFidelityEnhancer(pl.LightningModule):
    def __init__(self,
                 dataset_name: str,
                 input_length: int,
                 config: dict,
                 n_classes: int,
                 ):
        super().__init__()
        self.config = config
        self.n_fft = config['VQ-VAE']['n_fft']

        # domain shifter
        self.fidelity_enhancer = FidelityEnhancer(input_length, 1, config)

        # load the stage2 model
        self.stage2 = ExpMaskGIT.load_from_checkpoint(os.path.join('saved_models', f'stage2-{dataset_name}.ckpt'), 
                                                      dataset_name=dataset_name, 
                                                      input_length=input_length, 
                                                      n_classes=n_classes,
                                                      config=config,
                                                      map_location='cpu')
        freeze(self.stage2)
        self.stage2.eval()

        self.maskgit = self.stage2.maskgit
        self.encoder_l = self.maskgit.encoder_l
        self.decoder_l = self.maskgit.decoder_l
        self.vq_model_l = self.maskgit.vq_model_l
        self.encoder_h = self.maskgit.encoder_h
        self.decoder_h = self.maskgit.decoder_h
        self.vq_model_h = self.maskgit.vq_model_h

        self.svq_temp_rng = self.config['fidelity_enhancer']['svq_temp_rng']

    def forward(self, x):
        """
        :param x: (B, C, L)
        """
        pass

    def fidelity_enhancer_loss_fn(self, x, s_a_l, s_a_h):
        # s -> z -> x
        x_a_l = self.maskgit.decode_token_ind_to_timeseries(s_a_l, 'LF')  # (b 1 l)
        x_a_h = self.maskgit.decode_token_ind_to_timeseries(s_a_h, 'HF')  # (b 1 l)
        x_a = x_a_l + x_a_h  # (b c l)
        x_a = x_a.detach()

        xhat = self.fidelity_enhancer(x_a)
        recons_loss = F.l1_loss(xhat, x)

        fidelity_enhancer_loss = recons_loss
        return fidelity_enhancer_loss, (x_a, xhat)

    def training_step(self, batch, batch_idx):
        self.eval()
        self.fidelity_enhancer.train()

        x, y = batch
        x = x.float()

        svq_temp = np.random.uniform(*self.svq_temp_rng)
        _, s_a_l = self.maskgit.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq, svq_temp=svq_temp)  # (b n)
        _, s_a_h = self.maskgit.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq, svq_temp=svq_temp)  # (b m)

        fidelity_enhancer_loss, (x_a, xhat) = self.fidelity_enhancer_loss_fn(x, s_a_l, s_a_h)

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {'loss': fidelity_enhancer_loss,
                     }
        for k in loss_hist.keys():
            self.log(f'train/{k}', loss_hist[k])
        
        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()

        x, y = batch
        x = x.float()

        svq_temp = np.random.uniform(*self.svq_temp_rng)
        _, s_a_l = self.maskgit.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq, svq_temp=svq_temp)  # (b n)
        _, s_a_h = self.maskgit.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq, svq_temp=svq_temp)  # (b m)

        fidelity_enhancer_loss, (x_a, xhat) = self.fidelity_enhancer_loss_fn(x, s_a_l, s_a_h)

        # log
        loss_hist = {'loss': fidelity_enhancer_loss,
                     }
        for k in loss_hist.keys():
            self.log(f'val/{k}', loss_hist[k])

        # maskgit sampling
        if batch_idx == 0:
            class_index = np.random.choice(np.concatenate(([None], np.unique(y.cpu()))))

            # unconditional sampling
            s_l, s_h = self.maskgit.iterative_decoding(device=x.device, class_index=class_index)
            x_new_l = self.maskgit.decode_token_ind_to_timeseries(s_l, 'LF').cpu()
            x_new_h = self.maskgit.decode_token_ind_to_timeseries(s_h, 'HF').cpu()
            x_new = x_new_l + x_new_h
            x_new_corrected = self.fidelity_enhancer(x_new.to(x.device)).detach().cpu().numpy()

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
        opt = torch.optim.AdamW([{'params': self.parameters(), 'lr': self.config['exp_params']['LR']}], lr=self.config['exp_params']['LR'])
        T_max = self.config['trainer_params']['max_steps']['stage_fid_enhancer']
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, T_max, eta_min=1e-5)}
