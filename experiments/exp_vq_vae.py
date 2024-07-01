import torch
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import wandb
import pytorch_lightning as pl

from encoder_decoders.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from experiments.exp_base import ExpBase, detach_the_unnecessary
from vector_quantization import VectorQuantize
from supervised_FCN.example_pretrained_model_loading import load_pretrained_FCN
from utils import compute_downsample_rate, freeze, timefreq_to_time, time_to_timefreq, zero_pad_low_freq, zero_pad_high_freq, quantize


class ExpVQVAE(pl.LightningModule):
    def __init__(self,
                 input_length: int,
                 config: dict):
        """
        :param input_length: length of input time series
        :param config: configs/config.yaml
        :param n_train_samples: number of training samples
        """
        super().__init__()
        self.config = config

        self.n_fft = config['VQ-VAE']['n_fft']
        dim = config['encoder']['dim']
        in_channels = config['dataset']['in_channels']
        downsampled_width_l = config['encoder']['downsampled_width']['lf']
        downsampled_width_h = config['encoder']['downsampled_width']['hf']
        downsample_rate_l = compute_downsample_rate(input_length, self.n_fft, downsampled_width_l)
        downsample_rate_h = compute_downsample_rate(input_length, self.n_fft, downsampled_width_h)

        # encoder
        self.encoder_l = VQVAEEncoder(dim, 2 * in_channels, downsample_rate_l, config['encoder']['n_resnet_blocks'])
        self.decoder_l = VQVAEDecoder(dim, 2 * in_channels, downsample_rate_l, config['decoder']['n_resnet_blocks'])
        self.vq_model_l = VectorQuantize(dim, config['VQ-VAE']['codebook_sizes']['lf'], **config['VQ-VAE'])

        # decoder
        self.encoder_h = VQVAEEncoder(dim, 2 * in_channels, downsample_rate_h, config['encoder']['n_resnet_blocks'])
        self.decoder_h = VQVAEDecoder(dim, 2 * in_channels, downsample_rate_h, config['decoder']['n_resnet_blocks'])
        self.vq_model_h = VectorQuantize(dim, config['VQ-VAE']['codebook_sizes']['hf'], **config['VQ-VAE'])

        # pre-trained feature extractor in case the perceptual loss is used
        if config['VQ-VAE']['perceptual_loss_weight']:
            self.fcn = load_pretrained_FCN(config['dataset']['dataset_name']).to(self.device)
            self.fcn.eval()
            freeze(self.fcn)

    def forward(self, batch, batch_idx):
        """
        :param x: input time series (B, C, L)
        """
        x, y = batch

        recons_loss = {'LF.time': 0., 'HF.time': 0., 'LF.timefreq': 0., 'HF.timefreq': 0., 'perceptual': 0.}
        vq_losses = {'LF': None, 'HF': None}
        perplexities = {'LF': 0., 'HF': 0.}

        # time-frequency transformation: STFT(x)
        C = x.shape[1]
        xf = time_to_timefreq(x, self.n_fft, C)  # (B, C, H, W)
        u_l = zero_pad_high_freq(xf)  # (B, C, H, W)
        x_l = timefreq_to_time(u_l, self.n_fft, C)  # (B, C, L)

        # register `upsample_size` in the decoders
        for decoder in [self.decoder_l, self.decoder_h]:
            if not decoder.is_upsample_size_updated:
                decoder.register_upsample_size(torch.IntTensor(np.array(xf.shape[2:])))

        # forward: low-freq
        z_l = self.encoder_l(u_l)
        z_q_l, indices_l, vq_loss_l, perplexity_l = quantize(z_l, self.vq_model_l)
        xfhat_l = self.decoder_l(z_q_l)
        uhat_l = zero_pad_high_freq(xfhat_l)
        xhat_l = timefreq_to_time(uhat_l, self.n_fft, C)  # (B, C, L)

        recons_loss['LF.time'] = F.mse_loss(x_l, xhat_l)
        recons_loss['LF.timefreq'] = F.mse_loss(u_l, uhat_l)
        perplexities['LF'] = perplexity_l
        vq_losses['LF'] = vq_loss_l

        # forward: high-freq
        u_h = zero_pad_low_freq(xf)  # (B, C, H, W)
        x_h = timefreq_to_time(u_h, self.n_fft, C)  # (B, C, L)

        z_h = self.encoder_h(u_h)
        z_q_h, indices_h, vq_loss_h, perplexity_h = quantize(z_h, self.vq_model_h)
        xfhat_h = self.decoder_h(z_q_h)
        uhat_h = zero_pad_low_freq(xfhat_h)
        xhat_h = timefreq_to_time(uhat_h, self.n_fft, C)  # (B, C, L)

        recons_loss['HF.time'] = F.l1_loss(x_h, xhat_h)
        recons_loss['HF.timefreq'] = F.mse_loss(u_h, uhat_h)
        perplexities['HF'] = perplexity_h
        vq_losses['HF'] = vq_loss_h

        if self.config['VQ-VAE']['perceptual_loss_weight']:
            z_fcn = self.fcn(x.float(), return_feature_vector=True).detach()
            zhat_fcn = self.fcn(xhat_l.float() + xhat_h.float(), return_feature_vector=True)
            recons_loss['perceptual'] = F.mse_loss(z_fcn, zhat_fcn)

        # plot `x` and `xhat`
        # r = np.random.rand()
        # if self.training and r <= 0.05:
        if not self.training and batch_idx == 0:
            b = np.random.randint(0, x_h.shape[0])
            c = np.random.randint(0, x_h.shape[1])

            fig, axes = plt.subplots(3, 1, figsize=(4, 2*3))
            plt.suptitle(f'ep_{self.current_epoch}')
            axes[0].plot(x_l[b, c].cpu())
            axes[0].plot(xhat_l[b, c].detach().cpu())
            axes[0].set_title('x_l')
            axes[0].set_ylim(-4, 4)

            axes[1].plot(x_h[b, c].cpu())
            axes[1].plot(xhat_h[b, c].detach().cpu())
            axes[1].set_title('x_h')
            axes[1].set_ylim(-4, 4)

            axes[2].plot(x_l[b, c].cpu() + x_h[b, c].cpu())
            axes[2].plot(xhat_l[b, c].detach().cpu() + xhat_h[b, c].detach().cpu())
            axes[2].set_title('x')
            axes[2].set_ylim(-4, 4)

            plt.tight_layout()
            wandb.log({"x vs xhat (val)": wandb.Image(plt)})
            plt.close()

        return recons_loss, vq_losses, perplexities

    def training_step(self, batch, batch_idx):
        recons_loss, vq_losses, perplexities = self.forward(batch, batch_idx)
        loss = (recons_loss['LF.time'] + recons_loss['HF.time'] +
                recons_loss['LF.timefreq'] + recons_loss['HF.timefreq']) + \
                vq_losses['LF']['loss'] + vq_losses['HF']['loss'] + \
                recons_loss['perceptual']

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {'loss': loss,
                     'recons_loss.time': recons_loss['LF.time'] + recons_loss['HF.time'],
                     'recons_loss.LF.time': recons_loss['LF.time'],
                     'recons_loss.HF.time': recons_loss['HF.time'],

                     'recons_loss.LF.timefreq': recons_loss['LF.timefreq'],
                     'recons_loss.HF.timefreq': recons_loss['HF.timefreq'],

                     'commit_loss.LF': vq_losses['LF']['commit_loss'],
                     'commit_loss.HF': vq_losses['HF']['commit_loss'],
                     'perplexity.LF': perplexities['LF'],
                     'perplexity.HF': perplexities['HF'],

                     'perceptual': recons_loss['perceptual']
                     }
        
        # log
        for k in loss_hist.keys():
            self.log(f'train/{k}', loss_hist[k])

        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()

        recons_loss, vq_losses, perplexities = self.forward(batch, batch_idx)
        loss = (recons_loss['LF.time'] + recons_loss['HF.time'] +
                recons_loss['LF.timefreq'] + recons_loss['HF.timefreq']) + \
                vq_losses['LF']['loss'] + vq_losses['HF']['loss'] + \
                recons_loss['perceptual']

        # log
        loss_hist = {'loss': loss,
                     'recons_loss.time': recons_loss['LF.time'] + recons_loss['HF.time'],
                     'recons_loss.LF.time': recons_loss['LF.time'],
                     'recons_loss.HF.time': recons_loss['HF.time'],

                     'recons_loss.LF.timefreq': recons_loss['LF.timefreq'],
                     'recons_loss.HF.timefreq': recons_loss['HF.timefreq'],

                     'commit_loss.LF': vq_losses['LF']['commit_loss'],
                     'commit_loss.HF': vq_losses['HF']['commit_loss'],
                     'perplexity.LF': perplexities['LF'],
                     'perplexity.HF': perplexities['HF'],

                     'perceptual': recons_loss['perceptual']
                     }
        
        # log
        for k in loss_hist.keys():
            self.log(f'val/{k}', loss_hist[k])

        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), weight_decay=self.config['exp_params']['weight_decay'], lr=self.config['exp_params']['LR'])
        T_max = self.config['trainer_params']['max_steps']['stage1']
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, T_max, eta_min=1e-5)}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.eval()

        x = batch
        recons_loss, vq_losses, perplexities = self.forward(x)
        loss = (recons_loss['LF.time'] + recons_loss['HF.time'] +
                recons_loss['LF.timefreq'] + recons_loss['HF.timefreq']) + \
                vq_losses['LF']['loss'] + vq_losses['HF']['loss'] + \
                recons_loss['perceptual']

        # log
        loss_hist = {'loss': loss,
                     'recons_loss.time': recons_loss['LF.time'] + recons_loss['HF.time'],
                     'recons_loss.LF.time': recons_loss['LF.time'],
                     'recons_loss.HF.time': recons_loss['HF.time'],

                     'recons_loss.LF.timefreq': recons_loss['LF.timefreq'],
                     'recons_loss.HF.timefreq': recons_loss['HF.timefreq'],

                     'commit_loss.LF': vq_losses['LF']['commit_loss'],
                     'commit_loss.HF': vq_losses['HF']['commit_loss'],
                     'perplexity.LF': perplexities['LF'],
                     'perplexity.HF': perplexities['HF'],

                     'perceptual': recons_loss['perceptual']
                     }
        
        # log
        for k in loss_hist.keys():
            self.log(f'test/{k}', loss_hist[k])

        return loss_hist
