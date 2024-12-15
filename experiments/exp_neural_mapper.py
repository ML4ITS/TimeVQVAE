import os
import math
import warnings
import typing as tp
from typing import Callable, Optional, Sequence, Tuple, Union

import torchaudio
import matplotlib.pyplot as plt
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F
from sklearn.decomposition import PCA
from einops import rearrange
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch.nn as nn
from torch import Tensor

from evaluation.rocket_functions import MiniRocketTransform
from evaluation.metrics import Metrics
from generators.neural_mapper import NeuralMapper
from experiments.exp_stage2 import ExpStage2
from utils import freeze, zero_pad_low_freq, zero_pad_high_freq, linear_warmup_cosine_annealingLR, time_to_timefreq


class ExpNeuralMapper(pl.LightningModule):
    def __init__(self,
                 dataset_name: str,
                 in_channels:int,
                 input_length: int,
                 config: dict,
                 n_classes: int,
                 feature_extractor_type:str,
                 use_custom_dataset:bool=False
                 ):
        super().__init__()
        self.config = config
        self.use_custom_dataset = use_custom_dataset
        self.n_fft = config['VQ-VAE']['n_fft']
        self.tau_search_rng = config['neural_mapper']['tau_search_rng']

        # NM
        self.neural_mapper = NeuralMapper(input_length, in_channels, config)

        # load the stage2 model
        exp_maskgit_config = {'dataset_name':dataset_name, 
                              'in_channels':in_channels,
                              'input_length':input_length, 
                              'config':config, 
                              'n_classes':n_classes, 
                              'feature_extractor_type':'rocket',
                              'use_custom_dataset':use_custom_dataset
                              }
        ckpt_fname = os.path.join('saved_models', f'stage2-{dataset_name}.ckpt')
        stage2 = ExpStage2.load_from_checkpoint(ckpt_fname, **exp_maskgit_config, map_location='cpu', strict=False)
        print('\nThe pretrained ExpStage2 is loaded.\n')
        freeze(stage2)
        stage2.eval()

        self.maskgit = stage2.maskgit
        self.encoder_l = self.maskgit.encoder_l
        self.decoder_l = self.maskgit.decoder_l
        self.vq_model_l = self.maskgit.vq_model_l
        self.encoder_h = self.maskgit.encoder_h
        self.decoder_h = self.maskgit.decoder_h
        self.vq_model_h = self.maskgit.vq_model_h

        # log2_rates = np.arange(5, min(int(np.floor(np.log2(input_length))),11)+1)
        # window_lengths = [2**rate for rate in log2_rates]
        # self.msstft_loss_fn = MultiScaleSTFTLoss(window_lengths=window_lengths)

        self.minirocket = MiniRocketTransform(input_length)
        freeze(self.minirocket)

        self.metrics = Metrics(config, dataset_name, n_classes, feature_extractor_type, use_custom_dataset=use_custom_dataset)

    @torch.no_grad()
    def search_optimal_tau(self, 
                           X_train:np.ndarray, 
                           device, 
                           wandb_logger:WandbLogger,
                           n_samples:int=1024, 
                           batch_size:int=32) -> None:
        """
        must be run right after the instance is created.
        """
        maskgit = self.maskgit.to(device)

        n_iters = n_samples // batch_size + (0 if (n_samples%batch_size == 0) else 1)
        Xhat = torch.tensor([])
        for iter_idx in range(n_iters):
            s_l, s_h = maskgit.iterative_decoding(num=batch_size, device=device, class_index=None)
            x_new_l = maskgit.decode_token_ind_to_timeseries(s_l, 'lf').cpu()
            x_new_h = maskgit.decode_token_ind_to_timeseries(s_h, 'hf').cpu()
            x_new = x_new_l + x_new_h
            Xhat = torch.cat((Xhat, x_new))
        Xhat = Xhat.numpy().astype(float)
        Zhat = self.metrics.extract_feature_representations(Xhat, 'rocket')  # (b d)
        
        rind_hat_100 = np.random.randint(0, Xhat.shape[0], size=100)
        rind_prime_100 = np.random.randint(0, X_train.shape[0], size=100)
        rind_hat_200 = np.random.randint(0, Xhat.shape[0], size=200)
        rind_prime_200 = np.random.randint(0, X_train.shape[0], size=200)
        fids = []
        for i, tau in enumerate(self.tau_search_rng):
            print(f'searching optimal tau... ({round((i)/len(self.tau_search_rng) * 100, 1)}%)')
            Xprime = []
            n_iters = X_train.shape[0] // batch_size + (0 if X_train.shape[0] % batch_size == 0 else 1)
            for i in range(n_iters):
                x = X_train[i*batch_size:(i+1)*batch_size]
                x = torch.from_numpy(x).float().to(device)
                _, sprime_l = maskgit.encode_to_z_q(x, self.encoder_l, self.vq_model_l, svq_temp=tau)  # (b n)
                _, sprime_h = maskgit.encode_to_z_q(x, self.encoder_h, self.vq_model_h, svq_temp=tau)  # (b m)
                xprime_l = maskgit.decode_token_ind_to_timeseries(sprime_l, 'lf')  # (b 1 l)
                xprime_h = maskgit.decode_token_ind_to_timeseries(sprime_h, 'hf')  # (b 1 l)
                xprime = xprime_l + xprime_h  # (b c l)
                xprime = xprime.detach().cpu().numpy().astype(float)
                Xprime.append(xprime)
            Xprime = np.concatenate(Xprime)
            
            Z_prime = self.metrics.extract_feature_representations(Xprime, 'rocket')  # (b d)
            
            fid = self.metrics.fid_score(Zhat, Z_prime)
            fids.append(fid)
            print(f'tau:{tau} | fid:{round(fid,4)}')

            # ====
            fig, axes = plt.subplots(3, 1, figsize=(4, 2*3))
            # fig.suptitle(f'xhat vs x` (tau:{tau}, fid:{round(fid,4)})')
            fig.suptitle(r'$\hat{x}$ vs $\tilde{x}^\prime$' + f' | tau:{tau}, fid:{round(fid,4)}')
            axes[0].set_title(r'$\hat{x}$')
            axes[0].plot(Xhat[rind_hat_100,0,:].T, color='C0', alpha=0.2)
            axes[0].set_ylim(-5, 5)
            axes[1].set_title(r'$\tilde{x}^\prime$')
            axes[1].plot(Xprime[rind_prime_100,0,:].T, color='C0', alpha=0.2)
            axes[1].set_ylim(-5, 5)

            pca = PCA(n_components=2)
            Zhat_pca = pca.fit_transform(Zhat)
            Z_prime_pca = pca.transform(Z_prime)
            
            axes[2].set_title('comparison in a latent space')
            axes[2].scatter(Zhat_pca[rind_hat_200,0], Zhat_pca[rind_hat_200,1], alpha=0.2)
            axes[2].scatter(Z_prime_pca[rind_prime_200,0], Z_prime_pca[rind_prime_200,1], alpha=0.2)

            plt.tight_layout()
            wandb_logger.log_image(key='Xhat vs Xprime', images=[wandb.Image(plt),])
            plt.close()
            # plt.show()
            # ====
        print('{tau:fid} :', {tau: round(float(fid),4) for tau, fid in zip(self.tau_search_rng, fids)})
        optimal_idx = np.argmin(fids)
        optimal_tau = self.tau_search_rng[optimal_idx]
        print('** optimal_tau **:', optimal_tau)
        self.neural_mapper.tau = torch.tensor(optimal_tau).float()

    def _neural_mapper_loss_fn(self, x, sprime_l, sprime_h):
        # s -> z -> x
        xprime_l = self.maskgit.decode_token_ind_to_timeseries(sprime_l, 'lf')  # (b 1 l)
        xprime_h = self.maskgit.decode_token_ind_to_timeseries(sprime_h, 'hf')  # (b 1 l)
        xprime = xprime_l + xprime_h  # (b c l)
        xprime = xprime.detach()

        xhat = self.neural_mapper(xprime)
        # recons_loss = self.msstft_loss_fn(xhat, x) + 0.1*F.l1_loss(xhat, x)
        recons_loss = F.l1_loss(xhat, x)

        neural_mapper_loss = recons_loss
        return neural_mapper_loss, (xprime, xhat)

    def training_step(self, batch, batch_idx):
        self.eval()
        self.neural_mapper.train()

        x, y = batch
        x = x.float()

        tau = self.neural_mapper.tau.item()
        _, sprime_l = self.maskgit.encode_to_z_q(x, self.encoder_l, self.vq_model_l, svq_temp=tau)  # (b n)
        _, sprime_h = self.maskgit.encode_to_z_q(x, self.encoder_h, self.vq_model_h, svq_temp=tau)  # (b m)

        neural_mapper_loss, (xprime, xprime_R) = self._neural_mapper_loss_fn(x, sprime_l, sprime_h)

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss = neural_mapper_loss
        self.log('global_step', self.global_step)
        loss_hist = {'loss':loss,
                     'neural_mapper_loss':neural_mapper_loss,
                     }
        for k in loss_hist.keys():
            self.log(f'train/{k}', loss_hist[k])
        
        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()

        x, y = batch
        x = x.float()

        tau = self.neural_mapper.tau.item()
        _, sprime_l = self.maskgit.encode_to_z_q(x, self.encoder_l, self.vq_model_l, svq_temp=tau)  # (b n)
        _, sprime_h = self.maskgit.encode_to_z_q(x, self.encoder_h, self.vq_model_h, svq_temp=tau)  # (b m)

        neural_mapper_loss, (xprime, xprime_R) = self._neural_mapper_loss_fn(x, sprime_l, sprime_h)

        # log
        loss = neural_mapper_loss
        self.log('global_step', self.global_step)
        loss_hist = {'loss':loss,
                     'neural_mapper_loss':neural_mapper_loss,
                     }
        for k in loss_hist.keys():
            self.log(f'val/{k}', loss_hist[k])

        # maskgit sampling
        if batch_idx == 0:
            class_index = None

            # unconditional sampling
            num = 1024
            xhat_l, xhat_h, xhat = self.metrics.sample(self.maskgit, x.device, num, 'unconditional', class_index=class_index)
            xhat_R = self.neural_mapper(xhat.to(x.device)).detach().cpu().numpy()

            b = 0
            n_figs = 9
            fig, axes = plt.subplots(n_figs, 1, figsize=(4, 2 * n_figs))
            fig.suptitle(f'step-{self.global_step}; class idx: {class_index}')

            axes[0].set_title(r'$\hat{x}_l$')
            axes[0].plot(xhat_l[b, 0, :])
            axes[1].set_title(r'$\hat{x}_h$')
            axes[1].plot(xhat_h[b, 0, :])
            axes[2].set_title(r'$\hat{x}$')
            axes[2].plot(xhat[b, 0, :])
            axes[3].set_title(r'NM($\hat{x}$) = $\hat{x}_R$')
            axes[3].plot(xhat_R[b, 0, :])

            x = x.cpu().numpy()
            xprime = xprime.cpu().numpy()
            xprime_R = xprime_R.cpu().numpy()
            xhat = xhat.cpu().numpy()
            b_ = np.random.randint(0, x.shape[0])
            axes[4].set_title(r'$x$ vs NM($x^\prime$)')
            axes[4].plot(x[b_, 0, :], alpha=0.7)
            axes[4].plot(xprime_R[b_, 0, :], alpha=0.7)

            axes[5].set_title(r'$x^\prime$ vs NM($x^\prime$)')
            axes[5].plot(xprime[b_, 0, :], alpha=0.7)
            axes[5].plot(xprime_R[b_, 0, :], alpha=0.7)
            
            axes[6].set_title(r'$x$')
            axes[6].plot(x[b_, 0, :])

            axes[7].set_title(fr'$x^\prime$ ($\tau$={round(tau, 5)})')
            axes[7].plot(xprime[b_, 0, :])

            axes[8].set_title(r'NM($x^\prime$)')
            axes[8].plot(xprime_R[b_, 0, :])

            for ax in axes:
                ax.set_ylim(-4, 4)
            plt.tight_layout()
            self.logger.log_image(key='unconditionally generated sample', images=[wandb.Image(plt),])
            plt.close()
            
            # log the evaluation metrics

            zhat = self.metrics.z_gen_fn(xhat)
            fid_test_gen = self.metrics.fid_score(self.metrics.z_test, zhat)
            mdd, acd, sd, kd = self.metrics.stat_metrics(self.metrics.X_test, xhat)
            self.log('running_metrics/FID', fid_test_gen)
            self.log('running_metrics/MDD', mdd)
            self.log('running_metrics/ACD', acd)
            self.log('running_metrics/SD', sd)
            self.log('running_metrics/KD', kd)

            zhat_R = self.metrics.z_gen_fn(xhat_R)
            fid_test_gen_fe = self.metrics.fid_score(self.metrics.z_test, zhat_R)
            mdd, acd, sd, kd = self.metrics.stat_metrics(self.metrics.X_test, xhat_R)
            self.log('running_metrics/FID with NM', fid_test_gen_fe)
            self.log('running_metrics/MDD with NM', mdd)
            self.log('running_metrics/ACD with NM', acd)
            self.log('running_metrics/SD with NM', sd)
            self.log('running_metrics/KD with NM', kd)
            
            z_prime = self.metrics.z_gen_fn(xprime)
            fid_x_test_x_prime = self.metrics.fid_score(self.metrics.z_test, z_prime)
            # fid_train_x_prime, fid_test_x_prime = self.metrics.fid_score(xprime)
            mdd, acd, sd, kd = self.metrics.stat_metrics(x, xprime)
            self.log('running_metrics/FID (x, x`)', fid_x_test_x_prime)
            self.log('running_metrics/MDD (x, x`)', mdd)
            self.log('running_metrics/ACD (x, x`)', acd)
            self.log('running_metrics/SD (x, x`)', sd)
            self.log('running_metrics/KD (x, x`)', kd)

            plt.figure(figsize=(4, 4))
            plt.title(f'step-{self.global_step}')
            labels = ['Z_test', 'Zhat_R']
            pca = PCA(n_components=2, random_state=0)
            for i, (Z, label) in enumerate(zip([self.metrics.z_test, zhat_R], labels)):
                ind = np.random.choice(range(Z.shape[0]), size=num, replace=True)
                Z_embed = pca.fit_transform(Z[ind]) if i==0 else pca.transform(Z[ind])
                plt.scatter(Z_embed[:, 0], Z_embed[:, 1], alpha=0.1, label=label)
            plt.legend(loc='upper right')
            plt.tight_layout()
            self.logger.log_image(key=f"PCA on Z ({'-'.join(labels)})", images=[wandb.Image(plt),])
            plt.close()

        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config['exp_params']['lr'])
        scheduler = linear_warmup_cosine_annealingLR(opt, self.config['trainer_params']['max_steps']['stage_neural_mapper'], self.config['exp_params']['linear_warmup_rate'], min_lr=self.config['exp_params']['min_lr'])
        return {'optimizer': opt, 'lr_scheduler': scheduler}




class Spectrogram(torch.nn.Module):
    r"""Create a spectrogram from a audio signal.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float or None, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for magnitude, 2 for power, etc.
            If None, then the complex spectrum is returned instead. (Default: ``2``)
        normalized (bool or str, optional): Whether to normalize by magnitude after stft. If input is str, choices are
            ``"window"`` and ``"frame_length"``, if specific normalization type is desirable. ``True`` maps to
            ``"window"``. (Default: ``False``)
        wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy (Default: ``True``)
        return_complex (bool, optional):
            Deprecated and not used.

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = torchaudio.transforms.Spectrogram(n_fft=800)
        >>> spectrogram = transform(waveform)

    """
    __constants__ = ["n_fft", "win_length", "hop_length", "pad", "power", "normalized"]

    def __init__(
        self,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        pad: int = 0,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        power: Optional[float] = 2.0,
        normalized: Union[bool, str] = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        return_complex: Optional[bool] = None,
    ) -> None:
        super(Spectrogram, self).__init__()
        torch._C._log_api_usage_once("torchaudio.transforms.Spectrogram")
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequencies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer("window", window)
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        if return_complex is not None:
            warnings.warn(
                "`return_complex` argument is now deprecated and is not effective."
                "`torchaudio.transforms.Spectrogram(power=None)` always returns a tensor with "
                "complex dtype. Please remove the argument in the function call."
            )

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Dimension (..., freq, time), where freq is
            ``n_fft // 2 + 1`` where ``n_fft`` is the number of
            Fourier bins, and time is the number of window hops (n_frame).
        """
        return torchaudio.functional.spectrogram(
            waveform,
            self.pad,
            self.window,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.power,
            self.normalized,
            self.center,
            self.pad_mode,
            self.onesided,
        )


class InverseSpectrogram(torch.nn.Module):
    r"""Create an inverse spectrogram to recover an audio signal from a spectrogram.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        normalized (bool or str, optional): Whether the stft output was normalized by magnitude. If input is str,
            choices are ``"window"`` and ``"frame_length"``, dependent on normalization mode. ``True`` maps to
            ``"window"``. (Default: ``False``)
        wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)
        center (bool, optional): whether the signal in spectrogram was padded on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        onesided (bool, optional): controls whether spectrogram was used to return half of results to
            avoid redundancy (Default: ``True``)

    Example
        >>> batch, freq, time = 2, 257, 100
        >>> length = 25344
        >>> spectrogram = torch.randn(batch, freq, time, dtype=torch.cdouble)
        >>> transform = transforms.InverseSpectrogram(n_fft=512)
        >>> waveform = transform(spectrogram, length)
    """
    __constants__ = ["n_fft", "win_length", "hop_length", "pad", "power", "normalized"]

    def __init__(
        self,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        pad: int = 0,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        normalized: Union[bool, str] = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
    ) -> None:
        super(InverseSpectrogram, self).__init__()
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequencies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer("window", window)
        self.pad = pad
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided

    def forward(self, spectrogram: Tensor, length: Optional[int] = None) -> Tensor:
        r"""
        Args:
            spectrogram (Tensor): Complex tensor of audio of dimension (..., freq, time).
            length (int or None, optional): The output length of the waveform.

        Returns:
            Tensor: Dimension (..., time), Least squares estimation of the original signal.
        """
        return torchaudio.functional.inverse_spectrogram(
            spectrogram,
            length,
            self.pad,
            self.window,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.normalized,
            self.center,
            self.pad_mode,
            self.onesided,
        )
    
def sinusoidal_window_fn(L: int) -> torch.Tensor:
    return torch.sqrt(torch.hann_window(L))

def get_window_fn(window_name: str) -> tp.Callable[[int], torch.Tensor]:
    if window_name == "sinusoidal" or "sqrt_hann":
        return sinusoidal_window_fn
    return getattr(torch.signal.windows, window_name)

class STFT(nn.Module):
    def __init__(
        self,
        *,
        window_length: int,
        hop_length: int = -1,
        window_type: str = "sqrt_hann",
        padding_type: str = "reflect",
        normalized: bool = False,
    ) -> None:
        super().__init__()

        self.window_length = window_length
        if hop_length > 0:
            self.hop_length = hop_length
        else:
            self.hop_length = self.window_length // 4
        self.window_type = window_type
        self.padding_type = padding_type
        self.normalized = normalized
        self.stft = Spectrogram(
            n_fft=self.window_length,
            hop_length=self.hop_length,
            center=True,
            power=None,
            normalized=self.normalized,
            window_fn=get_window_fn(self.window_type),
        )
        self.istft = InverseSpectrogram(
            n_fft=self.window_length,
            hop_length=self.hop_length,
            center=True,
            normalized=self.normalized,
            window_fn=get_window_fn(self.window_type),
        )


class MultiScaleSTFTLoss(nn.Module):
    """
    Computes the multi-scale Short-Time Fourier Transform (STFT) loss.

    This class implements a loss function based on multiple scales of STFT to capture both
    fine and coarse details of audio signals, as described in Engel et al. [1]. The loss
    combines several components, including raw magnitude, log-magnitude, complex L1, and
    mean squared error (MSE) losses, to assess the similarity between an estimated and
    reference signal in the spectral domain.

    Parameters
    ----------
    window_lengths : list[int], optional
        A list of window lengths for the STFT computations, providing different scales.
        Default is [2096, 612].
    log_loss_eps : float, optional
        Small epsilon value added for numerical stability when computing log-magnitude loss.
        Default is 1.0e-6.
    mag_log_weight : float, optional
        Weight of the log-magnitude portion of the loss. Default is 1.0.
    cpx_log_weight : float, optional
        Weight of the complex log-magnitude portion of the loss. Default is 1.0.
    cpx_l1_weight : float, optional
        Weight of the complex L1 loss. Default is 1.0.
    mse_weight : float, optional
        Weight of the mean squared error (MSE) loss. Default is 0.1.
    window_type : str, optional
        Type of window function to use in STFT. Default is "sqrt_hann".
    normalize_fft : bool, optional
        Whether to normalize the FFT output. Default is False.

    References
    ----------
    1. Engel, Jesse, Chenjie Gu, and Adam Roberts.
       "DDSP: Differentiable Digital Signal Processing."
       International Conference on Learning Representations. 2019.

    Notes
    -----
    Implementation inspired by:
    https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        *,
        window_lengths: list[int] = [128, 64, 32], #[2096, 612, 226, 83],
        log_loss_eps: float = 1.0e-6,
        mag_log_weight: float = 1.0,
        cpx_log_weight: float = 1.0,
        cpx_l1_weight: float = 1.0,
        mse_weight: float = 0.1,
        spectral_convergence: float = 0.,  # Erlend said it causes some instability
        root_loss: float = 1.0,
        window_type: str = "sqrt_hann",
        normalize_fft: bool = False,
        use_weights: bool = False,
    ):
        super().__init__()
        self.stft_funcs = nn.ModuleList(
            [
                STFT(
                    window_length=w,
                    hop_length=w // 4,
                    window_type=window_type,
                    normalized=normalize_fft,
                )
                for w in window_lengths
            ]
        )
        total = sum([w**0.5 for w in window_lengths]) if use_weights else len(window_lengths)
        self.stft_weights = (
            [w**0.5 / total for w in window_lengths]
            if use_weights
            else [1.0 for _ in window_lengths]
        )
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.cpx_log_weight = cpx_log_weight
        self.mag_log_weight = mag_log_weight
        self.mse_weight = mse_weight
        self.cpx_l1_weight = cpx_l1_weight
        self.log_loss_eps = log_loss_eps
        self.spectral_convergence = spectral_convergence
        self.root_loss = root_loss

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Computes multi-scale STFT loss between an estimate and a reference signal."""
        loss = 0.0

        for stft, stft_weight in zip(self.stft_funcs, self.stft_weights):
            x_tf = stft.stft(x)
            y_tf = stft.stft(y)
            x_abs = x_tf.abs()
            y_abs = y_tf.abs()

            # Compute log-magnitude losses
            log_spec_abs = self.mag_log_weight * F.smooth_l1_loss(
                x_abs.clamp_min(self.log_loss_eps).log10(),
                y_abs.clamp_min(self.log_loss_eps).log10(),
                beta=1.0,
            )
            min_log_spec_cpx = math.log10(self.log_loss_eps)
            log_spec_cpx = (
                self.cpx_log_weight * self.l1(x_tf, y_tf).clamp_min(self.log_loss_eps).log10() -min_log_spec_cpx  # `-min_log_spec_cpx` to make this loss's minimum as zero.
                if self.cpx_log_weight > 0.0
                else 0.0
            )

            # Compute complex L1 and MSE losses if applicable
            l1_loss = (
                self.cpx_l1_weight * self.l1(x_tf, y_tf) if (self.cpx_l1_weight > 0.0) else 0.0
            )
            mse_loss = (
                self.mse_weight * ((x_tf - y_tf).abs() ** 2.0).mean().sqrt()
                if (self.mse_weight > 0.0)
                else 0.0
            )

            # Spectral convergence loss
            if self.spectral_convergence > 0.0:
                nom = torch.linalg.matrix_norm(x_abs - y_abs)
                den = torch.linalg.matrix_norm(y_abs)
                spect_conv = self.spectral_convergence * (nom / den).mean()
            else:
                spect_conv = 0.0

            # root_loss
            if self.root_loss > 0.0:
                root_loss = (
                    self.l2(torch.pow(x_abs + 1.0e-8, 0.3), torch.pow(y_abs + 1.0e-8, 0.3))
                    * self.root_loss
                )
            else:
                root_loss = 0.0

            # Accumulate total loss
            losses = log_spec_abs + log_spec_cpx + l1_loss + mse_loss + spect_conv + root_loss
            loss += losses * stft_weight

        return loss