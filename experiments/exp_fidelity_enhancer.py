import os
from typing import Callable
from pathlib import Path
import tempfile
from typing import List

import matplotlib.pyplot as plt
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from sklearn.decomposition import PCA

from generators.maskgit import MaskGIT
import pytorch_lightning as pl
from supervised_FCN_2.example_compute_FID import calculate_fid

from evaluation.metrics import Metrics
from generators.fidelity_enhancer import FidelityEnhancer
from experiments.exp_maskgit import ExpMaskGIT
from utils import get_root_dir, freeze, compute_downsample_rate, timefreq_to_time, time_to_timefreq, quantize, zero_pad_low_freq, zero_pad_high_freq


class ExpFidelityEnhancer(pl.LightningModule):
    def __init__(self,
                 dataset_name: str,
                 input_length: int,
                 config: dict,
                 n_classes: int,
                 use_pretrained_ExpMaskGIT:str,
                 feature_extractor_type:str,
                 ):
        super().__init__()
        self.config = config
        self.n_fft = config['VQ-VAE']['n_fft']

        # domain shifter
        self.fidelity_enhancer = FidelityEnhancer(input_length, 1, config)
        # self.tau = None  # set by `search_optimal_tau`

        # load the stage2 model
        exp_maskgit_config = {'dataset_name':dataset_name, 'input_length':input_length, 'config':config, 'n_classes':n_classes, 'use_fidelity_enhancer':False, 'feature_extractor_type':'rocket'}
        # ckpt_fname = ExpMaskGIT_ckpt_fname
        ckpt_fname = os.path.join('saved_models', f'stage2-{dataset_name}.ckpt')
        if use_pretrained_ExpMaskGIT and os.path.isfile(ckpt_fname):
            stage2 = ExpMaskGIT.load_from_checkpoint(ckpt_fname, **exp_maskgit_config, map_location='cpu')
            self.is_pretrained_maskgit_used = True
            print('\nThe pretrained ExpMaskGIT is loaded.\n')
        else:
            stage2 = ExpMaskGIT(**exp_maskgit_config)
            self.is_pretrained_maskgit_used = False
            print('\nno pretrained ExpMaskGIT is available.\n')
        freeze(stage2)
        stage2.eval()

        self.maskgit = stage2.maskgit
        self.encoder_l = self.maskgit.encoder_l
        self.decoder_l = self.maskgit.decoder_l
        self.vq_model_l = self.maskgit.vq_model_l
        self.encoder_h = self.maskgit.encoder_h
        self.decoder_h = self.maskgit.decoder_h
        self.vq_model_h = self.maskgit.vq_model_h

        self.metrics = Metrics(dataset_name, feature_extractor_type)

    @torch.no_grad()
    def search_optimal_tau(self, 
                           X_train:np.ndarray, 
                           device, 
                           tau_search_rng:List[float]=[0.1, 0.5, 1, 2, 4], 
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
            x_new_l = maskgit.decode_token_ind_to_timeseries(s_l, 'LF').cpu()
            x_new_h = maskgit.decode_token_ind_to_timeseries(s_h, 'HF').cpu()
            x_new = x_new_l + x_new_h
            Xhat = torch.cat((Xhat, x_new))
        Xhat = Xhat.numpy().astype(float)
        Zhat = self.metrics._extract_feature_representations(Xhat)  # (b d)
        
        fids = []
        for i, tau in enumerate(tau_search_rng):
            print(f'searching optimal tau... ({round((i)/len(tau_search_rng) * 100, 1)}%)')
            X_prime = []
            n_iters = X_train.shape[0] // batch_size + (0 if X_train.shape[0] % batch_size == 0 else 1)
            for i in range(n_iters):
                x = X_train[i*batch_size:(i+1)*batch_size]
                x = torch.from_numpy(x).float().to(device)
                _, s_a_l = maskgit.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq, svq_temp=tau)  # (b n)
                _, s_a_h = maskgit.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq, svq_temp=tau)  # (b m)
                x_a_l = maskgit.decode_token_ind_to_timeseries(s_a_l, 'LF')  # (b 1 l)
                x_a_h = maskgit.decode_token_ind_to_timeseries(s_a_h, 'HF')  # (b 1 l)
                x_a = x_a_l + x_a_h  # (b c l)
                x_a = x_a.detach().cpu().numpy().astype(float)
                X_prime.append(x_a)
            X_prime = np.concatenate(X_prime)
            
            Z_prime = self.metrics._extract_feature_representations(X_prime)  # (b d)
            
            fid = self.metrics.fid_score(Zhat, Z_prime)
            fids.append(fid)
            print(f'tau:{tau} | fid:{round(int(fid))}')

            # # ====
            # fig, axes = plt.subplots(3, 1, figsize=(4, 2*3))
            # fig.suptitle(f'xhat vs x` (tau:{tau}, fid:{round(fid)})')
            # axes[0].set_title('xhat')
            # axes[0].plot(Xhat[:100,0,:].T, color='C0', alpha=0.2)
            # axes[1].set_title('x`')
            # axes[1].plot(X_prime[:100,0,:].T, color='C0', alpha=0.2)

            # pca = PCA(n_components=2)
            # Zhat_pca = pca.fit_transform(Zhat)
            # Z_prime_pca = pca.transform(Z_prime)

            # axes[2].scatter(Zhat_pca[:100,0], Zhat_pca[:100,1], alpha=0.2)
            # axes[2].scatter(Z_prime_pca[:100,0], Z_prime_pca[:100,1], alpha=0.2)

            # plt.tight_layout()
            # plt.show()
            # # ====
        print('{tau:fid} :', {tau: round(float(fid),1) for tau, fid in zip(tau_search_rng, fids)})
        optimal_idx = np.argmin(fids)
        optimal_tau = tau_search_rng[optimal_idx]
        print('** optimal_tau **:', optimal_tau)
        self.fidelity_enhancer.tau = torch.tensor(optimal_tau).float()

    # @torch.no_grad()
    # def search_optimal_tau(self, X_train:np.ndarray, device, tau_search_rng:List[float]=[0.1, 0.5, 1, 2, 3], batch_size=32) -> None:
    #     """
    #     must be run right after the instance is created.
    #     """
    #     Z_train = self.metrics._extract_feature_representations(X_train)  # (b d)
    #     maskgit = self.maskgit.to(device)
        
    #     fids = []
    #     print('searching optimal tau...')
    #     for tau in tau_search_rng:
    #         print('searching optimal tau... ')
    #         X_prime = []
    #         n_iters = X_train.shape[0] // batch_size + (0 if X_train.shape[0] % batch_size == 0 else 1)
    #         for i in range(n_iters):
    #             x = X_train[i*batch_size:(i+1)*batch_size]
    #             x = torch.from_numpy(x).float().to(device)
    #             _, s_a_l = maskgit.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq, svq_temp=tau)  # (b n)
    #             _, s_a_h = maskgit.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq, svq_temp=tau)  # (b m)
    #             x_a_l = maskgit.decode_token_ind_to_timeseries(s_a_l, 'LF')  # (b 1 l)
    #             x_a_h = maskgit.decode_token_ind_to_timeseries(s_a_h, 'HF')  # (b 1 l)
    #             x_a = x_a_l + x_a_h  # (b c l)
    #             # x_a = x_a.numpy().astype(float)  
    #             x_a = x_a.detach().cpu().numpy().astype(float)
    #             X_prime.append(x_a)
    #         X_prime = np.concatenate(X_prime)
    #         Z_prime = self.metrics._extract_feature_representations(X_prime)  # (b d)
    #         fid = calculate_fid(Z_train, Z_prime)
    #         fids.append(fid)
    #     optimal_idx = np.argmin(fids)
    #     optimal_tau = tau_search_rng[optimal_idx]
    #     print('optimal_tau:', optimal_tau)
    #     self.tau = optimal_tau

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

    # def _sample_svq_temp(self):
    #     svq_temp = np.random.uniform(*self.svq_temp_rng)
    #     return svq_temp
    
    # def _sample_svq_temp(self):
    #     low, high = self.svq_temp_rng
    #     log_low = np.log(low)
    #     log_high = np.log(high)
    #     log_sample = np.random.uniform(log_low, log_high)
    #     svq_temp = np.e ** log_sample
    #     return svq_temp
    
    def _sample_svq_temp(self):
        sample = np.random.exponential(scale=0.6)  # lambda = 1/scale
        return np.clip(sample, a_min=0.01, a_max=None)  # clip for numerical stability
        # min_value = self.svq_temp_rng[0]
        # shifted_sample = sample + min_value
        # return shifted_sample

    def training_step(self, batch, batch_idx):
        self.eval()
        self.fidelity_enhancer.train()

        x, y = batch
        x = x.float()

        # svq_temp = np.random.uniform(*self.svq_temp_rng)
        # svq_temp = self._sample_svq_temp()
        svq_temp = self.fidelity_enhancer.tau.item()
        _, s_a_l = self.maskgit.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq, svq_temp=svq_temp)  # (b n)
        _, s_a_h = self.maskgit.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq, svq_temp=svq_temp)  # (b m)

        fidelity_enhancer_loss, (xprime, xprime_R) = self.fidelity_enhancer_loss_fn(x, s_a_l, s_a_h)

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

        # svq_temp = np.random.uniform(*self.svq_temp_rng)
        # svq_temp = self._sample_svq_temp()  # (b 1)
        svq_temp = self.fidelity_enhancer.tau.item()
        _, s_a_l = self.maskgit.encode_to_z_q(x, self.encoder_l, self.vq_model_l, zero_pad_high_freq, svq_temp=svq_temp)  # (b n)
        _, s_a_h = self.maskgit.encode_to_z_q(x, self.encoder_h, self.vq_model_h, zero_pad_low_freq, svq_temp=svq_temp)  # (b m)

        fidelity_enhancer_loss, (xprime, xprime_R) = self.fidelity_enhancer_loss_fn(x, s_a_l, s_a_h)

        # log
        loss_hist = {'loss': fidelity_enhancer_loss,
                     }
        for k in loss_hist.keys():
            self.log(f'val/{k}', loss_hist[k])

        # maskgit sampling
        if batch_idx == 0 and self.is_pretrained_maskgit_used:
            class_index = np.random.choice(np.concatenate(([None], np.unique(y.cpu()))))

            # unconditional sampling
            num = 1024
            xhat_l, xhat_h, xhat = self.metrics.sample(self.maskgit, x.device, num, 'unconditional')
            xhat_R = self.fidelity_enhancer(xhat.to(x.device)).detach().cpu().numpy()

            b = 0
            n_figs = 9
            fig, axes = plt.subplots(n_figs, 1, figsize=(4, 2 * n_figs))
            fig.suptitle(f'Epoch {self.current_epoch}; Class Index: {class_index}')

            axes[0].set_title('xhat_l')
            axes[0].plot(xhat_l[b, 0, :])
            axes[1].set_title('xhat_h')
            axes[1].plot(xhat_h[b, 0, :])
            axes[2].set_title('xhat')
            axes[2].plot(xhat[b, 0, :])
            axes[3].set_title('FE(xhat)')
            axes[3].plot(xhat_R[b, 0, :])

            x = x.cpu().numpy()
            xprime = xprime.cpu().numpy()
            xprime_R = xprime_R.cpu().numpy()
            xhat = xhat.cpu().numpy()
            b_ = np.random.randint(0, x.shape[0])
            axes[4].set_title('x vs FE(x`)')
            axes[4].plot(x[b_, 0, :], alpha=0.7)
            axes[4].plot(xprime_R[b_, 0, :], alpha=0.7)

            axes[5].set_title('x` vs FE(x`)')
            axes[5].plot(xprime[b_, 0, :], alpha=0.7)
            axes[5].plot(xprime_R[b_, 0, :], alpha=0.7)
            
            axes[6].set_title('x')
            axes[6].plot(x[b_, 0, :])

            axes[7].set_title(f'x` (tau={round(svq_temp, 2)})')
            axes[7].plot(xprime[b_, 0, :])

            axes[8].set_title('FE(x`)')
            axes[8].plot(xprime_R[b_, 0, :])

            for ax in axes:
                ax.set_ylim(-4, 4)
            plt.tight_layout()
            wandb.log({f"maskgit sample": wandb.Image(plt)})
            plt.close()
            
            # log the evaluation metrics
            # xhat = xhat.numpy()

            zhat = self.metrics.z_gen_fn(xhat)
            fid_test_gen = self.metrics.fid_score(self.metrics.z_test, zhat)
            mdd, acd, sd, kd = self.metrics.stat_metrics(self.metrics.X_test, xhat)
            self.log('metrics/FID', fid_test_gen)
            self.log('metrics/MDD', mdd)
            self.log('metrics/ACD', acd)
            self.log('metrics/SD', sd)
            self.log('metrics/KD', kd)

            zhat_R = self.metrics.z_gen_fn(xhat_R)
            fid_test_gen_fe = self.metrics.fid_score(self.metrics.z_test, zhat_R)
            mdd, acd, sd, kd = self.metrics.stat_metrics(self.metrics.X_test, xhat_R)
            self.log('metrics/FID with FE', fid_test_gen_fe)
            self.log('metrics/MDD with FE', mdd)
            self.log('metrics/ACD with FE', acd)
            self.log('metrics/SD with FE', sd)
            self.log('metrics/KD with FE', kd)
            
            z_prime = self.metrics.z_gen_fn(xprime)
            fid_x_test_x_prime = self.metrics.fid_score(self.metrics.z_test, z_prime)
            # fid_train_x_prime, fid_test_x_prime = self.metrics.fid_score(x_a)
            mdd, acd, sd, kd = self.metrics.stat_metrics(x, xprime)
            self.log('metrics/FID (x, x`)', fid_x_test_x_prime)
            self.log('metrics/MDD (x, x`)', mdd)
            self.log('metrics/ACD (x, x`)', acd)
            self.log('metrics/SD (x, x`)', sd)
            self.log('metrics/KD (x, x`)', kd)

            plt.figure(figsize=(4, 4))
            labels = ['z_test', 'zhat_R']
            pca = PCA(n_components=2, random_state=0)
            for i, (Z, label) in enumerate(zip([self.metrics.z_test, zhat_R], labels)):
                ind = np.random.choice(range(Z.shape[0]), size=num, replace=True)
                Z_embed = pca.fit_transform(Z[ind]) if i==0 else pca.transform(Z[ind])
                plt.scatter(Z_embed[:, 0], Z_embed[:, 1], alpha=0.1, label=label)
            plt.legend(loc='upper right')
            plt.tight_layout()
            wandb.log({f"PCA on Z ({labels})": wandb.Image(plt)})
            plt.close()

        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params': self.parameters(), 'lr': self.config['exp_params']['LR']}], lr=self.config['exp_params']['LR'])
        T_max = self.config['trainer_params']['max_steps']['stage_fid_enhancer']
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, T_max, eta_min=1e-5)}
