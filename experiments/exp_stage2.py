import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import pytorch_lightning as pl

from evaluation.metrics import Metrics, sample
from generators.maskgit import MaskGIT
from utils import linear_warmup_cosine_annealingLR


class ExpStage2(pl.LightningModule):
    def __init__(self,
                 dataset_name: str,
                 in_channels:int,
                 input_length: int,
                 config: dict,
                 n_classes: int,
                 feature_extractor_type:str='rocket',
                 use_custom_dataset:bool=False
                 ):
        super().__init__()
        self.n_classes = n_classes
        self.config = config
        self.use_custom_dataset = use_custom_dataset

        self.maskgit = MaskGIT(dataset_name, in_channels, input_length, **config['MaskGIT'], config=config, n_classes=n_classes)
        self.metrics = Metrics(config, dataset_name, n_classes, feature_extractor_type=feature_extractor_type, use_custom_dataset=use_custom_dataset)

    def training_step(self, batch, batch_idx):
        x, y = batch

        mask_pred_loss, (mask_pred_loss_l, mask_pred_loss_h) = self.maskgit(x, y)

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        self.log('global_step', self.global_step)
        loss_hist = {'loss': mask_pred_loss,
                     'mask_pred_loss': mask_pred_loss,
                     'mask_pred_loss_l': mask_pred_loss_l,
                     'mask_pred_loss_h': mask_pred_loss_h,
                     }
        for k in loss_hist.keys():
            self.log(f'train/{k}', loss_hist[k])

        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        x, y = batch

        mask_pred_loss, (mask_pred_loss_l, mask_pred_loss_h) = self.maskgit(x, y)

        # log
        self.log('global_step', self.global_step)
        loss_hist = {'loss': mask_pred_loss,
                     'mask_pred_loss': mask_pred_loss,
                     'mask_pred_loss_l': mask_pred_loss_l,
                     'mask_pred_loss_h': mask_pred_loss_h,
                     }
        for k in loss_hist.keys():
            self.log(f'val/{k}', loss_hist[k])
        
        # maskgit sampling & evaluation
        if batch_idx == 0 and (self.training == False):
            print('computing evaluation metrices...')
            self.maskgit.eval()

            n_samples = 1024
            xhat_l, xhat_h, xhat = self.metrics.sample(self.maskgit, x.device, n_samples, 'unconditional', class_index=None)

            self._visualize_generated_timeseries(xhat_l, xhat_h, xhat)

            # compute metrics
            xhat = xhat.numpy()
            zhat = self.metrics.z_gen_fn(xhat)
            fid_test_gen = self.metrics.fid_score(self.metrics.z_test, zhat)
            mdd, acd, sd, kd = self.metrics.stat_metrics(self.metrics.X_test, xhat)
            self.log('running_metrics/FID', fid_test_gen)
            self.log('running_metrics/MDD', mdd)
            self.log('running_metrics/ACD', acd)
            self.log('running_metrics/SD', sd)
            self.log('running_metrics/KD', kd)

        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config['exp_params']['lr'])
        scheduler = linear_warmup_cosine_annealingLR(opt, self.config['trainer_params']['max_steps']['stage2'], self.config['exp_params']['linear_warmup_rate'], min_lr=self.config['exp_params']['min_lr'])
        return {'optimizer': opt, 'lr_scheduler': scheduler}

    def _visualize_generated_timeseries(self, xhat_l, xhat_h, xhat):
        b = 0
        c = np.random.randint(0, xhat.shape[1])
        
        n_rows = 3
        fig, axes = plt.subplots(n_rows, 1, figsize=(4, 2*n_rows))
        fig.suptitle(f'step-{self.global_step} | channel idx:{c} \n unconditional sampling')
        axes = axes.flatten()
        axes[0].set_title(r'$\hat{x}_l$ (LF)')
        axes[0].plot(xhat_l[b,c,:])
        axes[1].set_title(r'$\hat{x}_h$ (HF)')
        axes[1].plot(xhat_h[b,c,:])
        axes[2].set_title(r'$\hat{x}$ (LF+HF)')
        axes[2].plot(xhat[b,c,:])
        for ax in axes:
            ax.set_ylim(-4, 4)
        plt.tight_layout()
        self.logger.log_image(key='generated sample', images=[wandb.Image(plt),])
        plt.close()