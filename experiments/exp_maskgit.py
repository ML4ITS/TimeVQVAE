import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from evaluation.metrics import Metrics
from generators.maskgit import MaskGIT
from generators.fidelity_enhancer import FidelityEnhancer
from utils import freeze


class ExpMaskGIT(pl.LightningModule):
    def __init__(self,
                 dataset_name: str,
                 input_length: int,
                 config: dict,
                 n_classes: int,
                 use_fidelity_enhancer:bool=False,
                 feature_extractor_type:str='rocket'
                 ):
        super().__init__()
        self.config = config
        self.use_fidelity_enhancer = use_fidelity_enhancer
        self.maskgit = MaskGIT(dataset_name, input_length, **config['MaskGIT'], config=config, n_classes=n_classes)

        self.metrics = Metrics(dataset_name, feature_extractor_type)

        # load the fidelity enhancer
        if self.use_fidelity_enhancer:
            self.fidelity_enhancer = FidelityEnhancer(input_length, config['dataset']['in_channels'], config)
            fname = f'fidelity_enhancer-{dataset_name}.ckpt'
            ckpt_fname = os.path.join('saved_models', fname)
            if not os.path.isfile(ckpt_fname):
                assert False, "There's no pretrained model checkpoint for the fidelity enhancer. Run `python stage_fid_enhancer.py` first."
            self.fidelity_enhancer.load_state_dict(torch.load(ckpt_fname))
            freeze(self.fidelity_enhancer)
        else:
            self.fidelity_enhancer = nn.Identity()

    def forward(self, x):
        """
        :param x: (B, C, L)
        """
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch

        mask_pred_loss, (mask_pred_loss_l, mask_pred_loss_h) = self.maskgit(x, y)

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
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
        loss_hist = {'loss': mask_pred_loss,
                     'mask_pred_loss': mask_pred_loss,
                     'mask_pred_loss_l': mask_pred_loss_l,
                     'mask_pred_loss_h': mask_pred_loss_h,
                     }
        for k in loss_hist.keys():
            self.log(f'val/{k}', loss_hist[k])
        
        # maskgit sampling & evaluation
        if batch_idx == 0 and (self.training == False):
            self.maskgit.eval()

            class_index = np.random.choice(np.concatenate(([None], np.unique(y.cpu()))))

            # unconditional sampling
            s_l, s_h = self.maskgit.iterative_decoding(num=1024, device=x.device, class_index=class_index)
            x_new_l = self.maskgit.decode_token_ind_to_timeseries(s_l, 'LF').cpu()
            x_new_h = self.maskgit.decode_token_ind_to_timeseries(s_h, 'HF').cpu()
            x_new = x_new_l + x_new_h

            # plot: generated sample
            fig, axes = plt.subplots(4, 1, figsize=(4, 2*3))
            fig.suptitle(f'Epoch {self.current_epoch}; Class Index: {class_index}')
            axes = axes.flatten()
            b = 0
            axes[0].set_xlabel('xhat_l')
            axes[0].plot(x_new_l[b,0,:])
            axes[1].set_xlabel('xhat_h')
            axes[1].plot(x_new_h[b, 0, :])
            axes[2].set_xlabel('xhat')
            axes[2].plot(x_new[b, 0, :])

            # compute metrics
            x_new = x_new.numpy()
            fid_train_gen, fid_test_gen = self.metrics.fid_score(x_new)
            self.log('metrics/FID', fid_test_gen)

            if self.use_fidelity_enhancer:
                x_new_fe = self.fidelity_enhancer(torch.from_numpy(x_new).to(self.device)).cpu().numpy()
                fid_train_gen_fe, fid_test_gen_fe = self.metrics.fid_score(x_new_fe)
                self.log('metrics/FID_with_FE', fid_test_gen_fe)
                axes[3].set_xlabel('FE(xhat)')
                axes[3].plot(x_new_fe[b,0,:])

            for ax in axes:
                ax.set_ylim(-4, 4)
            plt.tight_layout()
            self.logger.log_image(key='prior_model sample', images=[wandb.Image(plt),])
            plt.close()

        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config['exp_params']['LR'])
        T_max = self.config['trainer_params']['max_steps']['stage2']
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, T_max, eta_min=1e-5)}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.eval()
        x, y = batch

        mask_pred_loss, (mask_pred_loss_l, mask_pred_loss_h) = self.maskgit(x, y)

        # log
        loss_hist = {'loss': mask_pred_loss,
                     'mask_pred_loss': mask_pred_loss,
                     'mask_pred_loss_l': mask_pred_loss_l,
                     'mask_pred_loss_h': mask_pred_loss_h,
                     }
        for k in loss_hist.keys():
            self.log(f'test/{k}', loss_hist[k])

        return loss_hist
