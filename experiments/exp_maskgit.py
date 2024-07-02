import matplotlib.pyplot as plt
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from generators.maskgit import MaskGIT
from evaluation.metrics import Metrics


class ExpMaskGIT(pl.LightningModule):
    def __init__(self,
                 dataset_name: str,
                 input_length: int,
                 config: dict,
                 n_classes: int):
        super().__init__()
        self.config = config
        self.maskgit = MaskGIT(dataset_name, input_length, **config['MaskGIT'], config=config, n_classes=n_classes)
        
        metric_batch_size = 32
        self.metrics = Metrics(dataset_name, metric_batch_size)

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
            s_l, s_h = self.maskgit.iterative_decoding(num=64, device=x.device, class_index=class_index)
            x_new_l = self.maskgit.decode_token_ind_to_timeseries(s_l, 'LF').cpu()
            x_new_h = self.maskgit.decode_token_ind_to_timeseries(s_h, 'HF').cpu()
            x_new = x_new_l + x_new_h

            # plot: generated sample
            fig, axes = plt.subplots(3, 1, figsize=(4, 2*3))
            b = 0
            axes[0].plot(x_new_l[b,0,:])
            axes[1].plot(x_new_h[b, 0, :])
            axes[2].plot(x_new[b, 0, :])
            axes[0].set_ylim(-4, 4)
            axes[1].set_ylim(-4, 4)
            axes[2].set_ylim(-4, 4)
            plt.title(f'ep_{self.current_epoch}; class-{class_index}')
            plt.tight_layout()
            self.logger.log_image(key='prior_model sample', images=[wandb.Image(plt),])
            plt.close()

            # compute metrics
            x_new = x_new.numpy()
            fid_train_gen, fid_test_gen = self.metrics.fid_score(x_new)
            incept_score, _ = self.metrics.inception_score(x_new)
            self.log('metrics/fid_train_gen', fid_train_gen)
            self.log('metrics/fid_test_gen', fid_test_gen)
            self.log('metrics/incept_score', incept_score)

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
