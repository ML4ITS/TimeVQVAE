import matplotlib.pyplot as plt
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F

from experiments.exp_base import ExpBase, detach_the_unnecessary
from generators.maskgit import MaskGIT
from evaluation.metrics import Metrics


class ExpMaskGIT(ExpBase):
    def __init__(self,
                 dataset_name: str,
                 input_length: int,
                 config: dict,
                 n_train_samples: int,
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

        logits, target = self.maskgit(x, y)
        logits_l, logits_h = logits
        target_l, target_h = target
        prior_loss_l = F.cross_entropy(logits_l.reshape(-1, logits_l.size(-1)), target_l.reshape(-1))
        prior_loss_h = F.cross_entropy(logits_h.reshape(-1, logits_h.size(-1)), target_h.reshape(-1))
        prior_loss = (prior_loss_l + prior_loss_h)/2
        loss = (prior_loss_l + prior_loss_h)/2

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {'loss': loss,
                     'prior_loss': prior_loss,
                     }
        detach_the_unnecessary(loss_hist)
        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        x, y = batch

        logits, target = self.maskgit(x, y)
        logits_l, logits_h = logits
        target_l, target_h = target
        prior_loss_l = F.cross_entropy(logits_l.reshape(-1, logits_l.size(-1)), target_l.reshape(-1))
        prior_loss_h = F.cross_entropy(logits_h.reshape(-1, logits_h.size(-1)), target_h.reshape(-1))
        prior_loss = (prior_loss_l + prior_loss_h)/2
        loss = (prior_loss_l + prior_loss_h)/2

        # log
        loss_hist = {'loss': loss,
                     'prior_loss': prior_loss,
                     }
        
        # maskgit sampling
        if batch_idx == 0 and (self.training == False):
            self.maskgit.eval()

            class_index = np.random.choice(np.concatenate(([None], np.unique(y.cpu()))))

            # unconditional sampling
            n_samples = 1024 #128
            s_l, s_h = self.maskgit.iterative_decoding(num=n_samples, device=x.device, class_index=class_index)
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
            wandb.log({f"maskgit sample": wandb.Image(plt)})
            plt.close()

            # plot generated samples
            n_plot_cols = min(int(np.ceil(np.sqrt(x_new.shape[0]))), 8)
            fig, axes = plt.subplots(n_plot_cols, n_plot_cols, figsize=(3*n_plot_cols, 2*n_plot_cols))  #plt.subplots(1, 2, figsize=(16, 8))
            axes = axes.flatten()
            fig.suptitle(f'x_new | ep-{self.current_epoch}')
            for i, ax in enumerate(axes):
                c = 0
                ax.plot(x_new[i,c])
            plt.tight_layout()
            self.logger.log_image(key='prior_model sample', images=[wandb.Image(plt),])
            plt.close()

            # compute metrics
            x_new = x_new.numpy()
            fid_train_gen, fid_test_gen = self.metrics.fid_score(x_new)
            incept_score, _ = self.metrics.inception_score(x_new)
            wandb.log({'metrics/fid_train_gen': fid_train_gen,
                       'metrics/fid_test_gen': fid_test_gen,
                       'metrics/incept_score': incept_score,
                       'epoch': self.current_epoch})

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def configure_optimizers(self):
        T_max = self.config['trainer_params']['max_steps']['stage2']
        opt = torch.optim.AdamW([{'params': self.maskgit.parameters(), 'lr': self.config['exp_params']['LR']},
                                 ],
                                weight_decay=self.config['exp_params']['weight_decay'])
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, T_max, eta_min=1e-5)}

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits, target = self.maskgit(x, y)
        logits_l, logits_h = logits
        target_l, target_h = target
        prior_loss_l = F.cross_entropy(logits_l.reshape(-1, logits_l.size(-1)), target_l.reshape(-1))
        prior_loss_h = F.cross_entropy(logits_h.reshape(-1, logits_h.size(-1)), target_h.reshape(-1))
        prior_loss = (prior_loss_l + prior_loss_h)/2
        loss = (prior_loss_l + prior_loss_h)/2

        # log
        loss_hist = {'loss': loss,
                     'prior_loss': prior_loss,
                     }

        detach_the_unnecessary(loss_hist)
        return loss_hist
