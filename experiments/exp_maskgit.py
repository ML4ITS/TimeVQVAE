import matplotlib.pyplot as plt
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F

from experiments.exp_base import ExpBase, detach_the_unnecessary
from generators.maskgit import MaskGIT


class ExpMaskGIT(ExpBase):
    def __init__(self,
                 input_length: int,
                 config: dict,
                 n_train_samples: int,
                 n_classes: int):
        super().__init__()
        self.config = config
        self.maskgit = MaskGIT(input_length, **config['MaskGIT'], config=config, n_classes=n_classes)
        self.T_max = config['trainer_params']['max_epochs']['stage2'] * (np.ceil(n_train_samples / config['dataset']['batch_sizes']['stage2']) + 1)

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

        # maskgit sampling
        r = np.random.rand()
        if batch_idx == 0 and r <= 0.05:
            self.maskgit.eval()

            class_index = np.random.choice(np.concatenate(([None], np.unique(y.cpu()))))

            # unconditional sampling
            s_l, s_h = self.maskgit.iterative_decoding(device=x.device, class_index=class_index)
            x_new_l = self.maskgit.decode_token_ind_to_timeseries(s_l, 'LF').cpu()
            x_new_h = self.maskgit.decode_token_ind_to_timeseries(s_h, 'HF').cpu()
            x_new = x_new_l + x_new_h

            b = 0
            fig, axes = plt.subplots(3, 1, figsize=(4, 2*3))
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

        detach_the_unnecessary(loss_hist)
        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
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

    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params': self.maskgit.parameters(), 'lr': self.config['exp_params']['LR']},
                                 ],
                                weight_decay=self.config['exp_params']['weight_decay'])
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max)}

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
