from collections import Counter

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from supervised_FCN_2.experiments.exp_train import ExpFCN as ExpFCN_original
from supervised_FCN_2.models.fcn import ConvBlock
from torch.optim.lr_scheduler import CosineAnnealingLR

from evaluation.evaluation import Evaluation


class SmallFCN(nn.Module):
    """A PyTorch implementation of the FCN Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, 
                 in_channels:int, 
                 num_pred_classes:int=1,
                 dropout:float=0.5) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels, 16, 8, 1), nn.Dropout(dropout),
            ConvBlock(16, 32, 5, 1), nn.Dropout(dropout),
            ConvBlock(32, 32, 5, 1), nn.Dropout(dropout),
            ConvBlock(32, 16, 5, 1)
        ])
        self.final = nn.Linear(16, num_pred_classes)

    def forward(self, x: torch.Tensor, return_feature_vector: bool = False) -> torch.Tensor:  # type: ignore
        out = self.layers(x)

        if return_feature_vector:
            return out.mean(dim=-1)
        else:
            return self.final(out.mean(dim=-1))
        

class ExpFCN(ExpFCN_original):
    def __init__(self,
                 config_cas: dict,
                 n_classes: int,
                 ):
        super().__init__(config_cas, None, n_classes)
        self.config = config_cas
        self.fcn = SmallFCN(config_cas['dataset']['in_channels'], n_classes)

    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params': self.parameters(), 'lr': self.config['exp_params']['lr']}], )
        T_max = self.config['trainer_params']['max_steps']
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, T_max, eta_min=0.000001)}


class CASDataset(Dataset):
    def __init__(self,
                 real_train_data_loader: DataLoader,
                 min_n_synthetic_train_samples: int,
                 dataset_name:str,
                 config: dict,
                 device,
                 use_neural_mapper:bool,
                 ):
        super().__init__()
        self.use_neural_mapper = use_neural_mapper
        n_train_samples_per_class = dict(Counter(real_train_data_loader.dataset.Y.flatten()))
        n_classes = len(np.unique(real_train_data_loader.dataset.Y))
        input_length = real_train_data_loader.dataset.X.shape[1]

        # increase `n_train_samples_per_class` if the dataset that is too small.
        n_train_samples = np.sum(list(n_train_samples_per_class.values()))
        if n_train_samples < min_n_synthetic_train_samples:
            mul = min_n_synthetic_train_samples / n_train_samples
            n_train_samples_per_class = {k: round(v * mul) for k, v in n_train_samples_per_class.items()}
        n_train_samples_per_class = dict(sorted(n_train_samples_per_class.items()))  # sort
        print('n_train_samples_per_class:', n_train_samples_per_class)
        
        # sample synthetic dataset
        evaluation = Evaluation(dataset_name, input_length, n_classes, device, config, use_neural_mapper).to(device)
        self.Xhat, self.Xhat_R, self.Y = [], [], []
        for cls_idx, n_samples in n_train_samples_per_class.items():
            print(f'sampling synthetic data | cls_idx: {cls_idx}...')
            (_, _, xhat_c), xhat_c_R = evaluation.sample(n_samples, kind='conditional', class_index=cls_idx)
            self.Xhat.append(xhat_c)
            self.Xhat_R.append(xhat_c_R)
            self.Y.append(torch.Tensor([cls_idx] * n_samples))
        self.Xhat = torch.cat(self.Xhat).float()  # (b 1 l)
        self.Xhat_R = torch.cat(self.Xhat_R).float()  # (b 1 l)
        self.Y = torch.cat(self.Y)[:, None].long()  # (b 1)
    
    def __getitem__(self, idx):
        xhat = self.Xhat_R[idx] if self.use_neural_mapper else self.Xhat[idx]
        y = self.Y[idx]
        return xhat, y

    def __len__(self):
        return self.Xhat.shape[0]
