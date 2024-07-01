from collections import Counter

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from supervised_FCN_2.experiments.exp_train import ExpFCN as ExpFCN_original

from generators.sample import Sampler


class ExpFCN(ExpFCN_original):
    def __init__(self,
                 config_fcn: dict,
                 n_train_samples: int,
                 n_classes: int,
                 ):
        super().__init__(config_fcn, n_train_samples, n_classes)
        self.config = config_fcn
        self.T_max = config_fcn['trainer_params']['max_epochs'] * (np.ceil(n_train_samples / config_fcn['dataset']['batch_size']) + 1)


class SyntheticDataset(Dataset):
    def __init__(self,
                 real_train_data_loader: DataLoader,
                 min_n_synthetic_train_samples: int,
                 config: dict,
                 batch_size: int,
                 device):
        super().__init__()
        n_train_samples_per_class = dict(Counter(real_train_data_loader.dataset.Y.flatten()))

        # enlarge the dataset size that is too small.
        n_train_samples = np.sum(list(n_train_samples_per_class.values()))
        if n_train_samples < min_n_synthetic_train_samples:
            mul = min_n_synthetic_train_samples / n_train_samples
            n_train_samples_per_class = {k: round(v * mul) for k, v in n_train_samples_per_class.items()}
        print('n_train_samples_per_class:', n_train_samples_per_class)

        # sample synthetic dataset
        sampler = Sampler(real_train_data_loader, config, device)
        self.X_gen, self.Y_gen = [], []
        for class_idx, n_samples in n_train_samples_per_class.items():
            print(f'sampling synthetic data | class_idx: {class_idx}...')
            x_gen = sampler.sample('conditional', n_samples, class_idx, batch_size=batch_size)
            self.X_gen.append(x_gen)
            self.Y_gen.append(torch.Tensor([class_idx] * n_samples))
        self.X_gen = torch.cat(self.X_gen).float()  # (b c l)
        self.Y_gen = torch.cat(self.Y_gen)[:, None].long()  # (b 1)
        print('self.X_gen.shape:', self.X_gen.shape)
        print('self.Y_gen.shape:', self.Y_gen.shape)

        self._len = self.X_gen.shape[0]

    def __getitem__(self, idx):
        x_gen = self.X_gen[idx]
        y_gen = self.Y_gen[idx]
        return x_gen, y_gen

    def __len__(self):
        return self._len