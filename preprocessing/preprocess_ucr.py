"""
`Dataset` (pytorch) class is defined.
"""
from typing import Union
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from utils import get_root_dir, download_ucr_datasets


class DatasetImporterUCR(object):
    """
    This uses train and test sets as given.
    To compare with the results from ["Unsupervised scalable representation learning for multivariate time series"]
    """
    def __init__(self, dataset_name: str, data_scaling: bool, **kwargs):
        """
        :param dataset_name: e.g., "ElectricDevices"
        :param data_scaling
        """
        download_ucr_datasets()
        self.data_root = get_root_dir().joinpath("datasets", "UCRArchive_2018", dataset_name)

        # fetch an entire dataset
        df_train = pd.read_csv(self.data_root.joinpath(f"{dataset_name}_TRAIN.tsv"), sep='\t', header=None)
        df_test = pd.read_csv(self.data_root.joinpath(f"{dataset_name}_TEST.tsv"), sep='\t', header=None)

        self.X_train, self.X_test = df_train.iloc[:, 1:].values, df_test.iloc[:, 1:].values
        self.Y_train, self.Y_test = df_train.iloc[:, [0]].values, df_test.iloc[:, [0]].values

        # merge train and test sets and split it into 80-20 for training and testing.
        X = np.concatenate((df_train.iloc[:, 1:].values, df_test.iloc[:, 1:].values), axis=0)
        Y = np.concatenate((df_train.iloc[:, [0]].values, df_test.iloc[:, [0]].values), axis=0)
        # self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)  # Create a StratifiedShuffleSplit object
        for train_index, test_index in sss.split(X, Y):
            self.X_train, self.X_test = X[train_index], X[test_index]
            self.Y_train, self.Y_test = Y[train_index], Y[test_index]

        le = LabelEncoder()
        self.Y_train = le.fit_transform(self.Y_train.ravel())[:, None]
        self.Y_test = le.transform(self.Y_test.ravel())[:, None]

        if data_scaling:
            # following [https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/blob/dcc674541a94ca8a54fbb5503bb75a297a5231cb/ucr.py#L30]
            mean = np.nanmean(self.X_train)
            var = np.nanvar(self.X_train)
            self.X_train = (self.X_train - mean) / math.sqrt(var)
            self.X_test = (self.X_test - mean) / math.sqrt(var)

        np.nan_to_num(self.X_train, copy=False)
        np.nan_to_num(self.X_test, copy=False)

        print('self.X_train.shape:', self.X_train.shape)
        print('self.X_test.shape:', self.X_test.shape)

        print("# unique labels (train):", np.unique(self.Y_train.reshape(-1)))
        print("# unique labels (test):", np.unique(self.Y_test.reshape(-1)))


class UCRDataset(Dataset):
    def __init__(self,
                 kind: str,
                 dataset_importer: DatasetImporterUCR,
                 **kwargs):
        """
        :param kind: "train" / "test"
        :param dataset_importer: instance of the `DatasetImporter` class.
        """
        super().__init__()
        self.kind = kind

        if kind == "train":
            self.X, self.Y = dataset_importer.X_train, dataset_importer.Y_train
        elif kind == "test":
            self.X, self.Y = dataset_importer.X_test, dataset_importer.Y_test
        else:
            raise ValueError

        self._len = self.X.shape[0]

    @staticmethod
    def _assign_float32(*xs):
        """
        assigns `dtype` of `float32`
        so that we wouldn't have to change `dtype` later before propagating data through a model.
        """
        new_xs = []
        for x in xs:
            new_xs.append(x.astype(np.float32))
        return new_xs[0] if (len(xs) == 1) else new_xs

    def getitem_default(self, idx):
        x, y = self.X[idx, :], self.Y[idx, :]
        x = x[None, :]  # adds a channel dim
        return x, y

    def __getitem__(self, idx):
        return self.getitem_default(idx)

    def __len__(self):
        return self._len


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader
    os.chdir("../")

    # data pipeline
    dataset_importer = DatasetImporterUCR("ScreenType", data_scaling=True)
    dataset = UCRDataset("train", dataset_importer)
    data_loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=True)

    # get a mini-batch of samples
    for batch in data_loader:
        x, y = batch
        break
    print('x.shape:', x.shape)
    print('y.shape:', y.shape)
    print(y.flatten())

    # plot
    n_samples = 10
    c = 0
    fig, axes = plt.subplots(n_samples, 2, figsize=(3.5*2, 1.7*n_samples))
    for i in range(n_samples):
        axes[i,0].plot(x[i, c])
        axes[i,0].set_title(f'class: {y[i,0]}')
        xf = torch.stft(x[[i], c], n_fft=4, hop_length=1, normalized=False)
        print('xf.shape:', xf.shape)
        xf = np.sqrt(xf[0,:,:,0]**2 + xf[0,:,:,1]**2)
        axes[i,1].imshow(xf, aspect='auto')
        axes[i, 1].invert_yaxis()
    plt.tight_layout()
    plt.show()
