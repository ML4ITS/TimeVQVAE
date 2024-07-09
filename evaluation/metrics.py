"""
FID, IS
"""
import os
import copy
import tempfile
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from supervised_FCN_2.example_pretrained_model_loading import load_pretrained_FCN
from supervised_FCN_2.example_compute_FID import calculate_fid
from supervised_FCN_2.example_compute_IS import calculate_inception_score

from evaluation.rocket_functions import generate_kernels, apply_kernels
from preprocessing.preprocess_ucr import DatasetImporterUCR
from utils import freeze
from evaluation.stat_metrics import marginal_distribution_difference, auto_correlation_difference, skewness_difference, kurtosis_difference


class Metrics(nn.Module):
    """
    - FID
    - IS
    - visual inspection
    - PCA
    - t-SNE
    """
    def __init__(self, 
                 dataset_name: str, 
                 feature_extractor_type:str, 
                 rocket_num_kernels:int=1000,
                 batch_size: int=32,
                 ):
        super().__init__()
        self.dataset_name = dataset_name
        self.feature_extractor_type = feature_extractor_type
        self.batch_size = batch_size

        # load the pretrained FCN
        self.fcn = load_pretrained_FCN(dataset_name)
        freeze(self.fcn)
        self.fcn.eval()

        # load the numpy matrix of the test samples
        dataset_importer = DatasetImporterUCR(dataset_name, data_scaling=True)
        self.X_train = dataset_importer.X_train[:, None, :]  # (b c l)
        self.X_test = dataset_importer.X_test[:, None, :]  # (b c l)
        self.n_classes = len(np.unique(dataset_importer.Y_train))
        
        # load rocket (to extract unbiased representations)
        input_length = self.X_train.shape[-1]
        self.rocket_kernels = generate_kernels(input_length, num_kernels=rocket_num_kernels)

        # compute z_train, z_test
        self.z_train = self.compute_z(self.X_train)  # (b d)
        self.z_test = self.compute_z(self.X_test)  # (b d)

    def _extract_feature_representations(self, x:np.ndarray):
        """
        x: (b 1 l)
        """
        if self.feature_extractor_type == 'supervised_fcn':
            z = self.fcn(torch.from_numpy(x).float().to(self.device), return_feature_vector=True).cpu().detach().numpy()  # (b d)
        elif self.feature_extractor_type == 'rocket':
            x = x[:,0,:].astype(float)  # (b l)
            z = apply_kernels(x, self.rocket_kernels)
        else:
            raise ValueError
        return z
    
    def compute_z(self, x: np.ndarray) -> np.ndarray:
        device = next(self.fcn.parameters()).device

        n_samples = x.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get feature vectors
        zs = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            z = self._extract_feature_representations(x[s])
            zs.append(z)
        zs = np.concatenate(zs, axis=0)
        return zs
    
    def z_gen_fn(self, x_gen: np.ndarray):
        z_gen = self.compute_z(x_gen)
        return z_gen

    def fid_score(self, x_gen: np.ndarray):
        z_gen = self.z_gen_fn(x_gen)
        fid_train_gen = calculate_fid(self.z_train, z_gen)
        fid_test_gen = calculate_fid(self.z_test, z_gen)
        return fid_train_gen, fid_test_gen
    
    def inception_score(self, x_gen: np.ndarray):
        device = next(self.fcn.parameters()).device
        n_samples = x_gen.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get the softmax distribution from `x_gen`
        p_yx_gen = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)

            p_yx_g = self.fcn(torch.from_numpy(x_gen[s]).float().to(device))  # p(y|x)
            p_yx_g = torch.softmax(p_yx_g, dim=-1).cpu().detach().numpy()

            p_yx_gen.append(p_yx_g)
        p_yx_gen = np.concatenate(p_yx_gen, axis=0)

        IS_mean, IS_std = calculate_inception_score(p_yx_gen, n_split=5)
        return IS_mean, IS_std

    def stat_metrics(self, x_real:np.ndarray, x_gen:np.ndarray) -> Tuple[float, float, float, float]:
        """
        computes the statistical metrices introduced in the paper, [Ang, Yihao, et al. "Tsgbench: Time series generation benchmark." arXiv preprint arXiv:2309.03755 (2023).]

        x_real: (batch 1 length)
        x_gen: (batch 1 length)
        """
        mdd = marginal_distribution_difference(x_real, x_gen)
        acd = auto_correlation_difference(x_real, x_gen)
        sd = skewness_difference(x_real, x_gen)
        kd = kurtosis_difference(x_real, x_gen)
        return mdd, acd, sd, kd