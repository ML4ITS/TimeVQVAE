"""
FID, IS
"""
import os
import copy
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from supervised_FCN.example_pretrained_model_loading import load_pretrained_FCN
from supervised_FCN.example_compute_FID import calculate_fid
from supervised_FCN.example_compute_IS import calculate_inception_score

from preprocessing.preprocess_ucr import DatasetImporterUCR
from utils import freeze


class Metrics(object):
    """
    - FID
    - IS
    - visual inspection
    - PCA
    - t-SNE
    """
    def __init__(self, dataset_name: str, batch_size: int):
        self.dataset_name = dataset_name
        self.batch_size = batch_size

        # load the pretrained FCN
        self.fcn = load_pretrained_FCN(dataset_name)
        freeze(self.fcn)
        self.fcn.eval()

        # load the numpy matrix of the test samples
        dataset_importer = DatasetImporterUCR(dataset_name, data_scaling=True)
        X_train = dataset_importer.X_train[:, None, :]  # (b c l)
        X_test = dataset_importer.X_test[:, None, :]  # (b c l)
        self.n_classes = len(np.unique(dataset_importer.Y_train))

        # compute z_train, z_test
        self.z_train = self.compute_z(X_train)  # (b d)
        self.z_test = self.compute_z(X_test)  # (b d)
        self.z_gen = None

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
            z = self.fcn(torch.from_numpy(x[s]).float().to(device), return_feature_vector=True).cpu().detach().numpy()
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

    def log_visual_inspection(self, n_plot_samples: int, X1, X2, title: str, ylim: tuple = (-5, 5)):
        # `X_test`
        sample_ind = np.random.randint(0, X1.shape[0], n_plot_samples)
        fig, axes = plt.subplots(2, 1, figsize=(4, 4))
        plt.suptitle(title)
        for i in sample_ind:
            axes[0].plot(X1[i, 0, :], alpha=0.1)
        axes[0].set_xticks([])
        axes[0].set_ylim(*ylim)
        # axes[0].set_title('test samples')

        # `X_gen`
        sample_ind = np.random.randint(0, X2.shape[0], n_plot_samples)
        for i in sample_ind:
            axes[1].plot(X2[i, 0, :], alpha=0.1)
        axes[1].set_ylim(*ylim)
        # axes[1].set_title('generated samples')

        plt.tight_layout()
        wandb.log({f"visual comp ({title})": wandb.Image(plt)})
        plt.close()

    # def log_pca(self, n_plot_samples: int, X_gen, z_test: np.ndarray, z_gen: np.ndarray):
    def log_pca(self, n_plot_samples: int, Z1: np.ndarray, Z2: np.ndarray, labels):

        # X_gen = F.interpolate(X_gen, size=self.X_test.shape[-1], mode='linear', align_corners=True)
        # X_gen = X_gen.cpu().numpy()

        # sample_ind_test = np.random.choice(range(self.X_test.shape[0]), size=n_plot_samples, replace=True)
        ind1 = np.random.choice(range(Z1.shape[0]), size=n_plot_samples, replace=True)
        ind2 = np.random.choice(range(Z2.shape[0]), size=n_plot_samples, replace=True)

        # # PCA: data space
        # pca = PCA(n_components=2)
        # X_embedded_test = pca.fit_transform(self.X_test.squeeze()[sample_ind_test])
        # X_embedded_gen = pca.transform(X_gen.squeeze()[sample_ind_gen])
        #
        # plt.figure(figsize=(4, 4))
        # # plt.title("PCA in the data space")
        # plt.scatter(X_embedded_test[:, 0], X_embedded_test[:, 1], alpha=0.1, label='test')
        # plt.scatter(X_embedded_gen[:, 0], X_embedded_gen[:, 1], alpha=0.1, label='gen')
        # plt.legend()
        # plt.tight_layout()
        # wandb.log({"PCA-data_space": wandb.Image(plt)})
        # plt.close()

        # PCA: latent space
        pca = PCA(n_components=2)
        Z1_embed = pca.fit_transform(Z1[ind1])
        Z2_embed = pca.transform(Z2[ind2])

        plt.figure(figsize=(4, 4))
        # plt.title("PCA in the representation space by the trained encoder");
        plt.scatter(Z1_embed[:, 0], Z1_embed[:, 1], alpha=0.1, label=labels[0])
        plt.scatter(Z2_embed[:, 0], Z2_embed[:, 1], alpha=0.1, label=labels[1])
        plt.legend()
        plt.tight_layout()
        wandb.log({f"PCA on Z ({labels[0]} vs  {labels[1]})": wandb.Image(plt)})
        plt.close()

    def log_visual_inspection_train_test(self, n_plot_samples: int, ylim: tuple = (-5, 5)):
        # `X_train`
        sample_ind = np.random.randint(0, self.X_train.shape[0], n_plot_samples)
        fig, axes = plt.subplots(2, 1, figsize=(4, 4))
        for i in sample_ind:
            axes[0].plot(self.X_train[i, 0, :], alpha=0.1)
        axes[0].set_xticks([])
        axes[0].set_ylim(*ylim)
        axes[0].set_title('train samples')

        # `X_test`
        sample_ind = np.random.randint(0, self.X_test.shape[0], n_plot_samples)
        for i in sample_ind:
            axes[1].plot(self.X_test[i, 0, :], alpha=0.1)
        axes[1].set_ylim(*ylim)
        axes[1].set_title('test samples')

        plt.tight_layout()
        wandb.log({"visual inspection (X_train vs X_test)": wandb.Image(plt)})
        plt.close()

    def log_pca_ztrain_ztest(self, n_plot_samples: int, z1: np.ndarray, z2: np.ndarray):
        sample_ind1 = np.random.choice(range(z1.shape[0]), size=n_plot_samples, replace=True)
        sample_ind2 = np.random.choice(range(z2.shape[0]), size=n_plot_samples, replace=True)

        # PCA: latent space
        pca = PCA(n_components=2)
        z_embedded1 = pca.fit_transform(z1[sample_ind1].squeeze())
        z_embedded2 = pca.transform(z2[sample_ind2].squeeze())

        plt.figure(figsize=(4, 4))
        # plt.title("PCA in the representation space by the trained encoder");
        plt.scatter(z_embedded1[:, 0], z_embedded1[:, 1], alpha=0.1, label='train')
        plt.scatter(z_embedded2[:, 0], z_embedded2[:, 1], alpha=0.1, label='test')
        plt.legend()
        plt.tight_layout()
        wandb.log({"PCA-latent_space (z_train vs z_test)": wandb.Image(plt)})
        plt.close()

    def log_tsne(self, n_plot_samples: int, X_gen, z_test: np.ndarray, z_gen: np.ndarray):
        X_gen = F.interpolate(X_gen, size=self.X_test.shape[-1], mode='linear', align_corners=True)
        X_gen = X_gen.cpu().numpy()

        sample_ind_test = np.random.randint(0, self.X_test.shape[0], n_plot_samples)
        sample_ind_gen = np.random.randint(0, X_gen.shape[0], n_plot_samples)

        # TNSE: data space
        X = np.concatenate((self.X_test.squeeze()[sample_ind_test], X_gen.squeeze()[sample_ind_gen]), axis=0).squeeze()
        labels = np.array(['C0'] * len(sample_ind_test) + ['C1'] * len(sample_ind_gen))
        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X)

        plt.figure(figsize=(4, 4))
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, alpha=0.1)
        # plt.legend()
        plt.tight_layout()
        wandb.log({"TNSE-data_space": wandb.Image(plt)})
        plt.close()

        # TNSE: latent space
        Z = np.concatenate((z_test[sample_ind_test], z_gen[sample_ind_gen]), axis=0).squeeze()
        labels = np.array(['C0'] * len(sample_ind_test) + ['C1'] * len(sample_ind_gen))
        Z_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(Z)

        plt.figure(figsize=(4, 4))
        plt.scatter(Z_embedded[:, 0], Z_embedded[:, 1], c=labels, alpha=0.1)
        # plt.legend()
        plt.tight_layout()
        wandb.log({"TSNE-latent_space": wandb.Image(plt)})
        plt.close()
