"""
FID, IS, JS divergence.
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

from experiments.exp_maskgit import ExpMaskGIT
from generators.maskgit import MaskGIT
from preprocessing.data_pipeline import build_data_pipeline
from preprocessing.preprocess_ucr import DatasetImporterUCR
from generators.sample import unconditional_sample, conditional_sample
from supervised_FCN.example_pretrained_model_loading import load_pretrained_FCN
from supervised_FCN.example_compute_FID import calculate_fid
from supervised_FCN.example_compute_IS import calculate_inception_score
from utils import time_to_timefreq, timefreq_to_time
from generators.fidelity_enhancer import FidelityEnhancer


class Evaluation(nn.Module):
    """
    - FID
    - IS
    - visual inspection
    - PCA
    - t-SNE
    """
    def __init__(self, subset_dataset_name: str, input_length:int, n_classes:int, gpu_device_index: int, config: dict):
        super().__init__()
        self.subset_dataset_name = dataset_name = subset_dataset_name
        self.device = torch.device(gpu_device_index)
        self.config = config
        self.batch_size = self.config['evaluation']['batch_size']

        # load the pretrained FCN
        self.fcn = load_pretrained_FCN(subset_dataset_name)
        self.fcn.eval()

        # load the numpy matrix of the test samples
        dataset_importer = DatasetImporterUCR(subset_dataset_name, data_scaling=True)
        self.X_test = dataset_importer.X_test[:, None, :]
        self.X_train = dataset_importer.X_train[:, None, :]
        self.ts_len = self.X_train.shape[-1]  # time series length
        self.n_classes = len(np.unique(dataset_importer.Y_train))

        # # load maskgit
        # self.maskgit = MaskGIT(self.subset_dataset_name, self.ts_len, **self.config['MaskGIT'], config=self.config, n_classes=self.n_classes).to(self.device)
        # dataset_name = self.subset_dataset_name
        # fname = f'maskgit-{dataset_name}.ckpt'
        # try:
        #     ckpt_fname = os.path.join('saved_models', fname)
        #     self.maskgit.load_state_dict(torch.load(ckpt_fname), strict=False)
        # except FileNotFoundError:
        #     ckpt_fname = Path(tempfile.gettempdir()).joinpath(fname)
        #     self.maskgit.load_state_dict(torch.load(ckpt_fname), strict=False)
        # self.maskgit.eval()

        # load the stage2 model
        self.stage2 = ExpMaskGIT.load_from_checkpoint(os.path.join('saved_models', f'stage2-{dataset_name}.ckpt'), 
                                                      dataset_name=dataset_name, 
                                                      input_length=input_length, 
                                                      n_classes=n_classes,
                                                      config=config,
                                                      map_location='cpu')
        self.stage2.eval()
        self.maskgit = self.stage2.maskgit

        # load domain shifter
        if self.config['evaluation']['use_fidelity_enhancer']:
            self.fidelity_enhancer = FidelityEnhancer(self.ts_len, 1, config)
            fname = f'fidelity_enhancer-{dataset_name}.ckpt'
            ckpt_fname = os.path.join('saved_models', fname)
            self.fidelity_enhancer.load_state_dict(torch.load(ckpt_fname))
        else:
            self.fidelity_enhancer = nn.Identity()

    def sample(self, n_samples: int, kind: str, class_index: int = -1):
        assert kind in ['unconditional', 'conditional']

        # sampling
        if kind == 'unconditional':
            x_new_l, x_new_h, x_new = unconditional_sample(self.maskgit, n_samples, self.device, batch_size=self.batch_size)  # (b c l); b=n_samples, c=1 (univariate)
        elif kind == 'conditional':
            x_new_l, x_new_h, x_new = conditional_sample(self.maskgit, n_samples, self.device, class_index, self.batch_size)  # (b c l); b=n_samples, c=1 (univariate)
        else:
            raise ValueError

        # domain shifter
        with torch.no_grad():
            x_new = self.fidelity_enhancer(x_new)

        return x_new_l, x_new_h, x_new

    def compute_z_train_test(self, kind: str):
        assert kind in ['train', 'test']
        if kind == 'train':
            self.X = self.X_train
        elif kind == 'test':
            self.X = self.X_test

        n_samples = self.X.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get feature vectors
        zs = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            z = self.fcn(torch.from_numpy(self.X[s]).float().to(self.device), return_feature_vector=True).cpu().detach().numpy()
            zs.append(z)
        zs = np.concatenate(zs, axis=0)
        return zs

    def compute_z(self, kind: str) -> (np.ndarray, np.ndarray):
        """
        It computes representation z given input x
        :param X_gen: generated X
        :return: z_test (z on X_test), z_gen (z on X_generated)
        """
        assert kind in ['train', 'test']
        if kind == 'train':
            X = self.X_train
        elif kind == 'test':
            X = self.X_test
        else:
            raise ValueError

        n_samples = X.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get feature vectors from `X_test`
        z_test = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            z_t = self.fcn(torch.from_numpy(X[s]).float().to(self.device), return_feature_vector=True).cpu().detach().numpy()
            z_test.append(z_t)
        z_test = np.concatenate(z_test, axis=0)
        return z_test

    def compute_z_gen(self, X_gen: torch.Tensor) -> (np.ndarray, np.ndarray):
        """
        It computes representation z given input x
        :param X_gen: generated X
        :return: z_test (z on X_test), z_gen (z on X_generated)
        """
        n_samples = X_gen.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get feature vectors from `X_gen`
        z_gen = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)

            z_g = self.fcn(X_gen[s].float().to(self.device), return_feature_vector=True).cpu().detach().numpy()

            z_gen.append(z_g)
        z_gen = np.concatenate(z_gen, axis=0)
        return z_gen

    def fid_score(self, z_test: np.ndarray, z_gen: np.ndarray) -> (int, (np.ndarray, np.ndarray)):
        fid = calculate_fid(z_test, z_gen)
        return fid

    def inception_score(self, X_gen: torch.Tensor):
        # assert self.X_test.shape[0] == X_gen.shape[0], "shape of `X_test` must be the same as that of `X_gen`."

        n_samples = self.X_test.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get the softmax distribution from `X_gen`
        p_yx_gen = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)

            p_yx_g = self.fcn(X_gen[s].float().to(self.device))  # p(y|x)
            p_yx_g = torch.softmax(p_yx_g, dim=-1).cpu().detach().numpy()

            p_yx_gen.append(p_yx_g)
        p_yx_gen = np.concatenate(p_yx_gen, axis=0)

        IS_mean, IS_std = calculate_inception_score(p_yx_gen)
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
