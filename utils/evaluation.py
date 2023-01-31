"""
FID, IS, JS divergence.
"""
import os
import copy
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from generators.maskgit import MaskGIT
from preprocessing.data_pipeline import build_data_pipeline
from preprocessing.preprocess_ucr import DatasetImporterUCR
from generators.sample import unconditional_sample, conditional_sample
from supervised_FCN.example_pretrained_model_loading import load_pretrained_FCN
from supervised_FCN.example_compute_FID import calculate_fid
from supervised_FCN.example_compute_IS import calculate_inception_score
from utils import time_to_timefreq, timefreq_to_time


def KL_div(p_probs, q_probs, eps=1e-5):
    KL_div = p_probs * torch.log((p_probs + eps) / (q_probs + eps))
    return KL_div.sum()


def histogram_torch(x, n_bins, density=True):
    """
    x: flatten x (1-dimensional time series)
    """
    a, b = x.min().item(), x.max().item()
    delta = (b - a) / n_bins
    count = torch.histc(x, n_bins, min=a, max=b).float()
    if density:
        count = count / float(x.shape[0])  #count / delta / float(x.shape[0])
    return count


def JS_Div(x1: torch.Tensor, x2: torch.Tensor, bins=100):
    """
    JS divergence has
    - lower boundary of zero and
    - upper boundary of inf.
    """
    x1 = copy.deepcopy(x1)
    x2 = copy.deepcopy(x2)

    p = histogram_torch(x1, bins, density=True)
    q = histogram_torch(x2, bins, density=True)
    assert np.abs(p.sum() - 1.0) <= 0.1  # check if an integral of `p` sums up to 1.0.
    assert np.abs(q.sum() - 1.0) <= 0.1

    m = (p + q) / 2
    return (KL_div(p, m) + KL_div(q, m)) / 2


class Evaluation(object):
    def __init__(self, subset_dataset_name: str, gpu_device_index: int, config: dict, batch_size: int = 256):
        self.device = torch.device(gpu_device_index)
        self.batch_size = batch_size
        self.config = config

        # load the pretrained FCN
        self.fcn = load_pretrained_FCN(subset_dataset_name).to(self.device)
        self.fcn.eval()

        # load the numpy matrix of the test samples
        dataset_importer = DatasetImporterUCR(subset_dataset_name, data_scaling=True)
        self.X_test = dataset_importer.X_test[:, None, :]
        n_fft = self.config['VQ-VAE']['n_fft']
        self.X_test = timefreq_to_time(time_to_timefreq(torch.from_numpy(self.X_test), n_fft, 1), n_fft, 1)
        self.X_test = self.X_test.numpy()

    def sample(self, n_samples: int, input_length: int, n_classes: int, kind: str, class_index: int = -1):
        assert kind in ['unconditional', 'conditional']

        # build
        maskgit = MaskGIT(input_length, **self.config['MaskGIT'], config=self.config, n_classes=n_classes).to(self.device)

        # load
        subset_name = self.config['dataset']['subset_name']
        fname = f'maskgit-{subset_name}.ckpt'
        try:
            ckpt_fname = os.path.join('saved_models', fname)
            maskgit.load_state_dict(torch.load(ckpt_fname))
        except FileNotFoundError:
            ckpt_fname = Path(tempfile.gettempdir()).joinpath(fname)
            maskgit.load_state_dict(torch.load(ckpt_fname))

        # inference mode
        maskgit.eval()

        # sampling
        if kind == 'unconditional':
            x_new_l, x_new_h, x_new = unconditional_sample(maskgit, n_samples, self.device, batch_size=self.batch_size)  # (b c l); b=n_samples, c=1 (univariate)
        elif kind == 'conditional':
            x_new_l, x_new_h, x_new = conditional_sample(maskgit, n_samples, self.device, class_index, self.batch_size)  # (b c l); b=n_samples, c=1 (univariate)
        else:
            raise ValueError

        return x_new_l, x_new_h, x_new

    def compute_z(self, X_gen: torch.Tensor) -> (np.ndarray, np.ndarray):
        """
        It computes representation z given input x
        :param X_gen: generated X
        :return: z_test (z on X_test), z_gen (z on X_generated)
        """
        n_samples = self.X_test.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get feature vectors from `X_test` and `X_gen`
        z_test, z_gen = [], []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)

            z_t = self.fcn(torch.from_numpy(self.X_test[s]).float().to(self.device), return_feature_vector=True).cpu().detach().numpy()
            z_g = self.fcn(X_gen[s].float().to(self.device), return_feature_vector=True).cpu().detach().numpy()

            z_test.append(z_t)
            z_gen.append(z_g)
        z_test, z_gen = np.concatenate(z_test, axis=0), np.concatenate(z_gen, axis=0)
        return z_test, z_gen

    def fid_score(self, z_test: np.ndarray, z_gen: np.ndarray) -> (int, (np.ndarray, np.ndarray)):
        fid = calculate_fid(z_test, z_gen)
        return fid, (z_test, z_gen)

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

    def log_visual_inspection(self, n_plot_samples: int, X_gen, ylim: tuple = (-5, 5)):
        # `X_test`
        sample_ind = np.random.randint(0, self.X_test.shape[0], n_plot_samples)
        fig, axes = plt.subplots(2, 1, figsize=(4, 4))
        for i in sample_ind:
            axes[0].plot(self.X_test[i, 0, :], alpha=0.1)
        axes[0].set_ylim(*ylim)
        plt.grid()

        # `X_gen`
        sample_ind = np.random.randint(0, X_gen.shape[0], n_plot_samples)
        for i in sample_ind:
            axes[1].plot(X_gen[i, 0, :], alpha=0.1)
        axes[1].set_ylim(*ylim)
        plt.grid()

        plt.tight_layout()
        wandb.log({"visual inspection": wandb.Image(plt)})
        plt.close()

    def log_pca(self, n_plot_samples: int, X_gen, z_test: np.ndarray, z_gen: np.ndarray):
        X_gen = X_gen.cpu().numpy()

        sample_ind_test = np.random.choice(range(self.X_test.shape[0]), size=n_plot_samples, replace=False)
        sample_ind_gen = np.random.choice(range(X_gen.shape[0]), size=n_plot_samples, replace=False)

        # PCA: data space
        pca = PCA(n_components=2)
        X_embedded_test = pca.fit_transform(self.X_test.squeeze()[sample_ind_test])
        X_embedded_gen = pca.transform(X_gen.squeeze()[sample_ind_gen])

        plt.figure(figsize=(4, 4))
        # plt.title("PCA in the data space")
        plt.scatter(X_embedded_test[:, 0], X_embedded_test[:, 1], alpha=0.1, label='test')
        plt.scatter(X_embedded_gen[:, 0], X_embedded_gen[:, 1], alpha=0.1, label='gen')
        plt.legend()
        plt.tight_layout()
        wandb.log({"PCA-data_space": wandb.Image(plt)})
        plt.close()

        # PCA: latent space
        pca = PCA(n_components=2)
        z_embedded_test = pca.fit_transform(z_test.squeeze()[sample_ind_test].squeeze())
        z_embedded_gen = pca.transform(z_gen[sample_ind_gen].squeeze())

        plt.figure(figsize=(4, 4))
        # plt.title("PCA in the representation space by the trained encoder");
        plt.scatter(z_embedded_test[:, 0], z_embedded_test[:, 1], alpha=0.1, label='test')
        plt.scatter(z_embedded_gen[:, 0], z_embedded_gen[:, 1], alpha=0.1, label='gen')
        plt.legend()
        plt.tight_layout()
        wandb.log({"PCA-latent_space": wandb.Image(plt)})
        plt.close()

    def log_tsne(self, n_plot_samples: int, X_gen, z_test: np.ndarray, z_gen: np.ndarray):
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

    def js_divergence(self, X_gen):
        js_div = JS_Div(X_gen.cpu().flatten(), torch.from_numpy(self.X_test).flatten())
        return js_div
