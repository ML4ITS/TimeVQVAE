import os
from typing import List, Union, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from experiments.exp_stage2 import ExpStage2
from generators.maskgit import MaskGIT
from preprocessing.data_pipeline import build_data_pipeline
from preprocessing.preprocess_ucr import DatasetImporterUCR, DatasetImporterCustom
from generators.sample import unconditional_sample, conditional_sample
from supervised_FCN_2.example_pretrained_model_loading import load_pretrained_FCN
from supervised_FCN_2.example_compute_FID import calculate_fid
from supervised_FCN_2.example_compute_IS import calculate_inception_score
from utils import time_to_timefreq, timefreq_to_time
from generators.neural_mapper import NeuralMapper
from evaluation.rocket_functions import generate_kernels, apply_kernels
from utils import zero_pad_low_freq, zero_pad_high_freq, remove_outliers
from evaluation.stat_metrics import marginal_distribution_difference, auto_correlation_difference, skewness_difference, kurtosis_difference



class Evaluation(nn.Module):
    """
    - FID
    - IS
    - visual inspection
    - PCA
    - t-SNE
    """
    def __init__(self, 
                 dataset_name: str, 
                 in_channels:int,
                 input_length:int, 
                 n_classes:int, 
                 device:int, 
                 config:dict, 
                 use_neural_mapper:bool=False,
                 feature_extractor_type:str='rocket',
                 rocket_num_kernels:int=1000,
                 use_custom_dataset:bool=False
                 ):
        super().__init__()
        self.dataset_name = dataset_name
        self.device = torch.device(device)
        self.config = config
        self.batch_size = self.config['evaluation']['batch_size']
        self.feature_extractor_type = feature_extractor_type
        assert feature_extractor_type in ['supervised_fcn', 'rocket'], 'unavailable feature extractor type.'

        if not use_custom_dataset:
            self.fcn = load_pretrained_FCN(dataset_name).to(device)
            self.fcn.eval()
        if feature_extractor_type == 'rocket':
            self.rocket_kernels = generate_kernels(input_length, num_kernels=rocket_num_kernels)

        # load the numpy matrix of the test samples
        dataset_importer = DatasetImporterUCR(dataset_name, **config['dataset']) if not use_custom_dataset else DatasetImporterCustom(**config['dataset'])
        self.X_train = dataset_importer.X_train
        self.X_test = dataset_importer.X_test
        self.Y_train = dataset_importer.Y_train
        self.Y_test = dataset_importer.Y_test
        self.mean = dataset_importer.mean  # scaling coefficient
        self.std = dataset_importer.std  # scaling coefficient

        self.ts_len = self.X_train.shape[-1]  # time series length
        self.n_classes = len(np.unique(dataset_importer.Y_train))

        # load the stage2 model
        self.stage2 = ExpStage2.load_from_checkpoint(os.path.join('saved_models', f'stage2-{dataset_name}.ckpt'), 
                                                      dataset_name=dataset_name, 
                                                      in_channels=in_channels,
                                                      input_length=input_length, 
                                                      config=config,
                                                      n_classes=n_classes,
                                                    #   use_neural_mapper=False,
                                                      feature_extractor_type=feature_extractor_type,
                                                      use_custom_dataset=use_custom_dataset,
                                                      map_location='cpu',
                                                      strict=False)
        self.stage2.eval()
        self.maskgit = self.stage2.maskgit
        self.stage1 = self.stage2.maskgit.stage1

        # load the neural mapper
        if use_neural_mapper:
            self.neural_mapper = NeuralMapper(self.ts_len, 1, config)
            fname = f'neural_mapper-{dataset_name}.ckpt'
            ckpt_fname = os.path.join('saved_models', fname)
            self.neural_mapper.load_state_dict(torch.load(ckpt_fname))
        else:
            self.neural_mapper = nn.Identity()

        # fit PCA on a training set
        self.pca = PCA(n_components=2, random_state=0)
        self.z_train = self.compute_z('train')
        self.z_test = self.compute_z('test')

        z_test = remove_outliers(self.z_test)  # only used to fit pca because `def fid_score` already contains `remove_outliers`
        z_transform_pca = self.pca.fit_transform(z_test)

        self.xmin_pca, self.xmax_pca = np.min(z_transform_pca[:,0]), np.max(z_transform_pca[:,0])
        self.ymin_pca, self.ymax_pca = np.min(z_transform_pca[:,1]), np.max(z_transform_pca[:,1])

    @torch.no_grad()
    def sample(self, n_samples: int, kind: str, class_index:Union[int,None]=None, unscale:bool=False, batch_size=None):
        """
        
        unscale: unscale the generated sample with percomputed mean and std.
        """
        assert kind in ['unconditional', 'conditional']

        # sampling
        if kind == 'unconditional':
            x_new_l, x_new_h, x_new = unconditional_sample(self.maskgit, n_samples, self.device, batch_size=batch_size if not isinstance(batch_size, type(None)) else self.batch_size)  # (b c l); b=n_samples, c=1 (univariate)
        elif kind == 'conditional':
            x_new_l, x_new_h, x_new = conditional_sample(self.maskgit, n_samples, self.device, class_index, batch_size=batch_size if not isinstance(batch_size, type(None)) else self.batch_size)  # (b c l); b=n_samples, c=1 (univariate)
        else:
            raise ValueError

        # NM
        num_batches = x_new.shape[0] // self.batch_size + (1 if x_new.shape[0] % self.batch_size != 0 else 0)
        X_new_R = []
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            mini_batch = x_new[start_idx:end_idx]
            x_new_R = self.neural_mapper(mini_batch.to(self.device)).cpu()
            X_new_R.append(x_new_R)
        X_new_R = torch.cat(X_new_R)

        # unscale
        if unscale:
            x_new_l = x_new_l*self.std + self.mean
            x_new_h = x_new_h*self.std + self.mean
            x_new = x_new*self.std + self.mean
            X_new_R = X_new_R*self.std + self.mean

        return (x_new_l, x_new_h, x_new), X_new_R

    def _extract_feature_representations(self, x:np.ndarray):
        """
        x: (b 1 l)
        """
        if self.feature_extractor_type == 'supervised_fcn':
            z = self.fcn(torch.from_numpy(x).float().to(self.device), return_feature_vector=True).cpu().detach().numpy()  # (b d)
        elif self.feature_extractor_type == 'rocket':
            x = x[:,0,:]  # (b l)
            z = apply_kernels(x, self.rocket_kernels)
            z = F.normalize(torch.from_numpy(z), p=2, dim=1).numpy()
        else:
            raise ValueError
        return z

    def compute_z_rec(self, kind:str):
        """
        compute representations of X_rec
        """
        assert kind in ['train', 'test']
        if kind == 'train':
            X = self.X_train  # (b 1 l)
        elif kind == 'test':
            X = self.X_test  # (b 1 l)
        else:
            raise ValueError
        
        n_samples = X.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get feature vectors from `X_test`
        zs = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            x = X[s]  # (b 1 l)
            x = torch.from_numpy(x).float().to(self.device)
            x_rec = self.stage1.forward(batch=(x, None), batch_idx=-1, return_x_rec=True).cpu().detach().numpy().astype(float)  # (b 1 l)
            z_t = self._extract_feature_representations(x_rec)
            zs.append(z_t)
        zs = np.concatenate(zs, axis=0)
        return zs

    @torch.no_grad()
    def compute_z_svq(self, kind:str):
        """
        compute representations of X', a stochastic variant of X with SVQ
        """
        assert kind in ['train', 'test']
        if kind == 'train':
            X = self.X_train  # (b 1 l)
        elif kind == 'test':
            X = self.X_test  # (b 1 l)
        else:
            raise ValueError
        
        n_samples = X.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get feature vectors from `X_test`
        zs = []
        xs_a = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            x = X[s]  # (b 1 l)
            x = torch.from_numpy(x).float().to(self.device)
            
            # x_rec = self.stage1.forward(batch=(x, None), batch_idx=-1, return_x_rec=True).cpu().detach().numpy().astype(float)  # (b 1 l)
            # svq_temp_rng = self.config['neural_mapper']['svq_temp_rng']
            # svq_temp = np.random.uniform(*svq_temp_rng)
            # tau = self.config['neural_mapper']['tau']
            tau = self.neural_mapper.tau.item()
            _, s_a_l = self.maskgit.encode_to_z_q(x, self.stage1.encoder_l, self.stage1.vq_model_l, svq_temp=tau)  # (b n)
            _, s_a_h = self.maskgit.encode_to_z_q(x, self.stage1.encoder_h, self.stage1.vq_model_h, svq_temp=tau)  # (b m)
            x_a_l = self.maskgit.decode_token_ind_to_timeseries(s_a_l, 'lf')  # (b 1 l)
            x_a_h = self.maskgit.decode_token_ind_to_timeseries(s_a_h, 'hf')  # (b 1 l)
            x_a = x_a_l + x_a_h  # (b c l)
            x_a = x_a.cpu().numpy().astype(float)
            xs_a.append(x_a)

            z_t = self._extract_feature_representations(x_a)
            zs.append(z_t)
        zs = np.concatenate(zs, axis=0)
        xs_a = np.concatenate(xs_a, axis=0)
        return zs, xs_a

    def compute_z(self, kind: str) -> np.ndarray:
        """
        It computes representation z given input x
        :param X_gen: generated X
        :return: z_test (z on X_test), z_gen (z on X_generated)
        """
        assert kind in ['train', 'test']
        if kind == 'train':
            X = self.X_train  # (b 1 l)
        elif kind == 'test':
            X = self.X_test  # (b 1 l)
        else:
            raise ValueError

        n_samples = X.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get feature vectors from `X_test`
        zs = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            z_t = self._extract_feature_representations(X[s])
            zs.append(z_t)
        zs = np.concatenate(zs, axis=0)
        return zs

    def compute_z_gen(self, X_gen: torch.Tensor) -> np.ndarray:
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

            # z_g = self.fcn(X_gen[s].float().to(self.device), return_feature_vector=True).cpu().detach().numpy()
            z_g = self._extract_feature_representations(X_gen[s].numpy().astype(float))

            z_gen.append(z_g)
        z_gen = np.concatenate(z_gen, axis=0)
        return z_gen

    def fid_score(self, z1:np.ndarray, z2:np.ndarray) -> int:
        z1, z2 = remove_outliers(z1), remove_outliers(z2)
        fid = calculate_fid(z1, z2)
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
    
    def log_visual_inspection(self, X1, X2, title: str, ylim: tuple = (-5, 5), n_plot_samples:int=200, alpha:float=0.1):
        b, c, l = X1.shape

        # `X_test`
        sample_ind = np.random.randint(0, X1.shape[0], n_plot_samples)
        fig, axes = plt.subplots(2, c, figsize=(c*4, 4))
        if c == 1:
            axes = axes[:, np.newaxis]
        plt.suptitle(title)
        
        for channel_idx in range(c):
            # X1
            for i in sample_ind:
                axes[0,channel_idx].plot(X1[i, channel_idx, :], alpha=alpha, color='C0')
            axes[0,channel_idx].set_ylim(*ylim)
            axes[0,channel_idx].set_title(f'channel idx:{channel_idx}')

            # X2
            sample_ind = np.random.randint(0, X2.shape[0], n_plot_samples)
            for i in sample_ind:
                axes[1,channel_idx].plot(X2[i, channel_idx, :], alpha=alpha, color='C0')
            axes[1,channel_idx].set_ylim(*ylim)
            
            if channel_idx == 0:
                axes[0,channel_idx].set_ylabel('X_test')
                axes[1,channel_idx].set_ylabel('X_gen')

        plt.tight_layout()
        wandb.log({f"visual comp ({title})": wandb.Image(plt)})
        plt.close()

    def log_pca(self, Zs:List[np.ndarray], labels:List[str], n_plot_samples:int=1000):
        assert len(Zs) == len(labels)

        plt.figure(figsize=(4, 4))

        for Z, label in zip(Zs, labels):
            ind = np.random.choice(range(Z.shape[0]), size=n_plot_samples, replace=True)
            Z_embed = self.pca.transform(Z[ind])
            
            plt.scatter(Z_embed[:, 0], Z_embed[:, 1], alpha=0.1, label=label)
            
            xpad = (self.xmax_pca - self.xmin_pca) * 0.1
            ypad = (self.ymax_pca - self.ymin_pca) * 0.1
            plt.xlim(self.xmin_pca-xpad, self.xmax_pca+xpad)
            plt.ylim(self.ymin_pca-ypad, self.ymax_pca+ypad)

        # plt.legend(loc='upper right')
        plt.tight_layout()
        wandb.log({f"PCA on Z ({'-'.join(labels)})": wandb.Image(plt)})
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
