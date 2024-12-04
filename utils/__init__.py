import os
import pickle
import logging
import yaml
import tempfile
from pathlib import Path
from typing import Union
import argparse

import torch.jit as jit
from einops import rearrange, repeat
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from torch.optim.lr_scheduler import SequentialLR, LambdaLR, CosineAnnealingLR

from sklearn.preprocessing import MinMaxScaler
import requests
import tarfile
import zipfile


def get_root_dir():
    return Path(__file__).parent.parent
# root_dir = Path(__file__).parent.parent
prefix = os.path.join('datasets', 'processed')


def load_yaml_param_settings(yaml_fname: str):
    """
    :param yaml_fname: .yaml file that consists of hyper-parameter settings.
    """
    stream = open(yaml_fname, 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def preprocess(df, scaler: MinMaxScaler, kind: str):
    """returns normalized and standardized data.
    """

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num()

    # normalize data (corrected)
    if kind == 'train':
        df = scaler.fit_transform(df)
    elif kind == 'test':
        df = scaler.transform(df)
    # df = MinMaxScaler().fit_transform(df)  # -> previous incorrect scaling method
    print('Data normalized')

    return df


def minibatch_slices_iterator(length, step_size,
                              ignore_incomplete_batch=False):
    """
    Iterate through all the mini-batch slices.

    Args:
        length (int): Total length of data in an epoch.
        step_size (int):
        ignore_incomplete_batch (bool): If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of items.
            (default :obj:`False`)

    Yields
        slice: Slices of each mini-batch.  The last mini-batch may contain
               less indices than `batch_size`.
    """
    start = 0
    stop1 = (length // step_size) * step_size
    while start < stop1:
        yield slice(start, start + step_size, 1)
        start += step_size
    if not ignore_incomplete_batch and start < length:
        yield slice(start, length, 1)


class BatchSlidingWindow(object):
    """
    Class for obtaining mini-batch iterators of sliding windows.

    Each mini-batch will have `batch_size` windows.  If the final batch
    contains less than `batch_size` windows, it will be discarded if
    `ignore_incomplete_batch` is :obj:`True`.

    Args:
        array_size (int): Size of the arrays to be iterated.
        window_size (int): The size of the windows.
        batch_size (int): Size of each mini-batch.
        excludes (np.ndarray): 1-D `bool` array, indicators of whether
            or not to totally exclude a point.  If a point is excluded,
            any window which contains that point is excluded.
            (default :obj:`None`, no point is totally excluded)
        shuffle (bool): If :obj:`True`, the windows will be iterated in
            shuffled order. (default :obj:`False`)
        ignore_incomplete_batch (bool): If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of windows.
            (default :obj:`False`)
    """

    def __init__(self, array_size, window_size, step_size, batch_size, excludes=None,
                 shuffle=False, ignore_incomplete_batch=False):
        # check the parameters
        if window_size < 1:
            raise ValueError('`window_size` must be at least 1')
        if array_size < window_size:
            raise ValueError('`array_size` must be at least as large as '
                             '`window_size`')
        if excludes is not None:
            excludes = np.asarray(excludes, dtype=np.bool)
            expected_shape = (array_size,)
            if excludes.shape != expected_shape:
                raise ValueError('The shape of `excludes` is expected to be '
                                 '{}, but got {}'.
                                 format(expected_shape, excludes.shape))

        # compute which points are not excluded
        if excludes is not None:
            mask = np.logical_not(excludes)
        else:
            mask = np.ones([array_size], dtype=np.bool)
        mask[: window_size - 1] = False
        where_excludes = np.where(excludes)[0]
        for k in range(1, window_size):
            also_excludes = where_excludes + k
            also_excludes = also_excludes[also_excludes < array_size]
            mask[also_excludes] = False

        # generate the indices of window endings
        indices = np.arange(array_size)[mask]
        self._indices = indices.reshape([-1, 1])

        # the offset array to generate the windows
        self._offsets = np.arange(-window_size + 1, 1)

        # memorize arguments
        self._array_size = array_size
        self._window_size = window_size
        self._step_size = step_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._ignore_incomplete_batch = ignore_incomplete_batch

    def get_iterator(self, arrays):
        """
        Iterate through the sliding windows of each array in `arrays`.

        This method is not re-entrant, i.e., calling :meth:`get_iterator`
        would invalidate any previous obtained iterator.

        Args:
            arrays (Iterable[np.ndarray]): 1-D arrays to be iterated.

        Yields:
            tuple[np.ndarray]: The windows of arrays of each mini-batch.
        """
        # check the parameters
        arrays = tuple(np.asarray(a) for a in arrays)
        if not arrays:
            raise ValueError('`arrays` must not be empty')

        # shuffle if required
        if self._shuffle:
            np.random.shuffle(self._indices)

        # iterate through the mini-batches
        for s in minibatch_slices_iterator(
                length=len(self._indices),
                step_size=self._step_size,
                ignore_incomplete_batch=self._ignore_incomplete_batch):
            idx = self._indices[s] + self._offsets
            yield tuple(a[idx] if len(a.shape) == 1 else a[idx, :] for a in arrays)


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def save_model(models_dict: dict, dirname='saved_models', id: str = ''):
    """
    :param models_dict: {'model_name': model, ...}
    """
    try:
        if not os.path.isdir(get_root_dir().joinpath(dirname)):
            os.mkdir(get_root_dir().joinpath(dirname))

        id_ = id[:]
        if id != '':
            id_ = '-' + id_
        for model_name, model in models_dict.items():
            torch.save(model.state_dict(), get_root_dir().joinpath(dirname, model_name + id_ + '.ckpt'))
    except PermissionError:
        # dirname = tempfile.mkdtemp()
        dirname = tempfile.gettempdir()
        print(f'\nThe trained model is saved in the following temporary dirname due to some permission error: {dirname}.\n')

        id_ = id[:]
        if id != '':
            id_ = '-' + id_
        for model_name, model in models_dict.items():
            torch.save(model.state_dict(), get_root_dir().joinpath(dirname, model_name + id_ + '.ckpt'))


def time_to_timefreq(x, n_fft: int, C: int, norm:bool=True):
    """
    x: (B, C, L)
    """
    x = rearrange(x, 'b c l -> (b c) l')
    x = torch.stft(x, n_fft, normalized=norm, return_complex=True, window=torch.hann_window(window_length=n_fft, device=x.device))
    x = torch.view_as_real(x)  # (B, N, T, 2); 2: (real, imag)
    x = rearrange(x, '(b c) n t z -> b (c z) n t ', c=C)  # z=2 (real, imag)
    return x.float()  # (B, C, H, W)


def timefreq_to_time(x, n_fft: int, C: int, norm:bool=True):
    x = rearrange(x, 'b (c z) n t -> (b c) n t z', c=C).contiguous()
    x = x.contiguous()
    x = torch.view_as_complex(x)
    x = torch.istft(x, n_fft, normalized=norm, window=torch.hann_window(window_length=n_fft, device=x.device))
    x = rearrange(x, '(b c) l -> b c l', c=C)
    return x.float()


def compute_var_loss(z):
    return torch.relu(1. - torch.sqrt(z.var(dim=0) + 1e-4)).mean()


def compute_cov_loss(z):
    norm_z = (z - z.mean(dim=0))
    norm_z = F.normalize(norm_z, p=2, dim=0)  # (batch * feature); l2-norm
    fxf_cov_z = torch.mm(norm_z.T, norm_z)  # (feature * feature)
    ind = np.diag_indices(fxf_cov_z.shape[0])
    fxf_cov_z[ind[0], ind[1]] = torch.zeros(fxf_cov_z.shape[0]).to(norm_z.device)
    cov_loss = (fxf_cov_z ** 2).mean()
    return cov_loss


def quantize(z, vq_model, transpose_channel_length_axes=False, svq_temp:Union[float,None]=None):
    input_dim = len(z.shape) - 2
    if input_dim == 2:
        h, w = z.shape[2:]
        z = rearrange(z, 'b c h w -> b (h w) c')
        z_q, indices, vq_loss, perplexity = vq_model(z, svq_temp)
        z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h, w=w)
    elif input_dim == 1:
        if transpose_channel_length_axes:
            z = rearrange(z, 'b c l -> b (l) c')
        z_q, indices, vq_loss, perplexity = vq_model(z, svq_temp)
        if transpose_channel_length_axes:
            z_q = rearrange(z_q, 'b (l) c -> b c l')
    else:
        raise ValueError
    return z_q, indices, vq_loss, perplexity


# def zero_pad_high_freq(xf):
#     """
#     xf: (B, C, H, W); H: frequency-axis, W: temporal-axis
#     """
#     xf_l = torch.zeros(xf.shape, dtype=torch.float).to(xf.device)
#     xf_l[:, :, 0, :] = xf[:, :, 0, :]
#     return xf_l


# def zero_pad_low_freq(xf):
#     """
#     xf: (B, C, H, W); H: frequency-axis, W: temporal-axis
#     """
#     xf_h = torch.zeros(xf.shape, dtype=torch.float).to(xf.device)
#     xf_h[:, :, 1:, :] = xf[:, :, 1:, :]
#     return xf_h

def zero_pad_high_freq(xf, copy=False):
    """
    xf: (B, C, H, W); H: frequency-axis, W: temporal-axis
    """
    if not copy:
        xf_l = torch.zeros(xf.shape).to(xf.device)
        xf_l[:, :, 0, :] = xf[:, :, 0, :]  # (b c h w)
    else:
        # model input: copy the LF component and paste it to the rest of the frequency bands
        xf_l = xf[:, :, [0], :]  # (b c 1 w)
        xf_l = repeat(xf_l, 'b c 1 w -> b c h w', h=xf.shape[2]).float()  # (b c h w)
    return xf_l

def zero_pad_low_freq(xf, copy=False):
    """
    xf: (B, C, H, W); H: frequency-axis, W: temporal-axis
    """
    if not copy:
        xf_h = torch.zeros(xf.shape).to(xf.device)
        xf_h[:, :, 1:, :] = xf[:, :, 1:, :]
    else:
        # model input: copy the first HF component, and paste it to the LF band
        xf_h = xf[:, :, 1:, :]  # (b c h-1 w)
        xf_h = torch.cat((xf_h[:,:,[0],:], xf_h), dim=2).float()  # (b c h w)
    return xf_h

def compute_emb_loss(codebook, x, use_cosine_sim, esm_max_codes):
    embed = codebook.embed
    flatten = x.reshape(-1, x.shape[-1])

    if use_cosine_sim:
        flatten = F.normalize(flatten, p=2, dim=-1)
        embed = F.normalize(embed, p=2, dim=-1)

    # N samples can be sampled fro embed for the memory efficiency.
    ind = torch.randint(0, embed.shape[0], size=(min(esm_max_codes, embed.shape[0]),))
    embed = embed[ind]

    cov_embed = torch.cov(embed.t())  # (D, D)
    cov_x = torch.cov(flatten.t())  # (D, D)

    mean_embed = torch.mean(embed, dim=0)
    mean_x = torch.mean(flatten, dim=0)

    esm_loss = F.mse_loss(cov_x.detach(), cov_embed) + F.mse_loss(mean_x.detach(), mean_embed)
    return esm_loss


def compute_downsample_rate(input_length: int,
                            n_fft: int,
                            downsampled_width: int):
    return round(input_length / (np.log2(n_fft) - 1) / downsampled_width) if input_length >= downsampled_width else 1


def download_ucr_datasets_old(url='https://figshare.com/ndownloader/files/37909926', chunk_size=128, zip_fname='UCR_archive.zip'):
    """
    download the original datasets of the UCR archive
    """
    dirname = str(get_root_dir().joinpath("datasets"))
    if os.path.isdir(os.path.join(dirname, 'UCRArchive_2018')):
        return None

    if not os.path.isdir(dirname) or not os.path.isfile(os.path.join(dirname, zip_fname)):
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        # download
        r = requests.get(url, stream=True)
        print('downloading the UCR archive datasets...\n')
        fname = os.path.join(dirname, zip_fname)
        with open(fname, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)

        # unzip
        with tarfile.open(fname) as ff:
            ff.extractall(path=dirname)
    elif os.path.isfile(str(get_root_dir().joinpath("datasets", zip_fname))):
        # unzip
        fname = os.path.join(dirname, zip_fname)
        with tarfile.open(fname) as ff:
            ff.extractall(path=dirname)
    else:
        pass

def download_ucr_datasets(url='https://figshare.com/ndownloader/files/47494442', chunk_size=128, zip_fname='UCR_archive.zip'):
    """
    download the re-organized datasets of the UCR archive for time series generation
    """
    dirname = str(get_root_dir().joinpath("datasets", 'UCRArchive_2018_resplit'))
    fname = os.path.join(dirname, zip_fname)

    if os.path.isdir(dirname) and len(os.listdir(dirname)) > 1:
        return None
    
    if not os.path.isdir(dirname) or not os.path.isfile(fname):
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        # download
        r = requests.get(url, stream=True)
        print('downloading the UCR archive datasets...\n')
        
        with open(fname, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)

        # unzip
        with zipfile.ZipFile(fname, 'r') as zz:
            zz.extractall(path=dirname)
    elif os.path.isfile(fname):
        # unzip
        with zipfile.ZipFile(fname, 'r') as zz:
            zz.extractall(path=dirname)
    else:
        pass

def get_target_ucr_dataset_names(args, ):
    if args.dataset_names:
        dataset_names = args.dataset_names
    else:
        data_summary_ucr = pd.read_csv(get_root_dir().joinpath('datasets', 'DataSummary_UCR.csv'))
        dataset_names = data_summary_ucr['Name'].tolist()
        # if args.data_summary_ucr_condition_query:
        #     dataset_names = data_summary_ucr.query(args.data_summary_ucr_condition_query)['Name'].values  # e.g., "500 <= Train <= 1000" for `data_summary_ucr_condition_query`
        # if args.n_dataset_partitions == 1:
        #     pass
        # elif args.n_dataset_partitions >= 1:
        # b = int(np.ceil(len(dataset_names) / args.n_dataset_partitions))
        # dataset_names = dataset_names[args.partition_idx * b: (args.partition_idx + 1) * b]
        # else:
        #     raise ValueError
    return dataset_names


def remove_outliers(data:np.ndarray):
    """
    data: (b d)
    """
    iso_forest = IsolationForest(max_samples=0.9, contamination=0.1, random_state=0)
    outliers = iso_forest.fit_predict(data) == 1
    filtered_data = data[outliers]
    return filtered_data


class SnakeActivation(jit.ScriptModule):
    """
    this version allows multiple values of `a` for different channels/num_features
    """
    def __init__(self, num_features:int, dim:int, a_base=0.2, learnable=True, a_max=0.5):
        super().__init__()
        assert dim in [1, 2], '`dim` supports 1D and 2D inputs.'

        if learnable:
            if dim == 1:  # (b d l); like time series
                a = np.random.uniform(a_base, a_max, size=(1, num_features, 1))  # (1 d 1)
                self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
            elif dim == 2:  # (b d h w); like 2d images
                a = np.random.uniform(a_base, a_max, size=(1, num_features, 1, 1))  # (1 d 1 1)
                self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        else:
            self.register_buffer('a', torch.tensor(a_base, dtype=torch.float32))

    @jit.script_method
    def forward(self, x):
        return x + (1 / self.a) * torch.sin(self.a * x) ** 2


# @torch.jit.script
# def snakemod(x, alpha, beta):
#     shape = x.shape
#     c = (alpha.abs() + beta.abs())*alpha.sign()
#     x = x.reshape(shape[0], shape[1], -1)
#     x = x + (c + 1e-9).reciprocal() * torch.sin(alpha * x + torch.sin(beta*x)).pow(2)
#     x = x.reshape(shape)
#     return x

# class SnakeActivation(nn.Module):
#     def __init__(self, channels: int, init_min=0.2, init_max=0.6, mod=1.):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.rand(1, channels, 1) * (init_max-init_min) + init_min)
#         self.beta = nn.Parameter(torch.tensor(0., dtype=self.alpha.dtype))
#         self.mod = mod

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         in_beta = self.alpha * torch.arcsinh(self.mod*self.beta)/self.mod
#         return snakemod(x, self.alpha, in_beta)
    

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def linear_warmup_cosine_annealingLR(optimizer: torch.optim.Optimizer, max_steps:int, linear_warmup_rate:float=0.1, min_lr:float=1e-6):
    assert linear_warmup_rate > 0. and linear_warmup_rate < 1., '0 < linear_warmup_rate < 1.'

    warmup_steps = int(max_steps * linear_warmup_rate)  # n% of max_steps

    # Define the warmup scheduler
    def warmup_lambda(current_step):
        if current_step >= warmup_steps:
            return 1.0
        return float(current_step) / float(max(1, warmup_steps))

    # Create the warmup scheduler
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Create the cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(optimizer, max_steps - warmup_steps, eta_min=min_lr)

    # Combine the warmup and cosine annealing schedulers
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    return scheduler
