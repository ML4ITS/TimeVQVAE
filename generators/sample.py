"""
`python sample.py`

sample
    1) unconditional sampling
    2) class-conditional sampling
"""
import os
from argparse import ArgumentParser

import numpy as np
import torch
import matplotlib.pyplot as plt

from generators.maskgit import MaskGIT
from preprocessing.data_pipeline import build_data_pipeline
from utils import get_root_dir, load_yaml_param_settings


@torch.no_grad()
def unconditional_sample(maskgit: MaskGIT, n_samples: int, device, class_index=None, batch_size=32, return_representations=False):
    print('sampling...')
    n_iters = n_samples // batch_size
    is_residual_batch = False
    if n_samples % batch_size > 0:
        n_iters += 1
        is_residual_batch = True

    x_new_l, x_new_h, x_new = [], [], []
    quantize_new_l, quantize_new_h = [], []
    sample_callback = maskgit.iterative_decoding
    for i in range(n_iters):
        b = batch_size
        if (i+1 == n_iters) and is_residual_batch:
            b = n_samples - ((n_iters-1) * batch_size)
        embed_ind_l, embed_ind_h = sample_callback(num=b, device=device, class_index=class_index)

        if return_representations:
            x_l, quantize_l = maskgit.decode_token_ind_to_timeseries(embed_ind_l, 'lf', True)
            x_h, quantize_h = maskgit.decode_token_ind_to_timeseries(embed_ind_h, 'hf', True)
            x_l, quantize_l, x_h, quantize_h = x_l.cpu(), quantize_l.cpu(), x_h.cpu(), quantize_h.cpu()
            quantize_new_l.append(quantize_l)
            quantize_new_h.append(quantize_h)
        else:
            x_l = maskgit.decode_token_ind_to_timeseries(embed_ind_l, 'lf').cpu()
            x_h = maskgit.decode_token_ind_to_timeseries(embed_ind_h, 'hf').cpu()

        x_new_l.append(x_l)
        x_new_h.append(x_h)
        x_new.append(x_l + x_h)  # (b c l); b=n_samples, c=1 (univariate)

    x_new_l = torch.cat(x_new_l)
    x_new_h = torch.cat(x_new_h)
    x_new = torch.cat(x_new)

    if return_representations:
        quantize_new_l = torch.cat(quantize_new_l)
        quantize_new_h = torch.cat(quantize_new_h)
        return (x_new_l, x_new_h, x_new), (quantize_new_l, quantize_new_h)
    else:
        return x_new_l, x_new_h, x_new


@torch.no_grad()
def conditional_sample(maskgit: MaskGIT, n_samples: int, device, class_index: int, batch_size=32, return_representations=False):
    """
    class_index: starting from 0. If there are two classes, then `class_index` ∈ {0, 1}.
    """
    if return_representations:
        (x_new_l, x_new_h, x_new), (quantize_new_l, quantize_new_h) = unconditional_sample(maskgit, n_samples, device, class_index, batch_size)
        return (x_new_l, x_new_h, x_new), (quantize_new_l, quantize_new_h)
    else:
        x_new_l, x_new_h, x_new = unconditional_sample(maskgit, n_samples, device, class_index, batch_size)
        return x_new_l, x_new_h, x_new


def plot_generated_samples(x_new_l, x_new_h, x_new, title: str, max_len=20):
    """
    x_new: (n_samples, c, length); c=1 (univariate)
    """
    n_samples = x_new.shape[0]
    if n_samples > max_len:
        print(f"`n_samples` is too large for visualization. maximum is {max_len}")
        return None

    try:
        fig, axes = plt.subplots(1*n_samples, 1, figsize=(3.5, 1.7*n_samples))
        alpha = 0.5
        if n_samples > 1:
            for i, ax in enumerate(axes):
                ax.set_title(title)
                ax.plot(x_new[i, 0, :])
                ax.plot(x_new_l[i, 0, :], alpha=alpha)
                ax.plot(x_new_h[i, 0, :], alpha=alpha)
        else:
            axes.set_title(title)
            axes.plot(x_new[0, 0, :])
            axes.plot(x_new_l[0, 0, :], alpha=alpha)
            axes.plot(x_new_h[0, 0, :], alpha=alpha)

        plt.tight_layout()
        plt.show()
    except ValueError:
        print(f"`n_samples` is too large for visualization. maximum is {max_len}")


def save_generated_samples(x_new: np.ndarray, save: bool, fname: str = None):
    if save:
        fname = 'generated_samples.npy' if not fname else fname
        with open(get_root_dir().joinpath('generated_samples', fname), 'wb') as f:
            np.save(f, x_new)
            print("numpy matrix of the generated samples are saved as `generated_samples/generated_samples.npy`.")
