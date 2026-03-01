from typing import Union

import numpy as np
import torch
import torch.jit as jit
import torch.nn as nn
from einops import rearrange


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def quantize(z, vq_model, transpose_channel_length_axes=False, svq_temp: Union[float, None] = None):
    input_dim = len(z.shape) - 2
    if input_dim == 2:
        h, w = z.shape[2:]
        z = rearrange(z, "b c h w -> b (h w) c")
        z_q, indices, vq_loss, perplexity = vq_model(z, svq_temp)
        z_q = rearrange(z_q, "b (h w) c -> b c h w", h=h, w=w)
    elif input_dim == 1:
        if transpose_channel_length_axes:
            z = rearrange(z, "b c l -> b (l) c")
        z_q, indices, vq_loss, perplexity = vq_model(z, svq_temp)
        if transpose_channel_length_axes:
            z_q = rearrange(z_q, "b (l) c -> b c l")
    else:
        raise ValueError
    return z_q, indices, vq_loss, perplexity


class SnakeActivation(jit.ScriptModule):
    """Snake activation with channel-wise learnable frequencies."""

    def __init__(self, num_features: int, dim: int, a_base=0.2, learnable=True, a_max=0.5):
        super().__init__()
        assert dim in [1, 2], "`dim` supports 1D and 2D inputs."

        if learnable:
            if dim == 1:
                a = np.random.uniform(a_base, a_max, size=(1, num_features, 1))
                self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
            elif dim == 2:
                a = np.random.uniform(a_base, a_max, size=(1, num_features, 1, 1))
                self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        else:
            self.register_buffer("a", torch.tensor(a_base, dtype=torch.float32))

    @jit.script_method
    def forward(self, x):
        return x + (1 / self.a) * torch.sin(self.a * x) ** 2
