# Angus Dempster, Francois Petitjean, Geoff Webb
#
# @article{dempster_etal_2020,
#   author  = {Dempster, Angus and Petitjean, Fran\c{c}ois and Webb, Geoffrey I},
#   title   = {ROCKET: Exceptionally fast and accurate time classification using random convolutional kernels},
#   year    = {2020},
#   journal = {Data Mining and Knowledge Discovery},
#   doi     = {https://doi.org/10.1007/s10618-020-00701-z}
# }
#
# https://arxiv.org/abs/1910.13051 (preprint)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numba import njit, prange
import torch.jit as jit


@njit("Tuple((float64[:],int32[:],float64[:],int32[:],int32[:]))(int64,int64)")
def generate_kernels(input_length, num_kernels):

    candidate_lengths = np.array((7, 9, 11), dtype = np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)

    weights = np.zeros(lengths.sum(), dtype = np.float64)
    biases = np.zeros(num_kernels, dtype = np.float64)
    dilations = np.zeros(num_kernels, dtype = np.int32)
    paddings = np.zeros(num_kernels, dtype = np.int32)

    a1 = 0

    for i in range(num_kernels):

        _length = lengths[i]

        _weights = np.random.normal(0, 1, _length)

        b1 = a1 + _length
        weights[a1:b1] = _weights - _weights.mean()

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        a1 = b1

    return weights, lengths, biases, dilations, paddings

@njit(fastmath = True)
def apply_kernel(X, weights, length, bias, dilation, padding):

    input_length = len(X)

    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

    end = (input_length + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):

        _sum = bias

        index = i

        for j in range(length):

            if index > -1 and index < input_length:

                _sum = _sum + weights[j] * X[index]

            index = index + dilation

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return _ppv / output_length, _max

@njit("float64[:,:](float64[:,:],Tuple((float64[::1],int32[:],float64[:],int32[:],int32[:])))", parallel = True, fastmath = True)
def apply_kernels(X, kernels):

    weights, lengths, biases, dilations, paddings = kernels

    num_examples, _ = X.shape
    num_kernels = len(lengths)

    _X = np.zeros((num_examples, num_kernels * 2), dtype = np.float64) # 2 features per kernel

    for i in prange(num_examples):

        a1 = 0 # for weights
        a2 = 0 # for features

        for j in range(num_kernels):

            b1 = a1 + lengths[j]
            b2 = a2 + 2

            _X[i, a2:b2] = \
            apply_kernel(X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])

            a1 = b1
            a2 = b2

    return _X


# class MiniRocketTransform(jit.ScriptModule):
class MiniRocketTransform(nn.Module):
    def __init__(self, input_length:int, num_features:int=10000):
        super(MiniRocketTransform, self).__init__()
        self.num_features = num_features
        
        # Define kernel parameters
        self.kernel_length = 9
        self.num_kernels = 84  # Based on the paper's default 9{3} subset
        self.kernels = self._generate_kernels()
        
        # Compute dilations
        self.dilations = self._compute_dilations(input_length)
        
        # Compute bias quantiles
        self.biases = None
    
    def _generate_kernels(self):
        # Generate fixed set of two-valued kernels with weights {-1, 2}
        kernel_set = []
        for i in range(self.num_kernels):
            kernel = np.random.choice([-1, 2], size=self.kernel_length, p=[2/3, 1/3])
            if np.sum(kernel) != 0:
                kernel_set.append(kernel)
        return np.array(kernel_set)
    
    def _compute_dilations(self, input_length):
        max_dilation = (input_length - 1) // (self.kernel_length - 1)
        dilations = np.logspace(0, np.log10(max_dilation), num=self.num_kernels, base=2, dtype=int)
        return np.unique(dilations)
    
    @torch.no_grad()
    # @jit.script_method
    def forward(self, x, normalize=True):
        self.eval()

        batch_size, _, length = x.shape
        x_transformed = torch.zeros((batch_size, self.num_features), device=x.device)
        
        feature_idx = 0
        for kernel in self.kernels:
            for dilation in self.dilations:
                kernel_dilated = np.zeros(self.kernel_length + (self.kernel_length - 1) * (dilation - 1))
                kernel_dilated[::dilation] = kernel
                kernel_tensor = torch.tensor(kernel_dilated, dtype=torch.float32, device=x.device).view(1, 1, -1)
                
                conv_output = nn.functional.conv1d(x, kernel_tensor, padding=len(kernel_tensor) // 2)
                
                if self.biases is None:
                    self.biases = self._compute_biases(conv_output)
                
                for bias in self.biases:
                    ppv = (conv_output - bias > 0).float().mean(dim=2)
                    x_transformed[:, feature_idx] = ppv.view(-1)
                    feature_idx += 1
        
        if normalize:
            x_transformed = F.normalize(x_transformed, p=2, dim=-1)

        return x_transformed
    
    def _compute_biases(self, conv_output):
        # Compute biases as quantiles from the convolution output
        biases = []
        for i in range(3):
            bias = torch.quantile(conv_output, q=(i+1)/4.0, dim=2).mean(dim=0)
            biases.append(bias)
        return biases