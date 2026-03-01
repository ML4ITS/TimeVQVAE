# TimeVQVAE
This is an official Github repository for the PyTorch implementation of TimeVQVAE from our paper ["Vector Quantized Time Series Generation with a Bidirectional Prior Model", AISTATS 2023](https://arxiv.org/abs/2303.04743).

TimeVQVAE is a robust time series generation model that utilizes vector quantization for data compression into the discrete latent space (stage1) and a bidirectional transformer for the prior learning (stage2).

## Installation
Install from PyPI with `uv`:

```bash
uv add timevqvae
```

## Usage
Example of running `VQVAE` with a dummy 1D time-series input (`batch, channels, length`):
```python
import torch
from timevqvae.vqvae import VQVAE

model = VQVAE(
    in_channels=1,
    input_length=128,
    n_fft=4,
    init_dim=4,
    hid_dim=128,
    downsampled_width_l=8,
    downsampled_width_h=32,
    encoder_n_resnet_blocks=2,
    decoder_n_resnet_blocks=2,
    codebook_size_l=1024,
    codebook_size_h=1024,
    kmeans_init=True,
    codebook_dim=8,
)

x = torch.randn(4, 1, 128)  # (batch, channels, length)
out = model(x)

print(out.x_recon.shape)          # (4, 1, 128)
print(out.recons_loss.keys())     # dict_keys(['LF.time', 'HF.time'])
print(out.vq_losses.keys())       # dict_keys(['LF', 'HF'])
print(out.perplexities.keys())    # dict_keys(['LF', 'HF'])
```

## Google Colab
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ML4ITS/TimeVQVAE/blob/main/.google_colab/TimeVQVAE%20(generation%20only).ipynb) (NB! make sure to change your notebook setting to GPU.)

A Google Colab notebook is available for time series generation with the pretrained VQVAE. 
The usage is simple:
1. **User Settings**: specify `dataset_name` and `n_samples_to_generate`.
2. **Sampling**: Run the unconditional sampling and class-conditional sampling.

## Related Papers

### Neural Mapper for Vector Quantized Time Series Generator (NM-VQTSG)
If you want to improve realism of generated time series while preserving context, please see our Neural Mapper paper:
- Paper: https://arxiv.org/abs/2501.17553
- Citation entry: [3] below

### TimeVQVAE for Anomaly Detection (TimeVQVAE-AD)
If your focus is anomaly detection with explainability and counterfactual sampling, please see TimeVQVAE-AD:
- Paper: https://www.sciencedirect.com/science/article/pii/S0031320324008216
- Code: https://github.com/ML4ITS/TimeVQVAE-AnomalyDetection
- Citation entry: [4] below

## Citation
[1] Lee, Daesoo, Sara Malacarne, and Erlend Aune. "Vector Quantized Time Series Generation with a Bidirectional Prior Model." International Conference on Artificial Intelligence and Statistics. PMLR, 2023.

[3] Lee, Daesoo, Sara Malacarne, and Erlend Aune. "Closing the Gap Between Synthetic and Ground Truth Time Series Distributions via Neural Mapping." arXiv preprint arXiv:2501.17553 (2025).

[4] Lee, Daesoo, Sara Malacarne, and Erlend Aune. "Explainable time series anomaly detection using masked latent generative modeling." Pattern Recognition (2024): 110826.
