# TimeVQVAE
This is an official Github repository for the PyTorch implementation of TimeVQVAE from our paper ["Vector Quantized Time Series Generation with a Bidirectional Prior Model", AISTATS 2023](https://arxiv.org/abs/2303.04743).

TimeVQVAE is a robust time series generation model that utilizes vector quantization for data compression into the discrete latent space (stage1) and a bidirectional transformer for the prior learning (stage2).

## Installation
Install from PyPI with `uv`:

```bash
uv add timevqvae
```

## Usage

### Stage 1
Example of running `VQVAE` with a dummy 1D time-series input (`batch, channels, length`):
```python
import torch
from timevqvae.vqvae import VQVAE

vqvae = VQVAE(
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
out = vqvae(x)

print(out.x_recon.shape)          # (4, 1, 128)
print(out.recons_loss.keys())     # dict_keys(['LF.time', 'HF.time'])
print(out.vq_losses.keys())       # dict_keys(['LF', 'HF'])
print(out.perplexities.keys())    # dict_keys(['LF', 'HF'])
```

### Stage 2
Example of running `MaskGIT` for:
1. training loss computation (internally calls `compute_mask_prediction_loss`)
2. token sampling with `iterative_decoding` and decoding sampled tokens back to time series.

```python
import torch
from timevqvae.maskgit import MaskGIT, PriorModelConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

# Reuse `vqvae` from the Stage-1 example above.
# For Stage-2, this should be a pretrained Stage-1 model with trained weights loaded.
# Example:
# vqvae.load_state_dict(torch.load("vqvae_stage1.pt", map_location=device))
# vqvae = vqvae.to(device).eval()

# Stage-2 prior model.
# n_classes is dataset-dependent (example: 2 classes).
maskgit = MaskGIT(
    vqvae=vqvae,
    lf_choice_temperature=10.0,
    hf_choice_temperature=0.0,
    lf_num_sampling_steps=10,
    hf_num_sampling_steps=10,
    lf_codebook_size=1024,
    hf_codebook_size=1024,
    transformer_embedding_dim=128,
    lf_prior_model_config=PriorModelConfig(
        hidden_dim=128,
        n_layers=4,
        heads=2,
        ff_mult=1,
        use_rmsnorm=True,
        p_unconditional=0.2,
        model_dropout=0.3,
        emb_dropout=0.3,
    ),
    hf_prior_model_config=PriorModelConfig(
        hidden_dim=32,
        n_layers=1,
        heads=1,
        ff_mult=1,
        use_rmsnorm=True,
        p_unconditional=0.2,
        model_dropout=0.3,
        emb_dropout=0.3,
    ),
    classifier_free_guidance_scale=1.0,
    n_classes=2,
).to(device)

# ---------------------------------------------------------
# 1) Training logic example: dataclass loss from compute_mask_prediction_loss
# ---------------------------------------------------------
maskgit.train()
x = torch.randn(4, 1, 128, device=device)                      # (batch, channels, length)
class_condition = torch.randint(0, 2, (4, 1), device=device)   # (batch, 1)

# maskgit.forward(...) internally calls training_logic.compute_mask_prediction_loss(...)
losses = maskgit(x, class_condition)
print(
    losses.total_mask_prediction_loss.item(),
    losses.mask_pred_loss_l.item(),
    losses.mask_pred_loss_h.item(),
)

# ---------------------------------------------------------
# 2) Sampling logic example: iterative_decoding + token decoding
# ---------------------------------------------------------
maskgit.eval()
with torch.no_grad():
    token_ids_l, token_ids_h = maskgit.iterative_decoding(
        num_samples=4,
        mode="cosine",
        class_condition=1,   # int or tensor; normalized to (num_samples, 1)
        device=device,
    )

    x_l = maskgit.decode_token_ind_to_timeseries(token_ids_l, frequency="lf")  # (4, 1, 128)
    x_h = maskgit.decode_token_ind_to_timeseries(token_ids_h, frequency="hf")  # (4, 1, 128)
    x_gen = x_l + x_h

print(token_ids_l.shape, token_ids_h.shape)
print(x_l.shape, x_h.shape, x_gen.shape)
```


## Google Colab
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ML4ITS/TimeVQVAE/blob/main/.google_colab/TimeVQVAE%20(generation%20only).ipynb) (NB! make sure to change your notebook setting to GPU.)

A Google Colab notebook is available for time series generation with the pretrained VQVAE. 
The usage is simple:
1. **User Settings**: specify `dataset_name` and `n_samples_to_generate`.
2. **Sampling**: Run the unconditional sampling and class-conditional sampling.

## Related Papers

#### Neural Mapper for Vector Quantized Time Series Generator (NM-VQTSG)
If you want to improve realism of generated time series while preserving context, please see our Neural Mapper paper:
- Paper: https://arxiv.org/abs/2501.17553
- Citation entry: [3] below

#### TimeVQVAE for Anomaly Detection (TimeVQVAE-AD)
If your focus is anomaly detection with explainability and counterfactual sampling, please see TimeVQVAE-AD:
- Paper: https://www.sciencedirect.com/science/article/pii/S0031320324008216
- Code: https://github.com/ML4ITS/TimeVQVAE-AnomalyDetection
- Citation entry: [4] below

## References
[1] Lee, Daesoo, Sara Malacarne, and Erlend Aune. "Vector Quantized Time Series Generation with a Bidirectional Prior Model." International Conference on Artificial Intelligence and Statistics. PMLR, 2023.

[3] Lee, Daesoo, Sara Malacarne, and Erlend Aune. "Closing the Gap Between Synthetic and Ground Truth Time Series Distributions via Neural Mapping." arXiv preprint arXiv:2501.17553 (2025).

[4] Lee, Daesoo, Sara Malacarne, and Erlend Aune. "Explainable time series anomaly detection using masked latent generative modeling." Pattern Recognition (2024): 110826.
