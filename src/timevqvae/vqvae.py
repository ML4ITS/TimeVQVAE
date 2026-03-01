from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from timevqvae.models.vq_vae_encdec import VQVAEDecoder, VQVAEEncoder
from timevqvae.utils.nn import quantize
from timevqvae.utils.signal import (
    compute_downsample_rate,
    time_to_timefreq,
    timefreq_to_time,
    zero_pad_high_freq,
    zero_pad_low_freq,
)
from timevqvae.vector_quantization import VectorQuantize


@dataclass
class VQVAEOutput:
    """Structured output for a VQ-VAE forward pass."""

    x_recon: torch.Tensor
    recons_loss: dict[str, torch.Tensor]
    vq_losses: dict[str, Any]
    perplexities: dict[str, torch.Tensor]


class VQVAE(nn.Module):
    """Dual-branch VQ-VAE for low- and high-frequency time-series reconstruction."""

    def __init__(
        self,
        in_channels: int,
        input_length: int,
        n_fft: int = 4,
        init_dim: int = 4,
        hid_dim: int = 128,
        downsampled_width_l: int = 8,
        downsampled_width_h: int = 32,
        encoder_n_resnet_blocks: int = 2,
        decoder_n_resnet_blocks: int = 2,
        codebook_size_l: int = 1024,
        codebook_size_h: int = 1024,
        kmeans_init: bool = True,
        codebook_dim: int = 8,
    ):
        """
        Initialize encoder, quantizer, and decoder modules for LF/HF branches.

        Args:
            in_channels: Number of channels in the input time-series.
            input_length: Target temporal length of reconstructed outputs.
            n_fft: FFT size used for time-frequency transforms.
            init_dim: Initial channel dimension for encoder/decoder blocks.
            hid_dim: Bottleneck hidden dimension for latent features.
            downsampled_width_l: Target latent width for LF branch.
            downsampled_width_h: Target latent width for HF branch.
            encoder_n_resnet_blocks: Number of encoder ResNet blocks per branch.
            decoder_n_resnet_blocks: Number of decoder ResNet blocks per branch.
            codebook_size_l: Codebook size for LF vector quantizer.
            codebook_size_h: Codebook size for HF vector quantizer.
            kmeans_init: Whether to initialize codebooks with k-means.
            codebook_dim: Embedding dimension used inside codebooks.
        """
        super().__init__()
        self.input_length = input_length
        self.n_fft = n_fft
        downsample_rate_l = compute_downsample_rate(input_length, self.n_fft, downsampled_width_l)
        downsample_rate_h = compute_downsample_rate(input_length, self.n_fft, downsampled_width_h)

        # encoder
        self.encoder_l = VQVAEEncoder(
            init_dim,
            hid_dim,
            2 * in_channels,
            downsample_rate_l,
            encoder_n_resnet_blocks,
            "lf",
            self.n_fft,
            frequency_indepence=True,
        )
        self.encoder_h = VQVAEEncoder(
            init_dim,
            hid_dim,
            2 * in_channels,
            downsample_rate_h,
            encoder_n_resnet_blocks,
            "hf",
            self.n_fft,
            frequency_indepence=False,
        )

        # quantizer
        vq_kwargs = {
            "kmeans_init": kmeans_init,
            "codebook_dim": codebook_dim,
        }
        self.vq_model_l = VectorQuantize(hid_dim, codebook_size_l, **vq_kwargs)
        self.vq_model_h = VectorQuantize(hid_dim, codebook_size_h, **vq_kwargs)

        # decoder
        self.decoder_l = VQVAEDecoder(
            init_dim,
            hid_dim,
            2 * in_channels,
            downsample_rate_l,
            decoder_n_resnet_blocks,
            input_length,
            "lf",
            self.n_fft,
            in_channels,
            frequency_indepence=True,
        )
        self.decoder_h = VQVAEDecoder(
            init_dim,
            hid_dim,
            2 * in_channels,
            downsample_rate_h,
            decoder_n_resnet_blocks,
            input_length,
            "hf",
            self.n_fft,
            in_channels,
            frequency_indepence=False,
        )

    def _compute_frequency_targets(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build LF/HF time-domain targets from the input signal."""
        in_channels = x.shape[1]
        xf = time_to_timefreq(x, self.n_fft, in_channels)  # (b c h w)
        u_l = zero_pad_high_freq(xf)  # (b c h w)
        x_l = F.interpolate(timefreq_to_time(u_l, self.n_fft, in_channels), self.input_length, mode="linear")  # (b c l)
        u_h = zero_pad_low_freq(xf)  # (b c h w)
        x_h = F.interpolate(timefreq_to_time(u_h, self.n_fft, in_channels), self.input_length, mode="linear")  # (b c l)
        return x_l, x_h

    def _process_lf_branch(self, x: torch.Tensor) -> tuple[torch.Tensor, Any, torch.Tensor]:
        """Encode, quantize, and decode the low-frequency branch."""
        z_l = self.encoder_l(x)
        z_q_l, _, vq_loss_l, perplexity_l = quantize(z_l, self.vq_model_l)
        xhat_l = self.decoder_l(z_q_l)  # (b c l)
        return xhat_l, vq_loss_l, perplexity_l

    def _process_hf_branch(self, x: torch.Tensor) -> tuple[torch.Tensor, Any, torch.Tensor]:
        """Encode, quantize, and decode the high-frequency branch."""
        z_h = self.encoder_h(x)
        z_q_h, _, vq_loss_h, perplexity_h = quantize(z_h, self.vq_model_h)
        xhat_h = self.decoder_h(z_q_h)  # (b c l)
        return xhat_h, vq_loss_h, perplexity_h

    def _compute_losses(
        self,
        x_l: torch.Tensor,
        x_h: torch.Tensor,
        xhat_l: torch.Tensor,
        xhat_h: torch.Tensor,
        vq_loss_l: Any,
        vq_loss_h: Any,
        perplexity_l: torch.Tensor,
        perplexity_h: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, torch.Tensor]]:
        """Compute branch-wise reconstruction losses and quantization statistics."""
        recons_loss = {
            "LF.time": F.mse_loss(x_l, xhat_l),
            "HF.time": F.l1_loss(x_h, xhat_h),
        }
        vq_losses = {"LF": vq_loss_l, "HF": vq_loss_h}
        perplexities = {"LF": perplexity_l, "HF": perplexity_h}
        return recons_loss, vq_losses, perplexities

    def forward(self, x: torch.Tensor) -> VQVAEOutput:
        """
        Run a forward pass from an input 1D time-series.

        Args:
            x: Input tensor with shape `(B, C, L)`, where:
                `B` is batch size, `C` is channels, and `L` is temporal length.
                Each channel is one 1D signal over time.

        Returns:
            A `VQVAEOutput` dataclass containing:
            - `x_recon`: reconstructed signal `(B, C, L)`
            - `recons_loss`: LF/HF reconstruction losses
            - `vq_losses`: LF/HF quantization losses
            - `perplexities`: LF/HF codebook perplexities
        """
        x_l, x_h = self._compute_frequency_targets(x)
        xhat_l, vq_loss_l, perplexity_l = self._process_lf_branch(x)
        xhat_h, vq_loss_h, perplexity_h = self._process_hf_branch(x)

        recons_loss, vq_losses, perplexities = self._compute_losses(
            x_l,
            x_h,
            xhat_l,
            xhat_h,
            vq_loss_l,
            vq_loss_h,
            perplexity_l,
            perplexity_h,
        )
        x_recon = xhat_l + xhat_h
        return VQVAEOutput(
            x_recon=x_recon,
            recons_loss=recons_loss,
            vq_losses=vq_losses,
            perplexities=perplexities,
        )
