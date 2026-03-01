import numpy as np
import torch
from einops import rearrange, repeat


def time_to_timefreq(x, n_fft: int, C: int, norm: bool = True):
    """Convert time-domain signal (B, C, L) to time-frequency representation."""
    x = rearrange(x, "b c l -> (b c) l")
    x = torch.stft(
        x,
        n_fft,
        normalized=norm,
        return_complex=True,
        window=torch.hann_window(window_length=n_fft, device=x.device),
    )
    x = torch.view_as_real(x)
    x = rearrange(x, "(b c) n t z -> b (c z) n t ", c=C)
    return x.float()


def timefreq_to_time(x, n_fft: int, C: int, norm: bool = True):
    """Convert time-frequency representation back to time-domain signal."""
    x = rearrange(x, "b (c z) n t -> (b c) n t z", c=C).contiguous()
    x = torch.view_as_complex(x)
    x = torch.istft(
        x,
        n_fft,
        normalized=norm,
        window=torch.hann_window(window_length=n_fft, device=x.device),
    )
    x = rearrange(x, "(b c) l -> b c l", c=C)
    return x.float()


def zero_pad_high_freq(xf, copy=False):
    """Keep LF component only in frequency-axis of (B, C, H, W)."""
    if not copy:
        xf_l = torch.zeros(xf.shape).to(xf.device)
        xf_l[:, :, 0, :] = xf[:, :, 0, :]
    else:
        xf_l = xf[:, :, [0], :]
        xf_l = repeat(xf_l, "b c 1 w -> b c h w", h=xf.shape[2]).float()
    return xf_l


def zero_pad_low_freq(xf, copy=False):
    """Keep HF components only in frequency-axis of (B, C, H, W)."""
    if not copy:
        xf_h = torch.zeros(xf.shape).to(xf.device)
        xf_h[:, :, 1:, :] = xf[:, :, 1:, :]
    else:
        xf_h = xf[:, :, 1:, :]
        xf_h = torch.cat((xf_h[:, :, [0], :], xf_h), dim=2).float()
    return xf_h


def compute_downsample_rate(input_length: int, n_fft: int, downsampled_width: int):
    if input_length < downsampled_width:
        return 1
    return round(input_length / (np.log2(n_fft) - 1) / downsampled_width)
