"""
reference: https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import SnakeActivation


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, frequency_indepence:bool, mid_channels=None, dropout:float=0.):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels
        
        kernel_size = (1, 3) if frequency_indepence else (3, 3)
        padding = (0, 1) if frequency_indepence else (1, 1)

        layers = [
            SnakeActivation(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=kernel_size, stride=(1, 1), padding=padding),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=kernel_size, stride=(1, 1), padding=padding),
            nn.Dropout(dropout)
        ]
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class VQVAEEncBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 frequency_indepence:bool,
                 dropout:float=0.
                 ):
        super().__init__()
        
        kernel_size = (1, 4) if frequency_indepence else (3, 4)
        padding = (0, 1) if frequency_indepence else (1, 1)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=(1, 2), padding=padding,
                      padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(),
            nn.Dropout(dropout))

    def forward(self, x):
        out = self.block(x)
        return out


class VQVAEDecBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 frequency_indepence:bool,
                 dropout:float=0.
                 ):
        super().__init__()
        
        kernel_size = (1, 4) if frequency_indepence else (3, 4)
        padding = (0, 1) if frequency_indepence else (1, 1)

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=(1, 2), padding=padding),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(),
            nn.Dropout(dropout))

    def forward(self, x):
        out = self.block(x)
        return out


class VQVAEEncoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.
    """

    def __init__(self,
                 d: int,
                 num_channels: int,
                 downsample_rate: int,
                 n_resnet_blocks: int,
                 frequency_indepence:bool,
                 dropout:float=0.3,
                 **kwargs):
        """
        :param d: hidden dimension size
        :param num_channels: channel size of input
        :param downsample_rate: should be a factor of 2; e.g., 2, 4, 8, 16, ...
        :param n_resnet_blocks: number of ResNet blocks
        :param bn: use of BatchNorm
        :param kwargs:
        """
        super().__init__()
        self.encoder = nn.Sequential(
            VQVAEEncBlock(num_channels, d, frequency_indepence),
            *[nn.Sequential(VQVAEEncBlock(d, d, frequency_indepence, dropout=dropout), *[ResBlock(d, d, frequency_indepence, dropout=dropout) for _ in range(n_resnet_blocks)]) for _ in range(int(np.log2(downsample_rate)) - 1)]
        )

        self.is_num_tokens_updated = False
        self.register_buffer('num_tokens', torch.zeros(1).int())
        self.register_buffer('H_prime', torch.zeros(1).int())
        self.register_buffer('W_prime', torch.zeros(1).int())

    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return (B, C, H, W') where W' <= W
        """
        out = self.encoder(x)
        if not self.is_num_tokens_updated:
            self.H_prime += out.shape[2]
            self.W_prime += out.shape[3]
            self.num_tokens += self.H_prime * self.W_prime
            self.is_num_tokens_updated = True
        return out


class VQVAEDecoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.
    """

    def __init__(self,
                 d: int,
                 num_channels: int,
                 downsample_rate: int,
                 n_resnet_blocks: int,
                 frequency_indepence:bool,
                 dropout:float=0.3,
                 **kwargs):
        """
        :param d: hidden dimension size
        :param num_channels: channel size of input
        :param downsample_rate: should be a factor of 2; e.g., 2, 4, 8, 16, ...
        :param n_resnet_blocks: number of ResNet blocks
        :param kwargs:
        """
        super().__init__()
        
        kernel_size = (1, 4) if frequency_indepence else (3, 4)
        padding = (0, 1) if frequency_indepence else (1, 1)
        
        self.decoder = nn.Sequential(
            *[nn.Sequential(ResBlock(d, d, frequency_indepence, dropout=dropout), *[VQVAEDecBlock(d, d, frequency_indepence, dropout=dropout) for _ in range(int(np.log2(downsample_rate)) - 1)]) for _ in range(n_resnet_blocks)],
            # *[VQVAEDecBlock(d, d) for _ in range(int(np.log2(downsample_rate)) - 1)],
            nn.ConvTranspose2d(d, num_channels, kernel_size=kernel_size, stride=(1, 2), padding=padding),
            nn.ConvTranspose2d(num_channels, num_channels, kernel_size=kernel_size, stride=(1, 2), padding=padding),  # one more upsampling layer is added not to miss reconstruction details
        )

        self.is_upsample_size_updated = False
        self.register_buffer("upsample_size", torch.zeros(2))

    def register_upsample_size(self, hw: torch.IntTensor):
        """
        :param hw: (height H, width W) of input
        """
        self.upsample_size = hw
        self.is_upsample_size_updated = True

    def forward(self, x):
        """
        :param x: output from the encoder (B, C, H, W')
        :return  (B, C, H, W)
        """
        out = self.decoder(x)
        if isinstance(self.upsample_size, torch.Tensor):
            upsample_size = self.upsample_size.cpu().numpy().astype(int)
            upsample_size = [*upsample_size]
            out = F.interpolate(out, size=upsample_size, mode='bilinear', align_corners=True)
            return out
        else:
            raise ValueError('self.upsample_size is not yet registered.')


if __name__ == '__main__':
    import numpy as np

    x = torch.rand(1, 2, 4, 128)  # (batch, channels, height, width)

    encoder = VQVAEEncoder(d=32, num_channels=2, downsample_rate=4, n_resnet_blocks=2)
    decoder = VQVAEDecoder(d=32, num_channels=2, downsample_rate=4, n_resnet_blocks=2)
    decoder.upsample_size = torch.IntTensor(np.array(x.shape[2:]))

    z = encoder(x)
    x_recons = decoder(z)

    print('x.shape:', x.shape)
    print('z.shape:', z.shape)
    print('x_recons.shape:', x_recons.shape)
