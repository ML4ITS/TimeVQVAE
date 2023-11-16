"""
reference: https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.LeakyReLU(),
            nn.Conv1d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm1d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class VQVAEEncBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                      padding_mode='replicate'),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        out = self.block(x)
        return out


class VQVAEDecBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True))

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
                 bn: bool = True,
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
            VQVAEEncBlock(num_channels, d) if downsample_rate >= 2 else ResBlock(num_channels, d, bn=bn),
            *[VQVAEEncBlock(d, d) for _ in range(int(np.log2(downsample_rate)) - 1)],
            *[nn.Sequential(ResBlock(d, d, bn=bn), nn.BatchNorm1d(d)) for _ in range(n_resnet_blocks)],
        )

    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return (B, C, H, W') where W' <= W
        """
        out = self.encoder(x)
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
                 **kwargs):
        """
        :param d: hidden dimension size
        :param num_channels: channel size of input
        :param downsample_rate: should be a factor of 2; e.g., 2, 4, 8, 16, ...
        :param n_resnet_blocks: number of ResNet blocks
        :param kwargs:
        """
        super().__init__()
        self.decoder = nn.Sequential(
            *[nn.Sequential(ResBlock(d, d), nn.BatchNorm1d(d)) for _ in range(n_resnet_blocks)],
            *[VQVAEDecBlock(d, d) for _ in range(int(np.log2(downsample_rate)) - 1)],
            nn.ConvTranspose1d(d, num_channels, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose1d(num_channels, num_channels, kernel_size=4, stride=2, padding=1),  # one more upsampling layer is added not to miss reconstruction details
        )

        self.is_upsample_size_updated = False
        self.register_buffer("upsample_size", torch.zeros(0))

    def register_upsample_size(self, l):
        self.upsample_size = l
        self.is_upsample_size_updated = True

    def forward(self, z):
        """
        :param x: output from the encoder (B, C, H, W')
        :return  (B, C, H, W)
        """
        out = self.decoder(z)
        if isinstance(self.upsample_size, torch.Tensor):
            # print('self.upsample_size:', self.upsample_size)
            upsample_size = self.upsample_size.cpu().numpy().astype(int)
            # print('upsample_size:', upsample_size)
            out = F.interpolate(out, size=(upsample_size,), mode='linear', align_corners=False)
            return out


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
