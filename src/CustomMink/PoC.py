import torch

import MinkowskiEngine as ME

from utils.utils import *


class PoCMinkNet(ME.MinkowskiNetwork):
    """A class to represent MinkNet neural network."""

    def __init__(self, in_channels: int, out_channels: int, dimensions: int = 3):
        """
        :param conv_channels: list of integers describing consecutive convolution
        kernels sizes - it is assumed that convolution kernel is a cube, all
        kernel dimensions are equal
        :param in_channels: number of channels in input
        :param out_channels: number of channels in output, equal to number of classes
        :param dimensions: number of dimensions of input
        """
        ME.MinkowskiNetwork.__init__(self, dimensions)
        self.sparse_conv_blocks = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channels, out_channels=4, kernel_size=3, dimension=self.D
            ),
            ME.MinkowskiConvolution(
                in_channels=4, out_channels=4, kernel_size=3, dimension=self.D
            ),
            ME.MinkowskiConvolution(
                in_channels=4, out_channels=4, kernel_size=3, dimension=self.D
            ),
            ME.MinkowskiTanh(),
            ME.MinkowskiMaxPooling(kernel_size=2, dimension=self.D),
            ME.MinkowskiConvolution(
                in_channels=4, out_channels=8, kernel_size=3, dimension=self.D
            ),
            ME.MinkowskiConvolution(
                in_channels=8, out_channels=8, kernel_size=3, dimension=self.D
            ),
            ME.MinkowskiConvolution(
                in_channels=8, out_channels=8, kernel_size=3, dimension=self.D
            ),
            ME.MinkowskiTanh(),
            ME.MinkowskiMaxPooling(kernel_size=2, dimension=self.D),
            ME.MinkowskiConvolution(
                in_channels=8, out_channels=16, kernel_size=3, dimension=self.D
            ),
            ME.MinkowskiConvolution(
                in_channels=16, out_channels=16, kernel_size=3, dimension=self.D
            ),
            ME.MinkowskiConvolution(
                in_channels=16, out_channels=16, kernel_size=3, dimension=self.D
            ),
            ME.MinkowskiTanh(),
            ME.MinkowskiMaxPooling(kernel_size=2, dimension=self.D),
        )

        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.linear1 = torch.nn.Linear(in_features=2 * 16, out_features=64)
        self.tanh = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(in_features=64, out_features=out_channels)
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x: ME.SparseTensor) -> torch.Tensor:

        x = self.sparse_conv_blocks(x)

        x_avg = self.global_avg_pool(x)
        x_max = self.global_max_pool(x)

        x = torch.cat([x_avg.F, x_max.F], -1).squeeze(0)

        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.softmax(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return x
