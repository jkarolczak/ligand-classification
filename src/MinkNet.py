from typing import List

import torch

import MinkowskiEngine as ME

from utils import *


class SparseConvBlock(ME.MinkowskiNetwork):
    """A class to represent a single sparse convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimensions: int = 3,
        conv_channel: int = 3,
        pooling_kernel: int = 2,
        activation=ME.MinkowskiTanh,
        pooling=ME.MinkowskiMaxPooling,
    ):
        """
        :param in_channels: number of channels in input
        :param out_channels: number of channels in output, equal to number of classes
        :param conv_channel: integer describing convolution kernel size - it is
        assumed that convolution kernel is a cube, all kernel dimensions are equal
        :param dimensions: number of dimensions of input
        :param pooling_kernel: integer describing pooling kernel size, similarly
        to cov_kernel
        :param activation: activation function
        :param pooling: pooling function
        """
        ME.MinkowskiNetwork.__init__(self, dimensions)
        self.conv = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_channel,
            dimension=self.D,
        )
        self.activation = activation()
        self.pooling = pooling(kernel_size=pooling_kernel, dimension=self.D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        x = self.pooling(x)
        return x


class MinkNet(ME.MinkowskiNetwork):
    """A class to represent MinkNet neural network."""

    def __init__(
        self,
        conv_channels: List[int],
        in_channels: int,
        out_channels: int,
        dimensions: int = 3,
    ):
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
            *self.__get_sparse_conv_blocks(
                in_channels=in_channels, conv_channels=conv_channels
            )
        )

        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.linear1 = torch.nn.Linear(
            in_features=2 * conv_channels[-1], out_features=conv_channels[-1]
        )
        self.tanh = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(
            in_features=conv_channels[-1], out_features=out_channels
        )
        self.softmax = torch.nn.Softmax(-1)

    def __get_sparse_conv_blocks(self, in_channels: int, conv_channels: List[int]):
        channels = [in_channels] + conv_channels
        sparse_conv_blocks = []
        for i in range(len(conv_channels)):
            sparse_conv_blocks.append(
                SparseConvBlock(in_channels=channels[i], out_channels=channels[i + 1])
            )
        return sparse_conv_blocks

    def forward(self, x: ME.SparseTensor) -> torch.Tensor:

        x = self.sparse_conv_blocks(x)

        x_avg = self.global_avg_pool(x)
        x_max = self.global_max_pool(x)

        x = torch.cat([x_avg.F, x_max.F], -1).squeeze(0)

        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
