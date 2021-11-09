from math import exp
import os

from typing import List

import torch
from torch.utils.data import DataLoader

import MinkowskiEngine as ME

from utils import *
from simple_reader import LigandDataset

class SparseConvBlock(ME.MinkowskiNetwork):
    """A class to represent a single sparse convolution block."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimensions: int = 3,
        conv_kernel: int = 3,
        pooling_kernel: int = 2,
        activation = ME.MinkowskiSigmoid,
        pooling = ME.MinkowskiMaxPooling,
    ):
        """
        :param in_channels: number of channels in input 
        :param out_channels: number of channels in output, equal to number of classes
        :param conv_kernel: integer describing convolution kernel size - it is 
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
            kernel_size=conv_kernel,
            dimension=self.D
        )
        self.activation = activation()
        self.pooling = pooling(
            kernel_size=pooling_kernel, 
            dimension=self.D
        )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        x = self.pooling(x)
        return x

class PointNet(ME.MinkowskiNetwork):
    """A class to represent PointNet neural network."""
    def __init__(
        self,
        conv_kernels: List[int],
        in_channels: int,
        out_channels: int,
        dimensions: int = 3
    ):
        """
        :param conv_kernels: list of integers describing consecutive convolution
        kernels sizes - it is assumed that convolution kernel is a cube, all 
        kernel dimensions are equal
        :param in_channels: number of channels in input 
        :param out_channels: number of channels in output, equal to number of classes
        :param dimensions: number of dimensions of input
        """
        ME.MinkowskiNetwork.__init__(self, dimensions)
        self.sparse_conv_blocks = torch.nn.Sequential(
            *self.__get_sparse_conv_blocks(
                in_channels=in_channels,
                conv_kernels=conv_kernels
            )
        )

        self.global_sum_pool = ME.MinkowskiGlobalSumPooling()
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.linear1 = torch.nn.Linear(
            in_features=3 * conv_kernels[-1],
            out_features=conv_kernels[-1]
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(
            in_features=conv_kernels[-1], 
            out_features=out_channels
        )
        self.softmax = torch.nn.Softmax(-1)

    def __get_sparse_conv_blocks(
        self, 
        in_channels: int,
        conv_kernels: List[int]
    ):
        channels = [in_channels] + conv_kernels
        sparse_conv_blocks = []
        for i in range(len(conv_kernels)):
            sparse_conv_blocks.append(
                SparseConvBlock(
                    in_channels = channels[i],
                    out_channels = channels[i + 1]
                )
            )
        return sparse_conv_blocks

    def forward(
        self, 
        x: ME.SparseTensor
    ) -> torch.Tensor:
        
        x = self.sparse_conv_blocks(x)

        x_sum = self.global_sum_pool(x)
        x_avg = self.global_avg_pool(x)
        x_max = self.global_max_pool(x)

        x = torch.cat([x_sum.F, x_avg.F, x_max.F], -1).squeeze(0)
        
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

if __name__ == '__main__':

    experiment_dir = 'experiment'
    os.makedirs(experiment_dir, exist_ok=True)
    write_log_header(experiment_dir)
    
    # TODO: choose proper architecture of the NN

    model = PointNet(
        conv_kernels = [8, 32, 128, 512],
        in_channels = 1,
        out_channels = 44
    )
    dataset = LigandDataset('data', 'data/labels_ten_percent.csv')

    train, test = dataset_split(dataset=dataset)

    train_dataloader = DataLoader(
        dataset=train, 
        batch_size=16, 
        collate_fn=collation_fn,
        num_workers=4,
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test, 
        batch_size=16, 
        collate_fn=collation_fn,
        num_workers=4,
        shuffle=True
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-2
    )
    epochs = 10

    for e in range(epochs):
        model.train()
        for idx, (coords, feats, labels) in enumerate(train_dataloader):
            if idx >= 3: break
                
            batch = ME.SparseTensor(feats, coords)
            optimizer.zero_grad()
            labels_hat = model(batch)
            loss = criterion(labels_hat, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        groundtruth, predictions = None, None
        for idx, (coords, feats, labels) in enumerate(test_dataloader):
            if idx >= 3: break

            batch = ME.SparseTensor(feats, coords)
            preds = model(batch)
            if groundtruth is None:
                groundtruth = labels
                predictions = preds
            else:
                groundtruth = torch.cat([groundtruth, labels], -1)
                predictions = torch.cat([predictions, preds], -1)
        
        save_state_dict(
            model=model,
            directory=experiment_dir,
            epoch=e
        )
        log_epoch(
            preds=predictions, 
            target=groundtruth, 
            directory=experiment_dir, 
            epoch=e
        )

        # TODO: early stopping
