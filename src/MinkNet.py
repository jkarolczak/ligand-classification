import os

from typing import List

import torch
from torch.utils.data import DataLoader

import MinkowskiEngine as ME

from utils import *
from simple_reader import LigandDataset

import gc


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


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = neptune.init(
        project="LIGANDS/LIGANDS",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMGQ1ZDQwZS05YjhlLTRmMGUtYjZjZC0yYzk0OWE4OWJmYzkifQ==",
    )

    dataset_path = "data/labels_ten_percent.csv"
    dataset = LigandDataset("data", dataset_path)

    train, test = dataset_split(dataset=dataset)

    train_dataloader = DataLoader(
        dataset=train,
        batch_size=4,
        collate_fn=collation_fn,
        num_workers=4,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        dataset=test, batch_size=4, collate_fn=collation_fn, num_workers=4, shuffle=True
    )

    model = MinkNet(
        conv_channels=[64, 64, 128, 128, 256, 256, 512, 512],
        in_channels=1,
        out_channels=dataset.labels[0].shape[0],
    )
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    epochs = 100

    log_config(
        run=run, model=model, criterion=criterion, optimizer=optimizer, dataset=dataset
    )

    for e in range(epochs):
        model.train()
        for idx, (coords, feats, labels) in enumerate(train_dataloader):

            labels = labels.to(device=device)
            batch = ME.SparseTensor(feats, coords, device=device)

            optimizer.zero_grad()
            labels_hat = model(batch)
            loss = criterion(labels_hat, labels)
            loss.backward()
            optimizer.step()
            if device == torch.device("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            del batch, labels, labels_hat
            gc.collect()

        model.eval()
        with torch.no_grad():
            groundtruth, predictions = None, None
            for idx, (coords, feats, labels) in enumerate(test_dataloader):
                torch.cuda.empty_cache()
                batch = ME.SparseTensor(feats, coords, device=device)
                preds = model(batch)
                if groundtruth is None:
                    groundtruth = labels
                    predictions = preds
                else:
                    try:
                        groundtruth = torch.cat([groundtruth, labels], 0)
                        predictions = torch.cat([predictions, preds], 0)
                    except:
                        pass

        log_state_dict(run=run, model=model)
        log_epoch(run=run, preds=predictions, target=groundtruth)

    run.stop()
