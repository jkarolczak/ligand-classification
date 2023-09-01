# Author: Jacek Komorowski
# Warsaw University of Technology

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

import models.sparse.MinkLoc3Dv2.models.minkfpn
from .layers.pooling_wrapper import PoolingWrapper


class MinkLoc(torch.nn.Module):
    def __init__(self, backbone: nn.Module, pooling: PoolingWrapper, normalize_embeddings: bool = False):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.normalize_embeddings = normalize_embeddings
        self.stats = {}

    def forward(self, batch, nears):
        x: models.sparse.MinkLoc3Dv2.models.minkfpn.MinkFPN = self.backbone(batch)
        # x is (num_points, n_features) tensor
        assert x.shape[1] == self.pooling.in_dim, f'Backbone output tensor has: {x.shape[1]} channels. ' \
                                                  f'Expected: {self.pooling.in_dim}'
        x: models.sparse.MinkLoc3Dv2.models.layers.pooling_wrapper.PoolingWrapper = self.pooling(x)
        print(x.shape, nears.shape)
        if hasattr(self.pooling, 'stats'):
            self.stats.update(self.pooling.stats)

        assert x.dim() == 2, f'Expected 2-dimensional tensor (batch_size,output_dim). Got {x.dim()} dimensions.'
        assert x.shape[1] == self.pooling.output_dim, f'Output tensor has: {x.shape[1]} channels. ' \
                                                      f'Expected: {self.pooling.output_dim}'
        if self.normalize_embeddings:
            x = F.normalize(x, dim=1)
        return x

    def print_info(self):
        print('Model class: MinkLoc')
        n_params = sum([param.nelement() for param in self.parameters()])
        print(f'Total parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print(f'Backbone: {type(self.backbone).__name__} #parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print(f'Pooling method: {self.pooling.pool_method}   #parameters: {n_params}')
        print('# channels from the backbone: {}'.format(self.pooling.in_dim))
        print('# output channels : {}'.format(self.pooling.output_dim))
        print(f'Embedding normalization: {self.normalize_embeddings}')
