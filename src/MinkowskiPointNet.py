import MinkowskiEngine as ME
import torch.nn as nn


class MinkowskiPointNet(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, embedding_channel=1024, dimension=3):
        ME.MinkowskiNetwork.__init__(self, dimension)
        self.net = nn.Sequential(
            nn.Sequential(
                ME.MinkowskiLinear(in_channels, 64, bias=False),
                ME.MinkowskiBatchNorm(64),
                ME.MinkowskiReLU(),
            ),
            nn.Sequential(
                ME.MinkowskiLinear(64, 64, bias=False),
                ME.MinkowskiBatchNorm(64),
                ME.MinkowskiReLU(),
            ),
            nn.Sequential(
                ME.MinkowskiLinear(64, 64, bias=False),
                ME.MinkowskiBatchNorm(64),
                ME.MinkowskiReLU(),
            ),
            nn.Sequential(
                ME.MinkowskiLinear(64, 128, bias=False),
                ME.MinkowskiBatchNorm(128),
                ME.MinkowskiReLU(),
            ),
            nn.Sequential(
                ME.MinkowskiLinear(128, embedding_channel, bias=False),
                ME.MinkowskiBatchNorm(embedding_channel),
                ME.MinkowskiReLU(),
            ),
            ME.MinkowskiGlobalMaxPooling(),
            nn.Sequential(
                ME.MinkowskiLinear(embedding_channel, 512, bias=False),
                ME.MinkowskiBatchNorm(512),
                ME.MinkowskiReLU(),
            ),
            ME.MinkowskiDropout(),
            ME.MinkowskiLinear(512, out_channels, bias=True),
        )

    def forward(self, x: ME.TensorField):
        # Press F to pay respect
        return self.net(x).F
