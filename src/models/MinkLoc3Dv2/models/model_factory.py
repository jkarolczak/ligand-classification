# Warsaw University of Technology

import torch.nn as nn

from .minkloc import MinkLoc
from ..misc.utils import ModelParams
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from .layers.eca_block import ECABasicBlock
from .minkfpn import MinkFPN
from .layers.pooling_wrapper import PoolingWrapper


def model_factory(model_params: ModelParams):
    in_channels = model_params.input_dim   
    if model_params.model == 'MinkLoc3Dv2':
        block_module = create_resnet_block(model_params.block)
        backbone = MinkFPN(in_channels=in_channels, out_channels=model_params.feature_size,
                           num_top_down=model_params.num_top_down, conv0_kernel_size=model_params.conv0_kernel_size,
                           block=block_module, layers=model_params.layers, planes=model_params.planes)
        pooling = PoolingWrapper(pool_method=model_params.pooling, in_dim=model_params.feature_size,
                                 output_dim=model_params.output_dim)
        model = MinkLoc(backbone=backbone, pooling=pooling, normalize_embeddings=model_params.normalize_embeddings)
    else:
        raise NotImplementedError('Model not implemented: {}'.format(model_params.model))

    return model


def create_resnet_block(block_name: str) -> nn.Module:
    if block_name == 'BasicBlock':
        block_module = BasicBlock
    elif block_name == 'Bottleneck':
        block_module = Bottleneck
    elif block_name == 'ECABasicBlock':
        block_module = ECABasicBlock
    else:
        raise NotImplementedError('Unsupported network block: {}'.format(block_name))

    return block_module
