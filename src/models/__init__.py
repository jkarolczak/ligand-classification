import torch.nn as nn

from models.create import transloc3d, minkloc3dv2, pointnet

MODELS = {
    "transloc3d": transloc3d,
    "minkloc3dv2": minkloc3dv2,
    "pointnet": pointnet
}


def create(model_type: str) -> nn.Module:
    return MODELS[model_type.lower()]()
