import torch.nn as nn

from models.create import transloc3d, minkloc3dv2  # , pointnet, riconv2

MODELS = {
    "transloc3d": transloc3d,
    "minkloc3d": minkloc3dv2,
    # "pointnet": pointnet,
    # "riconv2": riconv2
}


def create(model_type: str) -> nn.Module:
    return MODELS[model_type.lower()]()
