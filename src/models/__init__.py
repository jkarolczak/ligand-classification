import torch.nn as nn

from .create import transloc3d, minkloc3dv2

MODELS = {
    "transloc3d": transloc3d,
    "minkloc3dv2": minkloc3dv2
}


def create(model_type: str) -> nn.Module:
    return MODELS[model_type.lower()]()
