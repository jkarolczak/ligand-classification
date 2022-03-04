import torch.nn as nn

from .create import transloc3d

MODELS = {
    "transloc3d": transloc3d
}


def create(model_type: str) -> nn.Module:
    return MODELS[model_type.lower()]()
