from .transloc3d.model import TransLoc3D
from .transloc3d_cfg import model_cfg, model_type
from .utils_config import Config

def create_model(model_type, model_cfg):
    type2model = dict(
        TransLoc3D=TransLoc3D,
    )
    return type2model[model_type](model_cfg)
