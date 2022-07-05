from . import MinkLoc3Dv2
from .MinkLoc3Dv2.misc.utils import ModelParams
from .MinkLoc3Dv2.models.model_factory import model_factory
from .TransLoc3D import create_model, Config
from .TransLoc3D.transloc3d.model import TransLoc3D
from cfg import read_config


def transloc3d() -> TransLoc3D:
    cfg = read_config("../cfg/transloc3d.yaml")
    model = create_model("TransLoc3D", Config(cfg))
    return model


def minkloc3dv2() -> MinkLoc3Dv2:
    minkloc_params = ModelParams('../cfg/minkloc3dv2.yaml')
    model = model_factory(minkloc_params)
    return model
