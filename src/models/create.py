import sys

# sys.path.append("../")
from .TransLoc3D import create_model, Config
from .TransLoc3D.transloc3d.model import TransLoc3D
from cfg import read_config


def transloc3d() -> TransLoc3D:
    cfg = read_config("../cfg/transloc3d.yaml")
    model = create_model("TransLoc3D", Config(cfg))
    return model
