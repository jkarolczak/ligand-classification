import importlib

from cfg import read_config
from models.contiguous.models import PointNet, Classifier
from models.sparse import MinkLoc3Dv2
from models.sparse.MinkLoc3Dv2.misc.utils import ModelParams
from models.sparse.MinkLoc3Dv2.models.model_factory import model_factory
from models.sparse.TransLoc3D import create_model, Config
from models.sparse.TransLoc3D.transloc3d.model import TransLoc3D
from models.contiguous.riconv2.riconv2_cls import get_model


def transloc3d() -> TransLoc3D:
    transloc_params = read_config("../cfg/models/transloc3d.yaml")
    model = create_model("TransLoc3D", Config(transloc_params))
    return model


def minkloc3dv2() -> MinkLoc3Dv2:
    minkloc_params = ModelParams('../cfg/models/minkloc3dv2.yaml')
    model = model_factory(minkloc_params)
    return model


def pointnet() -> PointNet:
    pointnet_params = read_config("../cfg/models/pointnet.yaml")
    num_classes = pointnet_params.pop("num_classes")
    pn = PointNet(**pointnet_params)
    model = Classifier(pn, num_classes=num_classes)
    return model


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def riconv2():
    riconv2_params = read_config("../cfg/models/riconv2.yaml")
    num_classes = riconv2_params.pop("num_classes")
    use_normals = riconv2_params.pop("use_normals")
    classifier = get_model(num_classes, 2, normal_channel=use_normals)
    classifier.apply(inplace_relu)
    return classifier
