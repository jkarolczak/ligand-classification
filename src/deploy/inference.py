import pickle
from typing import List, Tuple

import numpy as np
import MinkowskiEngine as ME
import pandas as pd
import torch
import yaml
from jinja2 import Environment, PackageLoader, select_autoescape

from data import SparseDataset
from models.sparse.MinkLoc3Dv2.misc.utils import ModelParams
from models.sparse.MinkLoc3Dv2.models.model_factory import model_factory

_ligand_dict = None
_encoder = None


def ligand_dict():
    global _ligand_dict
    if _ligand_dict is None:
        with open("/app/src/deploy/ligand_dict.yml", "r") as fp:
            _ligand_dict = yaml.safe_load(fp)
    return _ligand_dict


def encoder():
    global _encoder
    if _encoder is None:
        with open("/app/src/deploy/encoder.pkl", "rb") as fp:
            _encoder = pickle.load(fp)
    return _encoder


def render_table(classes: List[Tuple[int, str, float]]) -> str:
    env = Environment(loader=PackageLoader("deploy", "templates"), autoescape=select_autoescape())
    template = env.get_template("table.html")
    table = template.render(rows=classes)
    return table


def blob_to_me_tensor(blob: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    blob = torch.tensor(blob, dtype=torch.float32)
    coords, feats = SparseDataset._get_coords_feats(blob)
    coords = ME.utils.batched_coordinates([coords])
    me_tensor = ME.SparseTensor(features=feats, coordinates=coords, device=torch.device("cpu"))
    return me_tensor


def vals_to_probs(vals: torch.Tensor) -> torch.Tensor:
    if any(vals < 0.0):
        vals = torch.sigmoid(vals)
    vals = torch.nn.functional.softmax(vals, 0)
    vals = vals.detach().numpy()
    return vals


def idx_to_cls(idx: int) -> str:
    arr = np.zeros((1, 219))
    arr[0, idx] = 1.0
    cls = encoder().inverse_transform(arr)[0]
    return cls


def indices_to_cls(indices: torch.Tensor) -> List[str]:
    return [idx_to_cls(idx) for idx in indices]


def raw_pred_to_top10_dataframe(predictions: torch.Tensor) -> pd.DataFrame:
    top_k = torch.topk(predictions, 10)
    prob = vals_to_probs(top_k.values)
    cls = indices_to_cls(top_k.indices)
    df = pd.DataFrame(
        {
            "Class": cls,
            "Probability": prob
        }
    ).sort_values(by="Probability", ascending=False).reset_index().drop(columns="index")
    return df


def raw_pred_to_dataframe_probabilities(predictions: torch.Tensor) -> pd.DataFrame:
    prob, indices = torch.sort(predictions)
    prob = vals_to_probs(prob)
    cls = indices_to_cls(indices)
    df = pd.DataFrame(
        {
            "class": cls,
            "probability": prob
        }
    ).sort_values(by="probability", ascending=False).reset_index().drop(columns="index")
    return df


def predict(blob: np.ndarray, model) -> pd.DataFrame:
    """

    """
    me_tensor = blob_to_me_tensor(blob)
    pred = model(me_tensor).squeeze(0)
    return pred


def load_model():
    minkloc_params = ModelParams("cfg/models/minkloc3dv2.yaml")
    model = model_factory(minkloc_params)
    state_dict = torch.load("src/deploy/model.pt", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model = model.eval()
    return model
