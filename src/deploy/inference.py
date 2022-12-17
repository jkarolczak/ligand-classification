from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
from jinja2 import Environment, PackageLoader, select_autoescape

_ligand_dict = None

def ligand_dict():
    global _ligand_dict
    if _ligand_dict is None:
        with open("/app/src/deploy/ligand_dict.yml", "r") as fp:
            _ligand_dict = yaml.safe_load(fp)
    return _ligand_dict

def render_table(classes: List[Tuple[int, str, float]]) -> str:
    env = Environment(loader=PackageLoader("deploy", "templates"), autoescape=select_autoescape())
    template = env.get_template("table.html")
    table = template.render(rows=classes)
    return table


def predict(blob: np.ndarray) -> str:
    """

    """
    prob = np.random.random(10)
    prob = prob / np.sum(prob)
    df = pd.DataFrame(
        {
            "Class": ["HEM-like or OXY-like_HEM-like", "ZN-like_ZN-like", "2CV", "A2G", "2PE", "3BV", "1PE-like", "ADN-like", "ADP-like", "BCL"],
            "Probability": prob
        }
    ).sort_values(by="Probability", ascending=False).reset_index().drop(columns="index")
    table = render_table([(i + 1, ligand_dict()[row["Class"]], round(row["Probability"], 2)) for i, row in df.iterrows()])
    return table
