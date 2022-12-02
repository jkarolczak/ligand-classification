import numpy as np
import pandas as pd


def predict(blob: np.ndarray) -> pd.DataFrame:
    """

    """
    prob = np.random.random(10)
    prob = prob / np.sum(prob)
    return pd.DataFrame(
        {
            "Class": ["HOH-like", "13P-like", "2CV", "A2G", "2PE", "3BV", "1PE-like", "ADN-like", "ADP-like", "BCL"],
            "Probability": prob
        }
    ).sort_values(by="Probability", ascending=False).reset_index().drop(columns="index")
