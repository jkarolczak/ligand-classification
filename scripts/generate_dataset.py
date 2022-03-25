import os
from typing import Dict

import numpy as np

from cfg import read_config
from pipeline import Pipeline


def main(cfg: Dict) -> None:
    input_dir = cfg["input_dir"]
    files = os.listdir(input_dir)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    transformation_pipeline = Pipeline(cfg["steps"])


    for idx, f_name in enumerate(files):
        input_path = os.path.join(input_dir, f_name)
        blob = np.load(input_path)["blob"]
        blob = transformation_pipeline.preprocess(blob)
        output_path = os.path.join(output_dir, f_name)
        np.savez(output_path, blob=blob)
        if idx == 4:
            break


if __name__ == "__main__":
    config = read_config("../cfg/generate_dataset.yaml")
    main(config)
