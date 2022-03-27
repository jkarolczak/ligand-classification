import multiprocessing as mp
import os
from datetime import datetime

import numpy as np

from cfg import read_config
from pipeline import Pipeline


def main(x) -> None:
    idx, f_name = x
    input_path = os.path.join(input_dir, f_name)
    blob = np.load(input_path)["blob"]
    blob = transformation_pipeline.preprocess(blob)
    output_path = os.path.join(output_dir, f_name)
    np.savez_compressed(output_path, blob=blob)
    if not idx % 100:
        with open("../log/generate_dataset.txt", "a") as fp:
            fp.write(f"{idx + cfg['start']},{datetime.now()}\n")


if __name__ == "__main__":
    cfg = read_config("../cfg/generate_dataset.yaml")
    input_dir = cfg["input_dir"]
    files = os.listdir(input_dir)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    transformation_pipeline = Pipeline(cfg["steps"])

    start = cfg["start"]
    end = cfg["end"]
    files = files[start:end]
    idxs = list(range(len(files)))

    tot = [(idx, f_name) for idx, f_name in zip(idxs, files)]
    with mp.Pool() as pool:
        results = pool.map(main, tot, chunksize=1)
