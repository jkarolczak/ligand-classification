import os
from typing import Dict, Tuple, Union

import neptune.new as neptune
import numpy as np  # noqa
import yaml

from cfg import read_config


def get_last_run_epoch(config: Dict[str, Union[str, bool]]) -> Tuple[int, int]:
    project = neptune.init_project(name=config["project"], api_token=config["api_token"], mode="read-only")
    id = project.fetch_runs_table().to_pandas().iloc[0]["sys/id"]
    run = neptune.init_run(project=config["project"], api_token=config["api_token"], mode="read-only", with_id=id)
    epoch = run["eval/top5_accuracy"].fetch_values()["value"].argmax()
    return id, epoch


columns = ["dataset_dir", "batch_size", "lr", "accum_iter", "dataset_min_size", "dataset_max_size"]
values = [
    ["../data/blobs_uniform_2000_max", 128, 1e-3, 4, None, None],
    ["../data/blobs_uniform_2000_max", 128, 1e-3, 1, None, None],
    ["../data/blobs_uniform_2000_max", 128, 1e-2, 1, None, None],
    ["../data/blobs_uniform_2000_max", 128, 1e-2, 4, None, None],
    ["../data/blobs_uniform_2000_max", 128, 1e-3, 4, 2000, None],
    ["../data/blobs_uniform_2000_max", 128, 1e-3, 4, None, 50000],
    ["../data/blobs_uniform_2000_max", 128, 1e-3, 4, 2000, 50000],
]

if __name__ == "__main__":
    neptune_cfg = read_config("../cfg/neptune.yaml")
    train_config = read_config("../cfg/train.yaml")
    eval_config = read_config("../cfg/eval.yaml")

    for vals in values:
        for key, val in zip(columns, vals):
            train_config[key] = val
            if key in eval_config.keys():
                eval_config[key] = val

        with open("../cfg/train.yaml", "w") as fp:
            yaml.dump(train_config, fp)
        os.system("python ./train_sparse.py")

        run_id, epoch = get_last_run_epoch(config=neptune_cfg)
        eval_config["model_name"] = train_config["model"]
        eval_config["model_run_id"] = str(run_id)
        eval_config["model_epoch"] = int(epoch)
        with open("../cfg/eval.yaml", "w") as fp:
            yaml.dump(eval_config, fp)

        os.system("python ./eval.py")
