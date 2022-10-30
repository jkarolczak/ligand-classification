from typing import Dict, Tuple, Union

import neptune.new as neptune
import yaml


def get_last_run_epoch(config: Dict[str, Union[str, bool]]) -> Tuple[int, int]:
    project = neptune.init_project(name=config["project"], api_token=config["api_token"], mode="read-only")
    id = project.fetch_runs_table().to_pandas().iloc[0]["sys/id"]
    run = neptune.init_run(project=config["project"], api_token=config["api_token"], mode="read-only", with_id=id)
    epoch = run["eval/top5_accuracy"].fetch_values()["value"].argmax()
    return id, epoch


def get_model_from_run_epoch(model_name: str, run_id: int, epoch: int, config: Dict[str, Union[str, bool]]) -> str:
    full_model_name = f"LIGANDS-{model_name.upper()}"
    model = neptune.init_model(project=config["project"], api_token=config["api_token"], with_id=full_model_name)
    df = model.fetch_model_versions_table(columns=["run", "epoch"]).to_pandas()
    model_id = df[(df["run"] == run_id) & (df["epoch"] == epoch)].iloc[0]["sys/id"]
    return model_id


def read_neptune_cfg(file: str = "../cfg/neptune.yaml") -> Dict[str, Union[str, bool]]:
    with open(file) as fp:
        config = yaml.safe_load(fp)
    return config


if __name__ == "__main__":
    neptune_cfg = read_neptune_cfg()
    run_id, epoch = get_last_run_epoch(config=neptune_cfg)
    model_id = get_model_from_run_epoch("minkloc3d", run_id, epoch, neptune_cfg)
    print(model_id)
