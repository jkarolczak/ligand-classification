import gc
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
import numpy as np
import MinkowskiEngine as ME
import skopt
import yaml
from torch.utils.data import DataLoader

import models
import log
from cfg import read_config
from data import LigandDataset, dataset_split, collation_fn

DIM_DICT = {
    "real": skopt.space.Real,
    "integer": skopt.space.Integer,
    "categorical": skopt.space.Categorical
}


def construct_dims(cfg: Dict[str, Any]) -> List[skopt.space.Dimension]:
    dims = []
    for k, v in cfg.items():
        space = v["space"]
        if space == "categorical":
            dim = DIM_DICT[space](v["values"], name=k)
        else:
            dim = DIM_DICT[space](v["lower_bound"], v["upper_bound"], name=k)
        dims.append(dim)
    return dims


def replace_in_dict(dictionary: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    for k, v in dictionary.items():
        if isinstance(v, Dict):
            return replace_in_dict(dictionary, key, value)
        if k == key:
            dictionary[k] = value
    return dictionary


if __name__ == "__main__":
    torch.manual_seed(23)

    cfg = read_config("../cfg/tune_hparams.yaml")
    train_cfg_path = cfg["train_cfg"]
    cfg["train_cfg"] = read_config(train_cfg_path)
    cfg["train_cfg"] = replace_in_dict(cfg["train_cfg"], "epochs", cfg["epochs"])
    model_cfg_path = f"../cfg/{cfg['train_cfg']['model'].lower()}.yaml"
    cfg["model_cfg"] = read_config(model_cfg_path)
    hparams_names = cfg["hparams"].keys()

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train_cfg"]["device"] != "cpu" else "cpu")
    cpu = torch.device("cpu")


    def evaluate(hparams: Tuple) -> float:
        for path, conf in [(train_cfg_path, cfg["train_cfg"]), (model_cfg_path, cfg["model_cfg"])]:
            for k, v in zip(hparams_names, hparams):
                conf = replace_in_dict(conf, k, v)
        with open(model_cfg_path, "w") as fp:
            fp.write(yaml.dump(cfg["model_cfg"]))
        train_cfg = cfg["train_cfg"]

        run = log.get_run()

        dataset = LigandDataset(train_cfg["dataset_dir"], train_cfg["dataset_file"],
                                min_size=train_cfg["dataset_min_size"], max_size=train_cfg["dataset_max_size"])

        run["config/dataset/name"] = train_cfg["dataset_dir"].split("/")[-1]
        run["config/batch_accum"] = train_cfg["accum_iter"]
        run["config/batch_size"] = train_cfg["batch_size"]

        train, test = dataset_split(dataset=dataset)

        g_train, g_test = torch.Generator(), torch.Generator()
        g_train.manual_seed(42)
        g_test.manual_seed(42)
        train_dataloader = DataLoader(dataset=train, batch_size=int(train_cfg["batch_size"]), collate_fn=collation_fn,
                                      num_workers=train_cfg["no_workers"], generator=g_train, shuffle=True)
        test_dataloader = DataLoader(dataset=test, batch_size=int(train_cfg["batch_size"]), collate_fn=collation_fn,
                                     num_workers=train_cfg["no_workers"], generator=g_test, shuffle=True)

        model = models.create(train_cfg["model"])
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]),
                                      weight_decay=float(train_cfg["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()

        log.config(run=run, model=model, criterion=criterion, optimizer=optimizer, dataset=dataset)

        accum_iter = int(train_cfg["accum_iter"])
        for e in range(train_cfg["epochs"]):
            train_dataloader.dataset.sample(e)
            model.train()
            for idx, (coords, feats, labels) in enumerate(train_dataloader):
                labels = labels.to(device=device)
                batch = ME.SparseTensor(feats, coords, device=device)
                try:
                    labels_hat = model(batch)
                    loss = criterion(labels_hat, labels) / accum_iter
                    loss.backward()
                    del labels_hat

                    if not idx % accum_iter:
                        optimizer.step()
                        optimizer.zero_grad()
                except:
                    pass

                if device == torch.device("cuda"):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                del (batch, labels)
                gc.collect()
            model.eval()
            with torch.no_grad():
                groundtruth, predictions = None, None
                for idx, (coords, feats, labels) in enumerate(test_dataloader):
                    torch.cuda.empty_cache()

                    batch = ME.SparseTensor(feats, coords, device=device)
                    preds = model(batch)

                    labels = labels.to(cpu)
                    preds = preds.to(cpu)

                    if groundtruth is None:
                        groundtruth = labels
                        predictions = preds
                    else:
                        try:
                            groundtruth = torch.cat([groundtruth, labels], 0)
                            predictions = torch.cat([predictions, preds], 0)
                        except:
                            pass

            scheduler.step()
            log.state_dict(model=model, epoch=e)
            log.epoch(run=run, preds=predictions, target=groundtruth, epoch_num=e)

        score = run["eval/accuracy"].fetch_values().max().value

        run.stop()
        return score


    dims = construct_dims(cfg["hparams"])
    best_hparams = skopt.gp_minimize(evaluate, dims, n_calls=cfg["n_calls"]).x
    line = ", ".join([str(f"{k}: {v}") for k, v in zip(hparams_names, best_hparams)])
    print(line)

    path = os.path.join('logs', 'best_hparams.txt')
    with open(path, 'a') as fp:
        fp.write(f"{datetime.now()}, {line}\n")
