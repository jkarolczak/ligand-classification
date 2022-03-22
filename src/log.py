import os
from datetime import datetime
from typing import Union, List

import neptune.new as neptune
import torch
import torchmetrics
import yaml

from data import LigandDataset


def get_run(file: str = "../cfg/neptune.yaml", tags: Union[List[str], None] = None) -> neptune.Run:
    print(os.getcwd())
    with open(file) as fp:
        config = yaml.safe_load(fp)
    run = neptune.init(
        project=config["project"],
        api_token=config["api_token"],
        mode="async" if config["debug"] is False else "debug",
        tags=tags if tags is not None else []
    )
    return run


def state_dict(model: torch.nn.Module, epoch: int) -> None:
    models_path = os.path.join('logs', 'models')
    os.makedirs(models_path, exist_ok=True)
    time = str(datetime.now()).replace(' ', '-')
    file_name = f'{time}-epoch-{epoch}.pt'
    file_path = os.path.join(models_path, file_name)
    torch.save(model.state_dict(), file_path)


def config(
        run: neptune.Run,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.nn.Module,
        dataset: LigandDataset,
) -> None:
    run["config/model/class"] = type(model).__name__
    run["config/model/structure"] = str(model)

    run["config/dataset/instances"] = dataset.labels.shape[0]
    run["config/dataset/labels"] = dataset.labels.shape[1]

    run["config/criterion/class"] = type(criterion).__name__
    run["config/optimizer/class"] = type(optimizer).__name__
    run["config/optimizer/learning_rate"] = optimizer.__dict__["defaults"]["lr"]
    run["config/optimizer/weight_decay"] = optimizer.__dict__["defaults"]["weight_decay"]


def epoch(run: neptune.Run, preds: torch.Tensor, target: torch.Tensor,
          epoch_num: int) -> None:
    num_classes = target.shape[1]

    cross_entropy = torch.nn.functional.cross_entropy(preds, target)

    target = torch.argmax(target, axis=1)

    accuracy = torchmetrics.functional.accuracy(preds, target)
    top5_accuracy = (torchmetrics.functional.accuracy(preds, target, top_k=5) if num_classes > 5 else 1)
    top10_accuracy = (torchmetrics.functional.accuracy(preds, target, top_k=10) if num_classes > 10 else 1)
    top20_accuracy = (torchmetrics.functional.accuracy(preds, target, top_k=20) if num_classes > 20 else 1)

    macro_recall = torchmetrics.functional.recall(preds, target, average="macro", num_classes=num_classes)

    micro_recall = torchmetrics.functional.recall(preds, target, average="micro")
    micro_precision = torchmetrics.functional.precision(preds, target, average="micro")
    micro_f1 = torchmetrics.functional.f1(preds, target, average="micro")

    cohen_kappa = torchmetrics.functional.cohen_kappa(
        preds, target, num_classes=num_classes
    )

    run["eval/accuracy"].log(accuracy)
    run["eval/top5_accuracy"].log(top5_accuracy)
    run["eval/top10_accuracy"].log(top10_accuracy)
    run["eval/top20_accuracy"].log(top20_accuracy)
    run["eval/macro_recall"].log(macro_recall)
    run["eval/micro_recall"].log(micro_recall)
    run["eval/micro_precision"].log(micro_precision)
    run["eval/micro_f1"].log(micro_f1)
    run["eval/cohen_kappa"].log(cohen_kappa)
    run["eval/cross_entropy"].log(cross_entropy)

    time = str(datetime.now()).replace(' ', '-')
    line = f"{time},{epoch_num}"
    for metric in [accuracy, top5_accuracy, top10_accuracy, top20_accuracy, macro_recall, micro_recall, micro_precision,
                   micro_f1, cohen_kappa, cross_entropy]:
        line += f"{metric},"

    os.makedirs('logs', exist_ok=True)
    path = os.path.join('logs', 'log.txt')

    if not os.path.isfile(path):
        with open(path, 'a') as fp:
            fp.write(
                "time,epoch,accuracy,top5_accuracy,top10_accuracy,top20_accuracy,macro_recall,micro_recall,"
                "micro_precision,micro_f1,cohen_kappa,cross_entropy\n")

    with open(path, 'a') as fp:
        fp.write(line + '\n')