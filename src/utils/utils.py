import os
from datetime import datetime
from typing import Tuple
from copy import deepcopy

import torch
import torchmetrics
import MinkowskiEngine as ME
import numpy as np
import neptune.new as neptune
from sklearn.model_selection import train_test_split

from utils.simple_reader import LigandDataset

def collation_fn(blobel):
    """
    Implements collation function for batching the LigandsDataset using
    `torch.utils.data.DataLoader`

    :param blobel: tuple (coordinates, features, labels); blob+label => blobel; all credit to Witek T.
    """
    coords_batch, feats_batch, labels_batch = [], [], []

    for (coords, feats, label) in blobel:
        coords_batch.append(coords)
        feats_batch.append(feats)
        labels_batch.append(label)

    coords_batch = ME.utils.batched_coordinates(coords_batch)
    feats_batch = torch.tensor(np.concatenate(feats_batch, 0), dtype=torch.float32)
    labels_batch = torch.tensor(np.vstack(labels_batch), dtype=torch.float32)

    return coords_batch, feats_batch, labels_batch


def dataset_split(
    dataset: LigandDataset, train_size: float = 0.75, stratify: bool = True
) -> Tuple[LigandDataset, LigandDataset]:
    """
    Splits dataset into train and test sets.

    :param dataset: dataset of type LigandDataset
    :param train_size: the proportion of the dataset to include in the test split, must be float in [0.0, 1.0]
    :param stratify: boolean, if train and test sets should have the same proportions between classes like dataset
    :return Tuple[LigandDataset, LigandDataset]: train and test sets
    """
    files = dataset.files
    labels = dataset.labels

    stratify = labels if stratify else None

    files_train, files_test, labels_train, labels_test = train_test_split(
        files, labels, train_size=train_size, stratify=stratify
    )

    train = deepcopy(dataset)
    test = deepcopy(dataset)
    train.files, train.labels = files_train, labels_train
    test.files, test.labels = files_test, labels_test

    return (train, test)


def log_state_dict(
    model: torch.nn.Module,
    epoch: int
) -> None:
    """
    Serialize model weigts.
    :param model: torch.nn.Module to save its weights
    :param directory: path to the directory to log experiment results
    :param epoch: int describing epoch
    """
    models_path = os.path.join('logs', 'models')
    os.makedirs(models_path, exist_ok=True)
    time = str(datetime.now()).replace(' ', '-')
    file_name = f'{time}-epoch-{epoch}.pt'
    file_path = os.path.join(models_path, file_name)
    torch.save(model.state_dict(), file_path)


def log_config(
    run: neptune.Run,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.nn.Module,
    dataset: LigandDataset,
) -> None:
    """
    :param run: neptune.Run
    :param model: torch.nn.Module
    :param criterion: torch.nn.Module
    :param optimizer: torch.nn.Module
    :param datset:
    """

    run["config/model/class"] = type(model).__name__
    run["config/model/structure"] = str(model)

    run["config/dataset/instances"] = dataset.labels.shape[0]
    run["config/dataset/labels"] = dataset.labels.shape[1]

    run["config/criterion/class"] = type(criterion).__name__
    run["config/optimizer/class"] = type(optimizer).__name__
    run["config/optimizer/learning_rate"] = optimizer.__dict__["defaults"]["lr"]
    run["config/optimizer/weight_decay"] = optimizer.__dict__["defaults"][
        "weight_decay"
    ]


def log_epoch(run: neptune.Run, preds: torch.Tensor, target: torch.Tensor) -> None:
    """
    :param run: neptune.Run, object to log
    :param preds: torch.tensor, labels predictions (sotfmax output)
    :param target: torch.tensor, target one-hot encoded labels
    :returns: tuple of metrics
    """
    num_classes = target.shape[1]

    cross_entropy = torch.nn.functional.cross_entropy(preds, target)

    target = torch.argmax(target, axis=1)

    accuracy = torchmetrics.functional.accuracy(preds, target)
    top5_accuracy = (
        torchmetrics.functional.accuracy(preds, target, top_k=5)
        if num_classes > 5
        else 1
    )
    top10_accuracy = (
        torchmetrics.functional.accuracy(preds, target, top_k=10)
        if num_classes > 10
        else 1
    )
    top20_accuracy = (
        torchmetrics.functional.accuracy(preds, target, top_k=20)
        if num_classes > 20
        else 1
    )

    macro_recall = torchmetrics.functional.recall(
        preds, target, average="macro", num_classes=num_classes
    )

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
