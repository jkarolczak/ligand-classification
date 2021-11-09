import os
from datetime import date, datetime
from typing import Tuple
from copy import deepcopy

import numpy as np
from numpy.core.numeric import cross

import torch
import torchmetrics
import MinkowskiEngine as ME

from simple_reader import LigandDataset



from sklearn.model_selection import train_test_split

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
    dataset: LigandDataset,
    train_size: float = 0.75,
    stratify: bool = True
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
        files, 
        labels,
        train_size=train_size,
        stratify=stratify
    )

    train = deepcopy(dataset)
    test = deepcopy(dataset)
    train.files, train.labels = files_train, labels_train
    test.files, test.labels = files_test, labels_test

    return (train, test)

def save_state_dict(
    model: torch.nn.Module,
    directory: str,
    epoch: int
) -> None:
    """
    Serialize model weigts.
    :param model: torch.nn.Module to save its weights
    :param directory: path to the directory to log experiment results
    :param epoch: int describing epoch
    """
    models_path = os.path.join(directory, 'models')
    os.makedirs(models_path, exist_ok=True)
    time = str(datetime.now()).replace(' ', '-')
    file_name = f'{time}-epoch-{epoch}.pt'
    file_path = os.path.join(models_path, file_name)
    torch.save(model.state_dict(), file_path)

def write_log_header(
    directory: str,
    file_name: str = 'file.log'
) -> None:
    """
    :param directory: path to the directory to log experiment results
    :param file_name: name of the file to log metrics in it
    """
    file_path = os.path.join(directory, file_name)

    with open(file_path, 'a') as fp:
        fp.write(
            ','.join(['epoch', 'time', 'accuracy', 'top5_accuracy', 'top10_accuracy', 
            'top20_accuracy', 'macro_recall', 'micro_recall', 'micro_precision', 
            'micro_f1', 'cohen_kappa', 'cross_entropy']) + '\n'
        )

def log_epoch(
    preds: torch.Tensor, 
    target: torch.Tensor,
    directory: str,
    epoch: int,
    file_name: str = 'file.log'
) -> None:
    """
    :param preds: torch.tensor, labels predictions (sotfmax output)
    :param target: torch.tensor, target one-hot encoded labels
    :param directory: path to the directory to log experiment results
    :param epoch: int describing epoch
    :param file_name: name of the file to log metrics in it
    """
    file_path = os.path.join(directory, file_name)

    with open(file_path, 'a') as fp:
        fp.write(
            ','.join(
                [
                    str(epoch), 
                    str(datetime.now()).replace(' ', '-'), 
                    *[str(float(m)) for m in  compute_metrics(preds, target)]
                ]
            ) + '\n'
        )

def compute_metrics(
    preds: torch.Tensor, 
    target: torch.Tensor
) -> Tuple[torch.Tensor]:
    """
    :param preds: torch.tensor, labels predictions (sotfmax output)
    :param target: torch.tensor, target one-hot encoded labels
    :returns: tuple of metrics
    """
    num_classes = target.shape[1]

    target = torch.argmax(target, axis = 1)

    accuracy = torchmetrics.functional.accuracy(preds, target)
    top5_accuracy = torchmetrics.functional.accuracy(preds, target, top_k = 5)
    top10_accuracy = torchmetrics.functional.accuracy(preds, target, top_k = 10)
    top20_accuracy = torchmetrics.functional.accuracy(preds, target, top_k = 20)

    macro_recall = torchmetrics.functional.recall(preds, target, average = 'macro', num_classes=num_classes)

    micro_recall = torchmetrics.functional.recall(preds, target, average = 'micro')
    micro_precision = torchmetrics.functional.precision(preds, target, average = 'micro')
    micro_f1 = torchmetrics.functional.f1(preds, target, average = 'micro')

    cohen_kappa = torchmetrics.functional.cohen_kappa(torch.argmax(preds, axis = 1), target, num_classes = num_classes)

    cross_entropy = torch.nn.functional.cross_entropy(preds, target)

    return (
        accuracy,
        top5_accuracy,
        top10_accuracy,
        top20_accuracy,
        macro_recall,
        micro_recall,
        micro_precision,
        micro_f1,
        cohen_kappa,
        cross_entropy
    )