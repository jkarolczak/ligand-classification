import numpy as np

import torch
import MinkowskiEngine as ME

from simple_reader import LigandDataset

from typing import Tuple
from copy import deepcopy

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
