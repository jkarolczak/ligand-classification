import torch
import MinkowskiEngine as ME

from simple_reader import LigandDataset

from typing import Tuple
from copy import deepcopy

from sklearn.model_selection import train_test_split

def to_minkowski_tensor(
    batch: torch.tensor
) -> ME.SparseTensor: 
    """
    Converts torch tensor containing blob or batch of blobs into MinkowskiEngine sparse tensor.
    :param batch: torch tensor
    :return: MinkowskiEngine sparse tensor
    """
    coordinates = torch.nonzero(batch).int()
    features = []
    for idx in coordinates:
        features.append(batch[tuple(idx)])
    features = torch.tensor(features).unsqueeze(-1)
    coordinates, features = ME.utils.sparse_collate([coordinates], [features])
    return ME.SparseTensor(features=features, coordinates=coordinates)

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

    
