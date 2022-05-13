import os
import random
from copy import deepcopy
from random import seed
from typing import Tuple

import MinkowskiEngine as ME
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch import tensor
from torch.utils.data import Dataset, DataLoader


class LigandDataset(Dataset):
    """A class to represent a ligands' dataset."""

    def __init__(
            self,
            annotations_file_path: str,
            labels_file_path: str = None,
            rng_seed: int = 23,
            sample_size: int = None
    ):
        """
        :param annotations_file_path: path to the directory containing directory
        'blobs_full' (which contains .npz files)
        :param labels_file_path: string with path to the file containing csv definition
        of the dataset, default '{annotations_file_path}/cmb_blob_labels.csv', this
        file has to contain columns 'ligands' and 'blob_map_file'
        :param sample_size: maximal number of instances of each class present in the dataset
        """
        seed(rng_seed)
        self.annotations_file_path = annotations_file_path
        if labels_file_path is None:
            labels_file_path = os.path.join(
                self.annotations_file_path, "cmb_blob_labels.csv"
            )
        file_ligand_map = pd.read_csv(
            labels_file_path, usecols=["ligand", "blob_map_filename"]
        )
        self.file_ligand_map = file_ligand_map.set_index("blob_map_filename").to_dict()[
            "ligand"
        ]
        self.files = list(self.file_ligand_map.keys())
        self.labels_names = list(self.file_ligand_map.values())
        self.encoder = LabelBinarizer()
        self.labels = self.encoder.fit_transform(self.labels_names)
        """
        If specified, limit the maximum number of instances of each class up to sample_size
        """
        if sample_size:
            self.sample_size = sample_size
            self.class_counter = None
            # random seed that surely won't be encountered during training (no such epoch number)
            self.undersample(2137)

    def undersample(self, seed: int = None) -> None:
        """
        a utility method to create a dataset with maximum of 'sample_size' instances of each class

        :param seed: integer to be used as the random seed, by default it should simply be epoch number in the training
        loop
        """
        random.seed(seed)

        # initialize to default values -> entire dataset
        self.files = list(self.file_ligand_map.keys())
        self.labels = self.encoder.fit_transform(self.labels_names)

        files = []
        labels = []

        # pairwise shuffle of files and labels
        tmp = list(zip(self.files, self.labels))
        random.shuffle(tmp)
        self.files, self.labels = zip(*tmp)

        # initialize a dictionary with key = class, value = number of instances
        self.class_counter = dict(zip(list(self.encoder.classes_), [0 for _ in range(len(self.encoder.classes_))]))
        for file, class_name in zip(self.files, self.labels_names):
            if self.class_counter[class_name] < self.sample_size:
                self.class_counter[class_name] += 1
                files.append(file)
                labels.append(class_name)

        self.files = files
        self.labels = self.encoder.transform(labels)

    @staticmethod
    def _get_coords_feats(batch: torch.Tensor) -> ME.SparseTensor:
        coordinates = torch.nonzero(batch).int()
        features = []
        for idx in coordinates:
            features.append(batch[tuple(idx)])
        features = torch.tensor(features).unsqueeze(-1)
        return coordinates, features

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        idx = self.files[idx]
        blob_path = os.path.join(self.annotations_file_path, idx)
        blob = np.load(blob_path)["blob"]
        blob = tensor(blob, dtype=torch.float32)
        coordinates, features = self._get_coords_feats(blob)
        return coordinates, features, label


def collation_fn(blobel):
    """
    Implements collation function for batching the LigandsDataset using
    `torch.utils.data.DataLoader`

    :param blobel: tuple (coordinates, features, labels); blo(b)+(la)bel => blobel; all credit to Witek T.
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
        files, labels, train_size=train_size, stratify=stratify, random_state=23
    )

    train = deepcopy(dataset)
    test = deepcopy(dataset)
    train.files, train.labels = files_train, labels_train
    test.files, test.labels = files_test, labels_test

    return train, test
