import collections
import os
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from random import seed
from typing import List, Tuple

import MinkowskiEngine as ME
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """A class to represent a ligands' dataset."""

    def __init__(
            self,
            annotations_file_path: str,
            labels_file_path: str = None,
            rng_seed: int = 23,
            min_size: int = None,
            max_size: int = None,
            normalize: bool = False
    ):
        """
        :param annotations_file_path: path to the directory containing directory
        'blobs_full' (which contains .npz files)
        :param labels_file_path: string with path to the file containing csv definition
        of the dataset, default '{annotations_file_path}/cmb_blob_labels.csv', this
        file has to contain columns 'ligands' and 'blob_map_file'
        :param max_size: maximal number of instances of each class present in the dataset
        :param min_size: minimal number of instances of each class present in the dataset
        :param normalize: whether to normalize the point cloud
        """
        seed(rng_seed)
        self.annotations_file_path = annotations_file_path
        if labels_file_path is None:
            labels_file_path = os.path.join(
                self.annotations_file_path, "cmb_blob_labels.csv"
            )
        file_ligand_map = pd.read_csv(
            labels_file_path, usecols=["ligand", "blob_map_filename", "local_near_cut_count_N",
                                       "local_near_cut_count_O", "local_near_cut_count_C"]
        )
        self.file_ligand_map = file_ligand_map.set_index("blob_map_filename").to_dict()["ligand"]
        self.file_near_map = file_ligand_map.set_index("blob_map_filename")
        self.file_near_map["near"] = self.file_near_map.apply(lambda x: [x["local_near_cut_count_N"],
                                                                         x["local_near_cut_count_O"],
                                                                         x["local_near_cut_count_C"]], axis=1)
        self.file_near_map = self.file_near_map.to_dict()["near"]
        self.files = list(self.file_ligand_map.keys())
        self.labels_names = list(self.file_ligand_map.values())
        self.near = list(self.file_near_map.values())
        self.encoder = LabelBinarizer()
        self.labels = self.encoder.fit_transform(self.labels_names)

        self.label_files_map = collections.defaultdict(list)
        for k, v in sorted(self.file_ligand_map.items()):
            self.label_files_map[v].append(k)

        self.min_size = min_size
        self.max_size = max_size
        self.normalize = normalize

    def sample(self, seed: int = None) -> None:
        """
        a utility method to create a dataset with maximum of 'max_size' instances of each class and minimum of
        'min_size' instances of each class

        :param seed: integer to be used as the random seed, by default it should simply be epoch number in the training
        loop
        """
        random.seed(seed)

        files = []
        labels = []

        # initialize to default values -> entire dataset
        label_files_map = deepcopy(self.label_files_map)

        # perform undersampling
        if self.max_size:
            for key in label_files_map:
                l = len(label_files_map[key])
                if l > self.max_size:
                    label_files_map[key] = random.sample(label_files_map[key], k=self.max_size)

        # perform oversampling
        if self.min_size:
            for key in label_files_map:
                l = self.min_size - len(label_files_map[key])
                if l > 0:
                    choices = random.choices(label_files_map[key], k=l)
                    label_files_map[key].extend(choices)

        for key, values in label_files_map.items():
            keys = [key for _ in range(len(values))]
            labels.extend(keys)
            files.extend(values)

        # pairwise shuffle of files and labels
        tmp = list(zip(files, labels))
        random.shuffle(tmp)
        self.files, self.labels = zip(*tmp)
        self.labels = self.encoder.transform(self.labels)

    def __len__(self):
        return len(self.files)

    @abstractmethod
    def __getitem__(self, idx):
        pass


class SparseDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_coords_feats(batch: torch.Tensor) -> ME.SparseTensor:
        coordinates = torch.nonzero(batch).int()
        features = []
        for idx in coordinates:
            features.append(batch[tuple(idx)])
        features = torch.tensor(features).unsqueeze(-1)
        return coordinates, features

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        near = torch.tensor(self.near[idx], dtype=torch.float32)
        idx = self.files[idx]
        blob_path = os.path.join(self.annotations_file_path, idx)
        blob = np.load(blob_path)["blob"]
        blob = torch.tensor(blob, dtype=torch.float32)
        coordinates, features = self._get_coords_feats(blob)
        return coordinates, features, near, label


class CoordsDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self.num_points = 1
        super().__init__(*args, **kwargs)

    @staticmethod
    def _pc_normalize(pc: np.array) -> np.array:
        centroid = torch.mean(pc, axis=0)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        idx = self.files[idx]
        blob_path = os.path.join(self.annotations_file_path, idx)
        blob = np.load(blob_path)["blob"]
        blob = torch.tensor(blob, dtype=torch.float32)
        coordinates = torch.nonzero(blob).float()
        if self.normalize:
            coordinates = self._pc_normalize(coordinates)
        return coordinates, near, label


class RiconvDataset(CoordsDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        idx = self.files[idx]
        blob_path = os.path.join(self.annotations_file_path, idx)
        pcd = np.load(blob_path)['blob']
        pcd = torch.tensor(pcd, dtype=torch.float32)
        if self.normalize:
            pcd[:, :3] = self._pc_normalize(pcd[:, :3])
        return pcd, label


def collation_fn_contiguous(blobel: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implements collation function for batching torch tensors and adding padding (-1, -1, -1), where needed, to make
    the CoordsDataset working with blobels of different size and `torch.utils.data.DataLoader`.

    :param blobel: tuple of blo(b) + (la)bel => blobel
    :type blobel: List[Tuple[torch.Tensor, torch.Tensor]]
    """
    np.random.seed(23)
    coordinates = []
    labels = []
    max_len = 0
    for c, l in blobel:
        max_len = max(max_len, len(c))
        coordinates.append(c)
        labels.append(l)
    for idx, c in enumerate(coordinates):
        choose = np.random.choice(len(c), max_len, replace=True)
        coordinates[idx] = coordinates[idx][choose]

    coordinates = torch.stack(coordinates)
    labels = torch.stack(labels)
    return coordinates, labels


def collation_fn_sparse(blobel):
    """
    Implements collation function for batching the LigandsDataset using
    `torch.utils.data.DataLoader`

    :param blobel: tuple (coordinates, features, labels); blo(b)+(la)bel => blobel; all credit to Witek T.
    """
    coords_batch, feats_batch, near_batch, labels_batch = [], [], [], []

    for (coords, feats, near, label) in blobel:
        coords_batch.append(coords)
        feats_batch.append(feats)
        near_batch.append(near)
        labels_batch.append(label)

    coords_batch = ME.utils.batched_coordinates(coords_batch)
    feats_batch = torch.tensor(np.concatenate(feats_batch, 0), dtype=torch.float32)
    near_batch = torch.tensor(np.vstack(near_batch), dtype=torch.float32)
    labels_batch = torch.tensor(np.vstack(labels_batch), dtype=torch.float32)

    return coords_batch, feats_batch, near_batch, labels_batch


def dataset_split(
        dataset: BaseDataset, train_size: float = 0.75, stratify: bool = True
) -> Tuple[BaseDataset, BaseDataset]:
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
    train.labels_names = train.encoder.inverse_transform(train.labels)

    train_label_files_map = collections.defaultdict(list)
    for k, v in zip(train.files, train.labels_names):
        train_label_files_map[v].append(k)
    train.label_files_map = train_label_files_map

    test.files, test.labels = files_test, labels_test
    test.labels_names = test.encoder.inverse_transform(test.labels)

    test_label_files_map = collections.defaultdict(list)
    for k, v in zip(test.files, test.labels_names):
        test_label_files_map[v].append(k)
    test.label_files_map = test_label_files_map

    return train, test
