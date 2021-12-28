import os
from random import choices

import numpy as np
import pandas as pd

import torch
from torch import tensor
from torch.utils.data import Dataset

import MinkowskiEngine as ME

from sklearn.preprocessing import LabelBinarizer


class LigandDataset(Dataset):
    """A class to represent a ligands dataset."""

    def __init__(
        self, 
        annotations_file_path: str, 
        labels_file_path: str = None,
        max_blob_size: int = None
    ):
        """
        :param annotations_file_path: path to the directory containing directory
        'blobs_full' (which contains .npz files)
        :param labels_file_path: string with path to the file containing csv definition
        of the dataset, default '{annotations_file_path}/cmb_blob_labels.csv', this
        file has to contain columns 'ligands' and 'blob_map_file'
        """
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
        self.labels = list(self.file_ligand_map.values())
        self.encoder = LabelBinarizer()
        self.labels = self.encoder.fit_transform(self.labels)
        self.max_blob_size = max_blob_size

    def __get_coords_feats(self, batch: torch.Tensor) -> ME.SparseTensor:
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
        blob_path = os.path.join(self.annotations_file_path, "blobs_full", idx)
        blob = np.load(blob_path)["blob"]
        blob = tensor(blob, dtype=torch.float32)
        coordinates, features = self.__get_coords_feats(blob)
        blob_size = coordinates.shape[0]
        if self.max_blob_size and blob_size > self.max_blob_size:
            indices = choices(range(blob_size), k=self.max_blob_size)
            coordinates = coordinates[indices, :]
            features = features[indices, :]  
        features = (features - features.mean()) / features.std()
        return (coordinates, features, label)
