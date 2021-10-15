import os

import numpy as np
import pandas as pd

from torch import tensor
from torch.utils.data import Dataset

class LigandDataset(Dataset):
    def __init__(self, annotations_file_path: str):
        """
        Parameters:
        annotations_file_path (str): path to the directory containing directory 'blobs_full' (which contains .npz files) and 'cmb_blob_labels.csv' (which contains columns 'ligands' and 'blob_map_file')
        """
        self.annotations_file_path = annotations_file_path
        file_ligand_map = pd.read_csv(
            os.path.join(self.annotations_file_path, 'cmb_blob_labels.csv'),
            usecols=['ligand', 'blob_map_filename']
        )
        self.file_ligand_map = file_ligand_map.set_index('blob_map_filename').to_dict()['ligand']
        self.files = list(self.file_ligand_map.keys())
        self.labels = list(self.file_ligand_map.values())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        idx = self.files[idx]
        blob_path = os.path.join(self.annotations_file_path, 'blobs_full', idx)
        blob = np.load(blob_path)['blob']
        blob = tensor(blob)
        label = self.file_ligand_map[idx]
        return (blob, label)

class DataLoader:
    def __init__(self, dataset: Dataset):
        """
        Parameters:
        dataset (Dataset): dataset to be loaded
        """
        self.iter = iter(dataset)

    def __iter__(self):
        return self.iter
    
if __name__ == '__main__':
    dataset = LigandDataset('data')
    dataloader = DataLoader(dataset)

    for idx, (blob, label) in enumerate(dataloader):
        if idx >= 100:
            break
        print(blob.shape, label)