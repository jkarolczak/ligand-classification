import os

import sys

import plotly.graph_objects as go
import numpy as np
import pandas as pd


class Blob:
    def __init__(self, blob_dir_path : str, blob_file_name : str):
        """
        Parameters:
        blob_dir_path (str): path to the directory containing directory 'blobs_full' (which contains .npz files) and 'cmb_blob_labels.csv' (which contains columns 'ligands' and 'blob_map_file')
        blob_file_name (str): name of the .npz file corresponding to visualised blob
        """
        self.blob_file_path = os.path.join(blob_dir_path, 'blobs_full', blob_file_name)
        blob_file = np.load(self.blob_file_path)
        self.blob = blob_file['blob']
        blob_file.close()

    def nonzero_max(self):
        return float(self.blob[self.blob > 0].max())

    def nonzero_min(self):
        return float(self.blob[self.blob > 0].min())

    def display_volume_3d(self):
        x, y, z = self.blob.shape
        max_dim = max(x, y, z)
        x_pad = (max_dim - x) / 2
        y_pad = (max_dim - y) / 2
        z_pad = (max_dim - z) / 2

        data = np.pad(self.blob, 
            (
                (int(np.ceil(x_pad)), int(np.floor(x_pad))),
                (int(np.ceil(y_pad)), int(np.floor(y_pad))),
                (int(np.ceil(z_pad)), int(np.floor(z_pad)))
            ), 'constant', constant_values = 0)

        X, Y, Z = np.mgrid[-1:1:max_dim*1j, -1:1:max_dim*1j, -1:1:max_dim*1j]


        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=data.flatten(),
            isomin=self.nonzero_min(),
            isomax=self.nonzero_max(),
            opacity=0.1,
            surface_count=25,
            colorscale='brbg'
            )
        )
        fig.show()



if __name__ == '__main__':
    blob_dir_path = sys.argv[1] if len(sys.argv) >= 2 else '../data'
    blob_file_name = sys.argv[2] if len(sys.argv) >= 3 else '1a0h_0G6_1_B_2.npz'

    test_blob = Blob(blob_dir_path, blob_file_name)
    test_blob.display_volume_3d()
    