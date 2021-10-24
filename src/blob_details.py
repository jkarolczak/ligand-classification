import os

import sys

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import torch


class BlobDetails:
    def __init__(self, blob_dir_path : str, blob_file_name : str):
        """
        Parameters:
        blob_dir_path (str): path to the directory containing directory 'blobs_full' (which contains .npz files) and 'cmb_blob_labels.csv' (which contains columns 'ligands' and 'blob_map_file')
        blob_file_name (str): name of the .npz file corresponding to visualised blob
        """

        self.blob_dir_path = blob_dir_path
        self.blob_file_name = blob_file_name
        self.blob_file_path = os.path.join(blob_dir_path, 'blobs_full', blob_file_name)

        blob_file = np.load(self.blob_file_path)
        self.blob_name = blob_file_name[:-4]
        self.blob = torch.tensor(blob_file['blob'])

        file_ligand_map = pd.read_csv(
            os.path.join(self.blob_dir_path, 'cmb_blob_labels.csv'),
            usecols=['ligand', 'blob_map_filename']
        ).set_index('blob_map_filename')
        self.label = file_ligand_map.loc[blob_file_name][0]
        blob_file.close()

    def get_stats(self):
        stats = []
        labels = [
            'blob_name',
            'label',
            'blob_shape',
            'blob_n',
            'nonzero_n',
            'nonzero_%',        
            'nonzero_min',
            'nonzero_1_qrtl',
            'nonzero_mean',
            'nonzero_3_qrtl',
            'nonzero_max',
            'nonzero_sum',
            'nonzero_median',
            'nonzero_std',
            'nonzero_skewness',
            'nonzero_kurtosis',
            'nonzero_zscore_2_n',
            'nonzero_zscore_2_%',
            'nonzero_zscore_3_n',
            'nonzero_zscore_3_%'
        ]

        nonzero = self.blob[self.blob > 0]
        nonzero_n = nonzero.shape[0]                        # nonzero_n
        blob_n = self.blob.flatten().shape[0]                    # blob_n
        nonzero_mean = float(nonzero.mean())                # nonzero_mean
        nonzero_std = float(nonzero.std())                  # nonzero_std
        nonzero_1_qrtl = float(nonzero.quantile(0.25))      # nonzero_1_qrtl
        nonzero_3_qrtl = float(nonzero.quantile(0.75))      # nonzero_3_qrtl

        diffs = nonzero - nonzero_mean
        zscores = diffs / nonzero_std

        nonzero_zscore_2 = zscores[zscores > 2.0].shape[0]      # nonzero_zscore_2_n
        nonzero_zscore_3 = zscores[zscores > 3.0].shape[0]      # nonzero_zscore_3_n

        nonzero_skewness = float(torch.pow(zscores, 3.0).mean())        # nonzero_skewness
        nonzero_kurtosis = float(torch.pow(zscores, 4.0).mean() - 3.0)  # nonzero_kurtosis

        stats += [
            self.blob_name,
            self.label,                     # label
            list(self.blob.shape),               # blob_shape
            blob_n,                         # blob_n
            nonzero_n,                      # nonzero_n
            nonzero_n / blob_n,             # nonzero_%
            float(nonzero.min()),           # nonzero_min
            nonzero_1_qrtl,                 # nonzero_1_qrtl
            nonzero_mean,                   # nonzero_mean
            nonzero_3_qrtl,                 # nonzero_3_qrtl
            float(nonzero.max()),           # nonzero_max
            float(nonzero.sum()),           # nonzero_sum         
            float(nonzero.median()),        # nonzero_median
            nonzero_std,                    # nonzero_std
            nonzero_skewness,               # nonzero_skewness
            nonzero_kurtosis,               # nonzero_kurtosis
            nonzero_zscore_2,               # nonzero_zscore_2_n
            nonzero_zscore_2 / nonzero_n,   # nonzero_zscore_2_%
            nonzero_zscore_3,               # nonzero_zscore_3_n
            nonzero_zscore_3 / nonzero_n,   # nonzero_zscore_3_%
        ]

        return dict(zip(labels, stats))

    def plot_volume_3d(self, title, opacity = 0.1, surface_count = 15, colorscale = 'brbg'):
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
            isomin=float(self.blob[self.blob > 0].min()),
            isomax=float(self.blob[self.blob > 0].max()),
            opacity=opacity,
            surface_count=surface_count,
            colorscale=colorscale
            )
        )

        fig.update_layout(
            title={
                'text': title,
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})


        return fig



if __name__ == '__main__':
    blob_dir_path = sys.argv[1] if len(sys.argv) >= 2 else '../data'
    blob_file_name = sys.argv[2] if len(sys.argv) >= 3 else '1a0h_0G6_1_B_2.npz'

    test_blob = BlobDetails(blob_dir_path, blob_file_name)
    test_blob.plot_volume_3d().show()
    print(test_blob.get_stats())
    