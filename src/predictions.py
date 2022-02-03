import gc
import pickle

import pandas as pd
import torch
import MinkowskiEngine as ME
from torch.utils.data import DataLoader

from TransLoc3D import create_model
from TransLoc3D.transloc3d_cfg import model_cfg, model_type
from TransLoc3D.utils_config import Config
from utils.utils import *
from utils.simple_reader import LigandDataset

if __name__ == "__main__":
    dataset_path = "data/holdout.csv"
    model_path = "logs/models/2022-01-10-11:12:01.845989-epoch-13.pt"
    batch_size = 64
    no_workers = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")
    device = cpu

    run = neptune.init(
        project="LIGANDS/LIGANDS",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMGQ1ZDQwZS05YjhlLTRmMGUtYjZjZC0yYzk0OWE4OWJmYzkifQ==",
        tags=['holdout']
    )

    dataset = LigandDataset("data", dataset_path, max_blob_size=2000)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collation_fn,
        num_workers=no_workers,
        shuffle=True,
    )

    with open("encoder.pkl", "wb") as fp:
        pickle.dump(dataset.encoder, fp)

    cfg = Config(model_cfg)
    cfg.pool_cfg.out_channels = dataset.labels[0].shape[0]
    model = create_model(model_type, cfg)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    result_labels = []
    result_predicitons = []

    model.eval()
    with torch.no_grad():
        groundtruth, predictions = None, None
        for idx, (coords, feats, labels) in enumerate(dataloader):
            batch = ME.SparseTensor(feats, coords, device=device)
            preds = model(batch)
            labels = labels.to(cpu)
            preds = preds.to(cpu)

            if groundtruth is None:
                groundtruth = labels
                predictions = preds
            else:
                try:
                    groundtruth = torch.cat([groundtruth, labels], 0)
                    predictions = torch.cat([predictions, preds], 0)
                except:
                    pass
            preds_encoded = torch.zeros_like(preds)
            preds_encoded[list(range(preds.shape[0])), preds.max(axis=1).indices] = 1
            result_labels.extend(dataset.encoder.inverse_transform(labels))
            result_predicitons.extend(dataset.encoder.inverse_transform(preds_encoded))

        log_epoch(run=run, preds=predictions, target=groundtruth, epoch=0)

    df = pd.DataFrame({'id': dataset.files, 'labels': result_labels, 'predictions': result_predicitons})
    df.to_csv("predicitons.csv")

    run.stop()
