from datetime import datetime

import MinkowskiEngine as ME
import pandas as pd
import torch
from torch.utils.data import DataLoader

import models
from cfg import read_config
from data import SparseDataset, collation_fn

if __name__ == "__main__":
    cfg = read_config("../cfg/time.yaml")

    device = torch.device("cpu")
    dataset_path = "../data/eff_test.csv"
    dataset = SparseDataset(cfg["dataset_dir"], cfg["dataset_file"], max_blob_size=2000)

    model = models.create(cfg["model"])
    model.load_state_dict(torch.load("../best.pt"))
    model.to(device)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg["batch_size"],
        collate_fn=collation_fn,
        num_workers=cfg["no_workers"],
        shuffle=False
    )

    df = pd.read_csv(dataset_path)

    result = []
    model.eval()
    with torch.no_grad():
        for r in range(cfg["iterations"]):
            for idx, ((coords, feats, _), df_row) in enumerate(zip(dataloader, df.iterrows())):
                start_time = datetime.now()
                batch = ME.SparseTensor(feats, coords, device=device)
                preds = model(batch)
                inference_time = datetime.now() - start_time
                result.append([df_row[1]["Unnamed: 0"], df_row[1]["wout_0_n"], df_row[1]["ligand"], r,
                               inference_time.total_seconds() * 1000])

    pd.DataFrame(result, columns=["id", "ligand", "wout_0_n", "run", "inference_time (ms)"]).to_csv("times.csv",
                                                                                                    index=False)
