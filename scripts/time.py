from datetime import datetime

import MinkowskiEngine as ME
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import LigandDataset, collation_fn
from src.models.TransLoc3D import create_model, model_cfg, model_type, Config

if __name__ == "__main__":
    device = torch.device("cpu")
    dataset_path = "../data/eff_test.csv"
    dataset = LigandDataset("data", dataset_path, max_blob_size=2000)

    cfg = Config(model_cfg)
    cfg.pool_cfg.out_channels = 219
    model = create_model(model_type, cfg)
    model.load_state_dict(torch.load("../best.pt"))

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=collation_fn,
        num_workers=1,
        shuffle=False
    )

    df = pd.read_csv(dataset_path)

    result = []
    model.eval()
    with torch.no_grad():
        for r in range(10):
            for idx, ((coords, feats, _), df_row) in enumerate(zip(dataloader, df.iterrows())):
                start_time = datetime.now()
                batch = ME.SparseTensor(feats, coords, device=device)
                preds = model(batch)
                inference_time = datetime.now() - start_time
                result.append([df_row[1]["Unnamed: 0"], df_row[1]["wout_0_n"], df_row[1]["ligand"], r,
                               inference_time.total_seconds() * 1000])

    pd.DataFrame(result, columns=["id", "ligand", "wout_0_n", "run", "inference_time (ms)"]).to_csv("times.csv",
                                                                                                    index=False)
