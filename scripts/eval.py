from random import randrange

import MinkowskiEngine as ME
import pandas as pd
import torch
from torch.utils.data import DataLoader

import log
import models
from cfg import read_config
from data import SparseDataset, collation_fn_sparse
from log import get_run, epoch

if __name__ == "__main__":
    cfg = read_config("../cfg/eval.yaml")

    rng_seed = randrange(1000)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] != "cpu" else "cpu")
    cpu = torch.device("cpu")

    run = get_run(tags=["holdout"])
    run["seed"] = rng_seed

    dataset = SparseDataset(cfg["dataset_dir"], cfg["dataset_file"], rng_seed=rng_seed)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg["batch_size"],
        collate_fn=collation_fn_sparse,
        num_workers=cfg["no_workers"],
        shuffle=False,
    )

    run["config"] = cfg

    model = models.create(cfg["model_name"])
    state_dict = log.fetch_state_dict(cfg["model_name"], cfg["model_run_id"], cfg["model_epoch"])
    model.load_state_dict(state_dict)
    model.to(device)

    result_labels = []
    result_predictions = []

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
                groundtruth = torch.cat([groundtruth, labels], 0)
                predictions = torch.cat([predictions, preds], 0)

            preds_encoded = torch.zeros_like(preds)
            preds_encoded[list(range(preds.shape[0])), preds.max(axis=1).indices] = 1
            result_labels.extend(dataset.encoder.inverse_transform(labels))
            result_predictions.extend(dataset.encoder.inverse_transform(preds_encoded))

        epoch(run=run, preds=predictions, target=groundtruth, epoch_num=0, model_name=cfg["model_name"])
        run['seed'] = rng_seed

    df = pd.DataFrame({'id': dataset.files, 'labels': result_labels, 'predictions': result_predictions})
    df.to_csv("predictions.csv")

    run.stop()
