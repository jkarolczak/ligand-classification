import gc
import random

import MinkowskiEngine as ME
import numpy as np
import torch
from torch.utils.data import DataLoader

import log
import models
from cfg import read_config
from data import SparseDataset, collation_fn_sparse, concatenate_sparse_datasets

FOLD_FILES = ["../data/fold1.csv", "../data/fold2.csv", "../data/fold3.csv"]


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_training(cfg, train_files, test_file, fold_idx, device, run):
    train_datasets = [SparseDataset(cfg["dataset_dir"], f, min_size=cfg["dataset_min_size"],
                                    max_size=cfg["dataset_max_size"]) for f in train_files]
    train_dataset = concatenate_sparse_datasets(train_datasets)

    test_dataset = SparseDataset(cfg["dataset_dir"], test_file, min_size=cfg["dataset_min_size"],
                                 max_size=cfg["dataset_max_size"])

    g_train, g_test = torch.Generator(), torch.Generator()
    g_train.manual_seed(42)
    g_test.manual_seed(42)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg["batch_size"], collate_fn=collation_fn_sparse,
                                  num_workers=cfg["no_workers"], worker_init_fn=seed_worker, generator=g_train,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=cfg["batch_size"], collate_fn=collation_fn_sparse,
                                 num_workers=cfg["no_workers"], worker_init_fn=seed_worker, generator=g_test,
                                 shuffle=False)

    model = models.create(cfg["model"])
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    accum_iter = cfg["accum_iter"]
    for e in range(cfg["epochs"]):
        train_dataloader.dataset.sample(e)
        model.train()
        total_loss = 0
        for idx, (coords, feats, labels) in enumerate(train_dataloader):
            labels = labels.to(device=device)
            batch = ME.SparseTensor(feats, coords, device=device)
            try:
                labels_hat = model(batch)
                loss = criterion(labels_hat, labels) / accum_iter
                loss.backward()
                del labels_hat

                total_loss += loss.item()

                if not idx % accum_iter:
                    optimizer.step()
                    optimizer.zero_grad()
            except:
                pass

            if device == torch.device("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            del (batch, labels)
            gc.collect()

        run[f"train/fold-{fold_idx}/{criterion}"].log(total_loss / len(train_dataloader))

        model.eval()
        groundtruth, predictions = None, None
        with torch.no_grad():
            for idx, (coords, feats, labels) in enumerate(test_dataloader):
                torch.cuda.empty_cache()

                batch = ME.SparseTensor(feats, coords, device=device)
                preds = model(batch)

                labels = labels.cpu()
                preds = preds.cpu()

                if groundtruth is None:
                    groundtruth = labels
                    predictions = preds
                else:
                    try:
                        groundtruth = torch.cat([groundtruth, labels], 0)
                        predictions = torch.cat([predictions, preds], 0)
                    except:
                        pass

        scheduler.step()

    return groundtruth, predictions


if __name__ == "__main__":
    torch.manual_seed(23)
    torch.use_deterministic_algorithms(True)

    cfg = read_config("../cfg/train.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] != "cpu" else "cpu")

    run = log.get_run()

    run["config/dataset/name"] = cfg["dataset_dir"].split("/")[-1]
    run["config/batch_accum"] = cfg["accum_iter"]
    run["config/batch_size"] = cfg["batch_size"]
    run["config/epochs"] = cfg["epochs"]
    run["config/k_folds"] = len(FOLD_FILES)

    all_groundtruth = []
    all_predictions = []

    for fold_idx in range(len(FOLD_FILES)):
        test_file = FOLD_FILES[fold_idx]
        train_files = [f for i, f in enumerate(FOLD_FILES) if i != fold_idx]

        groundtruth, predictions = run_training(cfg, train_files, test_file, fold_idx, device, run)

        all_groundtruth.append(groundtruth)
        all_predictions.append(predictions)

    aggregated_groundtruth = torch.cat(all_groundtruth, dim=0)
    aggregated_predictions = torch.cat(all_predictions, dim=0)

    log.epoch(run=run, preds=aggregated_predictions, target=aggregated_groundtruth, epoch_num=cfg["epochs"] - 1,
              model_name=cfg["model"])

    run.stop()
