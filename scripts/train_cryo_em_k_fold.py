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

FOLD_FILES = ["../data/fold0.csv", "../data/fold1.csv", "../data/fold2.csv"]
CRYOEM_DIR = "../data/blobs_cryoem+xray"

def seed_worker(_):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_training(cfg, train_files, test_file, fold_idx, device, run):
    # cryo-em
    classes_file_path = cfg["dataset_file"] or "../data/cryoem-classes.csv"
    train_datasets = [SparseDataset(CRYOEM_DIR, f, min_size=cfg["dataset_min_size"],
                                    max_size=cfg["dataset_max_size"], classes_file_path=classes_file_path)
                      for f in train_files]

    # x-ray
    if cfg["dataset_file"]:
        train_datasets.append(SparseDataset(cfg["dataset_dir"], cfg["dataset_file"], min_size=cfg["dataset_min_size"],
                              max_size=cfg["dataset_max_size"]))

    train_dataset = concatenate_sparse_datasets(train_datasets)

    test_dataset = SparseDataset(CRYOEM_DIR, test_file, min_size=cfg["dataset_min_size"],
                                 max_size=cfg["dataset_max_size"], classes_file_path=classes_file_path)

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

    all_epoch_groundtruth = []
    all_epoch_predictions = []

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
            except Exception as e:
                print(e)
                pass

            if device == torch.device("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            del (batch, labels)
            gc.collect()

        run[f"train/fold-{fold_idx}-as-test/{criterion}"].log(total_loss / len(train_dataloader))

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

        all_epoch_groundtruth.append(groundtruth)
        all_epoch_predictions.append(predictions)

        scheduler.step()

    return all_epoch_groundtruth, all_epoch_predictions


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

    all_epoch_groundtruth_folds = [[] for _ in range(cfg["epochs"])]
    all_epoch_predictions_folds = [[] for _ in range(cfg["epochs"])]

    for fold_idx in range(len(FOLD_FILES)):
        test_file = FOLD_FILES[fold_idx]
        train_files = [f for i, f in enumerate(FOLD_FILES) if i != fold_idx]

        epoch_groundtruth, epoch_predictions = run_training(cfg, train_files, test_file, fold_idx, device, run)

        for e in range(cfg["epochs"]):
            all_epoch_groundtruth_folds[e].append(epoch_groundtruth[e])
            all_epoch_predictions_folds[e].append(epoch_predictions[e])

    for e in range(cfg["epochs"]):
        aggregated_groundtruth_epoch = torch.cat(all_epoch_groundtruth_folds[e], dim=0)
        aggregated_predictions_epoch = torch.cat(all_epoch_predictions_folds[e], dim=0)

        log.epoch(
            run=run,
            preds=aggregated_predictions_epoch,
            target=aggregated_groundtruth_epoch,
            epoch_num=e,
            model_name=cfg["model"]
        )

    run.stop()
