import gc
import random
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

import log
import models
from cfg import read_config
from data import CoordsDataset, dataset_split

warnings.simplefilter("ignore")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    torch.manual_seed(23)
    torch.use_deterministic_algorithms(True)

    cfg = read_config("../cfg/train.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] != "cpu" else "cpu")
    cpu = torch.device("cpu")

    run = log.get_run()

    dataset = CoordsDataset(cfg["dataset_dir"], cfg["dataset_file"], min_size=cfg["dataset_min_size"],
                            max_size=cfg["dataset_max_size"])

    run["config/dataset/name"] = cfg["dataset_dir"].split("/")[-1]
    run["config/batch_accum"] = cfg["accum_iter"]
    run["config/batch_size"] = cfg["batch_size"]

    train, test = dataset_split(dataset=dataset)

    g_train, g_test = torch.Generator(), torch.Generator()
    g_train.manual_seed(42)
    g_test.manual_seed(42)
    train_dataloader = DataLoader(dataset=train, batch_size=cfg["batch_size"], num_workers=cfg["no_workers"],
                                  worker_init_fn=seed_worker, generator=g_train, shuffle=True)
    test_dataloader = DataLoader(dataset=test, batch_size=cfg["batch_size"], num_workers=cfg["no_workers"],
                                 worker_init_fn=seed_worker, generator=g_test, shuffle=True)

    model = models.create(cfg["model"])
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    log.config(run=run, model=model, criterion=criterion, optimizer=optimizer, dataset=dataset)

    accum_iter = cfg["accum_iter"]
    for e in range(cfg["epochs"]):
        train_dataloader.dataset.sample(e)
        model.train()
        for idx, (batch, labels) in enumerate(train_dataloader):
            labels = labels.to(device=device)
            batch = batch.to(device=device)
            labels_hat = model(batch)
            try:
                labels_hat = model(batch)
                loss = criterion(labels_hat, labels) / accum_iter
                loss.backward()
                del labels_hat

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
        model.eval()
        with torch.no_grad():
            groundtruth, predictions = None, None
            for idx, (batch, labels) in enumerate(test_dataloader):
                labels = labels.to(device=device)
                batch = batch.to(device=device)
                torch.cuda.empty_cache()
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

        scheduler.step()
        log.model(run=run, model=model, epoch=e)
        log.epoch(run=run, preds=predictions, target=groundtruth, epoch_num=e)

    run.stop()
