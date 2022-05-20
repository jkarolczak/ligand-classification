import gc

import torch
import MinkowskiEngine as ME
from torch.utils.data import DataLoader

import models
import log
from cfg import read_config
from data import LigandDataset, dataset_split, collation_fn

if __name__ == "__main__":
    torch.manual_seed(23)

    cfg = read_config("../cfg/train.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] != "cpu" else "cpu")
    cpu = torch.device("cpu")

    run = log.get_run()

    dataset = LigandDataset(cfg["dataset_dir"], cfg["dataset_file"], min_size=cfg["dataset_min_size"],
                            max_size=cfg["dataset_max_size"])

    run["config/dataset/name"] = cfg["dataset_dir"].split("/")[-1]
    run["config/batch_accum"] = cfg["accum_iter"]
    run["config/batch_size"] = cfg["batch_size"]

    train, test = dataset_split(dataset=dataset)

    train_dataloader = DataLoader(dataset=train, batch_size=cfg["batch_size"], collate_fn=collation_fn,
                                  num_workers=cfg["no_workers"], shuffle=True)
    test_dataloader = DataLoader(dataset=test, batch_size=cfg["batch_size"], collate_fn=collation_fn,
                                 num_workers=cfg["no_workers"], shuffle=True)

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
        for idx, (coords, feats, labels) in enumerate(train_dataloader):
            labels = labels.to(device=device)
            batch = ME.SparseTensor(feats, coords, device=device)
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
            for idx, (coords, feats, labels) in enumerate(test_dataloader):
                torch.cuda.empty_cache()

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

        scheduler.step()
        log.state_dict(model=model, epoch=e)
        log.epoch(run=run, preds=predictions, target=groundtruth, epoch_num=e)

    run.stop()
