import os

from typing import List

import torch
from torch.utils.data import DataLoader

import MinkowskiEngine as ME

from utils import *
from simple_reader import LigandDataset

from MinkNet import MinkNet
from PoC import PoCMinkNet


import gc

if __name__ == "__main__":
    # ======================================================================
    # INITIAL CONFIG
    dataset_path = "data/labels_three.csv"
    batch_size = 16
    no_workers = 4
    epochs = 100
    # MODEL CONFIG LATER IN CODE
    # ======================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")

    run = neptune.init(
        project="LIGANDS/LIGANDS",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMGQ1ZDQwZS05YjhlLTRmMGUtYjZjZC0yYzk0OWE4OWJmYzkifQ==",
    )

    dataset = LigandDataset("data", dataset_path)
    train, test = dataset_split(dataset=dataset)
    train_dataloader = DataLoader(
        dataset=train,
        batch_size=batch_size,
        collate_fn=collation_fn,
        num_workers=no_workers,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        dataset=test,
        batch_size=batch_size,
        collate_fn=collation_fn,
        num_workers=no_workers,
        shuffle=True,
    )

    # ======================================================================
    # MODEL AND OPTIMIZER CONFIG
    # MODELS CONFIG
    modelMinkNet = MinkNet(
        conv_channels=[64, 64, 128, 128, 256, 256, 512, 512],
        in_channels=1,
        out_channels=dataset.labels[0].shape[0],
    )
    modelPoC = PoCMinkNet(in_channels=1, out_channels=dataset.labels[0].shape[0])
    # SET MODEL
    model = modelMinkNet
    model.to(device)
    # SET OPTIMIZER
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    # ======================================================================

    criterion = torch.nn.CrossEntropyLoss()

    log_config(
        run=run, model=model, criterion=criterion, optimizer=optimizer, dataset=dataset
    )

    for e in range(epochs):
        print(f"Current epoch {e}")
        model.train()
        for idx, (coords, feats, labels) in enumerate(train_dataloader):

            labels = labels.to(device=device)
            batch = ME.SparseTensor(feats, coords, device=device)

            # High fluctuations in batch size (up to 15x difference)
            # print(f"Batch shape: {batch.F.shape}")

            optimizer.zero_grad()
            labels_hat = model(batch)

            try:
                loss = criterion(labels_hat, labels)
                loss.backward()
                optimizer.step()
            except:
                pass

            if device == torch.device("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            del batch, labels, labels_hat
            gc.collect()

        print("You've reached eval")
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

        log_state_dict(run=run, model=model)
        log_epoch(run=run, preds=predictions, target=groundtruth)

    run.stop()
