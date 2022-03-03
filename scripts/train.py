import gc

import torch
import MinkowskiEngine as ME
from torch.utils.data import DataLoader

from src.models.TransLoc3D import create_model, model_cfg, model_type, Config
from src import log
from src.data import LigandDataset, dataset_split, collation_fn

if __name__ == "__main__":
    # ======================================================================
    # INITIAL CONFIG
    dataset_path = "../data/cmb_blob_labels.csv"
    batch_size = 64
    no_workers = 8
    epochs = 100
    weight_decay = 1e-4
    lr = 1e-3
    accum_iter = 4
    # MODEL CONFIG LATER IN CODE
    # ======================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")
    device = cpu

    run = log.get_run()

    dataset = LigandDataset("data", dataset_path, max_blob_size=2000)

    train, test = dataset_split(dataset=dataset)

    train_dataloader = DataLoader(
        dataset=train,
        batch_size=batch_size,
        collate_fn=collation_fn,
        num_workers=no_workers,
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test,
        batch_size=batch_size,
        collate_fn=collation_fn,
        num_workers=no_workers,
        shuffle=True
    )

    # ======================================================================
    # MODEL AND OPTIMIZER CONFIG
    # TRANSLOC3D CONFIGURATION (see transloc3d_cfg.py for information about options)
    cfg = Config(model_cfg)
    cfg.pool_cfg.out_channels = dataset.labels[0].shape[0]
    model = create_model(model_type, cfg)
    model.to(device)
    # SET OPTIMIZER
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
    # ======================================================================

    criterion = torch.nn.CrossEntropyLoss()

    log.config(
        run=run, model=model, criterion=criterion, optimizer=optimizer, dataset=dataset
    )

    for e in range(epochs):
        print(f"Current epoch {e}")
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

        scheduler.step()
        log.state_dict(model=model, epoch=e)
        log.epoch(run=run, preds=predictions, target=groundtruth, epoch_num=e)

    run.stop()
