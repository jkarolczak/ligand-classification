from random import randrange

import MinkowskiEngine as ME
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import models
from cfg import read_config
from data import SparseDataset, collation_fn_sparse


if __name__ == "__main__":
    cfg = read_config("../cfg/embeddings.yaml")

    rng_seed = randrange(1000)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] != "cpu" else "cpu")
    cpu = torch.device("cpu")

    dataset = SparseDataset(cfg["dataset_dir"], cfg["dataset_file"], rng_seed=rng_seed)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg["batch_size"],
        collate_fn=collation_fn_sparse,
        num_workers=cfg["no_workers"],
        shuffle=False,
    )

    model = models.create(cfg["model_name"])
    state_dict = torch.load("../model.pt", map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # if someone wants to use some of the intermediate features, they can use this hook - however not recommended as earlier layers produce sparse tensor
    # intermediate_features_storage = {}
    # def feature_extraction_hook(module, input_tensors, output_tensor):
    #     """Hook to capture the output of model.backbone."""
    #     if isinstance(output_tensor, ME.SparseTensor):
    #         intermediate_features_storage['backbone_output_features'] = output_tensor.F.detach().cpu()
    #     elif isinstance(output_tensor, torch.Tensor):
    #         intermediate_features_storage['backbone_output_features'] = output_tensor.detach().cpu()
    #     else:
    #         # Handle other types if necessary (e.g., tuples of tensors)
    #         print(f"Warning: Hooked output is of unexpected type: {type(output_tensor)}")
    #         intermediate_features_storage['backbone_output_features'] = output_tensor  # Store as is
    #
    # target_layer = model.backbone
    # hook_handle = target_layer.register_forward_hook(feature_extraction_hook)
    # print(f"Hook registered on 'model.backbone' ({type(target_layer).__name__}).")

    embeddings = None
    with torch.no_grad():
        for idx, (coords, feats, labels) in enumerate(dataloader):
            print(idx, len(dataloader))
            batch = ME.SparseTensor(feats, coords, device=device)
            preds = model(batch)
            labels = labels.to(cpu)
            preds = preds.to(cpu)
            if embeddings is not None:
                embeddings = np.concatenate((embeddings, preds.numpy()), axis=0)
            else:
                embeddings = preds.numpy()
            # if 'backbone_output_features' in intermediate_features_storage:
            #     current_backbone_features = intermediate_features_storage['backbone_output_features']
    with open(f"{cfg['output_filename']}", "wb") as f:
        np.save(f, embeddings)




