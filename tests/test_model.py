# import torch
# import pytest
# import MinkowskiEngine as ME
# 
# import models
# from cfg import read_config
# from data import LigandDataset
# 
# 
# @pytest.fixture
# def batch():
#     x = torch.ones((4, 8, 8, 8), dtype=torch.float32)
#     coords, feats = LigandDataset._get_coords_feats(x)
#     return ME.SparseTensor(feats, coords.contiguous())
# 
# 
# def test_softmax(batch):
#     cfg = read_config("../cfg/train.yaml")
#     model = models.create(cfg["model"])
#     y_hat = model(batch)
#     assert y_hat.sum(-1).mean() == 2
