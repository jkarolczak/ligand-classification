import torch
from torch.utils import data
from torch.utils.data import DataLoader

import MinkowskiEngine as ME

from utils import *
from simple_reader import LigandDataset

from TransLoc3D import create_model
from TransLoc3D.transloc3d_cfg import model_cfg, model_type
from TransLoc3D.utils_config import Config

if __name__ == "__main__":
    device = torch.device("cpu")
    
    dataset_path = "data/cmb_blob_labels.csv"
    dataset = LigandDataset("data", dataset_path, max_blob_size=10000)
    
    cfg = Config(model_cfg)
    cfg.pool_cfg.out_channels = dataset.labels[0].shape[0]
    model = create_model(model_type, cfg)
    model.load_state_dict(torch.load('src/best.pt'))
    
    torch.load('src/best.pt', map_location=torch.device('cpu'))
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=collation_fn,
        num_workers=1,
        shuffle=True
    )
    
    model.eval()
    with open('results.csv', 'w') as fp:
        labels = ['groundtruth']
        for i in range(219):
            bin = np.array([[0] * 219])
            bin[0, i] = 1
            labels.append(dataset.encoder.inverse_transform(bin)[0])
        
        fp.write(str(labels)[1:-1] + '\n')    
        with torch.no_grad():
            for idx, (coords, feats, labels) in enumerate(dataloader):
                batch = ME.SparseTensor(feats, coords, device=device)
                preds = model(batch)
                line = str(dataset.encoder.inverse_transform(labels.numpy())[0])
                line += ', '
                line += str(preds.numpy().tolist()[0])[1:-1]
                
                fp.write(line + '\n')